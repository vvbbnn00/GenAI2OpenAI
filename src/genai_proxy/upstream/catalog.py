import json
import os
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from genai_proxy.errors import ProxyError
from genai_proxy.logging_utils import safe_log_code
from genai_proxy.retry import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF,
    TRANSIENT_UPSTREAM_ERROR_CODE,
    is_retryable_business_error,
    is_retryable_status,
    schedule_retry,
    transient_upstream_error,
)
from genai_proxy.upstream.auth import is_genai_auth_failure

GENAI_MODEL_LIST_URL = (
    "https://genai.shanghaitech.edu.cn/htk/ai/aiModel/list"
    "?_t={timestamp}&pageNo=1&pageSize=999&showStatusList=2,3"
)
DEFAULT_MODEL = "GPT-4.1"
EXCLUDED_MODEL_IDS = {"gpt-image-1.5"}
MODEL_CACHE_TTL = 300
MODEL_REFRESH_FAILURE_COOLDOWN = 30
MODEL_CACHE_VERSION = 1
DEFAULT_FALLBACK_MODEL_IDS = (
    DEFAULT_MODEL,
    "deepseek-chat",
    "deepseek-pro",
    "chatglm",
    "qwen-instruct",
    "kimi-k3",
)


class ModelManager:
    def __init__(
        self,
        logger,
        token_manager,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        cache_path: str | None = None,
        fallback_model_ids=(),
    ):
        self._logger = logger
        self._token_manager = token_manager
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff = max(0.0, float(retry_backoff))
        self._models_cache_path = Path(cache_path).expanduser() if cache_path else None
        self._models_cache_lock = threading.Lock()
        self._models_refresh_lock = threading.Lock()
        self._models_refreshing = False
        self._models_refresh_after = 0.0
        self._fallback_models = _fallback_model_records(
            [
                *DEFAULT_FALLBACK_MODEL_IDS,
                *(fallback_model_ids or ()),
            ]
        )

        cached = self._load_persistent_cache()
        if cached is None:
            self._models_cache = list(self._fallback_models)
            self._models_cache_at = 0.0
        else:
            self._models_cache, self._models_cache_at = cached

    def resolve_model(self, model: str) -> str:
        return self.resolve_model_record(model)[0]

    def resolve_model_record(self, model: str) -> tuple[str, dict | None]:
        """Resolve an ID and its record from one stable catalog snapshot."""
        requested = model or DEFAULT_MODEL
        requested_key = requested.casefold()
        for record in self.list_genai_models():
            ai_type = record.get("aiType")
            if isinstance(ai_type, str) and ai_type.casefold() == requested_key:
                return ai_type, record
        return requested, None

    def root_ai_type_for(self, model: str) -> str:
        record = self.get_model_record(model)
        if record and record.get("rootAiType"):
            return record["rootAiType"]

        lowered = (model or "").lower()
        if lowered.startswith(("gpt-", "o1", "o3", "o4")):
            return "azure"
        return "xinference"

    def get_model_record(self, model: str):
        model_key = (model or "").casefold()
        for record in self.list_genai_models():
            ai_type = record.get("aiType")
            if isinstance(ai_type, str) and ai_type.casefold() == model_key:
                return record
        return None

    def list_openai_models(self) -> list[dict]:
        models = []
        for record in self.list_genai_models():
            models.append(
                {
                    "id": record["aiType"],
                    "object": "model",
                    "created": _parse_created_timestamp(record.get("createTime")),
                    "owned_by": _fallback_owner(record),
                }
            )
        return models

    def list_genai_models(self, force_refresh: bool = False) -> list[dict]:
        if force_refresh:
            return self._refresh_models()

        with self._models_cache_lock:
            models = self._models_cache
            if not self._cache_refresh_due_locked():
                return models
            self._start_background_refresh_locked()
            return models

    def refresh_in_background(self) -> None:
        with self._models_cache_lock:
            if self._cache_refresh_due_locked():
                self._start_background_refresh_locked()

    def _cache_refresh_due_locked(self) -> bool:
        now = time.time()
        return now >= max(
            self._models_cache_at + MODEL_CACHE_TTL,
            self._models_refresh_after,
        )

    def _start_background_refresh_locked(self) -> None:
        if self._models_refreshing:
            return
        self._models_refreshing = True
        threading.Thread(
            target=self._refresh_models_in_background,
            name="genai-model-refresh",
            daemon=True,
        ).start()

    def _refresh_models_in_background(self) -> None:
        try:
            self._refresh_models()
        finally:
            with self._models_cache_lock:
                self._models_refreshing = False

    def _refresh_models(self) -> list[dict]:
        with self._models_refresh_lock:
            try:
                models = _normalize_model_records(self._fetch_models())
                if not models:
                    raise ValueError("refresh returned no usable models")
            except Exception as exc:
                with self._models_cache_lock:
                    self._models_refresh_after = (
                        time.time() + MODEL_REFRESH_FAILURE_COOLDOWN
                    )
                    cached = self._models_cache or list(self._fallback_models)
                    self._models_cache = cached
                if isinstance(exc, ProxyError):
                    self._logger.warning(
                        "Using cached GenAI models after refresh failed"
                    )
                else:
                    self._logger.error(
                        "Using cached GenAI models after unexpected refresh failure (%s)",
                        type(exc).__name__,
                    )
                return cached

            cached_at = time.time()
            with self._models_cache_lock:
                self._models_cache = models
                self._models_cache_at = cached_at
                self._models_refresh_after = 0.0
            self._write_persistent_cache(models, cached_at)
            return models

    def _load_persistent_cache(self) -> tuple[list[dict], float] | None:
        if self._models_cache_path is None:
            return None
        try:
            with self._models_cache_path.open(encoding="utf-8") as cache_file:
                payload = json.load(cache_file)
            if (
                not isinstance(payload, dict)
                or payload.get("version") != MODEL_CACHE_VERSION
            ):
                raise ValueError("unsupported cache format")
            models = _normalize_model_records(payload.get("models"))
            if not models:
                raise ValueError("cache contains no models")
            cached_at = min(float(payload.get("fetched_at") or 0), time.time())
        except FileNotFoundError:
            return None
        except (OSError, TypeError, ValueError) as exc:
            self._logger.warning(
                "Ignoring invalid GenAI model cache (%s)",
                type(exc).__name__,
            )
            return None

        self._logger.info(
            "Loaded %d GenAI models from persistent cache",
            len(models),
        )
        return models, max(0.0, cached_at)

    def _write_persistent_cache(
        self,
        models: list[dict],
        cached_at: float,
    ) -> None:
        if self._models_cache_path is None:
            return
        temporary_path = None
        try:
            self._models_cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": MODEL_CACHE_VERSION,
                "fetched_at": cached_at,
                "models": models,
            }
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._models_cache_path.parent,
                prefix=f".{self._models_cache_path.name}.",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(
                    payload,
                    temporary,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            temporary_path.replace(self._models_cache_path)
        except (OSError, TypeError, ValueError) as exc:
            self._logger.warning(
                "Failed to persist GenAI model cache (%s)",
                type(exc).__name__,
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _fetch_models(self) -> list[dict]:
        return self._fetch_models_with_retry()

    def _fetch_models_with_retry(self) -> list[dict]:
        auth_retry_used = False
        retry_count = 0
        while True:
            token = self._token_manager.token
            try:
                return self._fetch_models_once(token)
            except ProxyError as exc:
                if exc.code == "upstream_auth_failed" and not auth_retry_used:
                    auth_retry_used = True
                    if not self._token_manager.refresh_after_auth_failure(
                        "model list rejected token",
                        rejected_token=token,
                    ):
                        raise
                    continue
                if exc.code == TRANSIENT_UPSTREAM_ERROR_CODE and schedule_retry(
                    self._logger,
                    max_retries=self._max_retries,
                    backoff=self._retry_backoff,
                    retry_count=retry_count,
                    operation="GenAI model list request",
                    reason=exc.message,
                ):
                    retry_count += 1
                    continue
                raise

    def _fetch_models_once(self, token: str | None) -> list[dict]:
        url = GENAI_MODEL_LIST_URL.format(timestamp=int(time.time() * 1000))
        try:
            response = requests.get(
                url,
                headers={
                    "Accept": "application/json",
                    "X-Access-Token": token,
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            self._logger.warning(
                "Failed to fetch GenAI model list (%s)",
                type(exc).__name__,
            )
            raise transient_upstream_error("Failed to fetch GenAI models") from exc

        if response.status_code in (401, 403):
            raise ProxyError(
                "Upstream GenAI token is invalid or expired",
                error_type="authentication_error",
                code="upstream_auth_failed",
                status=502,
            )
        if response.status_code != 200:
            self._logger.warning(
                "GenAI model list HTTP error %d",
                response.status_code,
            )
            if is_retryable_status(response.status_code):
                raise transient_upstream_error("Failed to fetch GenAI models")
            raise ProxyError(
                "Failed to fetch GenAI models", error_type="upstream_error", status=502
            )

        try:
            payload = response.json()
        except ValueError as exc:
            self._logger.warning(
                "Failed to decode GenAI model list JSON (%s)",
                type(exc).__name__,
            )
            raise transient_upstream_error("Failed to fetch GenAI models") from exc

        if is_genai_auth_failure(payload):
            raise ProxyError(
                "Upstream GenAI token is invalid or expired",
                error_type="authentication_error",
                code="upstream_auth_failed",
                status=502,
            )

        if payload.get("success") is False or payload.get("code", 200) >= 400:
            self._logger.warning(
                "GenAI model list business error (code=%s)",
                safe_log_code(payload.get("code")),
            )
            if is_retryable_business_error(
                payload.get("code"), payload.get("message", "")
            ):
                raise transient_upstream_error("Failed to fetch GenAI models")
            raise ProxyError(
                "Failed to fetch GenAI models", error_type="upstream_error", status=502
            )

        result = payload.get("result")
        if not isinstance(result, dict):
            self._logger.warning("GenAI model list response missing result object")
            raise transient_upstream_error("Failed to fetch GenAI models")

        records = result.get("records")
        if not isinstance(records, list):
            self._logger.warning("GenAI model list response missing records array")
            raise transient_upstream_error("Failed to fetch GenAI models")
        models = _normalize_model_records(records)
        if not models:
            self._logger.warning("GenAI model list response contained no usable models")
            raise transient_upstream_error("Failed to fetch GenAI models")

        self._logger.debug(
            "Fetched %d GenAI models from upstream",
            len(models),
        )
        return models


def _normalize_model_records(records) -> list[dict]:
    if not isinstance(records, list):
        raise TypeError("models must be a list")

    models = []
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        ai_type = record.get("aiType")
        if not isinstance(ai_type, str) or not ai_type:
            continue
        model_key = ai_type.casefold()
        if model_key in EXCLUDED_MODEL_IDS or model_key in seen:
            continue
        seen.add(model_key)
        models.append(record)
    return models


def _fallback_model_records(model_ids) -> list[dict]:
    records = []
    seen = set()
    for model_id in model_ids:
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        model_id = model_id.strip()
        model_key = model_id.casefold()
        if model_key in EXCLUDED_MODEL_IDS or model_key in seen:
            continue
        seen.add(model_key)
        records.append(
            {
                "aiType": model_id,
                "aiName": model_id,
                "rootAiType": (
                    "azure"
                    if model_key.startswith(("gpt-", "o1", "o3", "o4"))
                    else "xinference"
                ),
            }
        )
    return records


def _parse_created_timestamp(value) -> int:
    if not value:
        return 0

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return int(
                datetime.strptime(str(value), fmt).replace(tzinfo=UTC).timestamp()
            )
        except ValueError:
            continue
    return 0


def _fallback_owner(record: dict) -> str:
    root_model_name = (record.get("rootModelName") or "").strip().lower()
    root_ai_type = (record.get("rootAiType") or "").strip().lower()
    return root_model_name or root_ai_type or "genai"
