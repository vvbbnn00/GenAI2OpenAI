import time
from datetime import datetime, timezone

import requests

from genai_proxy.errors import ProxyError
from genai_proxy.services.token_manager import is_genai_auth_failure


GENAI_MODEL_LIST_URL = (
    "https://genai.shanghaitech.edu.cn/htk/ai/aiModel/list"
    "?_t={timestamp}&pageNo=1&pageSize=999&showStatusList=2,3"
)
DEFAULT_MODEL = "GPT-4.1"
EXCLUDED_MODEL_IDS = {"gpt-image-1.5"}
MODEL_CACHE_TTL = 300


class ModelManager:
    def __init__(self, logger, token_manager):
        self._logger = logger
        self._token_manager = token_manager
        self._models_cache = None
        self._models_cache_at = 0.0

    def resolve_model(self, model: str) -> str:
        return model or DEFAULT_MODEL

    def root_ai_type_for(self, model: str) -> str:
        record = self.get_model_record(model)
        if record and record.get("rootAiType"):
            return record["rootAiType"]

        lowered = (model or "").lower()
        if lowered.startswith(("gpt-", "o1", "o3", "o4")):
            return "azure"
        return "xinference"

    def get_model_record(self, model: str):
        for record in self.list_genai_models():
            if record.get("aiType") == model:
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
        if not force_refresh and self._models_cache is not None:
            if time.time() - self._models_cache_at < MODEL_CACHE_TTL:
                return self._models_cache

        models = self._fetch_models()
        self._models_cache = models
        self._models_cache_at = time.time()
        return models

    def _fetch_models(self) -> list[dict]:
        return self._fetch_models_with_retry()

    def _fetch_models_with_retry(self) -> list[dict]:
        refreshed = False
        while True:
            try:
                return self._fetch_models_once()
            except ProxyError as exc:
                if exc.code != "upstream_auth_failed" or refreshed:
                    raise
                refreshed = True
                if not self._token_manager.refresh_after_auth_failure("model list rejected token"):
                    raise

    def _fetch_models_once(self) -> list[dict]:
        url = GENAI_MODEL_LIST_URL.format(timestamp=int(time.time()))
        try:
            response = requests.get(
                url,
                headers={
                    "Accept": "application/json",
                    "X-Access-Token": self._token_manager.token,
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            self._logger.warning("Failed to fetch GenAI model list: %s", exc)
            raise ProxyError("Failed to fetch GenAI models", error_type="upstream_error", status=502) from exc

        if response.status_code in (401, 403):
            raise ProxyError(
                "Upstream GenAI token is invalid or expired",
                error_type="authentication_error",
                code="upstream_auth_failed",
                status=502,
            )
        if response.status_code != 200:
            self._logger.warning("GenAI model list HTTP error %d: %s", response.status_code, response.text[:500])
            raise ProxyError("Failed to fetch GenAI models", error_type="upstream_error", status=502)

        try:
            payload = response.json()
        except ValueError as exc:
            self._logger.warning("Failed to decode GenAI model list JSON: %s", exc)
            raise ProxyError("Failed to fetch GenAI models", error_type="upstream_error", status=502) from exc

        if is_genai_auth_failure(payload):
            raise ProxyError(
                "Upstream GenAI token is invalid or expired",
                error_type="authentication_error",
                code="upstream_auth_failed",
                status=502,
            )

        if payload.get("success") is False or payload.get("code", 200) >= 400:
            self._logger.warning("GenAI model list business error: %s", payload)
            raise ProxyError("Failed to fetch GenAI models", error_type="upstream_error", status=502)

        result = payload.get("result")
        if not isinstance(result, dict):
            self._logger.warning("GenAI model list response missing result object: %s", payload)
            raise ProxyError("Failed to fetch GenAI models", error_type="upstream_error", status=502)

        records = result.get("records") or []
        models = []
        seen = set()

        for record in records:
            ai_type = record.get("aiType")
            if not ai_type:
                continue
            if ai_type.lower() in EXCLUDED_MODEL_IDS:
                continue
            if ai_type in seen:
                continue
            seen.add(ai_type)
            models.append(record)

        self._logger.debug(
            "Fetched %d GenAI models from upstream: %s",
            len(models),
            [model.get("aiType") for model in models],
        )
        return models


def _parse_created_timestamp(value) -> int:
    if not value:
        return 0

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(str(value), fmt).replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return 0


def _fallback_owner(record: dict) -> str:
    root_model_name = (record.get("rootModelName") or "").strip().lower()
    root_ai_type = (record.get("rootAiType") or "").strip().lower()
    return root_model_name or root_ai_type or "genai"
