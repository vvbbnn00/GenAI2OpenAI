import time
from datetime import datetime, timezone

import requests


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
        url = GENAI_MODEL_LIST_URL.format(timestamp=int(time.time()))
        response = requests.get(
            url,
            headers={
                "Accept": "application/json",
                "X-Access-Token": self._token_manager.token,
            },
            timeout=30,
        )
        response.raise_for_status()

        payload = response.json()
        records = payload.get("result", {}).get("records", [])
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
