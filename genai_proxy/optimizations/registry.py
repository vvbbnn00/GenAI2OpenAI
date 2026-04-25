DEEPSEEK_LEGACY_ADAPTER = "deepseek_legacy"
DEEPSEEK_V4_FLASH_ADAPTER = "deepseek_v4_flash"
DEEPSEEK_V4_PRO_ADAPTER = "deepseek_v4_pro"
DEEPSEEK_ADAPTER = DEEPSEEK_LEGACY_ADAPTER
DEEPSEEK_V4_ADAPTERS = (DEEPSEEK_V4_FLASH_ADAPTER, DEEPSEEK_V4_PRO_ADAPTER)
DEEPSEEK_ADAPTERS = (*DEEPSEEK_V4_ADAPTERS, DEEPSEEK_LEGACY_ADAPTER)
GENERIC_ADAPTER = "generic"
GLM_ADAPTER = "glm"
MINIMAX_ADAPTER = "minimax"


def select_tool_adapter(model: str | None, record: dict | None = None) -> str:
    text = _model_text(model, record)
    model_key = (model or "").lower()

    if _has_non_xinference_root(record):
        return GENERIC_ADAPTER

    if "minimax" in text or "mini max" in text or "m2.7" in text or "m27" in text:
        return MINIMAX_ADAPTER
    if "chatglm" in text or "glm" in text:
        return GLM_ADAPTER
    if model_key == "deepseek-pro" or "deepseek-v4-pro" in text or "v4-pro" in text:
        return DEEPSEEK_V4_PRO_ADAPTER
    if (
        model_key == "deepseek-chat"
        or "deepseek-v4-flash" in text
        or "v4-flash" in text
        or ("deepseek" in text and "v4" in text)
    ):
        return DEEPSEEK_V4_FLASH_ADAPTER
    if "deepseek" in text:
        return DEEPSEEK_LEGACY_ADAPTER
    return GENERIC_ADAPTER


def tool_start_tags(adapter: str) -> tuple[str, ...]:
    if adapter in DEEPSEEK_V4_ADAPTERS:
        return ("<｜DSML｜tool_calls>", "<tool_call>", "<arg_key>")
    if adapter == DEEPSEEK_LEGACY_ADAPTER:
        return ("<｜DSML｜function_calls>", "<tool_call>", "<arg_key>")
    if adapter == MINIMAX_ADAPTER:
        return ("<minimax:tool_call>", "<tool_call>", "<arg_key>")
    if adapter == GLM_ADAPTER:
        return ("<tool_call>", "<arg_key>")
    return ("<tool_call>",)


def is_deepseek_adapter(adapter: str | None) -> bool:
    return adapter in DEEPSEEK_ADAPTERS


def is_deepseek_v4_adapter(adapter: str | None) -> bool:
    return adapter in DEEPSEEK_V4_ADAPTERS


def _model_text(model: str | None, record: dict | None) -> str:
    parts = [model or ""]
    if record:
        for key in (
            "aiType",
            "aiName",
            "simpleName",
            "descInfo",
            "descInfoEn",
            "rootAiType",
            "rootModelName",
        ):
            value = record.get(key)
            if value is not None:
                parts.append(str(value))
    return " ".join(parts).lower()


def _has_non_xinference_root(record: dict | None) -> bool:
    if not record:
        return False
    root = record.get("rootModelName")
    if root is None:
        return False
    return "xinference" not in str(root).lower()
