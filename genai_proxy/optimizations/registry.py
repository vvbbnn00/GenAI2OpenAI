DEEPSEEK_ADAPTER = "deepseek"
GENERIC_ADAPTER = "generic"
GLM_ADAPTER = "glm"
MINIMAX_ADAPTER = "minimax"


def select_tool_adapter(model: str | None, record: dict | None = None) -> str:
    text = _model_text(model, record)

    if "deepseek" in text:
        return DEEPSEEK_ADAPTER
    if "minimax" in text or "mini max" in text or "m2.7" in text or "m27" in text:
        return MINIMAX_ADAPTER
    if "chatglm" in text or "glm" in text:
        return GLM_ADAPTER
    return GENERIC_ADAPTER


def tool_start_tags(adapter: str) -> tuple[str, ...]:
    if adapter == DEEPSEEK_ADAPTER:
        return ("<｜DSML｜function_calls>", "<tool_call>")
    return ("<tool_call>",)


def native_tool_fields(adapter: str) -> dict:
    fields = {}
    if adapter in {DEEPSEEK_ADAPTER, GLM_ADAPTER, MINIMAX_ADAPTER}:
        fields["native_tools"] = True
    if adapter == GLM_ADAPTER:
        fields["tool_stream"] = True
    if adapter == MINIMAX_ADAPTER:
        fields["reasoning_split"] = True
    return fields




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
