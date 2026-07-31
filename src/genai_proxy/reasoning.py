from genai_proxy.errors import ProxyError
from genai_proxy.models.registry import (
    DEEPSEEK_V4_ADAPTERS,
    GLM_5_2_ADAPTER,
    KIMI_K3_ADAPTER,
)

OPENAI_REASONING_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
CLAUDE_REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")
CLAUDE_THINKING_TYPES = ("enabled", "adaptive", "disabled")
DEEPSEEK_HIGH_REASONING_EFFORTS = ("minimal", "low", "medium", "high")
DEEPSEEK_MAX_REASONING_EFFORTS = ("xhigh", "max")


def parse_reasoning_config(req_data: dict | None) -> dict:
    if not isinstance(req_data, dict):
        return {}

    effort = None
    reasoning = req_data.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
        effort = reasoning.get("effort")
    elif req_data.get("reasoning_effort") is not None:
        effort = req_data.get("reasoning_effort")

    return _normalize_reasoning_effort(
        effort,
        allowed_efforts=OPENAI_REASONING_EFFORTS,
        field_name="OpenAI reasoning.effort",
    )


def parse_claude_reasoning_config(req_data: dict | None) -> dict:
    if not isinstance(req_data, dict):
        return {}

    output_config = req_data.get("output_config")
    effort = output_config.get("effort") if isinstance(output_config, dict) else None
    config = _normalize_reasoning_effort(
        effort,
        allowed_efforts=CLAUDE_REASONING_EFFORTS,
        field_name="Claude output_config.effort",
    )
    thinking = req_data.get("thinking")
    if thinking is None:
        return config
    if not isinstance(thinking, dict):
        raise ProxyError(
            "Claude thinking must be an object",
            error_type="invalid_request_error",
            code="invalid_thinking_config",
            status=400,
        )
    thinking_type = str(thinking.get("type") or "").strip().lower()
    if thinking_type not in CLAUDE_THINKING_TYPES:
        supported = ", ".join(CLAUDE_THINKING_TYPES)
        raise ProxyError(
            f"Unsupported Claude thinking.type '{thinking_type}'. "
            f"Supported values are: {supported}.",
            error_type="invalid_request_error",
            code="invalid_thinking_config",
            status=400,
        )
    if thinking_type == "disabled":
        return {"effort": "none"}
    # Anthropic defines high as the default effort. DeepSeek V4 has no token
    # budget field, so enabled/adaptive thinking without an explicit effort
    # maps to its high tier rather than inventing a budget-to-tier heuristic.
    return config or {"effort": "high"}


def normalize_reasoning_for_adapter(
    adapter: str | None, reasoning_config: dict | None
) -> dict | None:
    effort = (reasoning_config or {}).get("effort")
    if not effort:
        return reasoning_config

    if adapter == GLM_5_2_ADAPTER:
        # GenAI does not expose GLM-5.2's chat-template reasoning_effort
        # argument. The upstream template therefore always uses its official
        # default, max; injecting a second system directive would duplicate or
        # contradict that template-owned directive.
        return {"effort": "max"}
    if adapter in DEEPSEEK_V4_ADAPTERS:
        if effort == "none":
            return {"effort": "none"}
        if effort in DEEPSEEK_HIGH_REASONING_EFFORTS:
            return {"effort": "high"}
        if effort in DEEPSEEK_MAX_REASONING_EFFORTS:
            return {"effort": "max"}
    if adapter == KIMI_K3_ADAPTER:
        return {"effort": "max"}

    return reasoning_config


def deepseek_thinking_enabled(
    adapter: str | None, reasoning_config: dict | None
) -> bool | None:
    if adapter not in DEEPSEEK_V4_ADAPTERS:
        return None
    effort = (reasoning_config or {}).get("effort")
    return effort not in (None, "none")


def validate_reasoning_for_adapter(
    adapter: str | None, reasoning_config: dict | None
) -> None:
    normalized = normalize_reasoning_for_adapter(adapter, reasoning_config)
    if reasoning_config is not None and normalized is not None:
        reasoning_config.clear()
        reasoning_config.update(normalized)


def _normalize_reasoning_effort(
    effort, *, allowed_efforts: tuple[str, ...], field_name: str
) -> dict:
    if effort is None:
        return {}
    normalized = str(effort).strip().lower()
    if not normalized:
        return {}
    if normalized not in allowed_efforts:
        supported = ", ".join(allowed_efforts)
        raise ProxyError(
            f"Unsupported {field_name} '{normalized}'. Supported values are: {supported}.",
            error_type="invalid_request_error",
            code="unsupported_reasoning_effort",
            status=400,
        )
    return {"effort": normalized}
