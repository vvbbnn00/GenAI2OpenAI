from genai_proxy.errors import ProxyError
from genai_proxy.optimizations.registry import GLM_5_2_ADAPTER


OPENAI_REASONING_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)
CLAUDE_REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")
GLM52_REASONING_EFFORTS = ("none", "high", "xhigh")


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
    if config.get("effort") == "max":
        return {"effort": "xhigh"}
    return config


def validate_reasoning_for_adapter(adapter: str | None, reasoning_config: dict | None) -> None:
    effort = (reasoning_config or {}).get("effort")
    if not effort:
        return

    if adapter == GLM_5_2_ADAPTER and effort not in GLM52_REASONING_EFFORTS:
        raise ProxyError(
            (
                f"Unsupported reasoning effort '{effort}' for GLM-5.2. "
                "Supported values are: none, high, xhigh."
            ),
            error_type="invalid_request_error",
            code="unsupported_reasoning_effort",
            status=400,
        )


def _normalize_reasoning_effort(effort, *, allowed_efforts: tuple[str, ...], field_name: str) -> dict:
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
