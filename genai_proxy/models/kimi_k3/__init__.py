"""Kimi K3 prompt and tool-call adapter."""

from genai_proxy.models.kimi_k3.tooling import (
    KIMI_FINAL_CLOSE,
    KIMI_FINAL_OPEN,
    KIMI_TOOL_TRANSPORT_ERROR,
    extract_kimi_final_response,
    extract_kimi_tool_calls,
    inject_kimi_tool_prompt,
    kimi_tool_retry_messages,
)

__all__ = [
    "KIMI_FINAL_CLOSE",
    "KIMI_FINAL_OPEN",
    "KIMI_TOOL_TRANSPORT_ERROR",
    "extract_kimi_final_response",
    "extract_kimi_tool_calls",
    "inject_kimi_tool_prompt",
    "kimi_tool_retry_messages",
]
