"""Kimi K3 prompt and tool-call adapter."""

from genai_proxy.models.kimi_k3.tooling import (
    KIMI_FINAL_CLOSE,
    KIMI_FINAL_OPEN,
    KIMI_TOOL_TRANSPORT_ERROR,
    collect_kimi_completed_actions,
    extract_kimi_final_response,
    extract_kimi_tool_calls,
    inject_kimi_tool_prompt,
    kimi_action_repeats_completed,
    kimi_duplicate_retry_messages,
    kimi_tool_retry_messages,
)

__all__ = [
    "KIMI_FINAL_CLOSE",
    "KIMI_FINAL_OPEN",
    "KIMI_TOOL_TRANSPORT_ERROR",
    "collect_kimi_completed_actions",
    "extract_kimi_final_response",
    "extract_kimi_tool_calls",
    "inject_kimi_tool_prompt",
    "kimi_action_repeats_completed",
    "kimi_duplicate_retry_messages",
    "kimi_tool_retry_messages",
]
