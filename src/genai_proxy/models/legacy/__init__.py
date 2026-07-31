"""Compatibility adapters for models no longer present in the active catalog."""

from genai_proxy.models.legacy.minimax import inject_minimax_tool_prompt

__all__ = ["inject_minimax_tool_prompt"]
