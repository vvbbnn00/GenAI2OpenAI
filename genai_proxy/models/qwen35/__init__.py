"""Qwen 3.5 prompt and tool-call adapter."""

from genai_proxy.models.qwen35.tooling import (
    extract_qwen35_tool_calls,
    inject_qwen35_tool_prompt,
)

__all__ = ["extract_qwen35_tool_calls", "inject_qwen35_tool_prompt"]
