"""DeepSeek V4 prompt and tool-call adapter."""

from genai_proxy.models.deepseek_v4.tooling import (
    extract_deepseek_tool_calls,
    inject_deepseek_reasoning_prompt,
    inject_deepseek_tool_prompt,
    is_deepseek_model,
)

__all__ = [
    "extract_deepseek_tool_calls",
    "inject_deepseek_reasoning_prompt",
    "inject_deepseek_tool_prompt",
    "is_deepseek_model",
]
