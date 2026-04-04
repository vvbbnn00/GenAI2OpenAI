from genai_proxy.optimizations.deepseek import (
    extract_deepseek_tool_calls,
    inject_deepseek_tool_prompt,
    is_deepseek_model,
    should_buffer_deepseek_tool_stream,
)

__all__ = [
    "extract_deepseek_tool_calls",
    "inject_deepseek_tool_prompt",
    "is_deepseek_model",
    "should_buffer_deepseek_tool_stream",
]
