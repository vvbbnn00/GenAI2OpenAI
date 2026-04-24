from genai_proxy.optimizations.deepseek import (
    extract_deepseek_tool_calls,
    inject_deepseek_tool_prompt,
    is_deepseek_model,
)
from genai_proxy.optimizations.glm import inject_glm_tool_prompt
from genai_proxy.optimizations.minimax import inject_minimax_tool_prompt
from genai_proxy.optimizations.registry import (
    DEEPSEEK_ADAPTER,
    GENERIC_ADAPTER,
    GLM_ADAPTER,
    MINIMAX_ADAPTER,
    native_tool_fields,
    select_tool_adapter,
    tool_start_tags,
)

__all__ = [
    "DEEPSEEK_ADAPTER",
    "GENERIC_ADAPTER",
    "GLM_ADAPTER",
    "MINIMAX_ADAPTER",
    "extract_deepseek_tool_calls",
    "inject_glm_tool_prompt",
    "inject_deepseek_tool_prompt",
    "inject_minimax_tool_prompt",
    "is_deepseek_model",
    "native_tool_fields",
    "select_tool_adapter",
    "tool_start_tags",
]
