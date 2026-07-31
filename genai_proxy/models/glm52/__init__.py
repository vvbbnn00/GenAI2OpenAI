"""GLM 5.2 prompt and tool-call adapter."""

from genai_proxy.models.glm52.tooling import (
    inject_glm_reasoning_prompt,
    inject_glm_tool_prompt,
)

__all__ = ["inject_glm_reasoning_prompt", "inject_glm_tool_prompt"]
