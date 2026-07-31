"""Backward-compatible OpenAI helpers and model-facing tool protocol."""

from genai_proxy.api.openai.errors import (
    make_error_chunk as make_error_chunk,
)
from genai_proxy.api.openai.errors import (
    openai_error as openai_error,
)
from genai_proxy.chat.tool_protocol import *  # noqa: F403
