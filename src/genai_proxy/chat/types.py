"""Shared request and model context types for chat orchestration."""

from dataclasses import dataclass


@dataclass(slots=True)
class PreparedChatRequest:
    messages: list
    model: str
    root_model_name: str | None
    root_ai_type: str
    max_tokens: int
    has_tools: bool
    tools: list
    tool_choice: object
    tool_adapter: str
    model_record: dict | None
    include_usage: bool
    prompt_tokens: int | None
    token_reasoning_config: dict | None
    thinking: bool | None
    image_sizes: tuple[tuple[int, int], ...] | None
    generated_usage: dict | None = None


@dataclass(frozen=True, slots=True)
class ResolvedModelContext:
    requested_model: str
    model: str
    model_record: dict | None
    tool_adapter: str
    tokenizer_family: str
    supports_vision: bool
    supports_thinking_toggle: bool
    transport: str
    root_ai_type: str
    root_model_name: str | None


__all__ = ["PreparedChatRequest", "ResolvedModelContext"]
