"""Prompt, completion, and reasoning usage accounting for chat requests."""

from genai_proxy.chat.types import PreparedChatRequest
from genai_proxy.token_usage import (
    count_openai_completion_tokens,
    count_openai_reasoning_tokens,
    count_openai_request_tokens,
)


class ChatUsageMixin:
    def _completion_tokens(
        self,
        prepared: PreparedChatRequest,
        message: dict,
        *,
        finish_reason: str = "stop",
    ) -> int:
        return count_openai_completion_tokens(
            message,
            prepared.model,
            model_record=prepared.model_record,
            tool_adapter=prepared.tool_adapter,
            prompt_messages=prepared.messages,
            reasoning_config=prepared.token_reasoning_config,
            thinking=prepared.thinking,
            finish_reason=finish_reason,
            image_sizes=prepared.image_sizes,
        )

    def _usage(
        self,
        prepared: PreparedChatRequest,
        message: dict,
        *,
        finish_reason: str = "stop",
    ) -> dict:
        if prepared.prompt_tokens is None:
            prepared.prompt_tokens = count_openai_request_tokens(
                prepared.messages,
                prepared.model,
                model_record=prepared.model_record,
                tool_adapter=prepared.tool_adapter,
                reasoning_config=prepared.token_reasoning_config,
                thinking=prepared.thinking,
                image_sizes=prepared.image_sizes,
            )
        completion_tokens = self._completion_tokens(
            prepared,
            message,
            finish_reason=finish_reason,
        )
        reasoning_tokens = count_openai_reasoning_tokens(
            str(message.get("reasoning_content") or ""),
            prepared.model,
            model_record=prepared.model_record,
            tool_adapter=prepared.tool_adapter,
            prompt_messages=prepared.messages,
            reasoning_config=prepared.token_reasoning_config,
            thinking=prepared.thinking,
            image_sizes=prepared.image_sizes,
        )
        return {
            "prompt_tokens": prepared.prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prepared.prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        }

    def _record_usage(
        self,
        prepared: PreparedChatRequest,
        message: dict,
        *,
        finish_reason: str,
    ) -> dict:
        prepared.generated_usage = self._usage(
            prepared,
            message,
            finish_reason=finish_reason,
        )
        return prepared.generated_usage

def responses_usage(openai_usage: dict | None) -> dict | None:
    if not openai_usage:
        return None
    prompt_details = openai_usage.get("prompt_tokens_details") or {}
    completion_details = openai_usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": openai_usage.get("prompt_tokens", 0),
        "input_tokens_details": {
            "cached_tokens": prompt_details.get("cached_tokens", 0),
        },
        "output_tokens": openai_usage.get("completion_tokens", 0),
        "output_tokens_details": {
            "reasoning_tokens": completion_details.get("reasoning_tokens", 0),
        },
        "total_tokens": openai_usage.get("total_tokens", 0),
    }


__all__ = ["ChatUsageMixin", "responses_usage"]
