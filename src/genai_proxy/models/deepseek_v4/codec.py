"""Pinned official tokenizer assets for DeepSeek V4 Flash and Pro."""

import json

from genai_proxy.errors import ProxyError
from genai_proxy.models.hf_assets import Artifact, TokenizerSpec
from genai_proxy.models.hf_assets import load_python_encoder, tokenizer_error

DEEPSEEK_V4_PRO_SPEC = TokenizerSpec(
    family="deepseek_v4_pro",
    repository="deepseek-ai/DeepSeek-V4-Pro",
    revision="b5968e9190ef611bbf34a7229255be88a0e937c1",
    tokenizer=Artifact(
        "tokenizer.json",
        "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
    ),
    encoder=Artifact(
        "encoding/encoding_dsv4.py",
        "bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0",
    ),
)

DEEPSEEK_V4_FLASH_SPEC = TokenizerSpec(
    family="deepseek_v4_flash",
    repository="deepseek-ai/DeepSeek-V4-Flash",
    revision="60d8d70770c6776ff598c94bb586a859a38244f1",
    tokenizer=DEEPSEEK_V4_PRO_SPEC.tokenizer,
    encoder=DEEPSEEK_V4_PRO_SPEC.encoder,
)


def official_tool_prompt(spec: TokenizerSpec, function_tools: list[dict]) -> str:
    encoder = load_python_encoder(spec)
    return encoder["render_tools"](
        encoder["tools_from_openai_format"](function_tools)
    )


def official_reasoning_prefix(spec: TokenizerSpec, effort: str | None) -> str:
    if effort != "max":
        return ""
    return str(load_python_encoder(spec)["REASONING_EFFORT_MAX"])


def serialize_completion(
    message: dict,
    *,
    finish_reason: str = "stop",
    thinking: bool | None = None,
) -> str:
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    parts = (
        [str(content)]
        if thinking is False
        else [str(reasoning), "</think>", str(content)]
    )
    if tool_calls:
        parts.append(_tool_calls(tool_calls))
    if finish_reason != "length":
        parts.append("<｜end▁of▁sentence｜>")
    return "".join(parts)


def official_transport_messages(
    spec: TokenizerSpec,
    messages,
    tools,
    *,
    reasoning_config: dict | None = None,
    tool_choice_suffix: str = "",
) -> list[dict]:
    encoder = load_python_encoder(spec)
    function_tools = [
        tool
        for tool in tools or []
        if isinstance(tool, dict) and tool.get("type") == "function"
    ]
    if not function_tools:
        raise ProxyError(
            "DeepSeek V4 requires at least one function tool",
            error_type="invalid_request_error",
            code="unsupported_tool_type",
            status=400,
        )

    official_messages = _normalize_text_messages(messages)
    if official_messages and official_messages[0].get("role") in {
        "system",
        "developer",
    }:
        official_messages[0]["tools"] = function_tools
    else:
        official_messages.insert(
            0,
            {"role": "system", "content": "", "tools": function_tools},
        )
    if tool_choice_suffix:
        official_messages.insert(
            1,
            {"role": "system", "content": tool_choice_suffix},
        )

    effort = (reasoning_config or {}).get("effort")
    thinking = effort not in (None, "none")
    reasoning_effort = effort if thinking else None
    thinking_mode = "thinking" if thinking else "chat"
    prompt = encoder["encode_messages"](
        official_messages,
        thinking_mode=thinking_mode,
        drop_thinking=True,
        add_default_bos_token=True,
        reasoning_effort=reasoning_effort,
    )

    processed = encoder["merge_tool_messages"](official_messages)
    processed = encoder["sort_tool_results_by_call_order"](processed)
    last_user_index = encoder["find_last_user_index"](processed)
    if last_user_index != len(processed) - 1:
        raise ProxyError(
            "DeepSeek V4 tool transport requires the final message to be user or tool",
            error_type="invalid_request_error",
            code="unsupported_message_sequence",
            status=400,
        )

    rendered_user = encoder["render_message"](
        last_user_index,
        processed,
        thinking_mode=thinking_mode,
        drop_thinking=False,
        reasoning_effort=reasoning_effort,
    )
    user_prefix = str(encoder["USER_SP_TOKEN"])
    assistant_suffix = str(encoder["ASSISTANT_SP_TOKEN"]) + str(
        encoder["thinking_start_token"] if thinking else encoder["thinking_end_token"]
    )
    bos = str(encoder["bos_token"])
    if (
        not prompt.startswith(bos)
        or not rendered_user.startswith(user_prefix)
        or not rendered_user.endswith(assistant_suffix)
        or not prompt.endswith(rendered_user)
    ):
        raise tokenizer_error(
            spec,
            "construct official DeepSeek V4 transport",
            ValueError("official encoder boundaries changed"),
        )

    prefix = prompt[len(bos) : -len(rendered_user)]
    user_content = rendered_user[
        len(user_prefix) : len(rendered_user) - len(assistant_suffix)
    ]
    transported = [
        {"role": "system", "content": prefix},
        {"role": "user", "content": user_content},
    ]
    verification = encoder["encode_messages"](
        transported,
        thinking_mode=thinking_mode,
        drop_thinking=True,
        add_default_bos_token=True,
        reasoning_effort=None,
    )
    if verification != prompt:
        raise tokenizer_error(
            spec,
            "verify official DeepSeek V4 transport",
            ValueError("transport rendering differs from official prompt"),
        )
    return transported


def _normalize_text_messages(messages) -> list[dict]:
    normalized = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if content is None or isinstance(content, str):
            normalized.append(copied)
            continue
        if not isinstance(content, list):
            raise ProxyError(
                "DeepSeek V4 message content must be a string or text content parts",
                error_type="invalid_request_error",
                code="unsupported_content_type",
                status=400,
            )

        text_parts = []
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else None
            if part_type not in {"text", "input_text", "output_text"}:
                raise ProxyError(
                    "DeepSeek V4 accepts only text content parts",
                    error_type="invalid_request_error",
                    code="unsupported_content_type",
                    status=400,
                )
            text = part.get("text")
            if not isinstance(text, str):
                raise ProxyError(
                    "DeepSeek V4 text content parts require a string 'text' field",
                    error_type="invalid_request_error",
                    code="unsupported_content_type",
                    status=400,
                )
            text_parts.append(text)
        copied["content"] = "".join(text_parts)
        normalized.append(copied)
    return normalized


def _tool_calls(tool_calls) -> str:
    calls = []
    for call in tool_calls:
        function = call.get("function") or {}
        arguments = _json_arguments(function.get("arguments"))
        parameters = []
        for key, value in arguments.items():
            is_string = isinstance(value, str)
            rendered = value if is_string else json.dumps(value, ensure_ascii=False)
            parameters.append(
                f'<｜DSML｜parameter name="{key}" string="{str(is_string).lower()}">'
                f"{rendered}</｜DSML｜parameter>"
            )
        calls.append(
            f'<｜DSML｜invoke name="{function.get("name", "")}">\n'
            + "\n".join(parameters)
            + "\n</｜DSML｜invoke>"
        )
    return "\n\n<｜DSML｜tool_calls>\n" + "\n".join(calls) + "\n</｜DSML｜tool_calls>"


def _json_arguments(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"arguments": str(value or "")}
    return parsed if isinstance(parsed, dict) else {"arguments": parsed}


__all__ = [
    "DEEPSEEK_V4_FLASH_SPEC",
    "DEEPSEEK_V4_PRO_SPEC",
    "official_reasoning_prefix",
    "official_tool_prompt",
    "official_transport_messages",
    "serialize_completion",
]
