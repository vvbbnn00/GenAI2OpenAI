"""Pinned official tokenizer and visual preprocessing constants for Qwen 3.5."""

import json
import math

from genai_proxy.models.hf_assets import (
    Artifact,
    TokenizerSpec,
    load_template,
    tokenizer_error,
)

QWEN_3_5_SPEC = TokenizerSpec(
    family="qwen_3_5",
    repository="Qwen/Qwen3.5-397B-A17B",
    revision="8472618112abcbd45acbcdc58436aff4233c23f7",
    tokenizer=Artifact(
        "tokenizer.json",
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    template=Artifact(
        "chat_template.jinja",
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    ),
)

IMAGE_PATCH_SIZE = 16
IMAGE_MERGE_SIZE = 2
IMAGE_MIN_PIXELS = 65536
IMAGE_MAX_PIXELS = 16777216
IMAGE_MAX_ASPECT_RATIO = 200


def official_tool_prompt(function_tools: list[dict]) -> str:
    serialized_tools = json.dumps(function_tools, ensure_ascii=False, sort_keys=True)
    sentinel = "__GENAI2OPENAI_SYSTEM_SENTINEL__"
    while sentinel in serialized_tools:
        sentinel += "_"
    prompt = load_template(QWEN_3_5_SPEC).render(
        messages=[
            {"role": "system", "content": sentinel},
            {"role": "user", "content": "__GENAI2OPENAI_USER_SENTINEL__"},
        ],
        tools=function_tools,
        add_generation_prompt=True,
        enable_thinking=True,
        clear_thinking=True,
        add_vision_id=False,
    )
    try:
        start_marker = "<|im_start|>system\n"
        end_marker = f"\n\n{sentinel}<|im_end|>"
        start = prompt.index(start_marker) + len(start_marker)
        end = prompt.index(end_marker, start)
        return prompt[start:end]
    except ValueError as exc:
        raise tokenizer_error(
            QWEN_3_5_SPEC,
            "extract official tool prompt",
            exc,
        ) from exc


def image_token_count(width: int, height: int) -> int:
    factor = IMAGE_PATCH_SIZE * IMAGE_MERGE_SIZE
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    resized_pixels = resized_height * resized_width
    if resized_pixels > IMAGE_MAX_PIXELS:
        beta = math.sqrt((height * width) / IMAGE_MAX_PIXELS)
        resized_height = max(
            factor,
            math.floor(height / beta / factor) * factor,
        )
        resized_width = max(
            factor,
            math.floor(width / beta / factor) * factor,
        )
    elif resized_pixels < IMAGE_MIN_PIXELS:
        beta = math.sqrt(IMAGE_MIN_PIXELS / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor

    grid_height = resized_height // IMAGE_PATCH_SIZE
    grid_width = resized_width // IMAGE_PATCH_SIZE
    return grid_height * grid_width // (IMAGE_MERGE_SIZE**2)


def serialize_completion(
    message: dict,
    *,
    finish_reason: str = "stop",
) -> str:
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    if finish_reason == "length" and not content and not tool_calls:
        return str(reasoning).strip()

    rendered_content = str(content).strip()
    parts = [str(reasoning).strip(), "\n</think>\n\n", rendered_content]
    for index, call in enumerate(tool_calls):
        if index == 0:
            parts.append("\n\n" if rendered_content else "")
        else:
            parts.append("\n")
        parts.append(_tool_call(call))
    if finish_reason != "length":
        parts.append("<|im_end|>\n")
    return "".join(parts)


def _tool_call(call) -> str:
    function = call.get("function") or {}
    arguments = _json_arguments(function.get("arguments"))
    parameters = "".join(
        f"<parameter={key}>\n{_argument_value(value)}\n</parameter>\n"
        for key, value in arguments.items()
    )
    return (
        f"<tool_call>\n<function={function.get('name', '')}>\n"
        f"{parameters}</function>\n</tool_call>"
    )


def _argument_value(value) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _json_arguments(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"arguments": str(value or "")}
    return parsed if isinstance(parsed, dict) else {"arguments": parsed}


__all__ = [
    "IMAGE_MAX_ASPECT_RATIO",
    "IMAGE_MAX_PIXELS",
    "IMAGE_MERGE_SIZE",
    "IMAGE_MIN_PIXELS",
    "IMAGE_PATCH_SIZE",
    "QWEN_3_5_SPEC",
    "image_token_count",
    "official_tool_prompt",
    "serialize_completion",
]
