"""Pinned tokenizer assets for the retired MiniMax compatibility adapter."""

import json

from genai_proxy.models.hf_assets import (
    Artifact,
    TokenizerSpec,
    load_template,
    tokenizer_error,
)

MINIMAX_M2_7_SPEC = TokenizerSpec(
    family="minimax_m2_7",
    repository="MiniMaxAI/MiniMax-M2.7",
    revision="d494266a4affc0d2995ba1fa35c8481cbd84294b",
    tokenizer=Artifact(
        "tokenizer.json",
        "757622126525aeeb131756849d93298070ff3f0319c455ec8c5bb0f6b1cebbe8",
    ),
    template=Artifact(
        "chat_template.jinja",
        "893d908f7b5cc65fdde270dcae5ea1a99647c6a7ce572ae874a57b7160069566",
    ),
)


def official_tool_prompt(function_tools: list[dict]) -> str:
    serialized_tools = json.dumps(function_tools, ensure_ascii=False, sort_keys=True)
    sentinel = "__GENAI2OPENAI_SYSTEM_SENTINEL__"
    while sentinel in serialized_tools:
        sentinel += "_"
    prompt = load_template(MINIMAX_M2_7_SPEC).render(
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
        start_marker = f"]~!b[]~b]system\n{sentinel}\n\n"
        start = prompt.index(start_marker) + len(start_marker)
        end = prompt.index("[e~[\n", start)
        return prompt[start:end]
    except ValueError as exc:
        raise tokenizer_error(
            MINIMAX_M2_7_SPEC,
            "extract official tool prompt",
            exc,
        ) from exc


def official_default_system_prompt() -> str:
    prompt = load_template(MINIMAX_M2_7_SPEC).render(
        messages=[{"role": "user", "content": "__GENAI2OPENAI_USER_SENTINEL__"}],
        tools=None,
        add_generation_prompt=True,
    )
    start_marker = "]~!b[]~b]system\n"
    try:
        start = prompt.index(start_marker) + len(start_marker)
        end = prompt.index("[e~[\n", start)
        return prompt[start:end]
    except ValueError as exc:
        raise tokenizer_error(
            MINIMAX_M2_7_SPEC,
            "extract official default system prompt",
            exc,
        ) from exc


def serialize_completion(
    message: dict,
    *,
    finish_reason: str = "stop",
) -> str:
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    parts = [str(reasoning)]
    if content or tool_calls or finish_reason != "length":
        parts.extend(
            (
                "\n</think>\n\n" if reasoning else "</think>\n\n",
                str(content),
            )
        )
    if tool_calls:
        parts.append(_tool_calls(tool_calls))
    if finish_reason != "length":
        parts.append("[e~[\n")
    return "".join(parts)


def _tool_calls(tool_calls) -> str:
    invocations = []
    for call in tool_calls:
        function = call.get("function") or {}
        arguments = _json_arguments(function.get("arguments"))
        parameters = "".join(
            f'\n<parameter name="{key}">'
            f"{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
            "</parameter>"
            for key, value in arguments.items()
        )
        invocations.append(
            f'<invoke name="{function.get("name", "")}">{parameters}\n</invoke>'
        )
    return "\n<minimax:tool_call>\n" + "\n".join(invocations) + "\n</minimax:tool_call>"


def _json_arguments(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"arguments": str(value or "")}
    return parsed if isinstance(parsed, dict) else {"arguments": parsed}


__all__ = [
    "MINIMAX_M2_7_SPEC",
    "official_default_system_prompt",
    "official_tool_prompt",
    "serialize_completion",
]
