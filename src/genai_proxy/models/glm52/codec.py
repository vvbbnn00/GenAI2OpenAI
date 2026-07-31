"""Pinned official tokenizer assets for GLM 5.1 and GLM 5.2."""

import json

from genai_proxy.models.hf_assets import (
    Artifact,
    TokenizerSpec,
    load_template,
    tokenizer_error,
)

GLM_5_2_SPEC = TokenizerSpec(
    family="glm_5_2",
    repository="zai-org/GLM-5.2",
    revision="b4734de4facf877f85769a911abafc5283eab3d9",
    tokenizer=Artifact(
        "tokenizer.json",
        "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    ),
    template=Artifact(
        "chat_template.jinja",
        "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679",
    ),
)

GLM_5_1_SPEC = TokenizerSpec(
    family="glm_5_1",
    repository="zai-org/GLM-5.1",
    revision="26e1bd6e011feb778d25ae34b09b07074139d92d",
    tokenizer=GLM_5_2_SPEC.tokenizer,
    template=Artifact(
        "chat_template.jinja",
        "03b1bbff20331e54647c68167e8ac7f0b5b7ceb40ead372f44826624a9ad79cd",
    ),
)


def official_tool_prompt(spec: TokenizerSpec, function_tools: list[dict]) -> str:
    serialized_tools = json.dumps(function_tools, ensure_ascii=False, sort_keys=True)
    sentinel = "__GENAI2OPENAI_SYSTEM_SENTINEL__"
    while sentinel in serialized_tools:
        sentinel += "_"
    messages = [
        {"role": "system", "content": sentinel},
        {"role": "user", "content": "__GENAI2OPENAI_USER_SENTINEL__"},
    ]
    template = load_template(spec)
    prompt = template.render(
        messages=messages,
        tools=function_tools,
        add_generation_prompt=True,
        enable_thinking=True,
        clear_thinking=True,
        add_vision_id=False,
    )
    try:
        system_marker = f"<|system|>{sentinel}"
        end = prompt.index(system_marker)
        baseline = template.render(
            messages=messages,
            tools=None,
            add_generation_prompt=True,
            enable_thinking=True,
            clear_thinking=True,
            add_vision_id=False,
        )
        baseline_prefix = baseline[: baseline.index(sentinel)]
        if not prompt.startswith(baseline_prefix):
            raise ValueError("official GLM tool prompt boundary changed")
        return prompt[len(baseline_prefix) : end]
    except ValueError as exc:
        raise tokenizer_error(spec, "extract official tool prompt", exc) from exc


def serialize_completion(message: dict) -> str:
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    rendered_content = (
        "None"
        if "content" in message and message.get("content") is None
        else str(content).strip()
    )
    parts = [str(reasoning), "</think>", rendered_content]
    parts.extend(_tool_call(call) for call in message.get("tool_calls") or [])
    return "".join(parts)


def _tool_call(call) -> str:
    function = call.get("function") or {}
    arguments = _json_arguments(function.get("arguments"))
    rendered = "".join(
        f"<arg_key>{key}</arg_key><arg_value>"
        f"{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
        "</arg_value>"
        for key, value in arguments.items()
    )
    return f"<tool_call>{function.get('name', '')}{rendered}</tool_call>"


def _json_arguments(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"arguments": str(value or "")}
    return parsed if isinstance(parsed, dict) else {"arguments": parsed}


__all__ = [
    "GLM_5_1_SPEC",
    "GLM_5_2_SPEC",
    "official_tool_prompt",
    "serialize_completion",
]
