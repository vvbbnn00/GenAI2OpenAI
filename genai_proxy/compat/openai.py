import json
import re
import uuid
from datetime import datetime

from flask import jsonify

from genai_proxy.optimizations import (
    DEEPSEEK_ADAPTER,
    GLM_ADAPTER,
    MINIMAX_ADAPTER,
    extract_deepseek_tool_calls,
    inject_deepseek_tool_prompt,
    inject_glm_tool_prompt,
    inject_minimax_tool_prompt,
    is_deepseek_model,
)


TOOL_SYSTEM_PROMPT = """\
You have access to the following tools:

<tools>
{tool_definitions}
</tools>

When you need to call a tool, you MUST use the following XML format. Do NOT use markdown code blocks.

<tool_call>
{{"name": "<function-name>", "arguments": {{<arguments-as-json>}}}}
</tool_call>

Rules:
1. You can call multiple tools by using multiple <tool_call> blocks.
2. If you don't need any tool, just respond normally in plain text without any <tool_call> tags.
3. After receiving tool results, analyze them and either call more tools or give a final answer in plain text.
4. The "arguments" field MUST be a valid JSON object matching the tool's parameter schema.
5. NEVER wrap <tool_call> in markdown code blocks like ```xml or ```json."""

TOOL_CHOICE_REQUIRED_PROMPT = (
    "\nYou MUST call at least one tool in your response. Do NOT respond with plain text only."
)
TOOL_CHOICE_SPECIFIC_PROMPT = (
    '\nYou MUST call the tool named "{name}" in your response.'
)


def openai_error(message, error_type="invalid_request_error", code=None, status=400):
    return (
        jsonify(
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "code": code,
                }
            }
        ),
        status,
    )


def make_error_chunk(message, model="unknown", completion_id=None):
    cid = completion_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    error_chunk = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": f"[Error] {message}"},
                "finish_reason": "error",
            }
        ],
    }
    return f"data: {json.dumps(error_chunk)}\n\ndata: [DONE]\n\n"


def format_tool_definitions(tools):
    definitions = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool["function"]
        params = func.get("parameters", {})
        params_json = json.dumps(params, ensure_ascii=False, indent=2)
        definitions.append(
            f"<tool_definition>\n"
            f"  <name>{func['name']}</name>\n"
            f"  <description>{func.get('description', '')}</description>\n"
            f"  <parameters>\n{params_json}\n  </parameters>\n"
            f"</tool_definition>"
        )
    return "\n".join(definitions)


def inject_tool_prompt(
    messages,
    tools,
    tool_choice=None,
    model=None,
    adapter=None,
):
    if adapter == DEEPSEEK_ADAPTER or (adapter is None and is_deepseek_model(model)):
        return inject_deepseek_tool_prompt(
            messages,
            tools,
            tool_choice,
        )
    if adapter == MINIMAX_ADAPTER:
        return inject_minimax_tool_prompt(
            messages,
            tools,
            tool_choice,
        )
    if adapter == GLM_ADAPTER:
        return inject_glm_tool_prompt(
            messages,
            tools,
            tool_choice,
        )

    tool_defs = format_tool_definitions(tools)
    tool_prompt = TOOL_SYSTEM_PROMPT.format(tool_definitions=tool_defs)

    if tool_choice == "required":
        tool_prompt += TOOL_CHOICE_REQUIRED_PROMPT
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice["function"]["name"]
        tool_prompt += TOOL_CHOICE_SPECIFIC_PROMPT.format(name=name)

    new_messages = []
    has_system = False

    for msg in messages:
        role = msg.get("role")

        if role == "system":
            new_messages.append(
                {
                    "role": "system",
                    "content": msg.get("content", "") + "\n\n" + tool_prompt,
                }
            )
            has_system = True
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "unknown")
            new_messages.append(
                {
                    "role": "user",
                    "content": (
                        "<tool_result>\n"
                        f"  <tool_call_id>{tool_call_id}</tool_call_id>\n"
                        f"  <result>\n{msg.get('content', '')}\n  </result>\n"
                        "</tool_result>"
                    ),
                }
            )
        elif role == "assistant" and msg.get("tool_calls"):
            tc_text = msg.get("content") or ""
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                call_obj = {
                    "name": func.get("name", ""),
                    "arguments": json.loads(func.get("arguments", "{}")),
                }
                tc_text += (
                    f"\n<tool_call>\n{json.dumps(call_obj, ensure_ascii=False)}\n</tool_call>"
                )
            new_messages.append({"role": "assistant", "content": tc_text.strip()})
        else:
            new_messages.append(msg)

    if not has_system:
        new_messages.insert(0, {"role": "system", "content": tool_prompt})

    return new_messages


def strip_think_blocks(content):
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _parse_tool_call_body(raw):
    raw = raw.strip()
    raw = re.sub(r"</?arg_value>", "", raw).strip()

    try:
        call = json.loads(raw)
        if "name" in call:
            call["arguments"] = _normalize_arguments(call.get("arguments", {}))
            return call
    except (json.JSONDecodeError, ValueError):
        pass

    json_obj = _extract_first_json_object(raw)
    if json_obj:
        try:
            call = json.loads(json_obj)
            if "name" in call:
                call["arguments"] = _normalize_arguments(call.get("arguments", {}))
                return call
        except (json.JSONDecodeError, ValueError):
            pass

    name_m = re.search(r"<name>\s*(.*?)\s*</name>", raw, re.DOTALL)
    args_m = re.search(r"<arguments>\s*(.*?)\s*</arguments>", raw, re.DOTALL)
    if name_m:
        name = name_m.group(1).strip()
        arguments = {}
        if args_m:
            args_str = args_m.group(1).strip()
            try:
                arguments = json.loads(args_str)
            except (json.JSONDecodeError, ValueError):
                arguments = {"raw": args_str}
        return {"name": name, "arguments": arguments}

    return None


def extract_tool_calls(content, logger=None, tools=None, model=None, adapter=None):
    cleaned = strip_think_blocks(content)
    cleaned = re.sub(
        r"```(?:xml|json|plaintext|text)?\s*\n?\s*(<tool_call>.*?</tool_call>)\s*\n?\s*```",
        r"\1",
        cleaned,
        flags=re.DOTALL,
    )

    if adapter == DEEPSEEK_ADAPTER or (adapter is None and is_deepseek_model(model)):
        repaired_tool_calls, repaired_remaining = extract_deepseek_tool_calls(
            cleaned,
            tools=tools,
            logger=logger,
        )
        if repaired_tool_calls:
            return repaired_tool_calls, repaired_remaining

    matches, spans = _find_tool_call_blocks(cleaned)

    if not matches:
        if logger:
            logger.debug(
                "No <tool_call> tags found in content (%d chars): %s",
                len(content),
                content[:500],
            )
        return None, content

    if logger:
        logger.debug("Found %d <tool_call> match(es)", len(matches))

    tool_calls = []
    for index, match in enumerate(matches):
        call = _parse_tool_call_body(match)
        if not call:
            if logger:
                logger.warning(
                    "Failed to parse tool_call[%d] — raw: %s",
                    index,
                    match[:300],
                )
            continue

        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(
                        _normalize_arguments(call.get("arguments", {})),
                        ensure_ascii=False,
                    ),
                },
            }
        )

    if not tool_calls:
        return None, content

    remaining = _remove_spans(cleaned, spans).strip()
    return tool_calls, remaining or None


def _normalize_arguments(arguments):
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except json.JSONDecodeError:
            return {"raw": arguments}
    if arguments is None:
        return {}
    return {"value": arguments}


def _find_tool_call_blocks(content: str):
    matches = []
    spans = []
    start_tag = "<tool_call>"
    end_pattern = re.compile(r"</(?:tool_call|arg_value)>", re.DOTALL)
    pos = 0

    while True:
        start = content.find(start_tag, pos)
        if start < 0:
            break

        body_start = start + len(start_tag)
        end_match = end_pattern.search(content, body_start)
        if end_match:
            body_end = end_match.start()
            block_end = end_match.end()
        else:
            json_start = content.find("{", body_start)
            json_end = _json_object_end(content, json_start) if json_start >= 0 else -1
            if json_end > 0:
                body_end = json_end
                block_end = json_end
            else:
                next_start = content.find(start_tag, body_start)
                body_end = next_start if next_start >= 0 else len(content)
                block_end = body_end

        matches.append(content[body_start:body_end])
        spans.append((start, block_end))
        pos = max(block_end, body_start + 1)

    return matches, spans


def _remove_spans(content: str, spans):
    pieces = []
    cursor = 0
    for start, end in spans:
        pieces.append(content[cursor:start])
        cursor = end
    pieces.append(content[cursor:])
    return "".join(pieces)


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    end = _json_object_end(text, start)
    if end < 0:
        return None
    return text[start:end]


def _json_object_end(text: str, start: int) -> int:
    if start < 0 or start >= len(text) or text[start] != "{":
        return -1

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def tag_prefix_len(text, tag):
    max_len = min(len(tag) - 1, len(text))
    for length in range(max_len, 0, -1):
        if text[-length:] == tag[:length]:
            return length
    return 0
