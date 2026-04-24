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


def _parse_tool_call_body(raw, tools=None):
    raw = raw.strip()

    arg_key_call = _parse_arg_key_tool_call(raw, tools)
    if arg_key_call:
        return arg_key_call

    raw = re.sub(r"</?arg_value>", "", raw).strip()

    call = _load_tool_call_json(raw)
    if call:
        return call

    json_obj = _extract_first_json_object(raw)
    if json_obj:
        call = _load_tool_call_json(json_obj)
        if call:
            return call

    jsonish_call = _repair_jsonish_tool_call(raw, tools)
    if jsonish_call:
        return jsonish_call

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


def _load_tool_call_json(raw: str):
    candidates = [raw]
    repaired = _escape_invalid_json_backslashes(raw)
    if repaired != raw:
        candidates.append(repaired)

    for candidate in candidates:
        try:
            call = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(call, dict) and "name" in call:
            call["arguments"] = _normalize_arguments(call.get("arguments", {}))
            return call
    return None


def _repair_jsonish_tool_call(raw: str, tools=None):
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
    if not name_match:
        return None

    name = _canonical_tool_name(name_match.group(1), tools)
    if name is None:
        return None

    arguments_match = re.search(r'"arguments"\s*:\s*\{', raw)
    if not arguments_match:
        return {"name": name, "arguments": {}}

    body_start = arguments_match.end()
    outer_end = raw.rfind("}")
    if outer_end < body_start:
        return None
    args_end = raw.rfind("}", 0, outer_end)
    if args_end < body_start:
        args_end = outer_end

    arguments_body = raw[body_start:args_end].strip()
    arguments = _parse_lenient_key_value_pairs(arguments_body)
    if arguments is None:
        return None
    return {"name": name, "arguments": arguments}


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

    matches, spans = _find_tool_call_blocks(cleaned, tools=tools)

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
        call = _parse_tool_call_body(match, tools=tools)
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


def _parse_arg_key_tool_call(raw: str, tools=None):
    match = re.match(
        r"\s*(?P<name>[^\s<>{}\[\],:]+)\s*<arg_key>\s*(?P<arguments>.*)\s*$",
        raw,
        re.DOTALL,
    )
    if not match:
        return None

    name = _canonical_tool_name(match.group("name"), tools)
    if name is None:
        return None

    arguments = _parse_arg_key_arguments(
        match.group("arguments"),
        arg_keys=_tool_argument_names(name, tools),
    )
    if arguments is None:
        return None
    return {"name": name, "arguments": arguments}


def _parse_arg_key_arguments(raw: str, arg_keys=None):
    raw = raw.strip()
    if not raw:
        return {}

    if "<arg_value>" in raw:
        arg_value_args = _parse_arg_value_arguments(raw)
        if arg_value_args is not None:
            return arg_value_args
    elif "</arg_value>" in raw:
        close_only_args = _parse_close_only_arg_value_arguments(raw, arg_keys or [])
        if close_only_args is not None:
            return close_only_args

    jsonish = re.sub(r"</?arg_value>", "", raw)
    jsonish = re.sub(r"\s*<arg_key>\s*", ", ", jsonish).strip().strip(",")
    if not jsonish:
        return {}

    object_text = jsonish if jsonish.startswith("{") else "{" + _quote_jsonish_keys(jsonish) + "}"
    object_text = _escape_invalid_json_backslashes(object_text)
    try:
        parsed = json.loads(object_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    return _parse_lenient_key_value_pairs(jsonish)


def _parse_arg_value_arguments(raw: str):
    text = raw.strip()
    if not text.startswith("<arg_key>"):
        text = "<arg_key>" + text

    matches = list(
        re.finditer(
            r"<arg_key>\s*\"?(?P<key>[A-Za-z_][\w./@-]*)\"?\s*(?:</arg_key>)?\s*(?:\"?\s*:\s*)?<arg_value>(?P<value>.*?)</arg_value>",
            text,
            re.DOTALL,
        )
    )
    if not matches:
        return None

    arguments = {}
    for match in matches:
        arguments[match.group("key")] = _parse_jsonish_scalar(match.group("value").strip())
    return arguments


def _parse_close_only_arg_value_arguments(raw: str, arg_keys):
    chunks = [chunk.strip() for chunk in raw.split("</arg_value>") if chunk.strip()]
    if not chunks:
        return None

    arguments = {}
    for chunk in chunks:
        parsed = _split_close_only_argument(chunk, arg_keys)
        if not parsed:
            return None
        key, value = parsed
        arguments[key] = value
    return arguments


def _split_close_only_argument(chunk: str, arg_keys):
    for key in sorted(arg_keys, key=len, reverse=True):
        if chunk == key:
            return key, ""
        if chunk.startswith(f"{key} ") or chunk.startswith(f"{key}:") or chunk.startswith(f'{key}"'):
            value = chunk[len(key) :].strip()
            value = re.sub(r'^"?\s*:\s*', "", value).strip()
            return key, _parse_jsonish_scalar(value)

    match = re.match(r"\"?(?P<key>[A-Za-z_][\w./@-]*)\"?\s*(?::\s*)?(?P<value>.*)$", chunk, re.DOTALL)
    if not match:
        return None
    return match.group("key"), _parse_jsonish_scalar(match.group("value").strip())


def _quote_jsonish_keys(text: str) -> str:
    return re.sub(
        r"(^|,)\s*\"?([A-Za-z_][\w./@-]*)\"?\s*:",
        lambda match: f'{match.group(1)} "{match.group(2)}":',
        text,
    ).strip()


def _escape_invalid_json_backslashes(text: str) -> str:
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _parse_lenient_key_value_pairs(text: str):
    key_pattern = re.compile(r"(^|,)\s*\"?([A-Za-z_][\w./@-]*)\"?\s*:\s*")
    matches = list(key_pattern.finditer(text))
    if not matches:
        return None

    arguments = {}
    for index, match in enumerate(matches):
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_value = text[value_start:value_end].strip()
        if raw_value.endswith(","):
            raw_value = raw_value[:-1].strip()
        arguments[match.group(2)] = _parse_jsonish_scalar(raw_value)

    return arguments


def _parse_jsonish_scalar(raw: str):
    value = raw.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        candidate = _escape_invalid_json_backslashes(value)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return value[1:-1]

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    if value.startswith(("{", "[")):
        try:
            return json.loads(_escape_invalid_json_backslashes(value))
        except json.JSONDecodeError:
            pass

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _canonical_tool_name(name: str, tools=None):
    requested = name.strip()
    names = _tool_name_set(tools)
    if not names:
        return requested
    if requested in names:
        return requested

    lowered = requested.lower()
    for candidate in names:
        if candidate.lower() == lowered:
            return candidate
    return None


def _tool_name_set(tools) -> set[str]:
    names = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function_data = tool.get("function", {})
        if not isinstance(function_data, dict):
            function_data = {}
        name = function_data.get("name") or tool.get("name")
        if name:
            names.add(name)
    return names


def _tool_argument_names(name: str, tools=None) -> list[str]:
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function_data = tool.get("function", {})
        if not isinstance(function_data, dict):
            function_data = {}
        tool_name = function_data.get("name") or tool.get("name")
        if tool_name != name:
            continue
        parameters = function_data.get("parameters", {}) or tool.get("input_schema", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        if isinstance(properties, dict):
            return list(properties.keys())
    return []


def _find_tool_call_blocks(content: str, tools=None):
    matches = []
    spans = []
    start_tag = "<tool_call>"
    pos = 0

    while True:
        start = content.find(start_tag, pos)
        if start < 0:
            break

        body_start = start + len(start_tag)
        tool_end = content.find("</tool_call>", body_start)
        arg_value_end = content.find("</arg_value>", body_start)
        if tool_end >= 0 and (
            arg_value_end < 0 or "<arg_key>" in content[body_start:tool_end]
        ):
            body_end = tool_end
            block_end = tool_end + len("</tool_call>")
        elif arg_value_end >= 0 and (tool_end < 0 or arg_value_end < tool_end):
            body_end = arg_value_end
            arg_value_close_end = arg_value_end + len("</arg_value>")
            if tool_end >= 0 and not content[arg_value_close_end:tool_end].strip():
                block_end = tool_end + len("</tool_call>")
            else:
                block_end = arg_value_close_end
        elif tool_end >= 0:
            body_end = tool_end
            block_end = tool_end + len("</tool_call>")
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

    raw_matches, raw_spans = _find_arg_key_tool_blocks(content, tools, spans)
    if raw_matches:
        combined = sorted(
            list(zip(spans, matches, strict=False))
            + list(zip(raw_spans, raw_matches, strict=False)),
            key=lambda item: item[0][0],
        )
        spans = [span for span, _ in combined]
        matches = [match for _, match in combined]

    return matches, spans


def _find_arg_key_tool_blocks(content: str, tools=None, occupied_spans=None):
    occupied_spans = occupied_spans or []
    tool_names = sorted(_tool_name_set(tools), key=len, reverse=True)
    if tool_names:
        name_pattern = "|".join(re.escape(name) for name in tool_names)
        pattern = re.compile(
            rf"(?<![\w./@-])(?P<name>{name_pattern})\s*<arg_key>",
            re.DOTALL,
        )
    else:
        pattern = re.compile(
            r"(?<![\w./@-])(?P<name>[A-Za-z_][\w./@-]*)\s*<arg_key>",
            re.DOTALL,
        )

    matches = list(pattern.finditer(content))
    blocks = []
    spans = []
    for index, match in enumerate(matches):
        start = match.start("name")
        if any(span_start <= start < span_end for span_start, span_end in occupied_spans):
            continue
        end = matches[index + 1].start("name") if index + 1 < len(matches) else len(content)
        blocks.append(content[start:end].strip())
        spans.append((start, end))
    return blocks, spans


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
