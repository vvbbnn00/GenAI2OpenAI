import json
import re
import uuid
from datetime import datetime

from flask import jsonify

from genai_proxy.optimizations import (
    GLM_ADAPTERS,
    KIMI_K3_ADAPTER,
    MINIMAX_ADAPTER,
    QWEN_3_5_ADAPTER,
    extract_deepseek_tool_calls,
    extract_kimi_tool_calls,
    extract_qwen35_tool_calls,
    inject_deepseek_tool_prompt,
    inject_glm_tool_prompt,
    inject_kimi_tool_prompt,
    inject_minimax_tool_prompt,
    inject_qwen35_tool_prompt,
    is_deepseek_adapter,
    is_deepseek_model,
    select_tool_adapter,
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
2. Call a tool only when it is needed to answer the user's current request and the tool's purpose matches the request.
3. If you can answer directly or no provided tool is relevant, respond normally in plain text without any <tool_call> tags.
4. After receiving tool results, analyze them and either call more tools or give a final answer in plain text.
5. The "arguments" field MUST be a valid JSON object matching the tool's parameter schema.
6. NEVER wrap <tool_call> in markdown code blocks like ```xml or ```json."""

TOOL_CHOICE_REQUIRED_PROMPT = "\nYou MUST call at least one tool in your response. Do NOT respond with plain text only."
TOOL_CHOICE_SPECIFIC_PROMPT = (
    '\nYou MUST call the tool named "{name}" in your response.'
)
TOOL_CHOICE_NONE_PROMPT = "\nFor this turn, do not call any tool."


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
    reasoning_config=None,
):
    resolved_adapter = adapter or (select_tool_adapter(model) if model else None)
    if is_deepseek_adapter(resolved_adapter) or (
        resolved_adapter is None and is_deepseek_model(model)
    ):
        return inject_deepseek_tool_prompt(
            messages,
            tools,
            tool_choice,
            adapter=resolved_adapter,
            reasoning_config=reasoning_config,
        )
    if resolved_adapter == MINIMAX_ADAPTER:
        return inject_minimax_tool_prompt(
            messages,
            tools,
            tool_choice,
        )
    if resolved_adapter == KIMI_K3_ADAPTER:
        return inject_kimi_tool_prompt(messages, tools, tool_choice)
    if resolved_adapter in GLM_ADAPTERS:
        return inject_glm_tool_prompt(
            messages,
            tools,
            tool_choice,
            adapter=resolved_adapter,
            reasoning_config=reasoning_config,
        )
    if resolved_adapter == QWEN_3_5_ADAPTER:
        return inject_qwen35_tool_prompt(messages, tools, tool_choice)

    tool_defs = format_tool_definitions(tools)
    tool_prompt = TOOL_SYSTEM_PROMPT.format(tool_definitions=tool_defs)

    if tool_choice == "required":
        tool_prompt += TOOL_CHOICE_REQUIRED_PROMPT
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice["function"]["name"]
        tool_prompt += TOOL_CHOICE_SPECIFIC_PROMPT.format(name=name)
    elif _tool_choice_is_none(tool_choice):
        tool_prompt += TOOL_CHOICE_NONE_PROMPT

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
                tc_text += f"\n<tool_call>\n{json.dumps(call_obj, ensure_ascii=False)}\n</tool_call>"
            new_messages.append({"role": "assistant", "content": tc_text.strip()})
        else:
            new_messages.append(msg)

    if not has_system:
        new_messages.insert(0, {"role": "system", "content": tool_prompt})

    return new_messages


def strip_think_blocks(content):
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _tool_choice_is_none(tool_choice) -> bool:
    return tool_choice == "none" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "none"
    )


def _parse_tool_call_body(raw, tools=None):
    raw = raw.strip()

    arg_key_call = _parse_arg_key_tool_call(raw, tools)
    if arg_key_call:
        return arg_key_call

    raw = re.sub(r"</?arg_value>", "", raw).strip()

    call = _load_tool_call_json(raw, tools)
    if call:
        return call

    json_obj = _extract_first_json_object(raw)
    if json_obj:
        call = _load_tool_call_json(json_obj, tools)
        if call:
            return call

    jsonish_call = _repair_jsonish_tool_call(raw, tools)
    if jsonish_call:
        return jsonish_call

    bare_name_call = _parse_bare_tool_name_call(raw, tools)
    if bare_name_call:
        return bare_name_call

    name_m = re.search(r"<name>\s*(.*?)\s*</name>", raw, re.DOTALL)
    args_m = re.search(r"<arguments>\s*(.*?)\s*</arguments>", raw, re.DOTALL)
    if name_m:
        name = name_m.group(1).strip()
        name = _canonical_tool_name(name, tools) or name
        arguments = {}
        if args_m:
            args_str = args_m.group(1).strip()
            try:
                arguments = json.loads(args_str)
            except (json.JSONDecodeError, ValueError):
                arguments = {"raw": args_str}
        arguments = _normalize_arguments(arguments)
        return {"name": name, "arguments": arguments}

    return None


def _parse_bare_tool_name_call(raw: str, tools=None):
    if not _tool_name_set(tools):
        return None
    if not re.fullmatch(r"[A-Za-z_][\w./@-]*", raw.strip()):
        return None

    name = _canonical_tool_name(raw, tools)
    if name is None:
        return None
    return {"name": name, "arguments": {}}


def _load_tool_call_json(raw: str, tools=None):
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
            name = call.get("name")
            if isinstance(name, str):
                canonical_name = _canonical_tool_name(name, tools)
                if canonical_name:
                    call["name"] = canonical_name
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
    arguments = _parse_lenient_key_value_pairs(
        arguments_body,
        argument_types=_tool_argument_types(name, tools),
    )
    if arguments is None:
        return None
    return {"name": name, "arguments": arguments}


def extract_tool_calls(
    content,
    logger=None,
    tools=None,
    model=None,
    adapter=None,
    tool_choice=None,
):
    cleaned = strip_think_blocks(content)
    cleaned = re.sub(
        r"```(?:xml|json|plaintext|text)?\s*\n?\s*(<tool_call>.*?</tool_call>)\s*\n?\s*```",
        r"\1",
        cleaned,
        flags=re.DOTALL,
    )

    resolved_adapter = adapter or (select_tool_adapter(model) if model else None)
    if resolved_adapter == KIMI_K3_ADAPTER:
        kimi_tool_calls, kimi_remaining = extract_kimi_tool_calls(
            cleaned,
            tools=tools,
            tool_choice=tool_choice,
            logger=logger,
        )
        if kimi_tool_calls:
            return kimi_tool_calls, kimi_remaining
    if is_deepseek_adapter(resolved_adapter) or (
        resolved_adapter is None and is_deepseek_model(model)
    ):
        repaired_tool_calls, repaired_remaining = extract_deepseek_tool_calls(
            cleaned,
            tools=tools,
            logger=logger,
            adapter=resolved_adapter,
        )
        if repaired_tool_calls:
            return repaired_tool_calls, repaired_remaining
    if resolved_adapter == QWEN_3_5_ADAPTER:
        qwen_tool_calls, qwen_remaining = extract_qwen35_tool_calls(
            cleaned,
            tools=tools,
            logger=logger,
        )
        if qwen_tool_calls:
            return qwen_tool_calls, qwen_remaining

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
        r"\s*(?P<name>[^\s<>{}\[\],:]+)\s*(?P<arguments><(?:arg_key|arg_value)>.*)\s*$",
        raw,
        re.DOTALL,
    )
    if not match:
        return None

    name = _canonical_tool_name(match.group("name"), tools)
    if name is None:
        return None

    argument_types = _tool_argument_types(name, tools)
    arguments = _parse_arg_key_arguments(
        match.group("arguments"),
        arg_keys=list(argument_types),
        argument_types=argument_types,
    )
    if arguments is None:
        return None
    return {"name": name, "arguments": arguments}


def _parse_arg_key_arguments(raw: str, arg_keys=None, argument_types=None):
    raw = raw.strip()
    if not raw:
        return {}

    argument_types = argument_types or {}
    if "<arg_value>" in raw:
        arg_value_args = _parse_arg_value_arguments(raw, argument_types=argument_types)
        if arg_value_args is not None:
            return arg_value_args
    elif "</arg_value>" in raw:
        close_only_raw = re.sub(r"^\s*<arg_key>\s*", "", raw)
        close_only_args = _parse_close_only_arg_value_arguments(
            close_only_raw,
            arg_keys or [],
            argument_types=argument_types,
        )
        if close_only_args is not None:
            return close_only_args

    jsonish = re.sub(r"</?arg_value>", "", raw)
    jsonish = re.sub(r"\s*<arg_key>\s*", ", ", jsonish).strip().strip(",")
    if not jsonish:
        return {}

    object_text = (
        jsonish if jsonish.startswith("{") else "{" + _quote_jsonish_keys(jsonish) + "}"
    )
    object_text = _escape_invalid_json_backslashes(object_text)
    try:
        parsed = json.loads(object_text)
        if isinstance(parsed, dict):
            return _coerce_jsonish_arguments(parsed, argument_types)
    except json.JSONDecodeError:
        pass

    return _parse_lenient_key_value_pairs(jsonish, argument_types=argument_types)


def _parse_arg_value_arguments(raw: str, argument_types=None):
    text = raw.strip()
    argument_types = argument_types or {}
    reversed_args = _parse_reversed_arg_value_arguments(
        text, argument_types=argument_types
    )
    if reversed_args is not None:
        return reversed_args

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
        key = match.group("key")
        arguments[key] = _parse_xml_argument_scalar(
            match.group("value").strip(),
            argument_types.get(key),
        )
    return arguments


def _parse_reversed_arg_value_arguments(raw: str, argument_types=None):
    argument_types = argument_types or {}
    matches = list(
        re.finditer(
            r"<arg_value>\s*\"?(?P<key>[A-Za-z_][\w./@-]*)\"?\s*</arg_key>\s*"
            r"<arg_value>(?P<value>.*?)(?=(?:</arg_value>)?\s*<arg_value>\s*\"?[A-Za-z_][\w./@-]*\"?\s*</arg_key>\s*<arg_value>|(?:</arg_value>)?\s*$)",
            raw,
            re.DOTALL,
        )
    )
    if not matches:
        return None

    arguments = {}
    for match in matches:
        value = match.group("value").strip()
        if value.endswith("</arg_value>"):
            value = value[: -len("</arg_value>")].strip()
        key = match.group("key")
        arguments[key] = _parse_xml_argument_scalar(value, argument_types.get(key))
    return arguments


def _parse_close_only_arg_value_arguments(raw: str, arg_keys, argument_types=None):
    chunks = [chunk.strip() for chunk in raw.split("</arg_value>") if chunk.strip()]
    if not chunks:
        return None

    argument_types = argument_types or {}
    arguments = {}
    for chunk in chunks:
        parsed = _split_close_only_argument(
            chunk,
            arg_keys,
            argument_types=argument_types,
        )
        if not parsed:
            return None
        key, value = parsed
        arguments[key] = value
    return arguments


def _split_close_only_argument(chunk: str, arg_keys, argument_types=None):
    argument_types = argument_types or {}
    for key in sorted(arg_keys, key=len, reverse=True):
        if chunk == key:
            return key, ""
        if chunk.startswith(f"{key} "):
            value = chunk[len(key) :].strip()
            return key, _parse_xml_argument_scalar(value, argument_types.get(key))
        if chunk.startswith(f"{key}:") or chunk.startswith(f'{key}"'):
            value = chunk[len(key) :].strip()
            value = re.sub(r'^"?\s*:\s*', "", value).strip()
            return key, _parse_jsonish_argument_scalar(
                value,
                argument_types.get(key),
                strip_unclosed_string=True,
            )

    match = re.match(
        r"\"?(?P<key>[A-Za-z_][\w./@-]*)\"?\s*(?::\s*)?(?P<value>.*)$", chunk, re.DOTALL
    )
    if not match:
        return None
    key = match.group("key")
    return key, _parse_jsonish_argument_scalar(
        match.group("value").strip(),
        argument_types.get(key),
        strip_unclosed_string=True,
    )


def _quote_jsonish_keys(text: str) -> str:
    return re.sub(
        r"(^|,)\s*\"?([A-Za-z_][\w./@-]*)\"?\s*:",
        lambda match: f'{match.group(1)} "{match.group(2)}":',
        text,
    ).strip()


def _escape_invalid_json_backslashes(text: str) -> str:
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _parse_lenient_key_value_pairs(text: str, argument_types=None):
    key_pattern = re.compile(r"(^|,)\s*\"?([A-Za-z_][\w./@-]*)\"?\s*:\s*")
    matches = list(key_pattern.finditer(text))
    if not matches:
        return None

    argument_types = argument_types or {}
    arguments = {}
    for index, match in enumerate(matches):
        value_start = match.end()
        value_end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(text)
        )
        raw_value = text[value_start:value_end].strip()
        if raw_value.endswith(","):
            raw_value = raw_value[:-1].strip()
        key = match.group(2)
        arguments[key] = _parse_jsonish_argument_scalar(
            raw_value,
            argument_types.get(key),
            strip_unclosed_string=True,
        )

    return arguments


def _parse_xml_argument_scalar(raw: str, expected_type: str | None = None):
    value = raw.strip()
    if expected_type in {None, "string"}:
        return value
    return _parse_typed_value(value, expected_type)


def _parse_jsonish_argument_scalar(
    raw: str,
    expected_type: str | None = None,
    *,
    strip_unclosed_string: bool = False,
):
    value = raw.strip()
    if expected_type in {None, "string"}:
        return _parse_jsonish_string_scalar(
            value,
            strip_unclosed_string=strip_unclosed_string,
        )
    return _parse_typed_value(value, expected_type)


def _parse_jsonish_string_scalar(raw: str, *, strip_unclosed_string: bool = False):
    value = raw.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        candidate = _escape_invalid_json_backslashes(value)
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, str) else str(parsed)
        except json.JSONDecodeError:
            return value[1:-1] if strip_unclosed_string else value
    if value.startswith('"') and not value.endswith('"'):
        if strip_unclosed_string and not _has_unescaped_quote_after_start(value):
            return value[1:]
        return value
    return value


def _coerce_jsonish_arguments(arguments: dict, argument_types: dict[str, str | None]):
    if not argument_types:
        return arguments
    coerced = {}
    for key, value in arguments.items():
        expected_type = argument_types.get(key)
        if expected_type == "string":
            coerced[key] = _coerce_string_value(value)
        elif expected_type and isinstance(value, str):
            coerced[key] = _parse_typed_value(value, expected_type)
        else:
            coerced[key] = value
    return coerced


def _coerce_string_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _has_unescaped_quote_after_start(text: str) -> bool:
    escaped = False
    for char in text[1:]:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            return True
    return False


def _parse_jsonish_scalar(raw: str):
    value = raw.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        candidate = _escape_invalid_json_backslashes(value)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return value[1:-1]
    if value.startswith('"') and not value.endswith('"'):
        return value[1:]

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
    return list(_tool_argument_types(name, tools))


def _tool_required_argument_names(name: str, tools=None) -> list[str]:
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
        required = (
            parameters.get("required", []) if isinstance(parameters, dict) else []
        )
        return required if isinstance(required, list) else []
    return []


def _tool_argument_types(name: str, tools=None) -> dict[str, str | None]:
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
        properties = (
            parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        )
        if isinstance(properties, dict):
            return {
                key: _schema_property_type(value) for key, value in properties.items()
            }
    return {}


def _schema_property_type(property_schema) -> str | None:
    if not isinstance(property_schema, dict):
        return None

    schema_type = property_schema.get("type")
    if isinstance(schema_type, str):
        return schema_type
    if isinstance(schema_type, list):
        type_names = [item for item in schema_type if isinstance(item, str)]
        if "string" in type_names:
            return "string"
        non_null_types = [item for item in type_names if item != "null"]
        if len(non_null_types) == 1:
            return non_null_types[0]
    for union_key in ("anyOf", "oneOf"):
        union_schemas = property_schema.get(union_key)
        if isinstance(union_schemas, list):
            union_types = [
                _schema_property_type(item)
                for item in union_schemas
                if isinstance(item, dict)
            ]
            if "string" in union_types:
                return "string"
            non_null_types = [item for item in union_types if item and item != "null"]
            if len(set(non_null_types)) == 1:
                return non_null_types[0]
    return None


def _parse_typed_value(value: str, expected_type: str | None):
    if expected_type in {"object", "array"}:
        try:
            return json.loads(_escape_invalid_json_backslashes(value))
        except json.JSONDecodeError:
            return value
    if expected_type == "integer":
        try:
            return int(value)
        except ValueError:
            return value
    if expected_type == "number":
        try:
            parsed = float(value)
            return int(parsed) if parsed == int(parsed) else parsed
        except ValueError:
            return value
    if expected_type == "boolean":
        lowered = value.lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
        return value
    return _parse_jsonish_scalar(value)


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
        next_start = content.find(start_tag, body_start)
        if (
            next_start >= 0
            and (tool_end < 0 or next_start < tool_end)
            and not _tool_call_prefix_looks_parseable(
                content[body_start:next_start], tools
            )
        ):
            body_end = next_start
            block_end = next_start
        elif tool_end >= 0 and (
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
                body_end = next_start if next_start >= 0 else len(content)
                block_end = body_end

        matches.append(content[body_start:body_end])
        spans.append((start, block_end))
        pos = max(block_end, body_start + 1)

    raw_matches, raw_spans = _find_arg_key_tool_blocks(content, tools, spans)
    minimax_matches, minimax_spans = _find_minimax_tool_call_blocks(
        content,
        tools,
        spans + raw_spans,
    )
    transcript_matches, transcript_spans = _find_claude_code_transcript_tool_blocks(
        content,
        tools,
        spans + raw_spans + minimax_spans,
    )
    inline_matches, inline_spans = _find_inline_tool_argument_blocks(
        content,
        tools,
        spans + raw_spans + minimax_spans + transcript_spans,
    )
    combined = sorted(
        list(zip(spans, matches, strict=False))
        + list(zip(raw_spans, raw_matches, strict=False))
        + list(zip(minimax_spans, minimax_matches, strict=False))
        + list(zip(transcript_spans, transcript_matches, strict=False)),
        key=lambda item: item[0][0],
    )
    combined = sorted(
        combined + list(zip(inline_spans, inline_matches, strict=False)),
        key=lambda item: item[0][0],
    )
    spans = [span for span, _ in combined]
    matches = [match for _, match in combined]

    return matches, spans


def _tool_call_prefix_looks_parseable(prefix: str, tools=None) -> bool:
    text = prefix.strip()
    if not text:
        return False
    if text.startswith(("{", "<name>", "<arg_key>")) or '"name"' in text[:80]:
        return True
    for name in _tool_name_set(tools):
        if re.match(
            rf"{re.escape(name)}(?:\s*<arg_key>|\s*<arg_value>|\s*$)",
            text,
            re.IGNORECASE,
        ):
            return True
    return False


def _find_arg_key_tool_blocks(content: str, tools=None, occupied_spans=None):
    occupied_spans = occupied_spans or []
    tool_names = sorted(_tool_name_set(tools), key=len, reverse=True)
    if tool_names:
        name_pattern = "|".join(re.escape(name) for name in tool_names)
        pattern = re.compile(
            rf"(?<![\w./@-])(?P<name>{name_pattern})\s*(?:<arg_key>|<arg_value>\s*[A-Za-z_][\w./@-]*\s*</arg_key>\s*<arg_value>)",
            re.DOTALL,
        )
    else:
        pattern = re.compile(
            r"(?<![\w./@-])(?P<name>[A-Za-z_][\w./@-]*)\s*(?:<arg_key>|<arg_value>\s*[A-Za-z_][\w./@-]*\s*</arg_key>\s*<arg_value>)",
            re.DOTALL,
        )

    matches = list(pattern.finditer(content))
    blocks = []
    spans = []
    for index, match in enumerate(matches):
        start = match.start("name")
        if any(
            span_start <= start < span_end for span_start, span_end in occupied_spans
        ):
            continue
        end = (
            matches[index + 1].start("name")
            if index + 1 < len(matches)
            else len(content)
        )
        blocks.append(content[start:end].strip())
        spans.append((start, end))
    return blocks, spans


def _find_minimax_tool_call_blocks(content: str, tools=None, occupied_spans=None):
    occupied_spans = occupied_spans or []
    pattern = re.compile(r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
    blocks = []
    spans = []

    for match in pattern.finditer(content):
        if any(
            span_start <= match.start() < span_end
            for span_start, span_end in occupied_spans
        ):
            continue
        inner = match.group(1)
        invocations = list(
            re.finditer(
                r"<invoke\s+name=(?P<quote>[\"']?)(?P<name>[^\"'>\s]+)(?P=quote)>(?P<body>.*?)</invoke>",
                inner,
                re.DOTALL,
            )
        )
        for invoke in invocations:
            name = _canonical_tool_name(invoke.group("name"), tools)
            if name is None:
                continue
            arguments = _parse_minimax_parameters(name, invoke.group("body"), tools)
            blocks.append(
                json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
            )
            spans.append((match.start(), match.end()))

    return blocks, spans


def _parse_minimax_parameters(name: str, body: str, tools=None):
    argument_types = _tool_argument_types(name, tools)
    arguments = {}
    parameter_pattern = re.compile(
        r"<parameter\s+name=(?P<quote>[\"']?)(?P<key>[^\"'>\s]+)(?P=quote)>(?P<value>.*?)</parameter>",
        re.DOTALL,
    )
    for parameter in parameter_pattern.finditer(body):
        key = parameter.group("key")
        value = parameter.group("value").strip()
        arguments[key] = _parse_xml_argument_scalar(value, argument_types.get(key))
    return arguments


def _find_inline_tool_argument_blocks(content: str, tools=None, occupied_spans=None):
    occupied_spans = occupied_spans or []
    tool_names = sorted(_tool_name_set(tools), key=len, reverse=True)
    if not tool_names:
        return [], []

    blocks = []
    spans = []
    lines = content.splitlines(keepends=True)
    offsets = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)

    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        line_start = offsets[line_index]
        if any(
            span_start <= line_start < span_end
            for span_start, span_end in occupied_spans
        ):
            continue

        parsed = _parse_inline_tool_line(stripped, tools)
        if not parsed:
            continue

        name, arguments = parsed
        end_index = line_index + 1
        end_index = _consume_inline_heredoc_arguments(
            arguments,
            _tool_argument_types(name, tools),
            lines,
            end_index,
        )
        while end_index < len(lines):
            next_stripped = lines[end_index].strip()
            if not next_stripped:
                end_index += 1
                continue
            continuation = _parse_inline_continuation_line(next_stripped, name, tools)
            if continuation is None:
                break
            _, key, value = continuation
            arguments[key] = value
            end_index += 1

        block_end = offsets[end_index] if end_index < len(offsets) else len(content)
        blocks.append(
            json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
        )
        spans.append((line_start, block_end))

    return blocks, spans


def _consume_inline_heredoc_arguments(
    arguments: dict,
    argument_types: dict[str, str | None],
    lines: list[str],
    start_index: int,
) -> int:
    for key, value in list(arguments.items()):
        if argument_types.get(key) != "string" or not isinstance(value, str):
            continue
        delimiter = _extract_heredoc_delimiter(value)
        if not delimiter:
            continue

        collected = []
        end_index = start_index
        while end_index < len(lines):
            line_value = lines[end_index].rstrip("\r\n")
            collected.append(line_value)
            end_index += 1
            if line_value.strip() == delimiter:
                arguments[key] = value + "\n" + "\n".join(collected)
                return end_index

    return start_index


def _extract_heredoc_delimiter(value: str) -> str | None:
    matches = list(
        re.finditer(
            r"<<-?\s*(?P<quote>['\"]?)(?P<delimiter>[A-Za-z_][\w.-]*)(?P=quote)",
            value,
        )
    )
    if not matches:
        return None
    return matches[-1].group("delimiter")


def _parse_inline_tool_line(line: str, tools=None):
    for name in sorted(_tool_name_set(tools), key=len, reverse=True):
        if not line.lower().startswith(name.lower()):
            continue
        rest = line[len(name) :].strip()
        if not rest:
            continue
        parsed = _parse_inline_argument_text(name, rest, tools)
        if parsed is not None:
            return name, parsed
    return None


def _parse_inline_continuation_line(line: str, name: str, tools=None):
    parsed = _parse_inline_argument_text(name, line, tools)
    if not parsed or len(parsed) != 1:
        return None
    key, value = next(iter(parsed.items()))
    return name, key, value


def _parse_inline_argument_text(name: str, text: str, tools=None):
    argument_names = _tool_argument_names(name, tools)
    if not argument_names:
        return None
    argument_types = _tool_argument_types(name, tools)

    key_pattern = "|".join(
        re.escape(key) for key in sorted(argument_names, key=len, reverse=True)
    )
    matches = list(
        re.finditer(
            rf"(?<![\w./@-])(?P<key>{key_pattern})\s*:\s*",
            text,
            re.IGNORECASE,
        )
    )
    if not matches:
        return None

    arguments = {}
    canonical_keys = {key.lower(): key for key in argument_names}
    for index, match in enumerate(matches):
        key = canonical_keys.get(match.group("key").lower(), match.group("key"))
        value_start = match.end()
        value_end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(text)
        )
        raw_value = text[value_start:value_end].strip()
        if not raw_value:
            return None
        arguments[key] = _parse_jsonish_argument_scalar(
            raw_value,
            argument_types.get(key),
        )
    return arguments


def _find_claude_code_transcript_tool_blocks(
    content: str, tools=None, occupied_spans=None
):
    occupied_spans = occupied_spans or []
    tool_names = sorted(_tool_name_set(tools), key=len, reverse=True)
    if not tool_names:
        return [], []

    name_pattern = "|".join(re.escape(name) for name in tool_names)
    pattern = re.compile(
        rf"(?m)(?<![\w./@-])(?P<name>{name_pattern})(?P<label>[^\n<]*)?\nIN\n",
        re.DOTALL,
    )
    matches = list(pattern.finditer(content))
    blocks = []
    spans = []

    for index, match in enumerate(matches):
        start = match.start("name")
        if any(
            span_start <= start < span_end for span_start, span_end in occupied_spans
        ):
            continue

        body_start = match.end()
        candidates = []
        if index + 1 < len(matches):
            candidates.append(matches[index + 1].start("name"))
        for marker in (
            "<tool_call>",
            "<minimax:tool_call>",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜function_calls>",
        ):
            marker_pos = content.find(marker, body_start)
            if marker_pos >= 0:
                candidates.append(marker_pos)
        next_tool_pos = min(candidates) if candidates else len(content)
        out_marker = _find_transcript_out_marker(content, body_start, next_tool_pos)
        body_end = out_marker[0] if out_marker else next_tool_pos
        span_end = next_tool_pos
        if out_marker:
            after_out = out_marker[1]
            next_after_out = [
                pos
                for pos in (
                    content.find("<tool_call>", after_out),
                    content.find("<minimax:tool_call>", after_out),
                    content.find("<｜DSML｜tool_calls>", after_out),
                    content.find("<｜DSML｜function_calls>", after_out),
                    matches[index + 1].start("name")
                    if index + 1 < len(matches)
                    else -1,
                )
                if pos >= 0
            ]
            span_end = min(next_after_out) if next_after_out else len(content)

        raw_input = content[body_start:body_end].strip()
        if not raw_input:
            continue

        name = _canonical_tool_name(match.group("name"), tools)
        if name is None:
            continue
        arguments = _transcript_tool_arguments(
            name,
            raw_input,
            tools,
            label=(match.group("label") or "").strip(),
        )
        if arguments is None:
            continue

        blocks.append(
            json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
        )
        spans.append((start, span_end))

    return blocks, spans


def _find_transcript_out_marker(
    content: str, start: int, end: int
) -> tuple[int, int] | None:
    search_area = content[start:end]
    match = re.search(r"(?:\r?\n)OUT(?:\r?\n|$)", search_area)
    if not match:
        return None
    return start + match.start(), start + match.end()


def _transcript_tool_arguments(name: str, raw_input: str, tools=None, label: str = ""):
    raw_input = raw_input.strip()
    if not raw_input:
        return {}
    if raw_input.startswith("{"):
        try:
            parsed = json.loads(_escape_invalid_json_backslashes(raw_input))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    argument_names = _tool_argument_names(name, tools)
    argument_types = _tool_argument_types(name, tools)
    required_names = _tool_required_argument_names(name, tools)
    target_names = required_names if len(required_names) == 1 else argument_names
    if len(target_names) == 1:
        target_name = target_names[0]
        arguments = {
            target_name: _parse_xml_argument_scalar(
                raw_input,
                argument_types.get(target_name),
            )
        }
        if label and "description" in argument_names and target_name != "description":
            arguments["description"] = label
        return arguments
    return None


def _remove_spans(content: str, spans):
    merged_spans = []
    for start, end in sorted(spans):
        if not merged_spans or start > merged_spans[-1][1]:
            merged_spans.append([start, end])
        else:
            merged_spans[-1][1] = max(merged_spans[-1][1], end)

    pieces = []
    cursor = 0
    for start, end in merged_spans:
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
