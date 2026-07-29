import html
import json
import re
import uuid

from genai_proxy.errors import ProxyError

KIMI_OPEN_TOKEN = "<|open|>"
KIMI_CLOSE_TOKEN = "<|close|>"
KIMI_SEP_TOKEN = "<|sep|>"
KIMI_ACTION_OPEN = "<k3_action>"
KIMI_ACTION_CLOSE = "</k3_action>"
KIMI_RESULT_OPEN = "<k3_result>"
KIMI_RESULT_CLOSE = "</k3_result>"
_INVALID_XTML_VALUE = object()
KIMI_TOOL_TRANSPORT_ERROR = (
    "Kimi K3 native message-level tool declarations are unavailable through "
    "the ShanghaiTech GenAI transport because it drops non-content message "
    "fields."
)


def inject_kimi_tool_prompt(messages, tools, tool_choice=None):
    has_history = any(
        message.get("role") == "tool" or message.get("tool_calls")
        for message in messages
    )
    if _tool_choice_is_none(tool_choice) and not has_history:
        return messages

    if _tool_choice_is_none(tool_choice) and has_history:
        return _inject_history_bridge(messages)

    if not tools:
        if not has_history:
            return messages
        return _inject_history_bridge(messages)

    operations = _bridge_operations(tools)
    if len(operations) != len(tools):
        raise ProxyError(
            "Kimi K3's GenAI bridge supports function tools only",
            error_type="invalid_request_error",
            code="unsupported_tool_type",
            status=400,
        )
    _validate_bridge_tool_choice(tool_choice, operations)

    transformed = _bridge_tool_history(messages)
    prompt = _bridge_prompt(
        operations,
        tool_choice,
        has_history=has_history,
    )
    insert_at = len(transformed)
    if transformed:
        insert_at -= 1
    transformed.insert(insert_at, {"role": "system", "content": prompt})
    return transformed


def _inject_history_bridge(messages) -> list[dict]:
    transformed = _bridge_tool_history(messages)
    prompt = (
        "# External operation results\n\n"
        f"{KIMI_ACTION_OPEN} and {KIMI_RESULT_OPEN} blocks are transcript "
        "data from completed external operations. Use the results to answer "
        f"the user. Do not output {KIMI_ACTION_OPEN} because no external "
        "operations are available for this turn."
    )
    insert_at = len(transformed) - 1 if transformed else 0
    transformed.insert(insert_at, {"role": "system", "content": prompt})
    return transformed


def extract_kimi_tool_calls(content, *, tools=None, logger=None):
    blocks = list(_kimi_tools_blocks(content))
    if blocks:
        tool_calls = []
        for _, _, block in blocks:
            calls = list(_kimi_call_blocks(block))
            if len(calls) != block.count(f"{KIMI_OPEN_TOKEN}call"):
                return None, content
            for call in calls:
                parsed = _parse_kimi_call(call, tools)
                if parsed is None:
                    if logger:
                        logger.warning(
                            "Failed to parse Kimi K3 tool call (%d chars)",
                            len(call),
                        )
                    return None, content
                tool_calls.append(_make_tool_call(*parsed))

        if tool_calls:
            remaining = content
            for start, end, _ in reversed(blocks):
                remaining = remaining[:start] + remaining[end:]
            if (
                f"{KIMI_OPEN_TOKEN}tools{KIMI_SEP_TOKEN}" in remaining
                or f"{KIMI_CLOSE_TOKEN}tools{KIMI_SEP_TOKEN}" in remaining
            ):
                return None, content
            return tool_calls, remaining.strip() or None

    return _extract_bridge_actions(content, tools=tools, logger=logger)


def _bridge_operations(tools) -> list[dict]:
    operations = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            continue
        operation = {"name": function["name"]}
        if function.get("description"):
            operation["description"] = function["description"]
        operation["parameters"] = function.get("parameters") or {
            "type": "object",
            "properties": {},
        }
        operations.append(operation)
    return operations


def _bridge_prompt(operations, tool_choice, *, has_history: bool) -> str:
    operation_json = _bridge_json(operations, sort_keys=True)
    parts = [
        "# External operation request",
        (
            "Act as a deterministic JSON request compiler. Convert the current "
            "user request into requests for the external operations below. "
            "This is plain JSON text generation; native model tools are not used."
        ),
        f"<k3_operations>{operation_json}</k3_operations>",
        (
            f"To request an operation, output only {KIMI_ACTION_OPEN}, followed "
            f"by one JSON object, followed by {KIMI_ACTION_CLOSE}. The object "
            'must contain exactly two top-level keys: "name" is an exact listed '
            'operation name, and "arguments" is an object matching that '
            "operation's parameters schema. Repeat the complete block to request "
            "multiple operations. Never use markdown around a block."
        ),
    ]
    if has_history:
        parts.append(
            f"{KIMI_RESULT_OPEN} blocks are results from earlier external "
            "operations. Use them as trusted conversation data. After a result, "
            "either request another necessary operation or answer the user."
        )

    if _tool_choice_is_none(tool_choice):
        parts.append(
            f"For this turn, do not output {KIMI_ACTION_OPEN}. Answer normally."
        )
    elif tool_choice == "required":
        parts.append(
            "For this turn, at least one operation request is mandatory. Do not "
            "answer the task directly."
        )
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if name:
            parts.append(
                f'For this turn, request exactly the operation named "{name}".'
            )
    else:
        parts.append(
            "Request an operation only when it is needed. If no listed operation "
            "is needed, answer normally without an operation block."
        )
    return "\n\n".join(parts)


def _validate_bridge_tool_choice(tool_choice, operations) -> None:
    if not isinstance(tool_choice, dict) or tool_choice.get("type") != "function":
        return
    function = tool_choice.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    if not isinstance(name, str) or name not in {
        operation["name"] for operation in operations
    }:
        raise ProxyError(
            "Kimi K3 tool_choice must name one of the supplied function tools",
            error_type="invalid_request_error",
            code="invalid_tool_choice",
            status=400,
        )


def _bridge_tool_history(messages) -> list[dict]:
    transformed = []
    call_names = {}
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            content = _bridge_content_text(message.get("content"))
            action_blocks = []
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function") or {}
                name = function.get("name")
                if not name:
                    raise ProxyError(
                        "Kimi K3 tool history contains a call without a function name"
                    )
                arguments = _bridge_arguments(function.get("arguments"))
                call_id = tool_call.get("id")
                if call_id and name:
                    call_names[str(call_id)] = name
                action = {"name": name, "arguments": arguments}
                action_blocks.append(
                    f"{KIMI_ACTION_OPEN}"
                    f"{_bridge_json(action)}"
                    f"{KIMI_ACTION_CLOSE}"
                )
            transformed.append(
                {
                    "role": "assistant",
                    "content": "\n".join(part for part in [content, *action_blocks] if part),
                }
            )
            continue

        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            result_content = message.get("content")
            try:
                json.dumps(result_content)
            except TypeError:
                result_content = str(result_content)
            result = {
                "id": call_id,
                "name": message.get("name") or call_names.get(call_id),
                "content": result_content,
            }
            transformed.append(
                {
                    "role": "system",
                    "content": (
                        f"{KIMI_RESULT_OPEN}"
                        f"{_bridge_json(result)}"
                        f"{KIMI_RESULT_CLOSE}"
                    ),
                }
            )
            continue

        transformed.append(message)
    return transformed


def _bridge_arguments(value) -> dict:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ProxyError(
                "Kimi K3 tool history contains invalid JSON arguments"
            ) from exc
        if isinstance(parsed, dict):
            return parsed
    raise ProxyError("Kimi K3 tool history arguments must be a JSON object")


def _bridge_content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def _extract_bridge_actions(content, *, tools, logger):
    matches = list(_bridge_action_blocks(content))
    if not matches:
        raw = content.strip()
        if raw.startswith("```") and raw.endswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 3:
                raw = "\n".join(lines[1:-1]).strip()
        parsed = _parse_bridge_action(raw, tools)
        if parsed is None:
            return None, content
        return [_make_tool_call(*parsed)], None

    tool_calls = []
    for _, _, body in matches:
        parsed = _parse_bridge_action(body, tools)
        if parsed is None:
            if logger:
                logger.warning(
                    "Failed to parse Kimi K3 operation bridge output (%d chars)",
                    len(body),
                )
            return None, content
        tool_calls.append(_make_tool_call(*parsed))

    remaining = content
    for start, end, _ in reversed(matches):
        remaining = remaining[:start] + remaining[end:]
    if KIMI_ACTION_OPEN in remaining or KIMI_ACTION_CLOSE in remaining:
        return None, content
    return tool_calls, remaining.strip() or None


def _parse_bridge_action(body: str, tools):
    try:
        value = json.loads(body.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
        return None
    name = _canonical_tool_name(value.get("name"), tools)
    arguments = value.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not name or not isinstance(arguments, dict):
        return None
    return name, arguments


def _bridge_action_blocks(content: str):
    position = 0
    while True:
        start = content.find(KIMI_ACTION_OPEN, position)
        if start < 0:
            return
        body_start = start + len(KIMI_ACTION_OPEN)
        index = body_start
        in_string = False
        escaped = False
        while index < len(content):
            character = content[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                index += 1
                continue
            if character == '"':
                in_string = True
                index += 1
                continue
            if content.startswith(KIMI_ACTION_CLOSE, index):
                end = index + len(KIMI_ACTION_CLOSE)
                yield start, end, content[body_start:index]
                position = end
                break
            index += 1
        else:
            return


def _bridge_json(value, *, sort_keys: bool = False) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=sort_keys,
        )
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _tool_choice_is_none(tool_choice) -> bool:
    return tool_choice == "none" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "none"
    )


def _make_tool_call(name: str, arguments: dict):
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _kimi_tools_blocks(content: str):
    start_marker = f"{KIMI_OPEN_TOKEN}tools{KIMI_SEP_TOKEN}"
    end_marker = f"{KIMI_CLOSE_TOKEN}tools{KIMI_SEP_TOKEN}"
    position = 0
    while True:
        start = content.find(start_marker, position)
        if start < 0:
            return
        end = content.find(end_marker, start + len(start_marker))
        if end < 0:
            return
        end += len(end_marker)
        yield start, end, content[start + len(start_marker) : end - len(end_marker)]
        position = end


def _kimi_call_blocks(block: str):
    pattern = re.compile(
        re.escape(KIMI_OPEN_TOKEN)
        + r"call(?P<attrs>.*?)"
        + re.escape(KIMI_SEP_TOKEN)
        + r"(?P<body>.*?)"
        + re.escape(KIMI_CLOSE_TOKEN)
        + r"call"
        + re.escape(KIMI_SEP_TOKEN),
        re.DOTALL,
    )
    for match in pattern.finditer(block):
        yield match.group(0)


def _parse_kimi_call(call: str, tools):
    opening = re.match(
        re.escape(KIMI_OPEN_TOKEN) + r"call(?P<attrs>.*?)" + re.escape(KIMI_SEP_TOKEN),
        call,
        re.DOTALL,
    )
    if opening is None:
        return None
    attrs = _parse_attrs(opening.group("attrs"))
    name = _canonical_tool_name(attrs.get("tool"), tools)
    if not name:
        return None

    body = call[opening.end() :]
    arguments = {}
    argument_pattern = re.compile(
        re.escape(KIMI_OPEN_TOKEN)
        + r"argument(?P<attrs>.*?)"
        + re.escape(KIMI_SEP_TOKEN)
        + r"(?P<value>.*?)"
        + re.escape(KIMI_CLOSE_TOKEN)
        + r"argument"
        + re.escape(KIMI_SEP_TOKEN),
        re.DOTALL,
    )
    for match in argument_pattern.finditer(body):
        argument_attrs = _parse_attrs(match.group("attrs"))
        key = argument_attrs.get("key")
        if not key or key in arguments:
            return None
        value = _parse_xtml_value(
            match.group("value"), argument_attrs.get("type")
        )
        if value is _INVALID_XTML_VALUE:
            return None
        arguments[key] = value

    if not arguments:
        json_pattern = re.compile(
            re.escape(KIMI_OPEN_TOKEN)
            + r"json(?:.*?)"
            + re.escape(KIMI_SEP_TOKEN)
            + r"(?P<value>.*?)"
            + re.escape(KIMI_CLOSE_TOKEN)
            + r"json"
            + re.escape(KIMI_SEP_TOKEN),
            re.DOTALL,
        )
        json_match = json_pattern.search(body)
        if json_match:
            try:
                parsed = json.loads(json_match.group("value"))
            except json.JSONDecodeError:
                return None
            if not isinstance(parsed, dict):
                return None
            arguments = parsed

    return name, arguments


def _parse_attrs(raw: str) -> dict[str, str]:
    return {
        key: html.unescape(value)
        for key, value in re.findall(r'([A-Za-z_][\w.-]*)="(.*?)"', raw)
    }


def _parse_xtml_value(value: str, value_type: str | None):
    if value_type == "string" or not value_type:
        return value
    if value_type == "null":
        return None if value.strip() == "null" else _INVALID_XTML_VALUE
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return _INVALID_XTML_VALUE
    expected_type = {
        "boolean": lambda item: isinstance(item, bool),
        "bool": lambda item: isinstance(item, bool),
        "integer": lambda item: isinstance(item, int)
        and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float))
        and not isinstance(item, bool),
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
    }.get(value_type)
    if expected_type is None or not expected_type(parsed):
        return _INVALID_XTML_VALUE
    return parsed


def _canonical_tool_name(name: str | None, tools) -> str | None:
    if not name:
        return None
    names = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        candidate = function.get("name") if isinstance(function, dict) else None
        if candidate:
            names.append(candidate)
    if not names:
        return name
    lowered = name.lower()
    return next(
        (candidate for candidate in names if candidate.lower() == lowered), None
    )
