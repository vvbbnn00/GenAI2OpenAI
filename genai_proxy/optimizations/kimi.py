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
KIMI_FINAL_OPEN = "<k3_final>"
KIMI_FINAL_CLOSE = "</k3_final>"
KIMI_RESULT_PREFIX = "Completed client action result: "
KIMI_STATE_OPEN = "<k3_state>"
KIMI_STATE_CLOSE = "</k3_state>"
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
    if transformed and transformed[-1].get("role") == "user":
        insert_at -= 1
    transformed.insert(insert_at, {"role": "system", "content": prompt})
    return transformed


def _inject_history_bridge(messages) -> list[dict]:
    return _bridge_tool_history(messages)


def kimi_tool_retry_messages(
    messages: list[dict],
    *,
    tool_choice=None,
    force_action: bool = False,
) -> list[dict]:
    if not messages:
        return messages

    requires_action = force_action or _tool_choice_requires_action(tool_choice)
    requirement = (
        "Return at least one complete action block and no other text."
        if requires_action
        else (
            f"Return complete {KIMI_ACTION_OPEN} blocks, or one "
            f"{KIMI_FINAL_OPEN} block if the task is complete."
        )
    )
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if name:
            requirement = (
                f'Return one complete action block named "{name}" and no other text.'
            )

    retry_prompt = (
        "The previous response did not use a complete valid client response "
        f"envelope and was discarded. {requirement} These envelopes are plain "
        "response data for the client, not native model tool calls. Use valid JSON "
        "in action blocks and no Markdown."
    )
    insert_at = len(messages)
    if messages[-1].get("role") == "user":
        insert_at -= 1
    return [
        *messages[:insert_at],
        {"role": "system", "content": retry_prompt},
        *messages[insert_at:],
    ]


def extract_kimi_tool_calls(
    content,
    *,
    tools=None,
    tool_choice=None,
    logger=None,
):
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

    return _extract_bridge_actions(
        content,
        tools=tools,
        allow_bare=_tool_choice_requires_action(tool_choice),
        logger=logger,
    )


def extract_kimi_final_response(content) -> tuple[bool, str | None]:
    if not isinstance(content, str):
        return False, content
    if (
        content.count(KIMI_FINAL_OPEN) != 1
        or content.count(KIMI_FINAL_CLOSE) != 1
    ):
        return False, content

    start = content.find(KIMI_FINAL_OPEN)
    end = content.find(KIMI_FINAL_CLOSE, start + len(KIMI_FINAL_OPEN))
    if end < 0:
        return False, content
    if content[:start].strip() or content[end + len(KIMI_FINAL_CLOSE) :].strip():
        return False, content
    return True, content[start + len(KIMI_FINAL_OPEN) : end]


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
        "# Client response protocol",
        f"<k3_actions>{operation_json}</k3_actions>",
        (
            "The response channel is a client data channel, not native model "
            "tool use. Return exactly one of these forms and no text outside it:\n"
            f"1. When external work is needed, write {KIMI_ACTION_OPEN}"
            '{"name":"exact listed name","arguments":{}}'
            f"{KIMI_ACTION_CLOSE}. Arguments must match the schema. Repeat the "
            "complete block for independent parallel work.\n"
            f"2. Only when the user request is complete, write {KIMI_FINAL_OPEN}"
            f"the normal answer{KIMI_FINAL_CLOSE}.\n"
            "Do not discuss, defer, or reinterpret this client response protocol. "
            "Do not use Markdown around the envelopes."
        ),
    ]
    if has_history:
        parts.append(
            f"Past user messages beginning `{KIMI_RESULT_PREFIX}` are completed "
            "client actions and results, not new requests. Match parallel results "
            "by `id`. Continue the plan in the single prior continuation state "
            "from the newest result. Do not restart analysis of the original task "
            "unless the result invalidates the plan."
        )
    if tool_choice == "required":
        parts.append(
            "This response must contain at least one complete action block and "
            f"must not contain {KIMI_FINAL_OPEN}."
        )
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if name:
            parts.append(
                f'This response must contain one complete action block named "{name}" '
                f"and must not contain {KIMI_FINAL_OPEN}."
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
    call_records = {}
    latest_tool_index = max(
        (
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant" and message.get("tool_calls")
        ),
        default=-1,
    )
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            content = _bridge_content_text(message.get("content"))
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                name = function.get("name")
                if not name:
                    raise ProxyError(
                        "Kimi K3 tool history contains a call without a function name"
                    )
                arguments = _bridge_arguments(function.get("arguments"))
                call_id = tool_call.get("id")
                if call_id:
                    call_records[str(call_id)] = {
                        "name": name,
                        "arguments": arguments,
                    }
            if content:
                transformed_message = {
                    "role": "assistant",
                    "content": content,
                }
                if message.get("name"):
                    transformed_message["name"] = message["name"]
                transformed.append(transformed_message)
            continue

        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            call_record = call_records.pop(call_id, {})
            result_content = message.get("content")
            try:
                json.dumps(result_content)
            except TypeError:
                result_content = str(result_content)
            result = {
                "id": call_id,
                "name": message.get("name") or call_record.get("name"),
                "arguments": call_record.get("arguments") or {},
                "content": result_content,
            }
            transformed.append(
                {
                    "role": "user",
                    "content": f"{KIMI_RESULT_PREFIX}{_bridge_json(result)}",
                }
            )
            continue

        transformed.append(message)

    if latest_tool_index >= 0:
        reasoning = messages[latest_tool_index].get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            state = {
                "role": "system",
                "content": (
                    "# Prior continuation state\n"
                    "This is the assistant's immediately preceding reasoning "
                    "before the latest completed client action. Use it only as "
                    "progress context; the completed result and current protocol "
                    "are authoritative.\n"
                    f"{KIMI_STATE_OPEN}{_bridge_json(reasoning)}{KIMI_STATE_CLOSE}"
                ),
            }
            insert_at = len(transformed)
            if transformed and transformed[-1].get("role") == "user":
                insert_at -= 1
            transformed.insert(insert_at, state)
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


def _extract_bridge_actions(content, *, tools, allow_bare: bool, logger):
    matches = list(_bridge_action_blocks(content))
    if not matches:
        if not allow_bare:
            return None, content
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
                    "Failed to parse Kimi K3 action bridge output (%d chars)",
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


def _tool_choice_requires_action(tool_choice) -> bool:
    return tool_choice == "required" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "function"
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
