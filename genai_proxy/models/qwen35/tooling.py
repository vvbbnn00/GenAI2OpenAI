import json
import re
import uuid

from genai_proxy.models.registry import QWEN_3_5_ADAPTER

QWEN35_REQUIRED_TOOL_SUFFIX = (
    "\n\nFor this turn, you must call at least one available function."
)
QWEN35_SPECIFIC_TOOL_SUFFIX = (
    '\n\nFor this turn, you must call the function named "{name}".'
)
QWEN35_NO_TOOL_SUFFIX = (
    "\n\nFor this turn, do not call a function or emit a <tool_call> block."
)

_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^>\r\n]+)>\s*"
    r"(?P<body>.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER_PATTERN = re.compile(
    r"<parameter=(?P<name>[^>\r\n]+)>\n?"
    r"(?P<value>.*?)\n?</parameter>",
    re.DOTALL,
)


def inject_qwen35_tool_prompt(messages, tools, tool_choice=None):
    tool_prompt = _render_qwen35_tools_prompt(tools, tool_choice)
    new_messages = []
    has_system = False
    index = 0

    while index < len(messages):
        message = messages[index]
        role = message.get("role")

        if role == "system" and not has_system:
            content = message.get("content", "")
            new_messages.append(
                {
                    **message,
                    "content": tool_prompt + ("\n\n" + content if content else ""),
                }
            )
            has_system = True
            index += 1
            continue

        new_messages.append(message)
        index += 1

    if not has_system:
        new_messages.insert(0, {"role": "system", "content": tool_prompt})
    return new_messages


def extract_qwen35_tool_calls(content, tools=None, logger=None):
    matches = list(_TOOL_CALL_PATTERN.finditer(content or ""))
    if not matches:
        return None, content

    tool_map = {
        str(tool.get("function", {}).get("name", "")).casefold(): tool
        for tool in tools or []
        if tool.get("type") == "function" and tool.get("function", {}).get("name")
    }
    calls = []
    spans = []
    for index, match in enumerate(matches):
        raw_name = match.group("name").strip()
        tool = tool_map.get(raw_name.casefold()) if tool_map else None
        if tool_map and tool is None:
            if logger:
                logger.warning(
                    "Qwen 3.5 returned unknown tool name in tool_call[%d]: %s",
                    index,
                    raw_name,
                )
            continue

        canonical_name = (
            tool.get("function", {}).get("name", raw_name) if tool else raw_name
        )
        properties = (
            tool.get("function", {}).get("parameters", {}).get("properties", {})
            if tool
            else {}
        )
        arguments = {}
        valid = True
        consumed = []
        for parameter in _PARAMETER_PATTERN.finditer(match.group("body")):
            parameter_name = parameter.group("name").strip()
            if parameter_name in arguments:
                valid = False
                break
            arguments[parameter_name] = _decode_qwen35_parameter(
                parameter.group("value"),
                properties.get(parameter_name, {}),
            )
            consumed.append(parameter.span())

        remainder = _remove_spans(match.group("body"), consumed).strip()
        if not valid or remainder:
            if logger:
                logger.warning(
                    "Failed to parse Qwen 3.5 tool_call[%d] parameters",
                    index,
                )
            continue

        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": canonical_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
        spans.append(match.span())

    if not calls:
        return None, content
    remaining = _remove_spans(content, spans).strip()
    return calls, remaining or None


def _render_qwen35_tools_prompt(tools, tool_choice=None):
    from genai_proxy.token_usage import official_tool_prompt_for_adapter

    prompt = official_tool_prompt_for_adapter(QWEN_3_5_ADAPTER, tools)
    if prompt is None:
        return ""
    if tool_choice == "required":
        prompt += QWEN35_REQUIRED_TOOL_SUFFIX
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        prompt += QWEN35_SPECIFIC_TOOL_SUFFIX.format(
            name=tool_choice.get("function", {}).get("name", "")
        )
    elif _tool_choice_is_none(tool_choice):
        prompt += QWEN35_NO_TOOL_SUFFIX
    return prompt


def _decode_qwen35_parameter(value, schema):
    value = value.removeprefix("\n").removesuffix("\n")
    expected_type = schema.get("type") if isinstance(schema, dict) else None
    if expected_type == "string":
        return value
    if expected_type in {"number", "integer", "boolean", "array", "object", "null"}:
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value
    if value[:1] in '[{"' or value in {"true", "false", "null"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _remove_spans(content, spans):
    if not spans:
        return content
    parts = []
    start = 0
    for span_start, span_end in sorted(spans):
        parts.append(content[start:span_start])
        start = span_end
    parts.append(content[start:])
    return "".join(parts)


def _tool_choice_is_none(tool_choice):
    return tool_choice == "none" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "none"
    )
