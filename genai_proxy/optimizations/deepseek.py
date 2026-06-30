import json
import re
import uuid
from typing import Any


DEEPSEEK_V4_TOOL_SYSTEM_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
<｜DSML｜invoke name="$TOOL_NAME2">
...
</｜DSML｜invoke>
</｜DSML｜tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

DEEPSEEK_LEGACY_TOOL_SYSTEM_TEMPLATE = """## Tools

You have access to a set of tools you can use to answer the user's question.

You can invoke functions by writing a "<｜DSML｜function_calls>" block like the following as part of your reply:
<｜DSML｜function_calls>
<｜DSML｜invoke name="$FUNCTION_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
</｜DSML｜function_calls>

String and scalar parameters should be specified as-is without any escaping or quotes.
Lists and objects should use JSON format.
The "string" attribute must be "true" for string parameters and "false" for numbers, booleans, arrays, and objects.

Here are the functions available in JSONSchema format:
<functions>
{tool_schemas}
</functions>
"""

DEEPSEEK_REQUIRED_TOOL_SUFFIX = (
    "\nFor this turn, a plain-text-only answer is invalid. You must emit a <｜DSML｜tool_calls> block."
)
DEEPSEEK_LEGACY_REQUIRED_TOOL_SUFFIX = (
    "\nFor this turn, a plain-text-only answer is invalid. You must emit a <｜DSML｜function_calls> block."
)
DEEPSEEK_NO_TOOL_SUFFIX = (
    "\nFor this turn, do not emit DSML tool_calls/function_calls or <tool_call> tags."
)

DSML_TOKEN = "｜DSML｜"
DSML_TOOL_CALLS_START = "<｜DSML｜tool_calls>"
DSML_TOOL_CALLS_END = "</｜DSML｜tool_calls>"
DSML_FUNCTION_CALLS_START = "<｜DSML｜function_calls>"
DSML_FUNCTION_CALLS_END = "</｜DSML｜function_calls>"


def is_deepseek_model(model: str | None) -> bool:
    return "deepseek" in (model or "").lower()


def inject_deepseek_tool_prompt(messages, tools, tool_choice=None, adapter=None):
    tool_prompt = _render_deepseek_tools_prompt(tools, tool_choice, adapter=adapter)
    new_messages = []
    has_system = False
    index = 0

    while index < len(messages):
        msg = messages[index]
        role = msg.get("role")

        if role == "system":
            new_messages.append(
                {
                    "role": "system",
                    "content": msg.get("content", "") + "\n\n" + tool_prompt,
                }
            )
            has_system = True
            index += 1
            continue

        if role == "assistant" and msg.get("tool_calls"):
            assistant_parts = []
            if msg.get("content"):
                assistant_parts.append(msg["content"])
            assistant_parts.append(_render_deepseek_tool_calls(msg["tool_calls"], adapter=adapter))
            new_messages.append(
                {
                    "role": "assistant",
                    "content": "\n\n".join(part for part in assistant_parts if part).strip(),
                }
            )
            index += 1
            continue

        if role == "tool":
            tool_messages = []
            while index < len(messages) and messages[index].get("role") == "tool":
                tool_messages.append(messages[index])
                index += 1
            new_messages.append(
                {
                    "role": "user",
                    "content": _render_function_results(
                        tool_messages,
                        adapter=adapter,
                        allow_additional_tool_calls=_allows_additional_tool_calls(tool_choice),
                    ),
                }
            )
            continue

        new_messages.append(msg)
        index += 1

    if not has_system:
        new_messages.insert(0, {"role": "system", "content": tool_prompt})

    return new_messages


def extract_deepseek_tool_calls(content, tools=None, logger=None, adapter=None):
    dsml_tool_calls, dsml_remaining = _extract_dsml_tool_calls(
        content,
        logger=logger,
        include_legacy=True,
    )
    if dsml_tool_calls:
        return dsml_tool_calls, dsml_remaining

    tool_schemas = _tool_schema_map(tools or [])
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    if not matches:
        return None, content

    repaired_tool_calls = []
    for index, match in enumerate(matches):
        repaired = _repair_tool_call_body(match, tool_schemas)
        if not repaired:
            if logger:
                logger.warning(
                    "DeepSeek repair failed for tool_call[%d] — raw: %s",
                    index,
                    match[:300],
                )
            continue

        repaired_tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": repaired["name"],
                    "arguments": json.dumps(repaired["arguments"], ensure_ascii=False),
                },
            }
        )

    if not repaired_tool_calls:
        return None, content

    remaining = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()
    return repaired_tool_calls, remaining or None


def _render_deepseek_tools_prompt(tools, tool_choice=None, adapter=None):
    tool_schemas = []
    for tool in tools:
        function_data = tool.get("function", {})
        if function_data:
            tool_schemas.append(json.dumps(function_data, ensure_ascii=False))

    is_legacy = adapter == "deepseek_legacy"
    template = DEEPSEEK_LEGACY_TOOL_SYSTEM_TEMPLATE if is_legacy else DEEPSEEK_V4_TOOL_SYSTEM_TEMPLATE
    prompt = template.format(
        tool_schemas="\n".join(tool_schemas),
    )

    if tool_choice == "required":
        prompt += DEEPSEEK_LEGACY_REQUIRED_TOOL_SUFFIX if is_legacy else DEEPSEEK_REQUIRED_TOOL_SUFFIX
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        if is_legacy:
            prompt += (
                f'\nFor this turn, you must emit a <｜DSML｜function_calls> block using the tool "{tool_choice["function"]["name"]}".'
            )
        else:
            prompt += (
                f'\nFor this turn, you must emit a <｜DSML｜tool_calls> block using the tool "{tool_choice["function"]["name"]}".'
            )
    elif _tool_choice_is_none(tool_choice):
        prompt += DEEPSEEK_NO_TOOL_SUFFIX

    return prompt


def _render_deepseek_tool_calls(tool_calls, adapter=None):
    block_name = "function_calls" if adapter == "deepseek_legacy" else "tool_calls"
    invocations = []
    for tool_call in tool_calls:
        function_data = tool_call.get("function", {})
        tool_name = function_data.get("name", "")
        arguments = _safe_json_loads(function_data.get("arguments", "{}"))
        param_lines = []
        if isinstance(arguments, dict):
            for key, value in arguments.items():
                param_lines.append(_render_dsml_parameter(key, value))
        invocations.append(
            "\n".join(
                [
                    f'<{DSML_TOKEN}invoke name="{tool_name}">',
                    *param_lines,
                    f"</{DSML_TOKEN}invoke>",
                ]
            )
        )

    return "\n".join(
        [
            f"<{DSML_TOKEN}{block_name}>",
            *invocations,
            f"</{DSML_TOKEN}{block_name}>",
        ]
    )


def _render_dsml_parameter(key: str, value: Any) -> str:
    is_string = isinstance(value, str)
    rendered = value if is_string else json.dumps(value, ensure_ascii=False)
    return (
        f'<{DSML_TOKEN}parameter name="{key}" string="{"true" if is_string else "false"}">'
        f"{rendered}</{DSML_TOKEN}parameter>"
    )


def _render_function_results(tool_messages, adapter=None, allow_additional_tool_calls=False) -> str:
    if adapter != "deepseek_legacy":
        return "\n".join(
            f"<tool_result>{_normalize_tool_content(msg.get('content'))}</tool_result>"
            for msg in tool_messages
        )

    result_lines = ["<function_results>"]
    if not allow_additional_tool_calls:
        result_lines.append("Answer the user normally using these function results.")
    for msg in tool_messages:
        result_content = _normalize_tool_content(msg.get("content"))
        result_lines.append(f"<tool_result>{result_content}</tool_result>")
    result_lines.append("</function_results>")
    return "\n".join(result_lines)


def _allows_additional_tool_calls(tool_choice) -> bool:
    return not _tool_choice_is_none(tool_choice)


def _tool_choice_is_none(tool_choice) -> bool:
    return tool_choice == "none" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "none"
    )


def _extract_dsml_tool_calls(content, logger=None, include_legacy=True):
    match = _find_dsml_tool_call_block(content, include_legacy=include_legacy)
    if not match:
        return None, content

    block = match.group(1)
    invocations = re.findall(
        r'<｜DSML｜invoke name="(.*?)">(.*?)</｜DSML｜invoke>',
        block,
        re.DOTALL,
    )
    if not invocations:
        return None, content

    tool_calls = []
    for tool_name, raw_params in invocations:
        arguments = {}
        for param_name, is_string, raw_value in re.findall(
            r'<｜DSML｜parameter name="(.*?)" string="(true|false)">(.*?)</｜DSML｜parameter>',
            raw_params,
            re.DOTALL,
        ):
            arguments[param_name] = _decode_dsml_parameter(raw_value, is_string == "true")

        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )

    if logger:
        logger.debug("Recovered %d DeepSeek DSML tool call(s)", len(tool_calls))

    remaining = (
        content[: match.start()] + content[match.end() :]
    ).strip()
    return tool_calls, remaining or None


def _find_dsml_tool_call_block(content: str, include_legacy=True):
    pairs = [(DSML_TOOL_CALLS_START, DSML_TOOL_CALLS_END)]
    if include_legacy:
        pairs.append((DSML_FUNCTION_CALLS_START, DSML_FUNCTION_CALLS_END))
    for start_tag, end_tag in pairs:
        match = re.search(
            rf"{re.escape(start_tag)}(.*?){re.escape(end_tag)}",
            content,
            re.DOTALL,
        )
        if match:
            return match
    return None


def _normalize_tool_content(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _safe_json_loads(raw: str):
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}


def _decode_dsml_parameter(raw_value: str, is_string: bool) -> Any:
    value = raw_value.strip()
    if is_string:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


def _tool_schema_map(tools) -> dict[str, dict[str, Any]]:
    schema_map = {}
    for tool in tools:
        function_data = tool.get("function", {})
        name = function_data.get("name")
        if not name:
            continue
        schema_map[name] = function_data.get("parameters", {})
    return schema_map


def _repair_tool_call_body(raw: str, tool_schemas: dict[str, dict[str, Any]]):
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
    if not name_match:
        name_match = re.search(r"<name>\s*(.*?)\s*</name>", raw, re.DOTALL)
    if not name_match:
        return None

    name = name_match.group(1).strip()
    name = _canonical_tool_name(name, tool_schemas)
    arguments = _extract_arguments(raw)
    if arguments is None:
        return None
    return {"name": name, "arguments": arguments}


def _canonical_tool_name(name: str, tool_schemas: dict[str, dict[str, Any]]) -> str:
    if name in tool_schemas:
        return name
    lowered = name.lower()
    for candidate in tool_schemas:
        if candidate.lower() == lowered:
            return candidate
    return name


def _extract_arguments(raw: str):
    xml_match = re.search(r"<arguments>\s*(.*?)\s*</arguments>", raw, re.DOTALL)
    if xml_match:
        raw_args = xml_match.group(1).strip()
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            pass

    args_match = re.search(r'"arguments"\s*:\s*', raw)
    if not args_match:
        return {}

    idx = args_match.end()
    while idx < len(raw) and raw[idx].isspace():
        idx += 1

    if idx >= len(raw):
        return {}

    if raw[idx] != "{":
        return {}

    try:
        parsed, _ = json.JSONDecoder().raw_decode(raw[idx:])
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        parsed, _ = _parse_lenient_json_object(raw, idx)
        return parsed
    except ValueError:
        return None


def _parse_lenient_json_object(text: str, start: int):
    if text[start] != "{":
        raise ValueError("Object must start with '{'")

    index = start + 1
    result = {}

    while index < len(text):
        index = _skip_ws(text, index)
        if index >= len(text):
            raise ValueError("Unexpected end of object")
        if text[index] == "}":
            return result, index + 1

        key, index = _parse_lenient_json_string(text, index)
        index = _skip_ws(text, index)
        if index >= len(text) or text[index] != ":":
            raise ValueError("Missing ':' after object key")

        index += 1
        index = _skip_ws(text, index)
        value, index = _parse_lenient_json_value(text, index)
        result[key] = value

        index = _skip_ws(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "}":
            return result, index + 1
        raise ValueError("Invalid object separator")

    raise ValueError("Object not terminated")


def _parse_lenient_json_array(text: str, start: int):
    if text[start] != "[":
        raise ValueError("Array must start with '['")

    index = start + 1
    result = []

    while index < len(text):
        index = _skip_ws(text, index)
        if index >= len(text):
            raise ValueError("Unexpected end of array")
        if text[index] == "]":
            return result, index + 1

        value, index = _parse_lenient_json_value(text, index)
        result.append(value)

        index = _skip_ws(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "]":
            return result, index + 1
        raise ValueError("Invalid array separator")

    raise ValueError("Array not terminated")


def _parse_lenient_json_value(text: str, start: int):
    if start >= len(text):
        raise ValueError("Missing value")

    char = text[start]
    if char == '"':
        return _parse_lenient_json_string(text, start)
    if char == "{":
        return _parse_lenient_json_object(text, start)
    if char == "[":
        return _parse_lenient_json_array(text, start)

    end = start
    while end < len(text) and text[end] not in ",]}":
        end += 1

    raw = text[start:end].strip()
    if not raw:
        raise ValueError("Empty scalar")

    if raw == "true":
        return True, end
    if raw == "false":
        return False, end
    if raw == "null":
        return None, end

    try:
        return int(raw), end
    except ValueError:
        try:
            return float(raw), end
        except ValueError:
            return raw, end


def _parse_lenient_json_string(text: str, start: int):
    if text[start] != '"':
        raise ValueError("String must start with quote")

    index = start + 1
    buffer = []
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            buffer.append(text[index + 1])
            index += 2
            continue
        if char == '"':
            lookahead = _skip_ws(text, index + 1)
            if lookahead >= len(text) or text[lookahead] in ",]}:":
                return "".join(buffer), index + 1
        buffer.append(char)
        index += 1

    raise ValueError("String not terminated")


def _skip_ws(text: str, start: int) -> int:
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    return index
