import json

from genai_proxy.optimizations.xml_tools import inject_xml_tool_prompt


MINIMAX_TOOL_SYSTEM_TEMPLATE = """\
# Tools
You may call one or more tools to assist with the user query.
Here are the tools available in JSONSchema format:

<tools>
{tool_definitions}
</tools>

When making tool calls, use XML format to invoke tools and pass parameters:

<minimax:tool_call>
<invoke name="tool-name-1">
<parameter name="param-key-1">param-value-1</parameter>
<parameter name="param-key-2">param-value-2</parameter>
...
</invoke>
</minimax:tool_call>"""

MINIMAX_REQUIRED_TOOL_SUFFIX = (
    "\nFor this turn, you must call at least one tool using a <minimax:tool_call> block."
)
MINIMAX_SPECIFIC_TOOL_SUFFIX = (
    '\nFor this turn, you must call the tool named "{name}" using a <minimax:tool_call> block.'
)
MINIMAX_NO_TOOL_SUFFIX = "\nFor this turn, do not call any tool or emit tool call tags."


def inject_minimax_tool_prompt(messages, tools, tool_choice=None):
    return inject_xml_tool_prompt(
        messages,
        _render_minimax_tools_prompt(tools, tool_choice),
        allow_additional_tool_calls=_allows_additional_tool_calls(tool_choice),
        render_tool_call_message=_render_minimax_tool_call_message,
        render_tool_results=_render_minimax_tool_results,
    )


def _render_minimax_tools_prompt(tools, tool_choice=None):
    tool_defs = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function_data = tool.get("function", {})
        if function_data:
            tool_defs.append(f"<tool>{json.dumps(function_data, ensure_ascii=False)}</tool>")

    prompt = MINIMAX_TOOL_SYSTEM_TEMPLATE.format(tool_definitions="\n".join(tool_defs))
    if tool_choice == "required":
        prompt += MINIMAX_REQUIRED_TOOL_SUFFIX
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        prompt += MINIMAX_SPECIFIC_TOOL_SUFFIX.format(
            name=tool_choice["function"]["name"],
        )
    elif _tool_choice_is_none(tool_choice):
        prompt += MINIMAX_NO_TOOL_SUFFIX
    return prompt


def _allows_additional_tool_calls(tool_choice) -> bool:
    return not _tool_choice_is_none(tool_choice)


def _tool_choice_is_none(tool_choice) -> bool:
    return tool_choice == "none" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "none"
    )


def _render_minimax_tool_call_message(message):
    parts = []
    if message.get("content"):
        parts.append(str(message["content"]))

    invocations = []
    for tool_call in message.get("tool_calls") or []:
        function_data = tool_call.get("function", {})
        arguments = _safe_json_loads(function_data.get("arguments", "{}"))
        param_lines = []
        if isinstance(arguments, dict):
            for key, value in arguments.items():
                param_lines.append(
                    f'<parameter name="{key}">'
                    f"{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
                    "</parameter>"
                )
        invocations.append(
            "\n".join(
                [
                    f'<invoke name="{function_data.get("name", "")}">',
                    *param_lines,
                    "</invoke>",
                ]
            )
        )

    if invocations:
        parts.append("\n".join(["<minimax:tool_call>", *invocations, "</minimax:tool_call>"]))
    return "\n\n".join(part for part in parts if part).strip()


def _render_minimax_tool_results(tool_messages, allow_additional_tool_calls=False):
    content = "\n".join(
        f"<response>{_normalize_content(msg.get('content'))}</response>"
        for msg in tool_messages
    )
    if allow_additional_tool_calls:
        return content + "\nUse these tool results to answer the user. Only call another tool if the current result is genuinely insufficient."
    return content + "\nAnswer the user normally using these tool results. Do not call any tool."


def _normalize_content(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _safe_json_loads(raw):
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
