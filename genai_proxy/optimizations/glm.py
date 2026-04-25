import json

from genai_proxy.optimizations.xml_tools import inject_xml_tool_prompt


GLM_TOOL_SYSTEM_TEMPLATE = """\
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_definitions}
</tools>

For each function call, output the function name and arguments within the following XML format:
<tool_call>{{function-name}}<arg_key>{{arg-key-1}}</arg_key><arg_value>{{arg-value-1}}</arg_value><arg_key>{{arg-key-2}}</arg_key><arg_value>{{arg-value-2}}</arg_value>...</tool_call>"""

GLM_REQUIRED_TOOL_SUFFIX = (
    "\nFor this turn, you must call at least one tool using a <tool_call> block."
)
GLM_SPECIFIC_TOOL_SUFFIX = (
    '\nFor this turn, you must call the tool named "{name}" using a <tool_call> block.'
)
GLM_NO_TOOL_SUFFIX = "\nFor this turn, do not call any tool or emit <tool_call> tags."


def inject_glm_tool_prompt(messages, tools, tool_choice=None):
    return inject_xml_tool_prompt(
        messages,
        _render_glm_tools_prompt(tools, tool_choice),
        allow_additional_tool_calls=_allows_additional_tool_calls(tool_choice),
        render_tool_call_message=_render_glm_tool_call_message,
        render_tool_results=_render_glm_tool_results,
    )


def _render_glm_tools_prompt(tools, tool_choice=None):
    tool_defs = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function_data = tool.get("function", {})
        if function_data:
            tool_defs.append(json.dumps(function_data, ensure_ascii=False))

    prompt = GLM_TOOL_SYSTEM_TEMPLATE.format(tool_definitions="\n".join(tool_defs))
    if tool_choice == "required":
        prompt += GLM_REQUIRED_TOOL_SUFFIX
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        prompt += GLM_SPECIFIC_TOOL_SUFFIX.format(name=tool_choice["function"]["name"])
    elif _tool_choice_is_none(tool_choice):
        prompt += GLM_NO_TOOL_SUFFIX
    return prompt


def _allows_additional_tool_calls(tool_choice) -> bool:
    return not _tool_choice_is_none(tool_choice)


def _tool_choice_is_none(tool_choice) -> bool:
    return tool_choice == "none" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "none"
    )


def _render_glm_tool_call_message(message):
    parts = []
    if message.get("content"):
        parts.append(str(message["content"]))

    for tool_call in message.get("tool_calls") or []:
        function_data = tool_call.get("function", {})
        arguments = _safe_json_loads(function_data.get("arguments", "{}"))
        arg_parts = []
        if isinstance(arguments, dict):
            for key, value in arguments.items():
                rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                arg_parts.append(f"<arg_key>{key}</arg_key><arg_value>{rendered}</arg_value>")
        parts.append(f"<tool_call>{function_data.get('name', '')}{''.join(arg_parts)}</tool_call>")
    return "\n\n".join(part for part in parts if part).strip()


def _render_glm_tool_results(tool_messages, allow_additional_tool_calls=False):
    return "<|observation|>" + "".join(
        f"<tool_response>{_normalize_content(msg.get('content'))}</tool_response>"
        for msg in tool_messages
    )


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
