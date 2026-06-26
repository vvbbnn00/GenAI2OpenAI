import json

from genai_proxy.optimizations.registry import GLM_5_2_ADAPTER, GLM_ADAPTER
from genai_proxy.optimizations.xml_tools import inject_xml_tool_prompt


GLM52_REASONING_TEMPLATE = "Reasoning Effort: {effort}"
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
GLM52_TOOL_RESULT_FINAL_SUFFIX = (
    "\nThe tool response above is sufficient for the current request. "
    "Return the final answer only. "
    "Do not call any tool again. "
    "Do not emit <tool_call>, <arg_key>, or <arg_value> tags."
)


def inject_glm_tool_prompt(
    messages,
    tools,
    tool_choice=None,
    *,
    adapter=GLM_ADAPTER,
    reasoning_config=None,
):
    tool_prompt = _render_glm_tools_prompt(
        tools,
        tool_choice,
        adapter=adapter,
        reasoning_config=reasoning_config,
    )
    return inject_xml_tool_prompt(
        messages,
        tool_prompt,
        allow_additional_tool_calls=_allows_additional_tool_calls(tool_choice),
        render_tool_call_message=_render_glm_tool_call_message,
        render_tool_results=lambda tool_messages, allow_additional_tool_calls=False: (
            _render_glm_tool_results(
                tool_messages,
                allow_additional_tool_calls=allow_additional_tool_calls,
                adapter=adapter,
            )
        ),
    )


def inject_glm_reasoning_prompt(messages, reasoning_config=None):
    reasoning_prompt = _render_glm52_reasoning_prompt(reasoning_config)
    if not reasoning_prompt:
        return messages

    new_messages = []
    has_system = False
    for msg in messages:
        if msg.get("role") == "system":
            new_messages.append(
                {
                    **msg,
                    "content": msg.get("content", "") + "\n\n" + reasoning_prompt,
                }
            )
            has_system = True
        else:
            new_messages.append(msg)

    if not has_system:
        new_messages.insert(0, {"role": "system", "content": reasoning_prompt})
    return new_messages


def _render_glm_tools_prompt(
    tools,
    tool_choice=None,
    *,
    adapter=GLM_ADAPTER,
    reasoning_config=None,
):
    tool_defs = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function_data = _render_glm_tool_definition(tool.get("function", {}), adapter=adapter)
        if function_data:
            tool_defs.append(json.dumps(function_data, ensure_ascii=False))

    prompt = GLM_TOOL_SYSTEM_TEMPLATE.format(tool_definitions="\n".join(tool_defs))
    reasoning_prompt = (
        _render_glm52_reasoning_prompt(reasoning_config)
        if adapter == GLM_5_2_ADAPTER
        else ""
    )
    if reasoning_prompt:
        prompt = reasoning_prompt + "\n\n" + prompt

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


def _render_glm_tool_definition(function_data, *, adapter=GLM_ADAPTER):
    if not function_data:
        return {}
    if adapter == GLM_5_2_ADAPTER and function_data.get("defer_loading"):
        return {}
    rendered = dict(function_data)
    if adapter == GLM_5_2_ADAPTER:
        rendered.pop("defer_loading", None)
        rendered.pop("strict", None)
    return rendered


def _render_glm52_reasoning_prompt(reasoning_config=None):
    effort = (reasoning_config or {}).get("effort")
    if effort == "none":
        return ""
    rendered_effort = "High" if effort == "high" else "Max"
    return GLM52_REASONING_TEMPLATE.format(effort=rendered_effort)


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


def _render_glm_tool_results(
    tool_messages,
    allow_additional_tool_calls=False,
    *,
    adapter=GLM_ADAPTER,
):
    content = "<|observation|>" + "".join(
        f"<tool_response>{_normalize_content(msg.get('content'))}</tool_response>"
        for msg in tool_messages
    )
    if adapter != GLM_5_2_ADAPTER or allow_additional_tool_calls:
        return content
    return content + GLM52_TOOL_RESULT_FINAL_SUFFIX


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
