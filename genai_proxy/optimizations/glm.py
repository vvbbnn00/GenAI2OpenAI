import json

from genai_proxy.optimizations.xml_tools import inject_xml_tool_prompt


GLM_TOOL_SYSTEM_TEMPLATE = """\
You have access to callable tools. GLM-5.1 supports function calling; on this
GenAI route, express each function call with the exact text fallback below.

Available tools in JSON Schema:
<tools>
{tool_definitions}
</tools>

When a tool is needed, output one or more blocks exactly like this:
<tool_call>
{{"name": "<function-name>", "arguments": {{<arguments-as-json>}}}}
</tool_call>

Rules:
1. The JSON object inside <tool_call> must be valid.
2. The arguments field must be a JSON object matching the schema.
3. The only valid closing tag is </tool_call>. Do not use </arg_value>.
4. Do not wrap tool calls in markdown fences.
5. After tool results are provided, answer the user normally unless another tool is truly needed.
6. If no tool is needed, answer normally without <tool_call> tags."""

GLM_REQUIRED_TOOL_SUFFIX = (
    "\nFor this turn, you must call at least one tool using a <tool_call> block."
)
GLM_SPECIFIC_TOOL_SUFFIX = (
    '\nFor this turn, you must call the tool named "{name}" using a <tool_call> block.'
)


def inject_glm_tool_prompt(messages, tools, tool_choice=None):
    return inject_xml_tool_prompt(
        messages,
        _render_glm_tools_prompt(tools, tool_choice),
        allow_additional_tool_calls=_allows_additional_tool_calls(tool_choice),
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
    return prompt


def _allows_additional_tool_calls(tool_choice) -> bool:
    return tool_choice == "required" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "function"
    )
