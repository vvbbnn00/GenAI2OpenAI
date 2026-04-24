import json

from genai_proxy.optimizations.xml_tools import inject_xml_tool_prompt


MINIMAX_TOOL_SYSTEM_TEMPLATE = """\
You have access to callable tools. MiniMax-M2.7 has strong tool-use ability, but this
GenAI route may expose tools only through text, so use this exact fallback format.

Available tools in JSON Schema:
<tools>
{tool_definitions}
</tools>

When you need a tool, output one or more tool call blocks exactly like this:
<tool_call>
{{"name": "<function-name>", "arguments": {{<arguments-as-json>}}}}
</tool_call>

Rules:
1. Use valid JSON for the object inside each <tool_call> block.
2. The arguments field must be a JSON object matching the tool schema.
3. Do not wrap tool calls in markdown fences.
4. If you include a <think> block, put the tool call after </think>, never inside it.
5. For Claude Code tools such as Bash, Read, Edit, or Write, use the exact tool name from the schema and still use the JSON object format above.
6. Do not print raw forms such as Bash<arg_key>command as final text.
7. After tool results are provided, answer the user normally unless another tool is truly needed.
8. If no tool is needed, answer normally without <tool_call> tags."""

MINIMAX_REQUIRED_TOOL_SUFFIX = (
    "\nFor this turn, you must call at least one tool using a <tool_call> block."
)
MINIMAX_SPECIFIC_TOOL_SUFFIX = (
    '\nFor this turn, you must call the tool named "{name}" using a <tool_call> block.'
)


def inject_minimax_tool_prompt(messages, tools, tool_choice=None):
    return inject_xml_tool_prompt(
        messages,
        _render_minimax_tools_prompt(tools, tool_choice),
        allow_additional_tool_calls=_allows_additional_tool_calls(tool_choice),
    )


def _render_minimax_tools_prompt(tools, tool_choice=None):
    tool_defs = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function_data = tool.get("function", {})
        if function_data:
            tool_defs.append(json.dumps(function_data, ensure_ascii=False))

    prompt = MINIMAX_TOOL_SYSTEM_TEMPLATE.format(tool_definitions="\n".join(tool_defs))
    if tool_choice == "required":
        prompt += MINIMAX_REQUIRED_TOOL_SUFFIX
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        prompt += MINIMAX_SPECIFIC_TOOL_SUFFIX.format(
            name=tool_choice["function"]["name"],
        )
    return prompt


def _allows_additional_tool_calls(tool_choice) -> bool:
    return tool_choice == "required" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "function"
    )
