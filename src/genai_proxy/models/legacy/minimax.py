from genai_proxy.models.registry import MINIMAX_ADAPTER
from genai_proxy.models.xml_tools import inject_xml_tool_prompt

MINIMAX_REQUIRED_TOOL_SUFFIX = "\nFor this turn, you must call at least one tool using a <minimax:tool_call> block."
MINIMAX_SPECIFIC_TOOL_SUFFIX = '\nFor this turn, you must call the tool named "{name}" using a <minimax:tool_call> block.'
MINIMAX_NO_TOOL_SUFFIX = "\nFor this turn, do not call any tool or emit tool call tags."


def inject_minimax_tool_prompt(messages, tools, tool_choice=None):
    from genai_proxy.token_usage import official_default_system_prompt_for_adapter

    constraint = _minimax_tool_choice_constraint(tool_choice)
    if not any(message.get("role") == "system" for message in messages):
        default_system_prompt = official_default_system_prompt_for_adapter(
            MINIMAX_ADAPTER
        )
        messages = [
            {
                "role": "system",
                "content": (default_system_prompt or "") + constraint,
            },
            *messages,
        ]
    elif constraint:
        messages = [dict(message) for message in messages]
        first_system = next(
            index
            for index, message in enumerate(messages)
            if message.get("role") == "system"
        )
        messages[first_system]["content"] = (
            messages[first_system].get("content", "") + constraint
        )
    return inject_xml_tool_prompt(
        messages,
        _render_minimax_tools_prompt(tools),
    )


def _render_minimax_tools_prompt(tools):
    from genai_proxy.token_usage import official_tool_prompt_for_adapter

    prompt = official_tool_prompt_for_adapter(MINIMAX_ADAPTER, tools)
    return prompt or ""


def _minimax_tool_choice_constraint(tool_choice) -> str:
    if tool_choice == "required":
        return MINIMAX_REQUIRED_TOOL_SUFFIX
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return MINIMAX_SPECIFIC_TOOL_SUFFIX.format(
            name=tool_choice["function"]["name"],
        )
    if _tool_choice_is_none(tool_choice):
        return MINIMAX_NO_TOOL_SUFFIX
    return ""


def _tool_choice_is_none(tool_choice) -> bool:
    return tool_choice == "none" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "none"
    )
