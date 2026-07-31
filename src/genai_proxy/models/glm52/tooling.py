from genai_proxy.models.registry import GLM_ADAPTER
from genai_proxy.models.xml_tools import inject_xml_tool_prompt

GLM_REQUIRED_TOOL_SUFFIX = (
    "\nFor this turn, you must call at least one tool using a <tool_call> block."
)
GLM_SPECIFIC_TOOL_SUFFIX = (
    '\nFor this turn, you must call the tool named "{name}" using a <tool_call> block.'
)
GLM_NO_TOOL_SUFFIX = "\nFor this turn, do not call any tool or emit <tool_call> tags."


def inject_glm_tool_prompt(
    messages,
    tools,
    tool_choice=None,
    *,
    adapter=GLM_ADAPTER,
    reasoning_config=None,
):
    tool_prompt = _render_glm_tools_prompt(tools, adapter=adapter)
    rendered = inject_xml_tool_prompt(
        messages,
        tool_prompt,
        prepend_tool_prompt=True,
    )
    constraint = _glm_tool_choice_constraint(tool_choice)
    if constraint:
        rendered.insert(1, {"role": "system", "content": constraint})
    return rendered


def inject_glm_reasoning_prompt(messages, reasoning_config=None):
    # GLM-5.2's official template emits its default max effort itself. GenAI
    # has no field for the template's reasoning_effort argument, so adding a
    # system directive here would duplicate that prompt (or contradict it).
    return messages


def _render_glm_tools_prompt(
    tools,
    *,
    adapter=GLM_ADAPTER,
):
    from genai_proxy.token_usage import official_tool_prompt_for_adapter

    prompt = official_tool_prompt_for_adapter(adapter, tools)
    return prompt or ""


def _glm_tool_choice_constraint(tool_choice) -> str:
    if tool_choice == "required":
        return GLM_REQUIRED_TOOL_SUFFIX.lstrip("\n")
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return GLM_SPECIFIC_TOOL_SUFFIX.format(
            name=tool_choice["function"]["name"]
        ).lstrip("\n")
    if _tool_choice_is_none(tool_choice):
        return GLM_NO_TOOL_SUFFIX.lstrip("\n")
    return ""


def _tool_choice_is_none(tool_choice) -> bool:
    return tool_choice == "none" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "none"
    )
