"""Shared tool-choice predicates for request preparation and execution."""


def tool_choice_is_none(tool_choice) -> bool:
    return tool_choice == "none" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "none"
    )


def tool_choice_requires_call(tool_choice) -> bool:
    return tool_choice == "required" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "function"
    )


def tool_calls_satisfy_choice(tool_calls, tool_choice) -> bool:
    if not tool_calls:
        return False
    if tool_choice_is_none(tool_choice):
        return False
    if not (isinstance(tool_choice, dict) and tool_choice.get("type") == "function"):
        return True
    function = tool_choice.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    return bool(name) and all(
        tool_call.get("function", {}).get("name") == name for tool_call in tool_calls
    )


__all__ = [
    "tool_calls_satisfy_choice",
    "tool_choice_is_none",
    "tool_choice_requires_call",
]
