import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime

from genai_proxy.errors import ProxyError


@dataclass(frozen=True, slots=True)
class ResponseToolMapping:
    kind: str
    model_name: str
    response_name: str
    namespace: str | None = None
    execution: str | None = None


@dataclass(slots=True)
class ResponsesRequestContext:
    openai_request: dict
    tool_map: dict[str, ResponseToolMapping]


def convert_responses_to_openai_request(
    req_data: dict | None,
) -> ResponsesRequestContext:
    if not isinstance(req_data, dict):
        raise ProxyError("Request body must be a JSON object")

    input_items = _normalize_input(req_data.get("input"))
    request_tools = req_data.get("tools") or []
    if not isinstance(request_tools, list) or any(
        not isinstance(tool, dict) for tool in request_tools
    ):
        raise ProxyError("'tools' must be a list of objects")
    request_tools = list(request_tools)
    request_tools.extend(_additional_tools_from_input(input_items))
    openai_tools, tool_map = _convert_responses_tools(request_tools)
    tool_choice = req_data.get("tool_choice")
    if tool_choice is None and _input_has_tool_output(input_items):
        tool_choice = "none"

    messages = []
    instructions = req_data.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})

    messages.extend(_convert_response_input_items(input_items, tool_map))

    openai_request = {
        "model": req_data.get("model", "GPT-4.1"),
        "messages": messages,
        "stream": bool(req_data.get("stream", False)),
    }

    if openai_tools:
        openai_request["tools"] = openai_tools
        openai_request["tool_choice"] = _convert_tool_choice(
            tool_choice,
            tool_map,
        )
    elif tool_choice in ("none", "required"):
        openai_request["tool_choice"] = tool_choice

    if req_data.get("max_output_tokens") is not None:
        openai_request["max_tokens"] = req_data["max_output_tokens"]
    elif req_data.get("max_tokens") is not None:
        openai_request["max_tokens"] = req_data["max_tokens"]

    reasoning = req_data.get("reasoning")
    if isinstance(reasoning, dict):
        openai_request["reasoning"] = reasoning
    elif req_data.get("reasoning_effort") is not None:
        openai_request["reasoning_effort"] = req_data["reasoning_effort"]

    return ResponsesRequestContext(openai_request=openai_request, tool_map=tool_map)


def make_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def make_event(event_name: str, payload: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def response_created_event(
    response_id: str, model: str, created: int | None = None
) -> str:
    created_at = int(created or datetime.now().timestamp())
    return make_event(
        "response.created",
        {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "status": "in_progress",
                "model": model,
            },
        },
    )


def response_output_text_delta(delta: str) -> str:
    return make_event(
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "delta": delta,
        },
    )


def response_reasoning_text_delta(delta: str, content_index: int = 0) -> str:
    return make_event(
        "response.reasoning_text.delta",
        {
            "type": "response.reasoning_text.delta",
            "delta": delta,
            "content_index": content_index,
        },
    )


def response_custom_tool_call_input_delta(
    item_id: str, call_id: str, delta: str
) -> str:
    return make_event(
        "response.custom_tool_call_input.delta",
        {
            "type": "response.custom_tool_call_input.delta",
            "item_id": item_id,
            "call_id": call_id,
            "delta": delta,
        },
    )


def response_output_item_done(item: dict) -> str:
    return make_event(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "item": item,
        },
    )


def response_output_item_added(item: dict) -> str:
    return make_event(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "item": item,
        },
    )


def response_completed_event(
    response_id: str,
    *,
    model: str,
    output: list[dict],
    end_turn: bool,
    created: int | None = None,
    usage: dict | None = None,
) -> str:
    response = {
        "id": response_id,
        "object": "response",
        "created_at": int(created or datetime.now().timestamp()),
        "status": "completed",
        "model": model,
        "output": output,
        "end_turn": end_turn,
    }
    if usage is not None:
        response["usage"] = usage
    return make_event(
        "response.completed",
        {
            "type": "response.completed",
            "response": response,
        },
    )


def response_failed_event(
    response_id: str, message: str, *, code: str | None = None
) -> str:
    return make_event(
        "response.failed",
        {
            "type": "response.failed",
            "response": {
                "id": response_id,
                "object": "response",
                "status": "failed",
                "error": {
                    "code": code or "upstream_error",
                    "message": message,
                },
            },
        },
    )


def make_message_item(text: str, item_id: str | None = None) -> dict:
    item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }
    if item_id:
        item["id"] = item_id
    return item


def make_message_added_item(item_id: str) -> dict:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "content": [],
    }


def make_response_tool_item(
    tool_call: dict, tool_map: dict[str, ResponseToolMapping]
) -> dict:
    function_data = tool_call.get("function") or {}
    model_name = function_data.get("name") or ""
    mapping = tool_map.get(model_name) or ResponseToolMapping(
        kind="function",
        model_name=model_name,
        response_name=model_name,
    )
    call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"
    arguments = function_data.get("arguments") or "{}"

    if mapping.kind == "custom":
        return {
            "type": "custom_tool_call",
            "call_id": call_id,
            "name": mapping.response_name,
            "input": _custom_input_from_arguments(arguments),
        }

    if mapping.kind == "tool_search":
        return {
            "type": "tool_search_call",
            "call_id": call_id,
            "status": "completed",
            "execution": mapping.execution or "client",
            "arguments": _safe_json_loads(arguments, default={}),
        }

    item = {
        "type": "function_call",
        "name": mapping.response_name,
        "arguments": arguments,
        "call_id": call_id,
    }
    if mapping.namespace:
        item["namespace"] = mapping.namespace
    return item


def response_output_text(response_items: list[dict]) -> str:
    parts = []
    for item in response_items:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text") or ""))
    return "".join(parts)


def _normalize_input(input_data) -> list:
    if input_data is None:
        return []
    if isinstance(input_data, str):
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": input_data}],
            }
        ]
    if isinstance(input_data, dict):
        return [input_data]
    if isinstance(input_data, list):
        if any(not isinstance(item, dict) for item in input_data):
            raise ProxyError("'input' arrays must contain objects")
        return input_data
    raise ProxyError("'input' must be a string, object, or list")


def _additional_tools_from_input(input_items: list) -> list:
    tools = []
    for item in input_items:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            additional_tools = item.get("tools") or []
            if not isinstance(additional_tools, list) or any(
                not isinstance(tool, dict) for tool in additional_tools
            ):
                raise ProxyError("'additional_tools.tools' must be a list of objects")
            tools.extend(additional_tools)
    return tools


def _input_has_tool_output(input_items: list) -> bool:
    return any(
        isinstance(item, dict)
        and item.get("type")
        in {"function_call_output", "custom_tool_call_output", "tool_search_output"}
        for item in input_items
    )


def _convert_response_input_items(
    input_items: list,
    tool_map: dict[str, ResponseToolMapping],
) -> list[dict]:
    messages = []
    for item in input_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "additional_tools":
            continue
        if _is_response_message_item(item):
            role = _response_role_to_chat_role(item.get("role"))
            content = _content_to_text(item.get("content"))
            if content:
                messages.append({"role": role, "content": content})
        elif item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id")
                            or f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": _model_name_for_response_call(item, tool_map),
                                "arguments": item.get("arguments") or "{}",
                            },
                        }
                    ],
                }
            )
        elif item_type == "custom_tool_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id")
                            or f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": item.get("name") or "custom_tool",
                                "arguments": json.dumps(
                                    {"input": item.get("input") or ""},
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                }
            )
        elif item_type in {
            "function_call_output",
            "custom_tool_call_output",
            "tool_search_output",
        }:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or "unknown",
                    "content": _output_to_text(
                        item.get("output", item.get("tools", ""))
                    ),
                }
            )
        elif item_type == "local_shell_call":
            messages.append(_local_shell_call_to_chat_message(item))
    return messages


def _response_role_to_chat_role(role) -> str:
    if role in {"system", "developer"}:
        return "system"
    if role == "assistant":
        return "assistant"
    return "user"


def _is_response_message_item(item: dict) -> bool:
    return item.get("type") == "message" or (
        item.get("type") is None and "role" in item and "content" in item
    )


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else json.dumps(content, ensure_ascii=False)

    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"input_text", "output_text", "text"}:
            parts.append(str(item.get("text") or ""))
        elif item_type == "input_image":
            image_url = item.get("image_url")
            if image_url:
                parts.append(f"[image: {image_url}]")
    return "\n".join(part for part in parts if part)


def _output_to_text(output) -> str:
    if isinstance(output, str):
        return output
    if output is None:
        return ""
    if isinstance(output, list):
        return _content_to_text(output)
    return json.dumps(output, ensure_ascii=False)


def _local_shell_call_to_chat_message(item: dict) -> dict:
    action = item.get("action") or {}
    if not isinstance(action, dict):
        action = {}
    exec_action = (
        action
        if action.get("type") == "exec"
        else action.get("exec") or action.get("Exec") or {}
    )
    if not isinstance(exec_action, dict):
        exec_action = {}
    command = exec_action["command"] if "command" in exec_action else []
    arguments = {
        "command": command,
        "timeout_ms": exec_action.get("timeout_ms"),
        "working_directory": exec_action.get("working_directory"),
    }
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": item.get("call_id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": "shell_command",
                    "arguments": json.dumps(
                        {
                            key: value
                            for key, value in arguments.items()
                            if value is not None
                        },
                        ensure_ascii=False,
                    ),
                },
            }
        ],
    }


def _convert_responses_tools(
    tools: list,
) -> tuple[list[dict], dict[str, ResponseToolMapping]]:
    openai_tools = []
    tool_map = {}
    used_names = set()

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == "function" or (tool_type is None and "function" in tool):
            model_name = _unique_tool_name(_tool_name(tool), used_names)
            openai_tools.append(_openai_function_tool(tool, model_name))
            tool_map[model_name] = ResponseToolMapping(
                kind="function",
                model_name=model_name,
                response_name=_tool_name(tool),
            )
        elif tool_type == "namespace":
            namespace = tool.get("name") or "namespace"
            for child in tool.get("tools") or []:
                if not isinstance(child, dict) or child.get("type") != "function":
                    continue
                response_name = _tool_name(child)
                model_name = _unique_tool_name(
                    _sanitize_tool_name(f"{namespace}__{response_name}"),
                    used_names,
                )
                openai_tools.append(
                    _openai_function_tool(
                        child,
                        model_name,
                        description_prefix=f"{namespace}.{response_name}",
                    )
                )
                tool_map[model_name] = ResponseToolMapping(
                    kind="function",
                    model_name=model_name,
                    response_name=response_name,
                    namespace=namespace,
                )
        elif tool_type == "custom":
            response_name = tool.get("name") or "custom_tool"
            model_name = _unique_tool_name(
                _sanitize_tool_name(response_name), used_names
            )
            openai_tools.append(_custom_tool_as_openai_function(tool, model_name))
            tool_map[model_name] = ResponseToolMapping(
                kind="custom",
                model_name=model_name,
                response_name=response_name,
            )
        elif tool_type == "tool_search":
            model_name = _unique_tool_name("tool_search", used_names)
            openai_tools.append(_tool_search_as_openai_function(tool, model_name))
            tool_map[model_name] = ResponseToolMapping(
                kind="tool_search",
                model_name=model_name,
                response_name="tool_search",
                execution=tool.get("execution") or "client",
            )
        elif tool_type in {"web_search", "image_generation"}:
            continue

    return openai_tools, tool_map


def _openai_function_tool(
    tool: dict, model_name: str, description_prefix: str | None = None
) -> dict:
    function_data = (
        tool.get("function") if isinstance(tool.get("function"), dict) else tool
    )
    description = function_data.get("description") or ""
    if description_prefix:
        description = f"{description_prefix}: {description}".strip()
    return {
        "type": "function",
        "function": {
            "name": model_name,
            "description": description,
            "parameters": function_data.get("parameters")
            or {"type": "object", "properties": {}},
        },
    }


def _custom_tool_as_openai_function(tool: dict, model_name: str) -> dict:
    description = (
        tool.get("description") or f"Call custom tool {tool.get('name') or model_name}."
    )
    fmt = tool.get("format")
    if isinstance(fmt, dict):
        syntax = fmt.get("syntax")
        if syntax:
            description += f" Input format syntax: {syntax}."
    return {
        "type": "function",
        "function": {
            "name": model_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Raw freeform input for the custom tool.",
                    }
                },
                "required": ["input"],
            },
        },
    }


def _tool_search_as_openai_function(tool: dict, model_name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": model_name,
            "description": tool.get("description") or "Search available client tools.",
            "parameters": tool.get("parameters")
            or {"type": "object", "properties": {}},
        },
    }


def _tool_name(tool: dict) -> str:
    function_data = (
        tool.get("function") if isinstance(tool.get("function"), dict) else {}
    )
    return function_data.get("name") or tool.get("name") or "tool"


def _unique_tool_name(name: str, used_names: set[str]) -> str:
    base = _sanitize_tool_name(name) or "tool"
    candidate = base
    index = 2
    while candidate in used_names:
        candidate = f"{base}_{index}"
        index += 1
    used_names.add(candidate)
    return candidate


def _sanitize_tool_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned or "tool"


def _convert_tool_choice(choice, tool_map: dict[str, ResponseToolMapping]):
    if choice in (None, "auto"):
        return "auto"
    if choice in ("none", "required"):
        return choice
    if isinstance(choice, dict):
        choice_type = choice.get("type")
        if choice_type == "none":
            return "none"
        name = choice.get("name")
        if choice_type == "function":
            function_data = choice.get("function")
            if isinstance(function_data, dict):
                name = function_data.get("name") or name
        if name:
            return {
                "type": "function",
                "function": {"name": _model_name_for_response_name(name, tool_map)},
            }
    return choice


def _model_name_for_response_name(
    response_name: str,
    tool_map: dict[str, ResponseToolMapping],
) -> str:
    for model_name, mapping in tool_map.items():
        if mapping.response_name == response_name:
            return model_name
    return response_name


def _model_name_for_response_call(
    item: dict,
    tool_map: dict[str, ResponseToolMapping],
) -> str:
    namespace = item.get("namespace")
    response_name = item.get("name") or "tool"
    for model_name, mapping in tool_map.items():
        if mapping.response_name == response_name and mapping.namespace == namespace:
            return model_name
    if namespace:
        return _sanitize_tool_name(f"{namespace}__{response_name}")
    return response_name


def _custom_input_from_arguments(arguments: str) -> str:
    parsed = _safe_json_loads(arguments, default=None)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("input"), str):
            return parsed["input"]
        if len(parsed) == 1:
            value = next(iter(parsed.values()))
            return (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            )
        return json.dumps(parsed, ensure_ascii=False)
    if parsed is not None:
        return (
            parsed
            if isinstance(parsed, str)
            else json.dumps(parsed, ensure_ascii=False)
        )
    return arguments or ""


def _safe_json_loads(raw: str, *, default):
    try:
        return json.loads(raw or "")
    except (TypeError, json.JSONDecodeError):
        return default
