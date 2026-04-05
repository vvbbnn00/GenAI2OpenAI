import json
import uuid

from flask import jsonify

from genai_proxy.errors import ProxyError
from genai_proxy.token_usage import estimate_claude_request_tokens, estimate_token_by_model


ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
ROLE_TOOL = "tool"
ROLE_USER = "user"
TOOL_FUNCTION = "function"

CONTENT_IMAGE = "image"
CONTENT_TEXT = "text"
CONTENT_TOOL_RESULT = "tool_result"
CONTENT_TOOL_USE = "tool_use"

DELTA_INPUT_JSON = "input_json_delta"
DELTA_TEXT = "text_delta"

EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
EVENT_CONTENT_BLOCK_START = "content_block_start"
EVENT_CONTENT_BLOCK_STOP = "content_block_stop"
EVENT_MESSAGE_DELTA = "message_delta"
EVENT_MESSAGE_START = "message_start"
EVENT_MESSAGE_STOP = "message_stop"
EVENT_PING = "ping"

STOP_END_TURN = "end_turn"
STOP_MAX_TOKENS = "max_tokens"
STOP_TOOL_USE = "tool_use"


def claude_error(message, error_type="invalid_request_error", status=400):
    return (
        jsonify(
            {
                "type": "error",
                "error": {
                    "type": error_type,
                    "message": message,
                },
            }
        ),
        status,
    )


def convert_claude_to_openai(req_data, model_manager):
    model = req_data.get("model")
    max_tokens = req_data.get("max_tokens")
    messages = req_data.get("messages")

    if not model:
        raise ProxyError("Missing 'model' field in request body")
    if max_tokens is None:
        raise ProxyError("Missing 'max_tokens' field in request body")
    if messages is None:
        raise ProxyError("Missing 'messages' field in request body")

    openai_messages = []

    system = req_data.get("system")
    system_text = _extract_system_text(system)
    if system_text:
        openai_messages.append({"role": ROLE_SYSTEM, "content": system_text})

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == ROLE_USER:
            user_message, tool_messages = _convert_claude_user_message(content)
            if user_message is not None:
                openai_messages.append(user_message)
            openai_messages.extend(tool_messages)
        elif role == ROLE_ASSISTANT:
            openai_messages.append(_convert_claude_assistant_message(content))
        else:
            raise ProxyError(f"Unsupported Claude message role: {role}")

    openai_request = {
        "model": model_manager.resolve_model(model),
        "messages": openai_messages,
        "max_tokens": max_tokens,
        "stream": bool(req_data.get("stream", False)),
    }

    if req_data.get("temperature") is not None:
        openai_request["temperature"] = req_data["temperature"]
    if req_data.get("top_p") is not None:
        openai_request["top_p"] = req_data["top_p"]
    if req_data.get("stop_sequences"):
        openai_request["stop"] = req_data["stop_sequences"]

    tools = req_data.get("tools") or []
    if tools:
        openai_request["tools"] = [
            {
                "type": TOOL_FUNCTION,
                TOOL_FUNCTION: {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            for tool in tools
            if tool.get("name")
        ]

    tool_choice = req_data.get("tool_choice")
    if tool_choice:
        choice_type = tool_choice.get("type")
        if choice_type == "any":
            openai_request["tool_choice"] = "required"
        elif choice_type == "tool" and tool_choice.get("name"):
            openai_request["tool_choice"] = {
                "type": TOOL_FUNCTION,
                TOOL_FUNCTION: {"name": tool_choice["name"]},
            }
        else:
            openai_request["tool_choice"] = "auto"

    return openai_request


def convert_openai_to_claude_response(openai_response, original_request):
    choices = openai_response.get("choices", [])
    if not choices:
        raise ProxyError("No choices in upstream response", error_type="api_error", status=502)

    choice = choices[0]
    message = choice.get("message", {})
    content_blocks = []

    text_content = message.get("content")
    if text_content is not None:
        content_blocks.append({"type": CONTENT_TEXT, "text": text_content})

    for tool_call in message.get("tool_calls", []) or []:
        if tool_call.get("type") != TOOL_FUNCTION:
            continue
        function_data = tool_call.get(TOOL_FUNCTION, {})
        try:
            arguments = json.loads(function_data.get("arguments", "{}"))
        except json.JSONDecodeError:
            arguments = {"raw_arguments": function_data.get("arguments", "")}

        content_blocks.append(
            {
                "type": CONTENT_TOOL_USE,
                "id": tool_call.get("id", f"tool_{uuid.uuid4()}"),
                "name": function_data.get("name", ""),
                "input": arguments,
            }
        )

    if not content_blocks:
        content_blocks.append({"type": CONTENT_TEXT, "text": ""})

    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = {
        "stop": STOP_END_TURN,
        "length": STOP_MAX_TOKENS,
        "tool_calls": STOP_TOOL_USE,
        "function_call": STOP_TOOL_USE,
    }.get(finish_reason, STOP_END_TURN)

    estimator_model = original_request.get("_estimator_model") or original_request.get("model")
    usage = openai_response.get("usage", {}) or {}
    input_tokens = usage.get("prompt_tokens") or estimate_claude_request_tokens(
        original_request.get("system"),
        original_request.get("messages"),
        estimator_model,
        original_request.get("tools"),
    )
    output_tokens = usage.get("completion_tokens")
    if output_tokens is None:
        output_text = text_content or ""
        for block in content_blocks:
            if block.get("type") == CONTENT_TOOL_USE:
                output_text += block.get("name", "")
                output_text += json.dumps(block.get("input", {}), ensure_ascii=False, sort_keys=True)
        output_tokens = estimate_token_by_model(estimator_model, output_text)
    return {
        "id": openai_response.get("id", f"msg_{uuid.uuid4()}"),
        "type": "message",
        "role": ROLE_ASSISTANT,
        "model": original_request.get("model"),
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def stream_openai_to_claude(openai_stream, original_request, logger):
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    estimator_model = original_request.get("_estimator_model") or original_request.get("model")
    input_tokens = estimate_claude_request_tokens(
        original_request.get("system"),
        original_request.get("messages"),
        estimator_model,
        original_request.get("tools"),
    )
    output_text_parts = []

    yield _claude_event(
        EVENT_MESSAGE_START,
        {
            "type": EVENT_MESSAGE_START,
            "message": {
                "id": message_id,
                "type": "message",
                "role": ROLE_ASSISTANT,
                "model": original_request.get("model"),
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        },
    )
    yield _claude_event(
        EVENT_CONTENT_BLOCK_START,
        {
            "type": EVENT_CONTENT_BLOCK_START,
            "index": 0,
            "content_block": {"type": CONTENT_TEXT, "text": ""},
        },
    )
    yield _claude_event(EVENT_PING, {"type": EVENT_PING})

    text_block_index = 0
    tool_block_counter = 0
    current_tool_calls = {}
    final_stop_reason = STOP_END_TURN

    try:
        for line in openai_stream:
            if not line.strip() or not line.startswith("data: "):
                continue

            chunk_data = line[6:].strip()
            if chunk_data == "[DONE]":
                break

            try:
                chunk = json.loads(chunk_data)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse OpenAI chunk: %s", exc)
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            if delta.get("content") is not None:
                output_text_parts.append(delta["content"])
                yield _claude_event(
                    EVENT_CONTENT_BLOCK_DELTA,
                    {
                        "type": EVENT_CONTENT_BLOCK_DELTA,
                        "index": text_block_index,
                        "delta": {"type": DELTA_TEXT, "text": delta["content"]},
                    },
                )

            for tc_delta in delta.get("tool_calls", []) or []:
                tc_index = tc_delta.get("index", 0)
                tool_call = current_tool_calls.setdefault(
                    tc_index,
                    {
                        "id": None,
                        "name": None,
                        "args_buffer": "",
                        "json_sent": False,
                        "claude_index": None,
                        "started": False,
                    },
                )

                if tc_delta.get("id"):
                    tool_call["id"] = tc_delta["id"]

                function_data = tc_delta.get(TOOL_FUNCTION, {})
                if function_data.get("name"):
                    tool_call["name"] = function_data["name"]
                    output_text_parts.append(function_data["name"])

                if tool_call["id"] and tool_call["name"] and not tool_call["started"]:
                    tool_block_counter += 1
                    tool_call["claude_index"] = text_block_index + tool_block_counter
                    tool_call["started"] = True
                    yield _claude_event(
                        EVENT_CONTENT_BLOCK_START,
                        {
                            "type": EVENT_CONTENT_BLOCK_START,
                            "index": tool_call["claude_index"],
                            "content_block": {
                                "type": CONTENT_TOOL_USE,
                                "id": tool_call["id"],
                                "name": tool_call["name"],
                                "input": {},
                            },
                        },
                    )

                if (
                    "arguments" in function_data
                    and tool_call["started"]
                    and function_data["arguments"] is not None
                ):
                    tool_call["args_buffer"] += function_data["arguments"]
                    output_text_parts.append(function_data["arguments"])
                    try:
                        json.loads(tool_call["args_buffer"])
                    except json.JSONDecodeError:
                        pass
                    else:
                        if not tool_call["json_sent"]:
                            yield _claude_event(
                                EVENT_CONTENT_BLOCK_DELTA,
                                {
                                    "type": EVENT_CONTENT_BLOCK_DELTA,
                                    "index": tool_call["claude_index"],
                                    "delta": {
                                        "type": DELTA_INPUT_JSON,
                                        "partial_json": tool_call["args_buffer"],
                                    },
                                },
                            )
                            tool_call["json_sent"] = True

            if finish_reason:
                final_stop_reason = {
                    "length": STOP_MAX_TOKENS,
                    "tool_calls": STOP_TOOL_USE,
                    "function_call": STOP_TOOL_USE,
                    "stop": STOP_END_TURN,
                }.get(finish_reason, STOP_END_TURN)

    except Exception as exc:
        logger.exception("Claude streaming conversion failed")
        yield _claude_event(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": f"Streaming error: {exc}"},
            },
        )
        return

    yield _claude_event(
        EVENT_CONTENT_BLOCK_STOP,
        {"type": EVENT_CONTENT_BLOCK_STOP, "index": text_block_index},
    )

    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield _claude_event(
                EVENT_CONTENT_BLOCK_STOP,
                {
                    "type": EVENT_CONTENT_BLOCK_STOP,
                    "index": tool_data["claude_index"],
                },
            )

    yield _claude_event(
        EVENT_MESSAGE_DELTA,
        {
            "type": EVENT_MESSAGE_DELTA,
            "delta": {"stop_reason": final_stop_reason, "stop_sequence": None},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": estimate_token_by_model(estimator_model, "".join(output_text_parts)),
            },
        },
    )
    yield _claude_event(EVENT_MESSAGE_STOP, {"type": EVENT_MESSAGE_STOP})


def estimate_claude_tokens(req_data):
    estimator_model = req_data.get("_estimator_model") or req_data.get("model")
    return {
        "input_tokens": estimate_claude_request_tokens(
            req_data.get("system"),
            req_data.get("messages"),
            estimator_model,
            req_data.get("tools"),
        )
    }


def _convert_claude_user_message(content):
    if content is None:
        return {"role": ROLE_USER, "content": ""}, []
    if isinstance(content, str):
        return {"role": ROLE_USER, "content": content}, []

    openai_content = []
    tool_messages = []

    for block in content:
        block_type = block.get("type")
        if block_type == CONTENT_TEXT:
            openai_content.append({"type": "text", "text": block.get("text", "")})
        elif block_type == CONTENT_IMAGE:
            source = block.get("source", {})
            if (
                isinstance(source, dict)
                and source.get("type") == "base64"
                and "media_type" in source
                and "data" in source
            ):
                openai_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{source['media_type']};base64,{source['data']}"
                        },
                    }
                )
        elif block_type == CONTENT_TOOL_RESULT:
            tool_messages.append(
                {
                    "role": ROLE_TOOL,
                    "tool_call_id": block.get("tool_use_id"),
                    "content": _normalize_tool_result(block.get("content")),
                }
            )

    if not openai_content:
        return None, tool_messages
    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return {"role": ROLE_USER, "content": openai_content[0]["text"]}, tool_messages
    return {"role": ROLE_USER, "content": openai_content}, tool_messages


def _convert_claude_assistant_message(content):
    if content is None:
        return {"role": ROLE_ASSISTANT, "content": None}
    if isinstance(content, str):
        return {"role": ROLE_ASSISTANT, "content": content}

    text_parts = []
    tool_calls = []
    for block in content:
        block_type = block.get("type")
        if block_type == CONTENT_TEXT:
            text_parts.append(block.get("text", ""))
        elif block_type == CONTENT_TOOL_USE:
            tool_calls.append(
                {
                    "id": block.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "type": TOOL_FUNCTION,
                    TOOL_FUNCTION: {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                }
            )

    message = {"role": ROLE_ASSISTANT, "content": "".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _extract_system_text(system):
    if not system:
        return ""
    if isinstance(system, str):
        return system.strip()
    if isinstance(system, list):
        return "\n\n".join(
            block.get("text", "").strip()
            for block in system
            if block.get("type") == CONTENT_TEXT and block.get("text")
        ).strip()
    return ""


def _normalize_tool_result(content):
    if content is None:
        return "No content provided"
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        result_parts = []
        for item in content:
            if isinstance(item, str):
                result_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == CONTENT_TEXT:
                result_parts.append(item.get("text", ""))
            elif isinstance(item, dict):
                result_parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(result_parts).strip()
    if isinstance(content, dict):
        if content.get("type") == CONTENT_TEXT:
            return content.get("text", "")
        return json.dumps(content, ensure_ascii=False)
    return str(content)
def _claude_event(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
