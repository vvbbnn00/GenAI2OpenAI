import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

import requests

from genai_proxy.compat.openai import (
    extract_tool_calls,
    inject_tool_prompt,
    make_error_chunk,
    tag_prefix_len,
)
from genai_proxy.errors import ProxyError
from genai_proxy.optimizations import native_tool_fields, select_tool_adapter, tool_start_tags
from genai_proxy.services.token_manager import parse_jwt_payload
from genai_proxy.token_usage import estimate_openai_request_tokens, estimate_token_by_model


GENAI_URL = "https://genai.shanghaitech.edu.cn/htk/chat/start/chat"
GENAI_BASE_HEADERS = {
    "Accept": "*/*, text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Origin": "https://genai.shanghaitech.edu.cn",
    "Referer": "https://genai.shanghaitech.edu.cn/dialogue",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    ),
    "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

GENAI_USER_INFO_URL = "https://genai.shanghaitech.edu.cn/htk/ai-user-info/list"
GENAI_CURRENT_USER_URL = "https://genai.shanghaitech.edu.cn/htk/user/info/{token}"


def _is_tool_result_continuation(messages) -> bool:
    for message in reversed(messages or []):
        if message.get("role") == "system":
            continue
        return message.get("role") == "tool"
    return False


def _requires_tool_choice(tool_choice) -> bool:
    return tool_choice == "required" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "function"
    )


@dataclass(slots=True)
class PreparedChatRequest:
    messages: list
    model: str
    max_tokens: int
    has_tools: bool
    tools: list
    tool_choice: object
    tool_adapter: str
    prompt_tokens: int


class GenAIService:
    def __init__(self, logger, token_manager, model_manager):
        self._logger = logger
        self._token_manager = token_manager
        self._model_manager = model_manager

    def build_openai_completion(self, req_data):
        prepared = self._prepare_chat_request(req_data)
        return self._build_openai_completion(prepared)

    def stream_openai_completion(self, req_data):
        prepared = self._prepare_chat_request(req_data)
        return self._stream_prepared_openai_completion(prepared)

    def fetch_openai_billing_subscription(self):
        token = self._token_manager.token
        access_until = self._extract_access_until(token)
        user_id = self._fetch_current_user_id(token)
        record = self._fetch_user_info_record(token, user_id)
        quota = self._coerce_amount(record.get("quota"))

        return {
            "object": "billing_subscription",
            "has_payment_method": True,
            "soft_limit_usd": quota,
            "hard_limit_usd": quota,
            "system_hard_limit_usd": quota,
            "access_until": access_until,
        }

    def fetch_openai_billing_usage(self):
        token = self._token_manager.token
        self._extract_access_until(token)
        user_id = self._fetch_current_user_id(token)
        record = self._fetch_user_info_record(token, user_id)
        month_usage_usd = self._coerce_amount(record.get("monthSurplus"))
        total_usage = max(month_usage_usd, 0.0) * 100

        return {
            "object": "list",
            "total_usage": round(total_usage, 2),
        }

    def _build_openai_completion(self, prepared: PreparedChatRequest):
        complete_content = ""
        collected_tool_calls = []
        finish_reason = "stop"

        for payload in self._stream_prepared_openai_completion(prepared):
            for line in _iter_sse_lines(payload):
                if not line.startswith("data: "):
                    continue

                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    continue

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if "choices" not in data or not data["choices"]:
                    continue

                choice = data["choices"][0]
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                if content:
                    complete_content += content

                for tool_call in delta.get("tool_calls", []) or []:
                    collected_tool_calls.append(tool_call)

                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]

        if collected_tool_calls:
            message_obj = {
                "role": "assistant",
                "content": complete_content or None,
                "tool_calls": _merge_tool_call_deltas(collected_tool_calls),
            }
        elif prepared.has_tools:
            tool_calls, remaining_text = extract_tool_calls(
                complete_content,
                self._logger,
                tools=prepared.tools,
                model=prepared.model,
                adapter=prepared.tool_adapter,
            )
            message_obj = {
                "role": "assistant",
                "content": remaining_text,
                "tool_calls": tool_calls,
            }
            if tool_calls:
                finish_reason = "tool_calls"
            else:
                message_obj = {"role": "assistant", "content": complete_content}
                finish_reason = "stop"
        else:
            message_obj = {"role": "assistant", "content": complete_content}
            finish_reason = "stop"

        completion_estimate_text = complete_content or ""
        if message_obj.get("tool_calls"):
            for tool_call in message_obj["tool_calls"]:
                function = tool_call.get("function", {})
                completion_estimate_text += function.get("name", "")
                completion_estimate_text += function.get("arguments", "")
        completion_tokens = estimate_token_by_model(prepared.model, completion_estimate_text)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": prepared.model,
            "choices": [
                {
                    "index": 0,
                    "message": message_obj,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prepared.prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prepared.prompt_tokens + completion_tokens,
            },
        }

    def _stream_prepared_openai_completion(self, prepared: PreparedChatRequest):
        if prepared.has_tools:
            return self._stream_genai_response_with_tools(prepared)
        return self._stream_genai_response(prepared)

    def _prepare_chat_request(self, req_data) -> PreparedChatRequest:
        if not req_data or "messages" not in req_data:
            raise ProxyError("Missing 'messages' field in request body")

        messages = req_data.get("messages", [])
        if not isinstance(messages, list):
            raise ProxyError("'messages' must be a list")

        model = self._model_manager.resolve_model(req_data.get("model", "GPT-4.1"))
        max_tokens = req_data.get("max_tokens", 30000)
        tools = req_data.get("tools") or []
        tool_choice = req_data.get("tool_choice")
        requested_tools = bool(tools)
        tool_result_continuation = _is_tool_result_continuation(
            messages,
        ) and not _requires_tool_choice(tool_choice)
        model_record = self._model_manager.get_model_record(model)
        tool_adapter = select_tool_adapter(model, model_record)

        if requested_tools:
            messages = inject_tool_prompt(
                messages,
                tools,
                tool_choice,
                model=model,
                adapter=tool_adapter,
            )

        has_tools = requested_tools and not tool_result_continuation
        if not self._extract_last_user_message(messages):
            raise ProxyError("No user message found in 'messages'")

        return PreparedChatRequest(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            has_tools=has_tools,
            tools=tools if has_tools else [],
            tool_choice=tool_choice if has_tools else None,
            tool_adapter=tool_adapter,
            prompt_tokens=estimate_openai_request_tokens(messages, model, None),
        )

    def _get_genai_headers(self):
        headers = dict(GENAI_BASE_HEADERS)
        headers["X-Access-Token"] = self._token_manager.token
        return headers

    def _get_user_genai_headers(self, user_token: str):
        return {
            "Accept": "application/json",
            "X-Access-Token": user_token,
        }

    def _extract_last_user_message(self, messages):
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                return json.dumps(content, ensure_ascii=False)
        return ""

    def _extract_delta_from_genai(self, response_data):
        try:
            if "choices" in response_data and response_data["choices"]:
                delta = response_data["choices"][0].get("delta", {})
                content = delta.get("content") or None
                reasoning = delta.get("reasoning_content") or None
                tool_calls = delta.get("tool_calls") or None
                return content, reasoning, tool_calls
        except (KeyError, IndexError, TypeError):
            pass
        return None, None, None

    def _fetch_user_info_record(self, user_token: str, user_id: str):
        try:
            response = requests.get(
                GENAI_USER_INFO_URL,
                params={
                    "_t": int(time.time()),
                    "pageNo": 1,
                    "pageSize": 1,
                    "userId": user_id,
                },
                headers=self._get_user_genai_headers(user_token),
                timeout=30,
            )
        except requests.RequestException as exc:
            self._logger.warning("Failed to fetch user billing info: %s", exc)
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502) from exc

        if response.status_code == 401:
            raise ProxyError("Upstream GenAI token is invalid or expired", error_type="authentication_error", status=502)
        if response.status_code != 200:
            self._logger.warning(
                "GenAI billing API error %d: %s",
                response.status_code,
                response.text[:500],
            )
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502)

        try:
            payload = response.json()
        except ValueError as exc:
            self._logger.warning("Failed to decode billing response JSON: %s", exc)
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502) from exc

        if payload.get("code", 200) >= 400 or payload.get("success") is False:
            self._logger.warning("GenAI billing business error: %s", payload)
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502)

        records = payload.get("result", {}).get("records") or []
        if not records:
            raise ProxyError(
                "Quota information for the current GenAI account was not found",
                error_type="invalid_request_error",
                status=404,
            )

        record = records[0]
        if str(record.get("id")) != str(user_id):
            self._logger.warning(
                "Billing record mismatch for user_id=%s, got id=%s",
                user_id,
                record.get("id"),
            )
            raise ProxyError(
                "Quota information for the current GenAI account was not found",
                error_type="invalid_request_error",
                status=404,
            )

        return record

    def _fetch_current_user_id(self, user_token: str) -> str:
        try:
            response = requests.get(
                GENAI_CURRENT_USER_URL.format(token=user_token),
                params={"_t": int(time.time() * 1000)},
                headers=self._get_user_genai_headers(user_token),
                timeout=30,
            )
        except requests.RequestException as exc:
            self._logger.warning("Failed to fetch current user info: %s", exc)
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502) from exc

        if response.status_code == 401:
            raise ProxyError("Upstream GenAI token is invalid or expired", error_type="authentication_error", status=502)
        if response.status_code != 200:
            self._logger.warning(
                "GenAI current user API error %d: %s",
                response.status_code,
                response.text[:500],
            )
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502)

        try:
            payload = response.json()
        except ValueError as exc:
            self._logger.warning("Failed to decode current user JSON: %s", exc)
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502) from exc

        user_info = payload.get("result", {}).get("userInfo") or {}
        user_id = user_info.get("id")
        if not user_id:
            self._logger.warning("Current user response missing id: %s", payload)
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502)
        return str(user_id)

    def _extract_access_until(self, user_token: str) -> int:
        try:
            access_until = int(parse_jwt_payload(user_token).get("exp") or 0)
        except Exception as exc:
            self._logger.warning("Failed to parse billing token expiry: %s", exc)
            raise ProxyError("Upstream GenAI token is invalid or expired", error_type="authentication_error", status=502) from exc

        if access_until <= int(time.time()):
            raise ProxyError("Upstream GenAI token is invalid or expired", error_type="authentication_error", status=502)
        return access_until

    def _coerce_amount(self, value) -> float:
        if value in (None, ""):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            self._logger.warning("Invalid billing amount from upstream: %r", value)
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502)

    def _stream_genai_response(self, prepared: PreparedChatRequest):
        root_ai_type = self._model_manager.root_ai_type_for(prepared.model)
        genai_data = {
            "chatInfo": "",
            "messages": prepared.messages,
            "type": "3",
            "stream": True,
            "aiType": prepared.model,
            "aiSecType": "1",
            "promptTokens": 0,
            "rootAiType": root_ai_type,
            "maxToken": prepared.max_tokens or 30000,
        }
        if prepared.has_tools:
            genai_data["tools"] = prepared.tools
            if prepared.tool_choice:
                genai_data["tool_choice"] = prepared.tool_choice
            genai_data.update(native_tool_fields(prepared.tool_adapter))
            genai_data.pop("native_tools", None)

        self._logger.debug("=== GenAI Request ===")
        self._logger.debug("Model: %s, rootAiType: %s", prepared.model, root_ai_type)
        self._logger.debug("Messages count: %d", len(prepared.messages))
        for index, message in enumerate(prepared.messages):
            role = message.get("role", "?")
            content = message.get("content", "")
            preview = (
                json.dumps(content, ensure_ascii=False)[:200] + "..."
                if not isinstance(content, str)
                else (content[:200] + "..." if len(content) > 200 else content)
            )
            self._logger.debug("  [%d] role=%s, content=%s", index, role, preview)

        try:
            response = requests.post(
                GENAI_URL,
                headers=self._get_genai_headers(),
                json=genai_data,
                stream=True,
                timeout=(10, 75),
            )
            self._logger.debug("GenAI Response Status: %d", response.status_code)

            if response.status_code != 200:
                self._logger.warning(
                    "GenAI API error %d: %s",
                    response.status_code,
                    response.text[:500],
                )
                if response.status_code == 401:
                    yield make_error_chunk("Upstream authentication failed", prepared.model)
                elif response.status_code == 429:
                    yield make_error_chunk("Upstream rate limit exceeded", prepared.model)
                else:
                    yield make_error_chunk(
                        f"Upstream API error: {response.status_code}",
                        prepared.model,
                    )
                return

            finished = False
            line_count = 0
            for line in response.iter_lines():
                if finished:
                    break

                if not line:
                    continue

                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                if line_count < 5:
                    self._logger.debug("Raw line [%d]: %s", line_count, line_str[:300])
                line_count += 1

                if line_str.startswith("data:"):
                    line_str = line_str[5:].strip()

                if not line_str:
                    continue

                try:
                    genai_json = json.loads(line_str)
                except json.JSONDecodeError as exc:
                    self._logger.debug("JSON decode error: %s, line: %s", exc, line_str[:200])
                    continue

                if isinstance(genai_json, dict) and genai_json.get("code", 200) >= 400:
                    err_msg = genai_json.get("message", "Unknown upstream error")
                    err_code = genai_json.get("code", 500)
                    self._logger.warning(
                        "GenAI business error (code=%s): %s",
                        err_code,
                        err_msg,
                    )
                    yield make_error_chunk(f"Upstream error: {err_msg}", prepared.model)
                    return

                choice = None
                finish_reason = None
                if "choices" in genai_json and genai_json["choices"]:
                    choice = genai_json["choices"][0]
                    finish_reason = choice.get("finish_reason")

                content, reasoning, tool_calls = self._extract_delta_from_genai(genai_json)
                delta = {}
                if content:
                    delta["content"] = content
                if reasoning:
                    delta["reasoning_content"] = reasoning
                if tool_calls:
                    delta["tool_calls"] = tool_calls

                if delta:
                    yield self._make_chunk(prepared.model, delta)

                if finish_reason is not None:
                    finished = True
                    yield self._make_chunk(prepared.model, {}, finish_reason=finish_reason)
                    yield "data: [DONE]\n\n"
                    break

            self._logger.debug("Total lines received: %d, finished: %s", line_count, finished)

            if not finished:
                self._logger.warning("Stream ended without finish_reason from GenAI")
                yield make_error_chunk(
                    "Stream ended unexpectedly without completion",
                    prepared.model,
                )
        except requests.Timeout as exc:
            self._logger.warning("GenAI stream timed out or stalled: %s", exc)
            yield make_error_chunk("Upstream stream timed out or stalled", prepared.model)
        except Exception as exc:
            self._logger.exception("Error in _stream_genai_response")
            yield make_error_chunk(str(exc), prepared.model)

    def _stream_genai_response_with_tools(self, prepared: PreparedChatRequest):
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())
        open_tags = _tool_start_tags_for_request(prepared.tool_adapter, prepared.tools)
        buffer = ""
        tool_buffer = ""
        sent_role = False
        tool_detected = False
        native_tool_detected = False
        think_buffer = ""
        in_think_block = False

        def make_chunk(delta, finish_reason=None):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": prepared.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }
                ],
            }
            return f"data: {json.dumps(chunk)}\n\n"

        def emit_text(text):
            nonlocal sent_role
            delta = {"content": text}
            if not sent_role:
                delta["role"] = "assistant"
                sent_role = True
            return make_chunk(delta)

        def strip_stream_think(text):
            nonlocal think_buffer, in_think_block
            current = think_buffer + text
            think_buffer = ""
            output = []

            while current:
                if in_think_block:
                    end_pos = current.find("</think>")
                    if end_pos < 0:
                        return "".join(output)
                    current = current[end_pos + len("</think>") :]
                    in_think_block = False
                    continue

                start_pos = current.find("<think>")
                if start_pos < 0:
                    prefix_len = tag_prefix_len(current, "<think>")
                    if prefix_len > 0:
                        think_buffer = current[-prefix_len:]
                        output.append(current[:-prefix_len])
                    else:
                        output.append(current)
                    return "".join(output)

                output.append(current[:start_pos])
                current = current[start_pos + len("<think>") :]
                in_think_block = True

            return "".join(output)

        for payload in self._stream_genai_response(prepared):
            for line in _iter_sse_lines(payload):
                if not line.startswith("data: "):
                    continue

                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    continue

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if "choices" not in data or not data["choices"]:
                    continue

                choice = data["choices"][0]
                if choice.get("finish_reason") == "error":
                    yield f"{line}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                chunk_delta = choice.get("delta", {})
                native_tool_calls = chunk_delta.get("tool_calls") or []
                if native_tool_calls:
                    native_tool_detected = True
                    if buffer:
                        yield emit_text(buffer)
                        buffer = ""
                    if not sent_role:
                        yield make_chunk({"role": "assistant"})
                        sent_role = True
                    for tool_call in native_tool_calls:
                        yield make_chunk({"tool_calls": [_normalize_stream_tool_call(tool_call)]})
                    continue

                content = chunk_delta.get("content", "")
                if not content:
                    finish_reason = choice.get("finish_reason")
                    if native_tool_detected and finish_reason is not None:
                        yield make_chunk({}, finish_reason="tool_calls")
                        yield "data: [DONE]\n\n"
                        return
                    continue

                if tool_detected:
                    tool_buffer += content
                    continue

                content = strip_stream_think(content)
                if not content:
                    continue

                buffer += content

                tag_pos, _ = _find_first_tag(buffer, open_tags)
                if tag_pos >= 0:
                    tool_detected = True
                    tool_buffer = buffer
                    buffer = ""
                    continue

                prefix_len = max(tag_prefix_len(buffer, tag) for tag in open_tags)
                if prefix_len > 0:
                    safe = buffer[:-prefix_len]
                    if safe.strip():
                        yield emit_text(safe)
                    buffer = buffer[-prefix_len:]
                else:
                    if buffer.strip():
                        yield emit_text(buffer)
                    buffer = ""

        if tool_detected:
            tool_calls, remaining = extract_tool_calls(
                tool_buffer,
                self._logger,
                tools=prepared.tools,
                model=prepared.model,
                adapter=prepared.tool_adapter,
            )
            if tool_calls:
                if remaining and remaining.strip():
                    yield emit_text(remaining.strip())
                if not sent_role:
                    yield make_chunk({"role": "assistant"})
                    sent_role = True

                for index, tool_call in enumerate(tool_calls):
                    yield make_chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": tool_call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tool_call["function"]["name"],
                                        "arguments": tool_call["function"]["arguments"],
                                    },
                                }
                            ]
                        }
                    )

                yield make_chunk({}, finish_reason="tool_calls")
                yield "data: [DONE]\n\n"
                return

            self._logger.warning("Tool tag detected but parsing failed — emitting as text")
            yield emit_text(tool_buffer)
            yield make_chunk({}, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        if buffer.strip():
            yield emit_text(buffer)
        if not sent_role:
            yield make_chunk({"role": "assistant", "content": ""})
        yield make_chunk({}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    def _make_chunk(self, model, delta, finish_reason=None):
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(response)}\n\n"

def _iter_sse_lines(payload: str):
    for line in str(payload).splitlines():
        if line.startswith("data: "):
            yield line


def _find_first_tag(text: str, tags: tuple[str, ...]) -> tuple[int, str | None]:
    first_pos = -1
    first_tag = None
    for tag in tags:
        pos = text.find(tag)
        if pos >= 0 and (first_pos < 0 or pos < first_pos):
            first_pos = pos
            first_tag = tag
    return first_pos, first_tag


def _tool_start_tags_for_request(adapter: str, tools: list | None) -> tuple[str, ...]:
    tags = list(tool_start_tags(adapter))
    for name in _request_tool_names(tools):
        tags.append(f"{name}<arg_key>")
        tags.append(f"{name} <arg_key>")
    return tuple(dict.fromkeys(tags))


def _request_tool_names(tools: list | None) -> list[str]:
    names = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function_data = tool.get("function", {})
        if not isinstance(function_data, dict):
            function_data = {}
        name = function_data.get("name") or tool.get("name")
        if name:
            names.append(name)
    return names


def _normalize_stream_tool_call(tool_call: dict) -> dict:
    function_data = tool_call.get("function") or {}
    return {
        "index": tool_call.get("index", 0),
        "id": tool_call.get("id"),
        "type": tool_call.get("type", "function"),
        "function": {
            "name": function_data.get("name"),
            "arguments": function_data.get("arguments", ""),
        },
    }


def _merge_tool_call_deltas(tool_call_deltas: list[dict]) -> list[dict]:
    merged = {}
    order = []

    for delta in tool_call_deltas:
        index = delta.get("index", len(order))
        if index not in merged:
            merged[index] = {
                "id": delta.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": delta.get("type", "function"),
                "function": {"name": "", "arguments": ""},
            }
            order.append(index)

        current = merged[index]
        if delta.get("id"):
            current["id"] = delta["id"]
        if delta.get("type"):
            current["type"] = delta["type"]

        function_data = delta.get("function") or {}
        if function_data.get("name"):
            current["function"]["name"] = function_data["name"]
        if function_data.get("arguments") is not None:
            current["function"]["arguments"] += function_data["arguments"]

    return [merged[index] for index in order]
