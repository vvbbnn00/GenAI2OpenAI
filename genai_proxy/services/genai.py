import json
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
from genai_proxy.optimizations import is_deepseek_model


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


@dataclass(slots=True)
class PreparedChatRequest:
    messages: list
    model: str
    max_tokens: int
    has_tools: bool
    tools: list


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

    def _build_openai_completion(self, prepared: PreparedChatRequest):
        complete_content = ""
        collected_tool_calls = []
        finish_reason = "stop"

        for line in self._stream_prepared_openai_completion(prepared):
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
                "tool_calls": collected_tool_calls,
            }
        elif prepared.has_tools:
            tool_calls, remaining_text = extract_tool_calls(
                complete_content,
                self._logger,
                tools=prepared.tools,
                model=prepared.model,
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
                "prompt_tokens": 0,
                "completion_tokens": len(complete_content),
                "total_tokens": len(complete_content),
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
        has_tools = bool(tools)

        if has_tools:
            messages = inject_tool_prompt(messages, tools, tool_choice, model=model)

        if not self._extract_last_user_message(messages):
            raise ProxyError("No user message found in 'messages'")

        return PreparedChatRequest(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            has_tools=has_tools,
            tools=tools,
        )

    def _get_genai_headers(self):
        headers = dict(GENAI_BASE_HEADERS)
        headers["X-Access-Token"] = self._token_manager.token
        return headers

    def _extract_last_user_message(self, messages):
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                return json.dumps(content, ensure_ascii=False)
        return ""

    def _extract_content_from_genai(self, response_data):
        try:
            if "choices" in response_data and response_data["choices"]:
                delta = response_data["choices"][0].get("delta", {})
                content = delta.get("content") or None
                reasoning = delta.get("reasoning_content") or None
                return content, reasoning
        except (KeyError, IndexError, TypeError):
            pass
        return None, None

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
                timeout=60,
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

                if "choices" in genai_json and genai_json["choices"]:
                    choice = genai_json["choices"][0]
                    if choice.get("finish_reason") is not None:
                        finished = True

                if finished:
                    yield self._final_chunk(prepared.model)
                    yield "data: [DONE]\n\n"
                    break

                content, reasoning = self._extract_content_from_genai(genai_json)
                delta = {}
                if content:
                    delta["content"] = content
                if reasoning:
                    delta["reasoning_content"] = reasoning

                if delta:
                    yield self._make_chunk(prepared.model, delta)

            self._logger.debug("Total lines received: %d, finished: %s", line_count, finished)

            if not finished:
                self._logger.warning("Stream ended without finish_reason from GenAI")
                yield make_error_chunk(
                    "Stream ended unexpectedly without completion",
                    prepared.model,
                )
        except Exception as exc:
            self._logger.exception("Error in _stream_genai_response")
            yield make_error_chunk(str(exc), prepared.model)

    def _stream_genai_response_with_tools(self, prepared: PreparedChatRequest):
        if is_deepseek_model(prepared.model):
            yield from self._stream_deepseek_tool_response(prepared)
            return

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())
        open_tag = "<tool_call>"
        buffer = ""
        tool_buffer = ""
        sent_role = False
        tool_detected = False

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

        for line in self._stream_genai_response(prepared):
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

            chunk_delta = data["choices"][0].get("delta", {})
            content = chunk_delta.get("content", "")
            if not content:
                continue

            if tool_detected:
                tool_buffer += content
                continue

            buffer += content

            tag_pos = buffer.find(open_tag)
            if tag_pos >= 0:
                prefix = buffer[:tag_pos]
                if prefix.strip():
                    yield emit_text(prefix)
                tool_detected = True
                tool_buffer = buffer[tag_pos:]
                buffer = ""
                continue

            prefix_len = tag_prefix_len(buffer, open_tag)
            if prefix_len > 0:
                safe = buffer[:-prefix_len]
                if safe:
                    yield emit_text(safe)
                buffer = buffer[-prefix_len:]
            else:
                if buffer:
                    yield emit_text(buffer)
                buffer = ""

        if tool_detected:
            tool_calls, remaining = extract_tool_calls(
                tool_buffer,
                self._logger,
                tools=prepared.tools,
                model=prepared.model,
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

        if buffer:
            yield emit_text(buffer)
        if not sent_role:
            yield make_chunk({"role": "assistant", "content": ""})
        yield make_chunk({}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    def _stream_deepseek_tool_response(self, prepared: PreparedChatRequest):
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())
        complete_content = ""

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

        for line in self._stream_genai_response(prepared):
            if not line.startswith("data: "):
                continue

            data_str = line[6:].strip()
            if data_str == "[DONE]":
                continue

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = data.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                complete_content += content

        tool_calls, remaining = extract_tool_calls(
            complete_content,
            self._logger,
            tools=prepared.tools,
            model=prepared.model,
        )

        sent_role = False
        if remaining:
            yield make_chunk({"role": "assistant", "content": remaining})
            sent_role = True

        if tool_calls:
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

        if not sent_role:
            yield make_chunk({"role": "assistant", "content": complete_content})
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

    def _final_chunk(self, model):
        return self._make_chunk(model, {}, finish_reason="stop")
