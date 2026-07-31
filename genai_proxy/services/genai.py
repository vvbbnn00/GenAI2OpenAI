import hashlib
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

import requests

from genai_proxy.compat.openai import (
    extract_tool_calls,
    inject_tool_prompt,
    make_error_chunk,
)
from genai_proxy.compat.responses import (
    convert_responses_to_openai_request,
    make_message_added_item,
    make_message_item,
    make_reasoning_added_item,
    make_reasoning_item,
    make_response_id,
    make_response_tool_added_item,
    make_response_tool_item,
    response_completed_event,
    response_content_part_added,
    response_content_part_done,
    response_created_event,
    response_custom_tool_call_input_delta,
    response_failed_event,
    response_output_item_added,
    response_output_item_done,
    response_output_text,
    response_output_text_delta,
    response_output_text_done,
    response_reasoning_text_delta,
    response_reasoning_text_done,
)
from genai_proxy.errors import ProxyError
from genai_proxy.messages import normalize_message_contents
from genai_proxy.optimizations import (
    DEEPSEEK_V4_ADAPTERS,
    GLM_5_2_ADAPTER,
    KIMI_FINAL_CLOSE,
    KIMI_FINAL_OPEN,
    KIMI_K3_ADAPTER,
    KIMI_TOOL_TRANSPORT_ERROR,
    extract_kimi_final_response,
    inject_deepseek_reasoning_prompt,
    inject_glm_reasoning_prompt,
    inject_kimi_tool_prompt,
    kimi_tool_retry_messages,
    select_tool_adapter,
    tool_start_tags,
)
from genai_proxy.reasoning import (
    deepseek_thinking_enabled,
    normalize_reasoning_for_adapter,
    parse_reasoning_config,
)
from genai_proxy.retry import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF,
    TRANSIENT_UPSTREAM_ERROR_CODE,
    is_retryable_business_error,
    is_retryable_status,
    schedule_retry,
    transient_upstream_error,
)
from genai_proxy.services.token_manager import is_genai_auth_failure, parse_jwt_payload
from genai_proxy.token_usage import (
    count_openai_completion_tokens,
    count_openai_reasoning_tokens,
    count_openai_request_tokens,
    kimi_image_sizes_for_messages,
    tokenizer_family_for_model,
)

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
GENAI_HISTORY_LIST_URL = (
    "https://genai.shanghaitech.edu.cn/htk/ai/history/listByContentGroup"
)
GENAI_HISTORY_DELETE_URL = (
    "https://genai.shanghaitech.edu.cn/htk/ai/history/delete/groupId"
)
GENAI_STREAM_TIMEOUT = (10, 90)
GENAI_TIMEOUT_MAX_RETRIES = 1
GENAI_HISTORY_TIMEOUT = (5, 15)
KIMI_EMPTY_CURRENT_INPUT = "\u200b"
KIMI_HISTORY_PAGE_SIZE = 200
KIMI_HISTORY_MAX_PAGES = 50
KIMI_HISTORY_POLL_ATTEMPTS = 20
KIMI_HISTORY_POLL_INTERVAL = 0.25
KIMI_TOOL_ATTEMPTS = 3


@dataclass(slots=True)
class PreparedChatRequest:
    messages: list
    model: str
    root_model_name: str | None
    max_tokens: int
    has_tools: bool
    tools: list
    tool_choice: object
    tool_adapter: str
    model_record: dict | None
    include_usage: bool
    prompt_tokens: int | None
    token_reasoning_config: dict | None
    thinking: bool | None
    image_sizes: tuple[tuple[int, int], ...] | None
    generated_usage: dict | None = None


@dataclass(frozen=True, slots=True)
class _KimiHistoryCleanup:
    question: str
    user_id: str
    existing_group_ids: frozenset[str]


class _ThinkTagDeltaParser:
    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self):
        self._buffer = ""
        self._in_reasoning = False
        self._can_open = True

    def disable(self):
        if not self._in_reasoning:
            self._can_open = False

    def feed(self, text: str | None) -> tuple[str, str]:
        if not text:
            return "", ""

        source = self._buffer + str(text)
        self._buffer = ""

        if self._in_reasoning:
            return self._feed_reasoning(source)
        if not self._can_open:
            return source, ""

        open_at = source.find(self._OPEN)
        if open_at >= 0:
            prefix = source[:open_at]
            if prefix.strip():
                self._can_open = False
                return source, ""
            self._in_reasoning = True
            content, reasoning = self._feed_reasoning(
                source[open_at + len(self._OPEN) :]
            )
            return prefix + content, reasoning

        suffix_length = self._partial_tag_suffix_length(source, self._OPEN)
        remainder = source[:-suffix_length] if suffix_length else source
        if remainder.strip():
            self._can_open = False
            return source, ""
        if suffix_length:
            self._buffer = source[-suffix_length:]
        return remainder, ""

    def finish(self) -> tuple[str, str]:
        pending = self._buffer
        self._buffer = ""
        if not pending:
            return "", ""
        return ("", pending) if self._in_reasoning else (pending, "")

    def _feed_reasoning(self, source: str) -> tuple[str, str]:
        close_at = source.find(self._CLOSE)
        if close_at >= 0:
            reasoning = source[:close_at]
            content = source[close_at + len(self._CLOSE) :]
            self._in_reasoning = False
            self._can_open = False
            return content, reasoning

        suffix_length = self._partial_tag_suffix_length(source, self._CLOSE)
        if suffix_length:
            self._buffer = source[-suffix_length:]
            source = source[:-suffix_length]
        return "", source

    @staticmethod
    def _partial_tag_suffix_length(text: str, tag: str) -> int:
        maximum = min(len(text), len(tag) - 1)
        for length in range(maximum, 0, -1):
            suffix = text[-length:]
            if tag.startswith(suffix):
                return length
        return 0


class GenAIService:
    def __init__(
        self,
        logger,
        token_manager,
        model_manager,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        cleanup_kimi_history: bool = False,
    ):
        self._logger = logger
        self._token_manager = token_manager
        self._model_manager = model_manager
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff = max(0.0, float(retry_backoff))
        self._cleanup_kimi_history = bool(cleanup_kimi_history)
        self._billing_user_id = getattr(token_manager, "billing_user_id", None)
        self._billing_user_id_lock = threading.Lock()
        self._kimi_history_locks = {}
        self._kimi_history_locks_guard = threading.Lock()

    def build_openai_completion(self, req_data):
        prepared = self._prepare_chat_request(req_data)
        return self._build_openai_completion(prepared)

    def stream_openai_completion(self, req_data):
        prepared = self._prepare_chat_request(req_data, count_usage=False)
        return self._stream_prepared_openai_completion(prepared)

    def count_openai_input_tokens(self, req_data) -> int:
        prompt_tokens = self._prepare_chat_request(req_data).prompt_tokens
        if prompt_tokens is None:
            raise RuntimeError("Input token counting completed without a token count")
        return prompt_tokens

    def build_response(self, req_data):
        response = None
        output = []
        for payload in self.stream_responses(req_data, buffer_upstream=True):
            for line in _iter_sse_lines(payload):
                data_str = line[6:].strip()
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                if event_type == "response.output_item.done" and isinstance(
                    event.get("item"), dict
                ):
                    output.append(event["item"])
                elif event_type == "response.completed":
                    response = event.get("response") or {}
                elif event_type == "response.failed":
                    error = (event.get("response") or {}).get("error") or {}
                    raise ProxyError(
                        error.get("message") or "Responses request failed",
                        error_type="upstream_error",
                        code=error.get("code"),
                        status=502,
                    )

        if response is None:
            raise ProxyError(
                "Responses stream ended without response.completed",
                error_type="upstream_error",
                status=502,
            )

        response = dict(response)
        response.setdefault("output", output)
        response["output_text"] = response_output_text(response.get("output") or [])
        return response

    def stream_responses(self, req_data, *, buffer_upstream=False):
        context = convert_responses_to_openai_request(req_data)
        openai_request = dict(context.openai_request)
        openai_request["stream"] = True

        model = openai_request.get("model", "unknown")
        response_id = make_response_id()
        created = int(datetime.now().timestamp())
        output_items_by_index = {}
        output_text = ""
        output_reasoning = ""
        tool_call_deltas = []
        message_item_id = None
        message_output_index = None
        reasoning_item_id = None
        reasoning_output_index = None
        next_output_index = 0
        sequence_number = 0

        def take_sequence_number():
            nonlocal sequence_number
            current = sequence_number
            sequence_number += 1
            return current

        def start_reasoning_item():
            nonlocal next_output_index, reasoning_item_id
            nonlocal reasoning_output_index
            if reasoning_item_id is not None:
                return
            reasoning_item_id = f"rs_{uuid.uuid4().hex[:24]}"
            reasoning_output_index = next_output_index
            next_output_index += 1
            yield response_output_item_added(
                make_reasoning_added_item(reasoning_item_id),
                output_index=reasoning_output_index,
                sequence_number=take_sequence_number(),
            )
            yield response_content_part_added(
                {"type": "reasoning_text", "text": ""},
                item_id=reasoning_item_id,
                output_index=reasoning_output_index,
                sequence_number=take_sequence_number(),
            )

        def finish_reasoning_item():
            if (
                reasoning_item_id is None
                or reasoning_output_index is None
                or reasoning_output_index in output_items_by_index
            ):
                return
            part = {"type": "reasoning_text", "text": output_reasoning}
            yield response_reasoning_text_done(
                output_reasoning,
                item_id=reasoning_item_id,
                output_index=reasoning_output_index,
                sequence_number=take_sequence_number(),
            )
            yield response_content_part_done(
                part,
                item_id=reasoning_item_id,
                output_index=reasoning_output_index,
                sequence_number=take_sequence_number(),
            )
            item = make_reasoning_item(output_reasoning, reasoning_item_id)
            output_items_by_index[reasoning_output_index] = item
            yield response_output_item_done(
                item,
                output_index=reasoning_output_index,
                sequence_number=take_sequence_number(),
            )

        def start_message_item():
            nonlocal message_item_id, message_output_index, next_output_index
            if message_item_id is not None:
                return
            message_item_id = f"msg_{uuid.uuid4().hex[:24]}"
            message_output_index = next_output_index
            next_output_index += 1
            yield response_output_item_added(
                make_message_added_item(message_item_id),
                output_index=message_output_index,
                sequence_number=take_sequence_number(),
            )
            yield response_content_part_added(
                {"type": "output_text", "text": "", "annotations": []},
                item_id=message_item_id,
                output_index=message_output_index,
                sequence_number=take_sequence_number(),
            )

        def finish_message_item(*, force: bool):
            if message_item_id is None:
                if not force:
                    return
                yield from start_message_item()
            if (
                message_item_id is None
                or message_output_index is None
                or message_output_index in output_items_by_index
            ):
                return
            part = {
                "type": "output_text",
                "text": output_text,
                "annotations": [],
            }
            yield response_output_text_done(
                output_text,
                item_id=message_item_id,
                output_index=message_output_index,
                sequence_number=take_sequence_number(),
            )
            yield response_content_part_done(
                part,
                item_id=message_item_id,
                output_index=message_output_index,
                sequence_number=take_sequence_number(),
            )
            item = make_message_item(output_text, message_item_id)
            output_items_by_index[message_output_index] = item
            yield response_output_item_done(
                item,
                output_index=message_output_index,
                sequence_number=take_sequence_number(),
            )

        def completed_output_items():
            return [item for _index, item in sorted(output_items_by_index.items())]

        # Responses always reports exact usage, but prompt tokenization does
        # not need to delay the upstream request or the first streamed event.
        # _usage() fills it lazily once the model has finished.
        prepared = self._prepare_chat_request(openai_request, count_usage=False)
        openai_stream = self._stream_prepared_openai_completion(
            prepared,
            buffer_until_complete=buffer_upstream,
        )

        yield response_created_event(
            response_id,
            model,
            created,
            sequence_number=take_sequence_number(),
        )

        try:
            for payload in openai_stream:
                for line in _iter_sse_lines(payload):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        continue

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    finish_reason = choice.get("finish_reason")

                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        yield from start_reasoning_item()
                        output_reasoning += reasoning
                        yield response_reasoning_text_delta(
                            reasoning,
                            item_id=reasoning_item_id,
                            output_index=reasoning_output_index,
                            sequence_number=take_sequence_number(),
                        )

                    content = delta.get("content")
                    if content:
                        yield from start_message_item()
                        output_text += content
                        yield response_output_text_delta(
                            content,
                            item_id=message_item_id,
                            output_index=message_output_index,
                            sequence_number=take_sequence_number(),
                        )

                    for tool_call in delta.get("tool_calls") or []:
                        tool_call_deltas.append(_normalize_stream_tool_call(tool_call))

                    if finish_reason == "error":
                        message = _strip_error_prefix(content or "Upstream error")
                        yield response_failed_event(
                            response_id,
                            message,
                            sequence_number=take_sequence_number(),
                        )
                        return

                    if finish_reason == "tool_calls":
                        merged_tool_calls = _merge_tool_call_deltas(tool_call_deltas)
                        yield from finish_reasoning_item()
                        yield from finish_message_item(force=False)
                        for tool_call in merged_tool_calls:
                            item = make_response_tool_item(tool_call, context.tool_map)
                            output_index = next_output_index
                            next_output_index += 1
                            yield response_output_item_added(
                                make_response_tool_added_item(item),
                                output_index=output_index,
                                sequence_number=take_sequence_number(),
                            )
                            if item.get("type") == "custom_tool_call":
                                item_id = (
                                    item.get("id") or item.get("call_id") or response_id
                                )
                                yield response_custom_tool_call_input_delta(
                                    item_id,
                                    item.get("call_id") or item_id,
                                    item.get("input") or "",
                                    output_index=output_index,
                                    sequence_number=take_sequence_number(),
                                )
                            output_items_by_index[output_index] = item
                            yield response_output_item_done(
                                item,
                                output_index=output_index,
                                sequence_number=take_sequence_number(),
                            )
                        yield response_completed_event(
                            response_id,
                            model=model,
                            output=completed_output_items(),
                            end_turn=False,
                            created=created,
                            usage=_responses_usage(
                                prepared.generated_usage
                                or self._usage(
                                    prepared,
                                    {
                                        "role": "assistant",
                                        "content": output_text or None,
                                        "reasoning_content": output_reasoning,
                                        "tool_calls": merged_tool_calls,
                                    },
                                    finish_reason="tool_calls",
                                )
                            ),
                            sequence_number=take_sequence_number(),
                        )
                        return

                    if finish_reason is not None:
                        yield from finish_reasoning_item()
                        yield from finish_message_item(force=True)
                        yield response_completed_event(
                            response_id,
                            model=model,
                            output=completed_output_items(),
                            end_turn=True,
                            created=created,
                            usage=_responses_usage(
                                prepared.generated_usage
                                or self._usage(
                                    prepared,
                                    {
                                        "role": "assistant",
                                        "content": output_text,
                                        "reasoning_content": output_reasoning,
                                    },
                                    finish_reason=finish_reason,
                                )
                            ),
                            sequence_number=take_sequence_number(),
                        )
                        return

            yield response_failed_event(
                response_id,
                "Responses stream ended without response.completed",
                sequence_number=take_sequence_number(),
            )
        except ProxyError as exc:
            yield response_failed_event(
                response_id,
                exc.message,
                code=exc.code,
                sequence_number=take_sequence_number(),
            )

    def fetch_openai_billing_subscription(self):
        def fetch(token):
            access_until = self._extract_access_until(token)
            user_id = self._get_billing_user_id(token)
            current_token = self._token_manager.token
            if current_token:
                access_until = self._extract_access_until(current_token)
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

        return self._with_token_auth_retry("billing subscription", fetch)

    def fetch_openai_billing_usage(self):
        def fetch(token):
            self._extract_access_until(token)
            user_id = self._get_billing_user_id(token)
            record = self._fetch_user_info_record(token, user_id)
            month_usage_usd = self._coerce_amount(record.get("monthSurplus"))
            total_usage = max(month_usage_usd, 0.0) * 100

            return {
                "object": "list",
                "total_usage": round(total_usage, 2),
            }

        return self._with_token_auth_retry("billing usage", fetch)

    def _build_openai_completion(self, prepared: PreparedChatRequest):
        complete_content = ""
        complete_reasoning = ""
        collected_tool_calls = []
        finish_reason = "stop"
        stream_error_message = None

        for payload in self._stream_prepared_openai_completion(
            prepared,
            buffer_until_complete=True,
        ):
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
                reasoning = delta.get("reasoning_content", "")
                if reasoning:
                    complete_reasoning += reasoning

                for tool_call in delta.get("tool_calls", []) or []:
                    collected_tool_calls.append(tool_call)

                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                    if finish_reason == "error":
                        stream_error_message = content or "Upstream error"

        if stream_error_message:
            raise ProxyError(
                _strip_error_prefix(stream_error_message),
                error_type="upstream_error",
                status=502,
            )

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
                tool_choice=prepared.tool_choice,
            )
            if tool_calls:
                remaining_text = _strip_visible_tool_syntax(
                    remaining_text or "",
                    _tool_start_tags_for_request(prepared.tool_adapter, prepared.tools),
                )
            message_obj = {
                "role": "assistant",
                "content": remaining_text or None,
                "tool_calls": tool_calls,
            }
            if tool_calls:
                finish_reason = "tool_calls"
            else:
                message_obj = {"role": "assistant", "content": complete_content}
        else:
            message_obj = {"role": "assistant", "content": complete_content}

        if complete_reasoning:
            message_obj["reasoning_content"] = complete_reasoning

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
            "usage": prepared.generated_usage
            or self._usage(prepared, message_obj, finish_reason=finish_reason),
        }

    def _stream_prepared_openai_completion(
        self,
        prepared: PreparedChatRequest,
        *,
        buffer_until_complete=False,
    ):
        if prepared.has_tools:
            return self._stream_genai_response_with_tools(
                prepared,
                stream_reasoning=not buffer_until_complete,
            )
        return self._stream_genai_response(
            prepared,
            buffer_until_complete=buffer_until_complete,
        )

    def _prepare_chat_request(
        self,
        req_data,
        *,
        count_usage: bool = True,
    ) -> PreparedChatRequest:
        if not req_data or "messages" not in req_data:
            raise ProxyError("Missing 'messages' field in request body")

        messages = req_data.get("messages", [])
        if not isinstance(messages, list) or any(
            not isinstance(message, dict) for message in messages
        ):
            raise ProxyError("'messages' must be a list of objects")
        _validate_openai_message_shapes(messages)

        requested_model = req_data.get("model", "GPT-4.1")
        if not isinstance(requested_model, str):
            raise ProxyError("'model' must be a string")
        model = self._model_manager.resolve_model(requested_model)
        max_tokens = req_data.get("max_tokens", 30000)
        tools = req_data.get("tools") or []
        if not isinstance(tools, list) or any(
            not isinstance(tool, dict) for tool in tools
        ):
            raise ProxyError("'tools' must be a list of objects")
        for tool in tools:
            if tool.get("type") != "function":
                continue
            function = tool.get("function")
            if not isinstance(function, dict):
                raise ProxyError("Function tools must contain a 'function' object")
            if not isinstance(function.get("name"), str) or not function["name"]:
                raise ProxyError("Function tools must contain a non-empty string name")
            if function.get("parameters") is not None and not isinstance(
                function["parameters"], dict
            ):
                raise ProxyError("Function tool parameters must be an object")
        tool_choice = req_data.get("tool_choice")
        model_record = self._model_manager.get_model_record(model)
        tool_adapter = select_tool_adapter(model, model_record)
        messages = normalize_message_contents(messages, adapter=tool_adapter)
        messages = _normalize_messages_for_model_template(
            messages,
            model,
            model_record=model_record,
            tool_adapter=tool_adapter,
        )
        reasoning_config = parse_reasoning_config(req_data)
        reasoning_config = normalize_reasoning_for_adapter(
            tool_adapter, reasoning_config
        )
        thinking = deepseek_thinking_enabled(tool_adapter, reasoning_config)

        requested_tools = bool(tools)
        has_tools = requested_tools and not _tool_choice_is_none(tool_choice)
        if requested_tools:
            messages = inject_tool_prompt(
                messages,
                tools,
                tool_choice,
                model=model,
                adapter=tool_adapter,
                reasoning_config=reasoning_config,
            )
        elif tool_adapter == KIMI_K3_ADAPTER and any(
            message.get("role") == "tool" or message.get("tool_calls")
            for message in messages
        ):
            messages = inject_kimi_tool_prompt(messages, [], tool_choice="none")
        elif tool_adapter == GLM_5_2_ADAPTER:
            messages = inject_glm_reasoning_prompt(messages, reasoning_config)
        elif tool_adapter in DEEPSEEK_V4_ADAPTERS:
            messages = inject_deepseek_reasoning_prompt(
                messages,
                reasoning_config,
                adapter=tool_adapter,
            )

        if tool_adapter == KIMI_K3_ADAPTER:
            messages = _normalize_kimi_messages_for_transport(messages)

        if not self._extract_last_user_message(messages):
            raise ProxyError("No user message found in 'messages'")

        # The GenAI transport carries two-level reasoning as injected message
        # text, not as a tokenizer/template argument. Reapplying the argument
        # here would count a prompt different from the one sent upstream.
        token_reasoning_config = (
            None
            if tool_adapter == GLM_5_2_ADAPTER or tool_adapter in DEEPSEEK_V4_ADAPTERS
            else reasoning_config
        )
        include_usage = bool(
            isinstance(req_data.get("stream_options"), dict)
            and req_data["stream_options"].get("include_usage")
        )
        prompt_tokens = None
        image_sizes = None
        family = tokenizer_family_for_model(model, model_record, tool_adapter)
        if family == "kimi_k3":
            image_sizes = kimi_image_sizes_for_messages(messages)
        if count_usage:
            prompt_tokens = count_openai_request_tokens(
                messages,
                model,
                model_record=model_record,
                tool_adapter=tool_adapter,
                reasoning_config=token_reasoning_config,
                thinking=thinking,
                image_sizes=image_sizes,
            )

        return PreparedChatRequest(
            messages=messages,
            model=model,
            root_model_name=(model_record or {}).get("rootModelName"),
            max_tokens=max_tokens,
            has_tools=has_tools,
            tools=tools if has_tools else [],
            tool_choice=tool_choice if has_tools else None,
            tool_adapter=tool_adapter,
            model_record=model_record,
            include_usage=include_usage,
            prompt_tokens=prompt_tokens,
            token_reasoning_config=token_reasoning_config,
            thinking=thinking,
            image_sizes=image_sizes,
        )

    def _completion_tokens(
        self,
        prepared: PreparedChatRequest,
        message: dict,
        *,
        finish_reason: str = "stop",
    ) -> int:
        return count_openai_completion_tokens(
            message,
            prepared.model,
            model_record=prepared.model_record,
            tool_adapter=prepared.tool_adapter,
            prompt_messages=prepared.messages,
            reasoning_config=prepared.token_reasoning_config,
            thinking=prepared.thinking,
            finish_reason=finish_reason,
            image_sizes=prepared.image_sizes,
        )

    def _usage(
        self,
        prepared: PreparedChatRequest,
        message: dict,
        *,
        finish_reason: str = "stop",
    ) -> dict:
        if prepared.prompt_tokens is None:
            prepared.prompt_tokens = count_openai_request_tokens(
                prepared.messages,
                prepared.model,
                model_record=prepared.model_record,
                tool_adapter=prepared.tool_adapter,
                reasoning_config=prepared.token_reasoning_config,
                thinking=prepared.thinking,
                image_sizes=prepared.image_sizes,
            )
        completion_tokens = self._completion_tokens(
            prepared,
            message,
            finish_reason=finish_reason,
        )
        reasoning_tokens = count_openai_reasoning_tokens(
            str(message.get("reasoning_content") or ""),
            prepared.model,
            model_record=prepared.model_record,
            tool_adapter=prepared.tool_adapter,
            prompt_messages=prepared.messages,
            reasoning_config=prepared.token_reasoning_config,
            thinking=prepared.thinking,
            image_sizes=prepared.image_sizes,
        )
        return {
            "prompt_tokens": prepared.prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prepared.prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        }

    def _record_usage(
        self,
        prepared: PreparedChatRequest,
        message: dict,
        *,
        finish_reason: str,
    ) -> dict:
        prepared.generated_usage = self._usage(
            prepared,
            message,
            finish_reason=finish_reason,
        )
        return prepared.generated_usage

    def _make_usage_chunk(self, prepared: PreparedChatRequest) -> str:
        if prepared.generated_usage is None:
            raise RuntimeError(
                "Completion usage was requested before generation finished"
            )
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": prepared.model,
            "choices": [],
            "usage": prepared.generated_usage,
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def _get_genai_headers(self, token: str | None = None):
        headers = dict(GENAI_BASE_HEADERS)
        headers["X-Access-Token"] = (
            token if token is not None else self._token_manager.token
        )
        return headers

    def _get_user_genai_headers(self, user_token: str):
        return {
            "Accept": "application/json",
            "X-Access-Token": user_token,
        }

    def _get_billing_user_id(self, token: str) -> str:
        if self._billing_user_id:
            return self._billing_user_id

        cached_user_id = getattr(self._token_manager, "billing_user_id", None)
        if cached_user_id:
            self._billing_user_id = cached_user_id
            return cached_user_id

        with self._billing_user_id_lock:
            if not self._billing_user_id:
                cached_user_id = getattr(self._token_manager, "billing_user_id", None)
                self._billing_user_id = cached_user_id or self._fetch_current_user_id(
                    token
                )
            return self._billing_user_id

    def _with_token_auth_retry(self, reason: str, fetch):
        auth_retry_used = False
        retry_count = 0
        while True:
            token = self._token_manager.token
            try:
                return fetch(token)
            except ProxyError as exc:
                if exc.code == "upstream_auth_failed" and not auth_retry_used:
                    if not self._token_manager.refresh_after_auth_failure(
                        reason,
                        rejected_token=token,
                    ):
                        raise
                    auth_retry_used = True
                    continue
                if exc.code == TRANSIENT_UPSTREAM_ERROR_CODE and schedule_retry(
                    self._logger,
                    max_retries=self._max_retries,
                    backoff=self._retry_backoff,
                    retry_count=retry_count,
                    operation=reason,
                    reason=exc.message,
                ):
                    retry_count += 1
                    continue
                raise

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
                reasoning = (
                    delta.get("reasoning_content") or delta.get("reasoning") or None
                )
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
            raise transient_upstream_error(
                "Failed to fetch subscription quota"
            ) from exc

        if response.status_code in (401, 403):
            raise ProxyError(
                "Upstream GenAI token is invalid or expired",
                error_type="authentication_error",
                code="upstream_auth_failed",
                status=502,
            )
        if response.status_code != 200:
            self._logger.warning(
                "GenAI billing API error %d: %s",
                response.status_code,
                response.text[:500],
            )
            if is_retryable_status(response.status_code):
                raise transient_upstream_error("Failed to fetch subscription quota")
            raise ProxyError(
                "Failed to fetch subscription quota",
                error_type="upstream_error",
                status=502,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            self._logger.warning("Failed to decode billing response JSON: %s", exc)
            raise transient_upstream_error(
                "Failed to fetch subscription quota"
            ) from exc

        if is_genai_auth_failure(payload):
            raise ProxyError(
                "Upstream GenAI token is invalid or expired",
                error_type="authentication_error",
                code="upstream_auth_failed",
                status=502,
            )

        if payload.get("code", 200) >= 400 or payload.get("success") is False:
            self._logger.warning("GenAI billing business error: %s", payload)
            if is_retryable_business_error(
                payload.get("code"), payload.get("message", "")
            ):
                raise transient_upstream_error("Failed to fetch subscription quota")
            raise ProxyError(
                "Failed to fetch subscription quota",
                error_type="upstream_error",
                status=502,
            )

        result = payload.get("result")
        records = result.get("records") if isinstance(result, dict) else []
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
            raise transient_upstream_error(
                "Failed to fetch subscription quota"
            ) from exc

        if response.status_code in (401, 403):
            raise ProxyError(
                "Upstream GenAI token is invalid or expired",
                error_type="authentication_error",
                code="upstream_auth_failed",
                status=502,
            )
        if response.status_code != 200:
            self._logger.warning(
                "GenAI current user API error %d: %s",
                response.status_code,
                response.text[:500],
            )
            if is_retryable_status(response.status_code):
                raise transient_upstream_error("Failed to fetch subscription quota")
            raise ProxyError(
                "Failed to fetch subscription quota",
                error_type="upstream_error",
                status=502,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            self._logger.warning("Failed to decode current user JSON: %s", exc)
            raise transient_upstream_error(
                "Failed to fetch subscription quota"
            ) from exc

        if is_genai_auth_failure(payload):
            raise ProxyError(
                "Upstream GenAI token is invalid or expired",
                error_type="authentication_error",
                code="upstream_auth_failed",
                status=502,
            )

        if payload.get("code", 200) >= 400 or payload.get("success") is False:
            self._logger.warning("GenAI current user business error: %s", payload)
            if is_retryable_business_error(
                payload.get("code"), payload.get("message", "")
            ):
                raise transient_upstream_error("Failed to fetch subscription quota")
            raise ProxyError(
                "Failed to fetch subscription quota",
                error_type="upstream_error",
                status=502,
            )

        result = payload.get("result")
        user_info = result.get("userInfo") if isinstance(result, dict) else {}
        user_id = user_info.get("id")
        if not user_id:
            self._logger.warning("Current user response missing id: %s", payload)
            raise ProxyError(
                "Failed to fetch subscription quota",
                error_type="upstream_error",
                status=502,
            )

        self._token_manager.update_billing_user_id(user_id)
        refreshed_token = result.get("token") if isinstance(result, dict) else None
        if refreshed_token:
            self._token_manager.update_token_from_upstream(
                refreshed_token,
                "current user response",
            )
        return str(user_id)

    def _extract_access_until(self, user_token: str) -> int:
        try:
            access_until = int(parse_jwt_payload(user_token).get("exp") or 0)
        except Exception as exc:
            self._logger.warning("Failed to parse billing token expiry: %s", exc)
            raise ProxyError(
                "Upstream GenAI token is invalid or expired",
                error_type="authentication_error",
                code="upstream_auth_failed",
                status=502,
            ) from exc

        if access_until <= int(time.time()):
            raise ProxyError(
                "Upstream GenAI token is invalid or expired",
                error_type="authentication_error",
                code="upstream_auth_failed",
                status=502,
            )
        return access_until

    def _coerce_amount(self, value) -> float:
        if value in (None, ""):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            self._logger.warning("Invalid billing amount from upstream: %r", value)
            raise ProxyError(
                "Failed to fetch subscription quota",
                error_type="upstream_error",
                status=502,
            )

    def _stream_genai_response(
        self,
        prepared: PreparedChatRequest,
        messages: list | None = None,
        *,
        buffer_until_complete=False,
        stream_reasoning=False,
    ):
        if prepared.tool_adapter != KIMI_K3_ADAPTER or not self._cleanup_kimi_history:
            return self._stream_genai_response_raw(
                prepared,
                messages=messages,
                buffer_until_complete=buffer_until_complete,
                stream_reasoning=stream_reasoning,
            )
        return self._stream_kimi_response_with_history_cleanup(
            prepared,
            messages=messages,
            buffer_until_complete=buffer_until_complete,
            stream_reasoning=stream_reasoning,
        )

    def _stream_kimi_response_with_history_cleanup(
        self,
        prepared: PreparedChatRequest,
        messages: list | None = None,
        *,
        buffer_until_complete=False,
        stream_reasoning=False,
    ):
        messages = messages if messages is not None else prepared.messages
        _, chat_info, _ = _genai_transport_input(
            messages,
            prepared.tool_adapter,
            prepared.image_sizes,
        )
        lock_key, history_lock = self._acquire_kimi_history_lock(chat_info)
        upstream = None
        try:
            cleanup = self._prepare_kimi_history_cleanup(chat_info)
            upstream = self._stream_genai_response_raw(
                prepared,
                messages=messages,
                buffer_until_complete=buffer_until_complete,
                stream_reasoning=stream_reasoning,
            )
            cleanup_attempted = False
            for payload in upstream:
                if (
                    cleanup is not None
                    and not cleanup_attempted
                    and _has_successful_finish_reason(payload)
                ):
                    cleanup_attempted = True
                    self._delete_completed_kimi_history(cleanup)
                yield payload
        finally:
            if upstream is not None:
                try:
                    upstream.close()
                except Exception as exc:
                    self._logger.debug("Failed to close Kimi K3 stream: %s", exc)
            self._release_kimi_history_lock(lock_key, history_lock)

    def _stream_genai_response_raw(
        self,
        prepared: PreparedChatRequest,
        messages: list | None = None,
        *,
        buffer_until_complete=False,
        stream_reasoning=False,
    ):
        if stream_reasoning and not buffer_until_complete:
            raise ValueError("stream_reasoning requires buffered completion content")
        root_ai_type = self._model_manager.root_ai_type_for(prepared.model)
        messages = messages if messages is not None else prepared.messages
        transport_messages, chat_info, image_fields = _genai_transport_input(
            messages,
            prepared.tool_adapter,
            prepared.image_sizes,
        )
        genai_data = {
            "chatInfo": chat_info,
            "messages": transport_messages,
            "type": "3",
            "stream": True,
            "aiType": prepared.model,
            "aiSecType": "1",
            "promptTokens": 0,
            "rootAiType": root_ai_type,
            "maxToken": prepared.max_tokens or 30000,
        }
        genai_data.update(image_fields)
        if prepared.thinking is not None:
            genai_data["thinking"] = prepared.thinking
        if prepared.root_model_name:
            genai_data["rootModelName"] = prepared.root_model_name

        self._logger.debug("=== GenAI Request ===")
        self._logger.debug(
            "Model: %s, rootAiType: %s, rootModelName: %s, tool_prompt: %s",
            prepared.model,
            root_ai_type,
            prepared.root_model_name,
            prepared.has_tools,
        )
        self._logger.debug("Messages count: %d", len(transport_messages))
        for index, message in enumerate(transport_messages):
            role = message.get("role", "?")
            content = message.get("content", "")
            preview = (
                json.dumps(content, ensure_ascii=False)[:200] + "..."
                if not isinstance(content, str)
                else (content[:200] + "..." if len(content) > 200 else content)
            )
            self._logger.debug("  [%d] role=%s, content=%s", index, role, preview)

        auth_retry_used = False
        network_retry_count = 0
        sent_any_chunk = False
        while True:
            response = None
            retry_after_refresh = False
            retry_after_transient_error = False
            attempt_chunks = []
            attempt_content = ""
            attempt_reasoning = ""
            attempt_tool_calls = []
            received_any_chunk = False
            think_tag_parser = _ThinkTagDeltaParser()
            request_token = self._token_manager.token
            try:
                response = requests.post(
                    GENAI_URL,
                    headers=self._get_genai_headers(request_token),
                    json=genai_data,
                    stream=True,
                    timeout=GENAI_STREAM_TIMEOUT,
                )
                self._logger.debug("GenAI Response Status: %d", response.status_code)

                if response.status_code != 200:
                    self._logger.warning(
                        "GenAI API error %d: %s",
                        response.status_code,
                        response.text[:500],
                    )
                    if is_retryable_status(
                        response.status_code
                    ) and self._schedule_chat_retry(
                        network_retry_count,
                        f"HTTP {response.status_code}",
                        sent_any_chunk=sent_any_chunk,
                    ):
                        network_retry_count += 1
                        continue
                    if response.status_code in (401, 403):
                        if (
                            not sent_any_chunk
                            and not auth_retry_used
                            and self._token_manager.refresh_after_auth_failure(
                                "chat request",
                                rejected_token=request_token,
                            )
                        ):
                            auth_retry_used = True
                            retry_after_refresh = True
                            continue
                        if not sent_any_chunk:
                            raise _upstream_auth_error()
                        yield make_error_chunk(
                            "Upstream authentication failed", prepared.model
                        )
                    elif response.status_code == 429:
                        yield _error_chunk_or_raise(
                            sent_any_chunk,
                            "Upstream rate limit exceeded",
                            prepared.model,
                            error_type="rate_limit_error",
                            status=429,
                        )
                    else:
                        yield _error_chunk_or_raise(
                            sent_any_chunk,
                            f"Upstream API error: {response.status_code}",
                            prepared.model,
                        )
                    return

                finished = False
                line_count = 0
                for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                    if finished:
                        break

                    if not line:
                        continue

                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line_count < 5:
                        self._logger.debug(
                            "Raw line [%d]: %s", line_count, line_str[:300]
                        )
                    line_count += 1

                    if line_str.startswith("data:"):
                        line_str = line_str[5:].strip()

                    if not line_str:
                        continue

                    try:
                        genai_json = json.loads(line_str)
                    except json.JSONDecodeError as exc:
                        self._logger.debug(
                            "JSON decode error: %s, line: %s", exc, line_str[:200]
                        )
                        continue

                    if is_genai_auth_failure(genai_json):
                        err_msg = genai_json.get("message", "Unknown upstream error")
                        self._logger.warning(
                            "GenAI authentication business error: %s", err_msg
                        )
                        if (
                            not sent_any_chunk
                            and not auth_retry_used
                            and self._token_manager.refresh_after_auth_failure(
                                "chat stream",
                                rejected_token=request_token,
                            )
                        ):
                            auth_retry_used = True
                            retry_after_refresh = True
                            break
                        if not sent_any_chunk:
                            raise _upstream_auth_error()
                        yield make_error_chunk(
                            "Upstream authentication failed", prepared.model
                        )
                        return

                    structured_error = _structured_upstream_error(genai_json)
                    if structured_error is not None:
                        err_msg, error_type, error_code, error_status = structured_error
                        self._logger.warning("GenAI structured error: %s", err_msg)
                        yield _error_chunk_or_raise(
                            sent_any_chunk,
                            err_msg,
                            prepared.model,
                            error_type=error_type,
                            code=error_code,
                            status=error_status,
                        )
                        return

                    if (
                        isinstance(genai_json, dict)
                        and genai_json.get("code", 200) >= 400
                    ):
                        err_msg = (
                            genai_json.get("message")
                            or genai_json.get("errMsg")
                            or "Unknown upstream error"
                        )
                        err_code = genai_json.get("code", 500)
                        self._logger.warning(
                            "GenAI business error (code=%s): %s",
                            err_code,
                            err_msg,
                        )
                        if is_retryable_business_error(
                            err_code, err_msg
                        ) and self._schedule_chat_retry(
                            network_retry_count,
                            f"stream business error {err_code}: {err_msg}",
                            sent_any_chunk=sent_any_chunk,
                        ):
                            network_retry_count += 1
                            retry_after_transient_error = True
                            break
                        yield _error_chunk_or_raise(
                            sent_any_chunk,
                            f"Upstream error: {err_msg}",
                            prepared.model,
                        )
                        return

                    choice = None
                    finish_reason = None
                    if "choices" in genai_json and genai_json["choices"]:
                        choice = genai_json["choices"][0]
                        finish_reason = choice.get("finish_reason")

                    content, reasoning, tool_calls = self._extract_delta_from_genai(
                        genai_json
                    )
                    if reasoning:
                        think_tag_parser.disable()
                    content, tagged_reasoning = think_tag_parser.feed(content)
                    if tagged_reasoning:
                        reasoning = (reasoning or "") + tagged_reasoning
                    if finish_reason is not None:
                        flush_content, flush_reasoning = think_tag_parser.finish()
                        if flush_content:
                            content = (content or "") + flush_content
                        if flush_reasoning:
                            reasoning = (reasoning or "") + flush_reasoning
                    delta = {}
                    if content:
                        delta["content"] = content
                        attempt_content += content
                    if reasoning:
                        delta["reasoning_content"] = reasoning
                        attempt_reasoning += reasoning
                    if tool_calls:
                        delta["tool_calls"] = tool_calls
                        attempt_tool_calls.extend(tool_calls)

                    if finish_reason == "error":
                        err_msg = content or reasoning or "Upstream error"
                        if self._schedule_chat_retry(
                            network_retry_count,
                            err_msg,
                            sent_any_chunk=sent_any_chunk,
                        ):
                            network_retry_count += 1
                            retry_after_transient_error = True
                            break
                        yield _error_chunk_or_raise(
                            sent_any_chunk,
                            _strip_error_prefix(err_msg),
                            prepared.model,
                        )
                        return

                    if delta:
                        received_any_chunk = True
                        if buffer_until_complete:
                            if stream_reasoning:
                                buffered_delta = {}
                                if content:
                                    buffered_delta["content"] = content
                                if tool_calls:
                                    buffered_delta["tool_calls"] = tool_calls
                                if reasoning:
                                    sent_any_chunk = True
                                    yield self._make_chunk(
                                        prepared.model,
                                        {"reasoning_content": reasoning},
                                    )
                                if buffered_delta:
                                    attempt_chunks.append(
                                        self._make_chunk(
                                            prepared.model,
                                            buffered_delta,
                                        )
                                    )
                            else:
                                attempt_chunks.append(
                                    self._make_chunk(prepared.model, delta)
                                )
                        else:
                            sent_any_chunk = True
                            yield self._make_chunk(prepared.model, delta)

                    if finish_reason is not None:
                        finished = True
                        terminal_message = {
                            "role": "assistant",
                            "content": attempt_content,
                            "reasoning_content": attempt_reasoning,
                        }
                        if attempt_tool_calls:
                            terminal_message["tool_calls"] = _merge_tool_call_deltas(
                                attempt_tool_calls
                            )
                        if prepared.prompt_tokens is not None or prepared.include_usage:
                            self._record_usage(
                                prepared,
                                terminal_message,
                                finish_reason=finish_reason,
                            )
                        finish_chunk = self._make_chunk(
                            prepared.model,
                            {},
                            finish_reason=finish_reason,
                        )
                        terminal_chunks = [finish_chunk]
                        if prepared.include_usage:
                            terminal_chunks.append(self._make_usage_chunk(prepared))
                        terminal_chunks.append("data: [DONE]\n\n")
                        if buffer_until_complete:
                            attempt_chunks.extend(terminal_chunks)
                        else:
                            sent_any_chunk = True
                            yield from terminal_chunks
                        break

                if retry_after_refresh or retry_after_transient_error:
                    continue

                self._logger.debug(
                    "Total lines received: %d, finished: %s", line_count, finished
                )

                if not finished:
                    self._logger.warning(
                        "Stream ended without finish_reason from GenAI"
                    )
                    stream_state = (
                        "after partial response data"
                        if received_any_chunk
                        else "before any response data"
                    )
                    if self._schedule_chat_retry(
                        network_retry_count,
                        f"stream ended {stream_state}",
                        sent_any_chunk=sent_any_chunk,
                    ):
                        network_retry_count += 1
                        continue
                    yield _error_chunk_or_raise(
                        sent_any_chunk,
                        "Stream ended unexpectedly without completion",
                        prepared.model,
                    )
                elif buffer_until_complete:
                    sent_any_chunk = sent_any_chunk or bool(attempt_chunks)
                    yield from attempt_chunks
                return
            except (requests.RequestException, OSError, EOFError) as exc:
                self._logger.warning("GenAI chat request failed: %s", exc)
                if self._schedule_chat_retry(
                    network_retry_count,
                    str(exc),
                    sent_any_chunk=sent_any_chunk,
                    max_retries=(
                        GENAI_TIMEOUT_MAX_RETRIES
                        if isinstance(exc, requests.Timeout)
                        else None
                    ),
                ):
                    network_retry_count += 1
                    continue
                message = (
                    "Upstream stream timed out or stalled"
                    if isinstance(exc, requests.Timeout)
                    else "Failed to connect to upstream GenAI"
                )
                yield _error_chunk_or_raise(
                    sent_any_chunk,
                    message,
                    prepared.model,
                )
                return
            except ProxyError:
                raise
            except Exception as exc:
                self._logger.exception("Error in _stream_genai_response")
                if not sent_any_chunk:
                    raise
                yield make_error_chunk(str(exc), prepared.model)
                return
            finally:
                if response is not None:
                    try:
                        response.close()
                    except Exception as exc:
                        self._logger.debug("Failed to close GenAI response: %s", exc)

    def _schedule_chat_retry(
        self,
        retry_count: int,
        reason: str,
        *,
        sent_any_chunk: bool,
        max_retries: int | None = None,
    ) -> bool:
        retry_limit = (
            self._max_retries
            if max_retries is None
            else min(self._max_retries, max(0, int(max_retries)))
        )
        if sent_any_chunk or retry_count >= retry_limit:
            return False

        return schedule_retry(
            self._logger,
            max_retries=retry_limit,
            backoff=self._retry_backoff,
            retry_count=retry_count,
            operation="GenAI chat request",
            reason=reason,
        )

    def _acquire_kimi_history_lock(self, question: str):
        key = hashlib.sha256(question.encode("utf-8")).digest()
        with self._kimi_history_locks_guard:
            entry = self._kimi_history_locks.get(key)
            if entry is None:
                entry = [threading.Lock(), 0]
                self._kimi_history_locks[key] = entry
            entry[1] += 1
            lock = entry[0]
        lock.acquire()
        return key, lock

    def _release_kimi_history_lock(self, key: bytes, lock) -> None:
        lock.release()
        with self._kimi_history_locks_guard:
            entry = self._kimi_history_locks.get(key)
            if entry is None or entry[0] is not lock:
                return
            entry[1] -= 1
            if entry[1] == 0:
                del self._kimi_history_locks[key]

    def _prepare_kimi_history_cleanup(
        self,
        question: str,
    ) -> _KimiHistoryCleanup | None:
        try:

            def fetch(token):
                user_id = self._get_billing_user_id(token)
                records = self._fetch_kimi_history_records(token, user_id)
                return user_id, records

            user_id, records = self._with_token_auth_retry(
                "Kimi K3 history snapshot",
                fetch,
            )
        except Exception as exc:
            self._logger.warning(
                "Kimi K3 history cleanup disabled for this request: %s",
                exc,
            )
            return None

        return _KimiHistoryCleanup(
            question=question,
            user_id=user_id,
            existing_group_ids=frozenset(_history_group_ids(records)),
        )

    def _delete_completed_kimi_history(
        self,
        cleanup: _KimiHistoryCleanup,
    ) -> None:
        try:
            candidates = []
            for attempt in range(KIMI_HISTORY_POLL_ATTEMPTS):
                records = self._with_token_auth_retry(
                    "Kimi K3 history lookup",
                    lambda token: self._fetch_kimi_history_records(
                        token,
                        cleanup.user_id,
                    ),
                )
                candidates_by_id = {
                    str(record["chatGroupId"]): record
                    for record in records
                    if isinstance(record, dict)
                    and record.get("question") == cleanup.question
                    and str(record.get("chatGroupId") or "")
                    not in cleanup.existing_group_ids
                    and record.get("chatGroupId")
                }
                candidates = list(candidates_by_id.values())
                if candidates or attempt == KIMI_HISTORY_POLL_ATTEMPTS - 1:
                    break
                time.sleep(KIMI_HISTORY_POLL_INTERVAL)

            if not candidates:
                self._logger.warning(
                    "Kimi K3 completed, but its history record was not found"
                )
                return
            if len(candidates) != 1:
                self._logger.warning(
                    "Kimi K3 history cleanup found %d new matching records; "
                    "skipping ambiguous deletion",
                    len(candidates),
                )
                return

            group_id = str(candidates[0]["chatGroupId"])
            self._with_token_auth_retry(
                "Kimi K3 history deletion",
                lambda token: self._delete_kimi_history_group(token, group_id),
            )
            self._logger.debug("Deleted completed Kimi K3 history record")
        except Exception as exc:
            self._logger.warning("Failed to delete Kimi K3 history record: %s", exc)

    def _fetch_kimi_history_records(
        self,
        token: str,
        user_id: str,
    ) -> list[dict]:
        records = []
        page_number = 1
        page_count = 1
        while page_number <= page_count:
            try:
                response = requests.get(
                    GENAI_HISTORY_LIST_URL,
                    params={
                        "_t": int(time.time() * 1000),
                        "pageNo": page_number,
                        "pageSize": KIMI_HISTORY_PAGE_SIZE,
                        "userId": user_id,
                        "question": "",
                    },
                    headers=self._get_user_genai_headers(token),
                    timeout=GENAI_HISTORY_TIMEOUT,
                )
            except requests.RequestException as exc:
                raise transient_upstream_error(
                    "Failed to fetch Kimi K3 history"
                ) from exc

            payload = self._decode_kimi_history_response(
                response,
                "fetch Kimi K3 history",
            )
            result = payload.get("result")
            if not isinstance(result, dict) or not isinstance(
                result.get("records"), list
            ):
                raise transient_upstream_error("Failed to fetch Kimi K3 history")
            page_records = result["records"]
            records.extend(page_records)

            raw_page_count = result.get("pages", 1)
            try:
                reported_page_count = max(int(raw_page_count), 1)
            except (TypeError, ValueError) as exc:
                raise transient_upstream_error(
                    "Failed to fetch Kimi K3 history"
                ) from exc
            if reported_page_count > KIMI_HISTORY_MAX_PAGES:
                raise ProxyError(
                    "Kimi K3 history is too large for safe cleanup",
                    error_type="upstream_error",
                    status=502,
                )
            page_count = max(page_count, reported_page_count)
            page_number += 1

        return records

    def _delete_kimi_history_group(self, token: str, group_id: str) -> None:
        try:
            response = requests.get(
                GENAI_HISTORY_DELETE_URL,
                params={
                    "_t": int(time.time() * 1000),
                    "id": group_id,
                },
                headers=self._get_user_genai_headers(token),
                timeout=GENAI_HISTORY_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise transient_upstream_error("Failed to delete Kimi K3 history") from exc

        self._decode_kimi_history_response(
            response,
            "delete Kimi K3 history",
        )

    def _decode_kimi_history_response(self, response, operation: str) -> dict:
        if response.status_code in (401, 403):
            raise ProxyError(
                "Upstream GenAI token is invalid or expired",
                error_type="authentication_error",
                code="upstream_auth_failed",
                status=502,
            )
        if response.status_code != 200:
            if is_retryable_status(response.status_code):
                raise transient_upstream_error(f"Failed to {operation}")
            raise ProxyError(
                f"Failed to {operation}",
                error_type="upstream_error",
                status=502,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise transient_upstream_error(f"Failed to {operation}") from exc

        if not isinstance(payload, dict):
            raise transient_upstream_error(f"Failed to {operation}")
        raw_code = payload.get("code", 200)
        try:
            status_code = int(raw_code) if raw_code is not None else 200
        except (TypeError, ValueError) as exc:
            raise transient_upstream_error(f"Failed to {operation}") from exc
        if is_genai_auth_failure(payload) or status_code in (401, 403):
            raise ProxyError(
                "Upstream GenAI token is invalid or expired",
                error_type="authentication_error",
                code="upstream_auth_failed",
                status=502,
            )
        if status_code >= 400 or payload.get("success") is False:
            message = payload.get("message") or payload.get("errMsg") or ""
            if is_retryable_business_error(status_code, message):
                raise transient_upstream_error(f"Failed to {operation}")
            raise ProxyError(
                f"Failed to {operation}",
                error_type="upstream_error",
                status=502,
            )
        return payload

    def _stream_genai_response_with_tools(
        self,
        prepared: PreparedChatRequest,
        *,
        stream_reasoning: bool,
    ):
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())
        open_tags = _tool_start_tags_for_request(prepared.tool_adapter, prepared.tools)

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

        def emit_response_text(text, sent_role):
            if not text:
                return sent_role
            delta = {"content": text}
            if not sent_role:
                delta["role"] = "assistant"
                sent_role = True
            yield make_chunk(delta)
            return sent_role

        def emit_tool_calls(tool_calls, sent_role):
            if not sent_role:
                yield make_chunk({"role": "assistant"})
                sent_role = True
            for index, tool_call in enumerate(tool_calls):
                yield make_chunk(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": tool_call.get("id")
                                or f"call_{uuid.uuid4().hex[:24]}",
                                "type": tool_call.get("type", "function"),
                                "function": {
                                    "name": tool_call.get("function", {}).get("name"),
                                    "arguments": tool_call.get("function", {}).get(
                                        "arguments", ""
                                    ),
                                },
                            }
                        ]
                    }
                )

        sent_role = False
        tool_calls = []
        content = ""
        remaining = ""
        reasoning_content = ""
        visible_reasoning = ""
        choice_satisfied = False
        final_response = False
        invalid_syntax = False
        max_attempts = (
            KIMI_TOOL_ATTEMPTS if prepared.tool_adapter == KIMI_K3_ADAPTER else 1
        )
        attempt_messages = prepared.messages
        for attempt_index in range(max_attempts):
            attempt = None
            try:
                for event_type, value in self._iter_tool_attempt_events(
                    prepared,
                    attempt_messages,
                    stream_reasoning=stream_reasoning,
                ):
                    if event_type == "reasoning":
                        delta = {"reasoning_content": value}
                        visible_reasoning += value
                        if not sent_role:
                            delta["role"] = "assistant"
                            sent_role = True
                        yield make_chunk(delta)
                    else:
                        attempt = value
            except ProxyError as exc:
                if not sent_role:
                    raise
                yield make_error_chunk(
                    exc.message,
                    prepared.model,
                    completion_id,
                )
                return

            if attempt is None:
                raise RuntimeError("Tool stream ended without a completed attempt")

            tool_calls = attempt.get("tool_calls") or []
            content = attempt.get("content") or ""
            remaining = content
            final_response = False
            if prepared.tool_adapter == KIMI_K3_ADAPTER and not tool_calls:
                final_response, remaining = extract_kimi_final_response(content)
                remaining = remaining or ""
            if not tool_calls and not final_response:
                tool_calls, remaining = extract_tool_calls(
                    content,
                    self._logger,
                    tools=prepared.tools,
                    model=prepared.model,
                    adapter=prepared.tool_adapter,
                    tool_choice=prepared.tool_choice,
                )
                tool_calls = tool_calls or []
            reasoning_content = attempt.get("reasoning_content") or ""

            final_marker = prepared.tool_adapter == KIMI_K3_ADAPTER and (
                KIMI_FINAL_OPEN in content or KIMI_FINAL_CLOSE in content
            )
            invalid_syntax = (
                any(tag in content for tag in open_tags) and not tool_calls
            ) or (final_marker and not final_response)
            choice_satisfied = bool(tool_calls) and (
                prepared.tool_adapter != KIMI_K3_ADAPTER
                or _tool_calls_satisfy_choice(
                    tool_calls,
                    prepared.tool_choice,
                )
            )
            protocol_missing = (
                prepared.tool_adapter == KIMI_K3_ADAPTER
                and not tool_calls
                and not final_response
            )
            should_retry = (
                invalid_syntax
                or (
                    _tool_choice_requires_call(prepared.tool_choice)
                    and not choice_satisfied
                )
                or protocol_missing
            )
            if (
                choice_satisfied
                or (
                    final_response
                    and not _tool_choice_requires_call(prepared.tool_choice)
                )
                or not should_retry
                or attempt_index == max_attempts - 1
            ):
                break
            attempt_messages = kimi_tool_retry_messages(
                prepared.messages,
                tool_choice=prepared.tool_choice,
                force_action=(
                    _tool_choice_requires_call(prepared.tool_choice)
                    or (protocol_missing and not invalid_syntax)
                ),
            )
            self._logger.warning(
                "Kimi K3 did not produce a valid client response; retrying "
                "(attempt %d/%d)",
                attempt_index + 2,
                max_attempts,
            )

        if (
            prepared.tool_adapter == KIMI_K3_ADAPTER
            and visible_reasoning != reasoning_content
            and prepared.prompt_tokens is not None
        ):
            self._record_usage(
                prepared,
                {
                    "role": "assistant",
                    "reasoning_content": visible_reasoning,
                    "content": content,
                },
                finish_reason="tool_calls" if choice_satisfied else "stop",
            )

        if choice_satisfied:
            clean_remaining = _strip_visible_tool_syntax(
                remaining or "", open_tags
            ).strip()
            if (
                prepared.tool_adapter == KIMI_K3_ADAPTER
                and "<|open|>tools<|sep|>" in content
                and prepared.prompt_tokens is not None
            ):
                self._record_usage(
                    prepared,
                    {
                        "role": "assistant",
                        "reasoning_content": (visible_reasoning or reasoning_content),
                        "content": clean_remaining or None,
                        "tool_calls": tool_calls,
                    },
                    finish_reason="tool_calls",
                )
            if clean_remaining:
                for chunk in emit_response_text(clean_remaining, sent_role):
                    if isinstance(chunk, bool):
                        sent_role = chunk
                    else:
                        yield chunk
                sent_role = True
            for chunk in emit_tool_calls(tool_calls, sent_role):
                yield chunk
            yield make_chunk({}, finish_reason="tool_calls")
            if prepared.include_usage:
                yield self._make_usage_chunk(prepared)
            yield "data: [DONE]\n\n"
            return

        if prepared.tool_adapter == KIMI_K3_ADAPTER and invalid_syntax:
            self._logger.warning(
                "Tool adapter output contained unparseable tool syntax (%d chars)",
                len(content),
            )
            yield make_error_chunk(
                "Upstream returned an invalid tool call",
                prepared.model,
                completion_id,
            )
            return

        if prepared.tool_adapter == KIMI_K3_ADAPTER and _tool_choice_requires_call(
            prepared.tool_choice
        ):
            yield make_error_chunk(
                "Upstream did not return the required tool call",
                prepared.model,
                completion_id,
            )
            return

        if prepared.tool_adapter == KIMI_K3_ADAPTER and not final_response:
            yield make_error_chunk(
                "Upstream returned neither a valid client action nor a final response",
                prepared.model,
                completion_id,
            )
            return

        clean_content = (
            remaining
            if final_response
            else _strip_visible_tool_syntax(content, open_tags)
        )
        if clean_content.strip():
            for chunk in emit_response_text(clean_content, sent_role):
                if isinstance(chunk, bool):
                    sent_role = chunk
                else:
                    yield chunk
            sent_role = True
        if not sent_role:
            yield make_chunk({"role": "assistant", "content": ""})
        yield make_chunk({}, finish_reason="stop")
        if prepared.include_usage:
            yield self._make_usage_chunk(prepared)
        yield "data: [DONE]\n\n"

    def _iter_tool_attempt_events(
        self,
        prepared: PreparedChatRequest,
        messages: list,
        *,
        stream_reasoning: bool,
    ):
        complete_content = ""
        complete_reasoning = ""
        collected_tool_calls = []
        stream = self._stream_genai_response(
            prepared,
            messages=messages,
            buffer_until_complete=True,
            stream_reasoning=stream_reasoning,
        )
        try:
            for payload in stream:
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
                    if choice.get("finish_reason") == "error":
                        raise ProxyError(
                            _strip_error_prefix(
                                delta.get("content") or "Upstream error"
                            ),
                            error_type="upstream_error",
                            status=502,
                        )

                    content = delta.get("content") or ""
                    if content:
                        complete_content += content
                    reasoning = delta.get("reasoning_content") or ""
                    if reasoning:
                        complete_reasoning += reasoning
                        yield "reasoning", reasoning
                    for tool_call in delta.get("tool_calls", []) or []:
                        collected_tool_calls.append(
                            _normalize_stream_tool_call(tool_call)
                        )

        finally:
            close = getattr(stream, "close", None)
            if close:
                close()

        yield (
            "complete",
            {
                "content": complete_content,
                "reasoning_content": complete_reasoning,
                "tool_calls": _merge_tool_call_deltas(collected_tool_calls)
                if collected_tool_calls
                else [],
            },
        )

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


def _normalize_kimi_messages_for_transport(messages: list[dict]) -> list[dict]:
    if any(
        message.get("role") not in {"system", "user", "assistant", "tool"}
        for message in messages
    ):
        raise ProxyError("Kimi K3 received an unsupported message role")
    for message in messages:
        _validate_kimi_message_content(message)
        if message.get("tools"):
            raise ProxyError(
                KIMI_TOOL_TRANSPORT_ERROR,
                error_type="invalid_request_error",
                code="unsupported_tool_transport",
                status=400,
            )
        if message.get("role") == "tool" or message.get("tool_calls"):
            raise ProxyError(
                "Kimi K3 tool history cannot be forwarded through the "
                "ShanghaiTech GenAI transport without changing Moonshot's "
                "official XTML encoding",
                error_type="invalid_request_error",
                code="unsupported_tool_transport",
                status=400,
            )
    messages = [
        {**message, "content": ""} if message.get("content") is None else message
        for message in messages
    ]
    if not messages or not any(message.get("role") == "user" for message in messages):
        return messages
    if messages[-1].get("role") != "user":
        return [
            *messages,
            {"role": "user", "content": KIMI_EMPTY_CURRENT_INPUT},
        ]

    content = messages[-1].get("content")
    if isinstance(content, str):
        if content:
            return [
                *messages[:-1],
                _kimi_current_user_message(content),
            ]
        return [
            *messages[:-1],
            _kimi_current_user_message(
                KIMI_EMPTY_CURRENT_INPUT,
            ),
        ]
    if not isinstance(content, list):
        raise ProxyError(
            "Kimi K3 message content must be a string or a list of content parts"
        )

    text_parts = []
    image_parts = []
    for part in content:
        if part.get("type") in {"image", "image_url"}:
            image_parts.append(part)
            continue
        text = part.get("text")
        if not isinstance(text, str):
            raise ProxyError(
                "Kimi K3 message content supports only text and image parts"
            )
        text_parts.append(text)

    text = "".join(text_parts)
    if not image_parts:
        normalized_content = text or KIMI_EMPTY_CURRENT_INPUT
    else:
        normalized_content = []
        normalized_content.append(
            {
                "type": "text",
                "text": text or KIMI_EMPTY_CURRENT_INPUT,
            }
        )
        normalized_content.extend(image_parts)

    normalized_message = _kimi_current_user_message(
        normalized_content,
    )
    return [*messages[:-1], normalized_message]


def _validate_kimi_message_content(message: dict) -> None:
    role = message.get("role")
    content = message.get("content")
    if role == "tool" or content is None or isinstance(content, str):
        return
    if not isinstance(content, list):
        raise ProxyError(
            "Kimi K3 message content must be a string or a list of content parts"
        )
    for part in content:
        part_type = part.get("type")
        if part_type in {"image", "image_url"}:
            if role != "user":
                raise ProxyError("Kimi K3 accepts image content only in user messages")
            continue
        if part_type != "text" or not isinstance(part.get("text"), str):
            raise ProxyError(
                "Kimi K3 message content supports only text and image parts"
            )


def _kimi_current_user_message(content) -> dict:
    return {
        "role": "user",
        "content": content,
    }


def _genai_transport_input(
    messages: list[dict],
    tool_adapter: str,
    image_sizes: tuple[tuple[int, int], ...] | None,
):
    if tool_adapter != KIMI_K3_ADAPTER:
        return messages, "", {}

    current = messages[-1]
    if current.get("role") != "user":
        raise RuntimeError("Kimi K3 transport requires a final user message")

    content = current.get("content")
    image_urls = []
    if isinstance(content, str):
        chat_info = content
    else:
        text_parts = []
        for part in content or []:
            if part.get("type") in {"image", "image_url"}:
                image_urls.append(_kimi_image_url(part))
            else:
                text_parts.append(str(part.get("text") or ""))
        chat_info = "".join(text_parts) or KIMI_EMPTY_CURRENT_INPUT

    image_fields = {}
    if image_urls:
        current_sizes = (image_sizes or ())[-len(image_urls) :]
        if len(current_sizes) != len(image_urls):
            raise RuntimeError("Kimi K3 image dimensions were not prepared")
        width, height = current_sizes[0]
        image_fields = {
            "imageUrl": image_urls[0],
            "imageUrls": image_urls,
            "width": width,
            "height": height,
        }

    return messages[:-1], chat_info, image_fields


def _kimi_image_url(part: dict) -> str:
    source = part.get(part.get("type"))
    if source is None and part.get("type") == "image":
        source = part.get("url")
    if isinstance(source, dict):
        source = source.get("url", source.get("data"))
    if not isinstance(source, str) or not source:
        raise ProxyError("Kimi K3 image content is missing its URL")
    return source


def _validate_openai_message_shapes(messages: list[dict]) -> None:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            not isinstance(part, dict) for part in content
        ):
            raise ProxyError("Message content arrays must contain objects")

        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            continue
        if not isinstance(tool_calls, list) or any(
            not isinstance(tool_call, dict) for tool_call in tool_calls
        ):
            raise ProxyError("'tool_calls' must be a list of objects")
        for tool_call in tool_calls:
            if not isinstance(tool_call.get("function"), dict):
                raise ProxyError("Each tool call must contain a 'function' object")


def _normalize_messages_for_model_template(
    messages: list[dict],
    model: str,
    *,
    model_record: dict | None,
    tool_adapter: str,
) -> list[dict]:
    family = tokenizer_family_for_model(model, model_record, tool_adapter)
    if family not in {
        "glm_5_1",
        "glm_5_2",
        "qwen_3_5",
        "minimax_m2_7",
        "kimi_k3",
    }:
        return messages

    normalized = [
        {**message, "role": "system"} if message.get("role") == "developer" else message
        for message in messages
    ]
    if family not in {"qwen_3_5", "minimax_m2_7"}:
        return normalized

    system_messages = [
        message for message in normalized if message.get("role") == "system"
    ]
    if not system_messages:
        return normalized
    non_system_messages = [
        message for message in normalized if message.get("role") != "system"
    ]
    return [
        {
            "role": "system",
            "content": _merge_system_contents(system_messages),
        },
        *non_system_messages,
    ]


def _merge_system_contents(messages: list[dict]):
    contents = [message.get("content", "") for message in messages]
    if all(isinstance(content, str) for content in contents):
        return "\n\n".join(contents)

    parts = []
    for index, content in enumerate(contents):
        if index:
            parts.append({"type": "text", "text": "\n\n"})
        if isinstance(content, list):
            parts.extend(content)
        else:
            parts.append({"type": "text", "text": str(content or "")})
    return parts


def _responses_usage(openai_usage: dict | None) -> dict | None:
    if not openai_usage:
        return None
    prompt_details = openai_usage.get("prompt_tokens_details") or {}
    completion_details = openai_usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": openai_usage.get("prompt_tokens", 0),
        "input_tokens_details": {
            "cached_tokens": prompt_details.get("cached_tokens", 0),
        },
        "output_tokens": openai_usage.get("completion_tokens", 0),
        "output_tokens_details": {
            "reasoning_tokens": completion_details.get("reasoning_tokens", 0),
        },
        "total_tokens": openai_usage.get("total_tokens", 0),
    }


def _upstream_auth_error() -> ProxyError:
    return ProxyError(
        "Upstream GenAI token is invalid or expired",
        error_type="authentication_error",
        code="upstream_auth_failed",
        status=502,
    )


def _structured_upstream_error(payload):
    if not isinstance(payload, dict) or not payload.get("error"):
        return None

    error = payload["error"]
    if isinstance(error, dict):
        message = (
            error.get("message")
            or payload.get("message")
            or payload.get("errMsg")
            or "Unknown upstream error"
        )
        error_type = error.get("type") or "upstream_error"
        raw_code = error.get("code")
    else:
        message = str(error)
        error_type = "upstream_error"
        raw_code = None

    status = raw_code if isinstance(raw_code, int) and 400 <= raw_code < 500 else 502
    code = raw_code if isinstance(raw_code, str) else None
    return str(message), str(error_type), code, status


def _error_chunk_or_raise(
    sent_any_chunk: bool,
    message: str,
    model: str,
    *,
    error_type: str = "upstream_error",
    code: str | None = None,
    status: int = 502,
) -> str:
    if not sent_any_chunk:
        raise ProxyError(message, error_type=error_type, code=code, status=status)
    return make_error_chunk(message, model)


def _strip_error_prefix(message: str) -> str:
    return message.removeprefix("[Error] ").strip() or "Upstream error"


def _iter_sse_lines(payload: str):
    for line in str(payload).splitlines():
        if line.startswith("data: "):
            yield line


def _has_successful_finish_reason(payload) -> bool:
    for line in _iter_sse_lines(payload):
        data_str = line[6:].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        choices = data.get("choices") or []
        if not choices:
            continue
        finish_reason = choices[0].get("finish_reason")
        if finish_reason is not None:
            return finish_reason != "error"
    return False


def _history_group_ids(records: list[dict]) -> set[str]:
    return {
        str(record["chatGroupId"])
        for record in records
        if isinstance(record, dict) and record.get("chatGroupId")
    }


def _tool_start_tags_for_request(adapter: str, tools: list | None) -> tuple[str, ...]:
    tags = list(tool_start_tags(adapter))
    if adapter == KIMI_K3_ADAPTER:
        return tuple(tags)
    for name, argument_names in _request_tool_specs(tools).items():
        tags.append(f"{name}<arg_key>")
        tags.append(f"{name} <arg_key>")
        for argument_name in argument_names:
            tags.append(f"{name}{argument_name}:")
            tags.append(f"{name} {argument_name}:")
    return tuple(dict.fromkeys(tags))


def _request_tool_specs(tools: list | None) -> dict[str, list[str]]:
    specs = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function_data = tool.get("function", {})
        if not isinstance(function_data, dict):
            function_data = {}
        name = function_data.get("name") or tool.get("name")
        if name:
            parameters = function_data.get("parameters", {}) or tool.get(
                "input_schema", {}
            )
            properties = (
                parameters.get("properties", {}) if isinstance(parameters, dict) else {}
            )
            specs[name] = (
                list(properties.keys()) if isinstance(properties, dict) else []
            )
    return specs


def _tool_choice_is_none(tool_choice) -> bool:
    return tool_choice == "none" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "none"
    )


def _tool_choice_requires_call(tool_choice) -> bool:
    return tool_choice == "required" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "function"
    )


def _tool_calls_satisfy_choice(tool_calls, tool_choice) -> bool:
    if not tool_calls:
        return False
    if _tool_choice_is_none(tool_choice):
        return False
    if not (isinstance(tool_choice, dict) and tool_choice.get("type") == "function"):
        return True
    function = tool_choice.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    return bool(name) and all(
        tool_call.get("function", {}).get("name") == name for tool_call in tool_calls
    )


def _strip_visible_tool_syntax(text: str, tags: tuple[str, ...]) -> str:
    if not text:
        return ""
    positions = [text.find(tag) for tag in tags if text.find(tag) >= 0]
    transcript_pos = _find_claude_code_transcript_pos(text, tags)
    if transcript_pos >= 0:
        positions.append(transcript_pos)
    if not positions:
        return text
    return text[: min(positions)].strip()


def _find_claude_code_transcript_pos(text: str, tags: tuple[str, ...]) -> int:
    names = []
    for tag in tags:
        if "<" in tag:
            name = tag.split("<", 1)[0].strip()
            if name:
                names.append(re.escape(name))
    if not names:
        return -1
    pattern = re.compile(
        rf"(?m)(?<![\w./@-])(?:{'|'.join(dict.fromkeys(names))})(?:[^\n<]*)?\nIN\n"
    )
    match = pattern.search(text)
    return match.start() if match else -1


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
