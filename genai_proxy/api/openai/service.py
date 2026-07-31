"""OpenAI Chat Completions and Responses protocol service."""

import json
import uuid
from datetime import datetime

from genai_proxy.api.openai.errors import make_error_chunk
from genai_proxy.api.openai.responses import (
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
from genai_proxy.chat.service import ChatService
from genai_proxy.chat.streaming import _iter_sse_lines, _strip_error_prefix
from genai_proxy.chat.tool_calls import (
    merge_tool_call_deltas as _merge_tool_call_deltas,
)
from genai_proxy.chat.tool_calls import (
    normalize_stream_tool_call as _normalize_stream_tool_call,
)
from genai_proxy.chat.tool_loop import (
    _strip_visible_tool_syntax,
    _tool_start_tags_for_request,
)
from genai_proxy.chat.tool_protocol import extract_tool_calls
from genai_proxy.chat.types import PreparedChatRequest, ResolvedModelContext
from genai_proxy.chat.usage import responses_usage as _responses_usage
from genai_proxy.errors import ProxyError


class OpenAIProtocolMixin:
    """Translate chat orchestration results into OpenAI wire formats."""

    def build_openai_completion(
        self,
        req_data,
        *,
        model_context: ResolvedModelContext | None = None,
    ):
        prepared = self._prepare_chat_request(
            req_data,
            model_context=model_context,
        )
        return self._build_openai_completion(prepared)

    def stream_openai_completion(
        self,
        req_data,
        *,
        model_context: ResolvedModelContext | None = None,
    ):
        prepared = self._prepare_chat_request(
            req_data,
            count_usage=False,
            model_context=model_context,
        )
        return self._stream_prepared_openai_completion(prepared)

    def count_openai_input_tokens(
        self,
        req_data,
        *,
        model_context: ResolvedModelContext | None = None,
    ) -> int:
        prompt_tokens = self._prepare_chat_request(
            req_data,
            model_context=model_context,
        ).prompt_tokens
        if prompt_tokens is None:
            raise RuntimeError("Input token counting completed without a token count")
        return prompt_tokens

    def count_responses_input_tokens(self, req_data) -> int:
        context, model_context = self._convert_responses_request(req_data)
        return self.count_openai_input_tokens(
            context.openai_request,
            model_context=model_context,
        )

    def _convert_responses_request(self, req_data):
        if not isinstance(req_data, dict):
            raise ProxyError("Request body must be a JSON object")
        requested_model = req_data.get("model", "GPT-4.1")
        model_context = self.resolve_model_context(requested_model)
        context = convert_responses_to_openai_request(req_data)
        return context, model_context

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
        context, model_context = self._convert_responses_request(req_data)
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
        prepared = self._prepare_chat_request(
            openai_request,
            count_usage=False,
            model_context=model_context,
        )
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

    def _new_completion_metadata(self):
        return f"chatcmpl-{uuid.uuid4().hex[:24]}", int(datetime.now().timestamp())

    def _make_chunk(
        self,
        model,
        delta,
        finish_reason=None,
        *,
        completion_id=None,
        created=None,
    ):
        completion_id = completion_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(created if created is not None else datetime.now().timestamp())
        response = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
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

    def _make_usage_chunk(
        self,
        prepared: PreparedChatRequest,
        *,
        completion_id=None,
        created=None,
    ) -> str:
        if prepared.generated_usage is None:
            raise RuntimeError(
                "Completion usage was requested before generation finished"
            )
        completion_id = completion_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(created if created is not None else datetime.now().timestamp())
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": prepared.model,
            "choices": [],
            "usage": prepared.generated_usage,
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def _make_error_chunk(
        self,
        message,
        model="unknown",
        completion_id=None,
        created=None,
    ):
        return make_error_chunk(message, model, completion_id, created)


class GenAIService(OpenAIProtocolMixin, ChatService):
    """Concrete service exposed by the OpenAI-compatible API."""


__all__ = ["GenAIService", "OpenAIProtocolMixin"]
