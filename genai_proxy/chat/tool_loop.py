"""Tool-call attempt loop, validation, retries, and client chunk emission."""

import json
import re
import uuid
from datetime import datetime

from genai_proxy.chat.streaming import _iter_sse_lines, _strip_error_prefix
from genai_proxy.chat.tool_calls import (
    merge_tool_call_deltas as _merge_tool_call_deltas,
    normalize_stream_tool_call as _normalize_stream_tool_call,
)
from genai_proxy.chat.tool_choice import (
    tool_calls_satisfy_choice as _tool_calls_satisfy_choice,
    tool_choice_requires_call as _tool_choice_requires_call,
)
from genai_proxy.chat.types import PreparedChatRequest
from genai_proxy.compat.openai import extract_tool_calls, make_error_chunk
from genai_proxy.errors import ProxyError
from genai_proxy.models import (
    KIMI_FINAL_CLOSE,
    KIMI_FINAL_OPEN,
    KIMI_K3_ADAPTER,
    extract_kimi_final_response,
    kimi_tool_retry_messages,
    tool_start_tags,
)

KIMI_TOOL_ATTEMPTS = 3
REQUIRED_TOOL_ATTEMPTS = 3


class ToolLoopMixin:
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
        if prepared.tool_adapter == KIMI_K3_ADAPTER:
            max_attempts = KIMI_TOOL_ATTEMPTS
        elif _tool_choice_requires_call(prepared.tool_choice):
            max_attempts = REQUIRED_TOOL_ATTEMPTS
        else:
            max_attempts = 1
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
            choice_satisfied = bool(tool_calls) and _tool_calls_satisfy_choice(
                tool_calls,
                prepared.tool_choice,
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
                or (
                    prepared.tool_adapter != KIMI_K3_ADAPTER
                    and sent_role
                )
            ):
                break
            if prepared.tool_adapter == KIMI_K3_ADAPTER:
                attempt_messages = kimi_tool_retry_messages(
                    prepared.messages,
                    tool_choice=prepared.tool_choice,
                    force_action=(
                        _tool_choice_requires_call(prepared.tool_choice)
                        or (protocol_missing and not invalid_syntax)
                    ),
                )
                warning = "Kimi K3 did not produce a valid client response"
            else:
                attempt_messages = _required_tool_retry_messages(
                    prepared.messages,
                    prepared.tool_choice,
                )
                warning = "Upstream did not satisfy the explicit tool choice"
            self._logger.warning(
                "%s; retrying (attempt %d/%d)",
                warning,
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

        if _tool_choice_requires_call(prepared.tool_choice):
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


def _required_tool_retry_messages(messages: list[dict], tool_choice) -> list[dict]:
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
    else:
        name = None
    requirement = (
        f"Call exactly the function {json.dumps(name, ensure_ascii=False)}."
        if name
        else "Call at least one of the available functions."
    )
    reminder = (
        "The previous response was discarded because it did not satisfy the "
        f"client's explicit tool_choice. {requirement} Use the tool-call format "
        "already defined in this conversation and do not answer in prose."
    )

    retried = [dict(message) for message in messages]
    if retried and retried[-1].get("role") == "user":
        content = retried[-1].get("content")
        if isinstance(content, str):
            retried[-1]["content"] = f"{content}\n\n{reminder}"
        elif isinstance(content, list):
            retried[-1]["content"] = [
                *content,
                {"type": "text", "text": f"\n\n{reminder}"},
            ]
        else:
            retried[-1]["content"] = reminder
        return retried
    return [*retried, {"role": "user", "content": reminder}]


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


__all__ = ["KIMI_TOOL_ATTEMPTS", "REQUIRED_TOOL_ATTEMPTS", "ToolLoopMixin"]
