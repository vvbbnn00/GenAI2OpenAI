"""Incremental upstream stream parsing, retries, and Kimi cleanup ordering."""

import json

import requests

from genai_proxy.chat.preparation import (
    genai_transport_input as _genai_transport_input,
)
from genai_proxy.chat.tool_calls import (
    merge_tool_call_deltas as _merge_tool_call_deltas,
)
from genai_proxy.chat.types import PreparedChatRequest
from genai_proxy.errors import ProxyError
from genai_proxy.models import KIMI_K3_ADAPTER
from genai_proxy.retry import (
    is_retryable_business_error,
    is_retryable_status,
    schedule_retry,
)
from genai_proxy.upstream import transport as upstream_transport
from genai_proxy.upstream.auth import is_genai_auth_failure

GENAI_TIMEOUT_MAX_RETRIES = 1


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


class ChatStreamingMixin:
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
        completion_id, created = self._new_completion_metadata()

        def make_chunk(delta, finish_reason=None):
            return self._make_chunk(
                prepared.model,
                delta,
                finish_reason,
                completion_id=completion_id,
                created=created,
            )

        def make_usage_chunk():
            return self._make_usage_chunk(
                prepared,
                completion_id=completion_id,
                created=created,
            )

        def make_error_chunk(message):
            return self._make_error_chunk(
                message,
                prepared.model,
                completion_id,
                created,
            )

        def error_chunk_or_raise(
            sent_chunk,
            message,
            *,
            error_type="upstream_error",
            code=None,
            status=502,
        ):
            if not sent_chunk:
                raise ProxyError(
                    message,
                    error_type=error_type,
                    code=code,
                    status=status,
                )
            return make_error_chunk(message)

        root_ai_type = prepared.root_ai_type
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
                response = upstream_transport.post_chat(request_token, genai_data)
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
                        yield make_error_chunk("Upstream authentication failed")
                    elif response.status_code == 429:
                        yield error_chunk_or_raise(
                            sent_any_chunk,
                            "Upstream rate limit exceeded",
                            error_type="rate_limit_error",
                            status=429,
                        )
                    else:
                        yield error_chunk_or_raise(
                            sent_any_chunk,
                            f"Upstream API error: {response.status_code}",
                        )
                    return

                finished = False
                line_count = 0
                for line_count, genai_json in upstream_transport.iter_sse_json(
                    response,
                    self._logger,
                ):
                    if finished:
                        break

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
                        yield make_error_chunk("Upstream authentication failed")
                        return

                    structured_error = _structured_upstream_error(genai_json)
                    if structured_error is not None:
                        err_msg, error_type, error_code, error_status = structured_error
                        self._logger.warning("GenAI structured error: %s", err_msg)
                        yield error_chunk_or_raise(
                            sent_any_chunk,
                            err_msg,
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
                        yield error_chunk_or_raise(
                            sent_any_chunk,
                            f"Upstream error: {err_msg}",
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
                        yield error_chunk_or_raise(
                            sent_any_chunk,
                            _strip_error_prefix(err_msg),
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
                                    yield make_chunk(
                                        {"reasoning_content": reasoning}
                                    )
                                if buffered_delta:
                                    attempt_chunks.append(make_chunk(buffered_delta))
                            else:
                                attempt_chunks.append(make_chunk(delta))
                        else:
                            sent_any_chunk = True
                            yield make_chunk(delta)

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
                        finish_chunk = make_chunk({}, finish_reason=finish_reason)
                        terminal_chunks = [finish_chunk]
                        if prepared.include_usage:
                            terminal_chunks.append(make_usage_chunk())
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
                    yield error_chunk_or_raise(
                        sent_any_chunk,
                        "Stream ended unexpectedly without completion",
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
                yield error_chunk_or_raise(
                    sent_any_chunk,
                    message,
                )
                return
            except ProxyError:
                raise
            except Exception as exc:
                self._logger.exception("Error in _stream_genai_response")
                if not sent_any_chunk:
                    raise
                yield make_error_chunk(str(exc))
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


__all__ = ["ChatStreamingMixin", "GENAI_TIMEOUT_MAX_RETRIES"]
