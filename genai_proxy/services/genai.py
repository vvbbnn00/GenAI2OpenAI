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
    make_response_id,
    make_response_tool_item,
    response_completed_event,
    response_created_event,
    response_custom_tool_call_input_delta,
    response_failed_event,
    response_output_item_added,
    response_output_item_done,
    response_output_text,
    response_output_text_delta,
    response_reasoning_text_delta,
)
from genai_proxy.errors import ProxyError
from genai_proxy.optimizations import (
    DEEPSEEK_V4_ADAPTERS,
    GLM_5_2_ADAPTER,
    inject_deepseek_reasoning_prompt,
    inject_glm_reasoning_prompt,
    select_tool_adapter,
    tool_start_tags,
)
from genai_proxy.reasoning import normalize_reasoning_for_adapter, parse_reasoning_config
from genai_proxy.services.token_manager import is_genai_auth_failure, parse_jwt_payload
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
GENAI_STREAM_TIMEOUT = (10, 600)
GENAI_RETRYABLE_STATUS_CODES = frozenset({408, 502, 503, 504})


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
    prompt_tokens: int


class GenAIService:
    def __init__(
        self,
        logger,
        token_manager,
        model_manager,
        *,
        max_retries: int = 5,
        retry_backoff: float = 0.5,
    ):
        self._logger = logger
        self._token_manager = token_manager
        self._model_manager = model_manager
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff = max(0.0, float(retry_backoff))
        self._billing_user_id = getattr(token_manager, "billing_user_id", None)
        self._billing_user_id_lock = threading.Lock()

    def build_openai_completion(self, req_data):
        prepared = self._prepare_chat_request(req_data)
        return self._build_openai_completion(prepared)

    def stream_openai_completion(self, req_data):
        prepared = self._prepare_chat_request(req_data)
        return self._stream_prepared_openai_completion(prepared)

    def build_response(self, req_data):
        response = None
        output = []
        for payload in self.stream_responses(req_data):
            for line in _iter_sse_lines(payload):
                data_str = line[6:].strip()
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                if event_type == "response.output_item.done" and isinstance(event.get("item"), dict):
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
            raise ProxyError("Responses stream ended without response.completed", error_type="upstream_error", status=502)

        response = dict(response)
        response.setdefault("output", output)
        response["output_text"] = response_output_text(response.get("output") or [])
        return response

    def stream_responses(self, req_data):
        context = convert_responses_to_openai_request(req_data)
        openai_request = dict(context.openai_request)
        openai_request["stream"] = True

        model = openai_request.get("model", "unknown")
        response_id = make_response_id()
        created = int(datetime.now().timestamp())
        output_items = []
        output_text = ""
        tool_call_deltas = []
        message_item_id = None
        openai_stream = self.stream_openai_completion(openai_request)

        yield response_created_event(response_id, model, created)

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
                        yield response_reasoning_text_delta(reasoning)

                    content = delta.get("content")
                    if content:
                        if message_item_id is None:
                            message_item_id = f"msg_{uuid.uuid4().hex[:24]}"
                            yield response_output_item_added(
                                make_message_added_item(message_item_id)
                            )
                        output_text += content
                        yield response_output_text_delta(content)

                    for tool_call in delta.get("tool_calls") or []:
                        tool_call_deltas.append(_normalize_stream_tool_call(tool_call))

                    if finish_reason == "error":
                        message = _strip_error_prefix(content or "Upstream error")
                        yield response_failed_event(response_id, message)
                        return

                    if finish_reason == "tool_calls":
                        if output_text:
                            item = make_message_item(output_text, message_item_id)
                            output_items.append(item)
                            yield response_output_item_done(item)
                        for tool_call in _merge_tool_call_deltas(tool_call_deltas):
                            item = make_response_tool_item(tool_call, context.tool_map)
                            if item.get("type") == "custom_tool_call":
                                item_id = item.get("id") or item.get("call_id") or response_id
                                yield response_custom_tool_call_input_delta(
                                    item_id,
                                    item.get("call_id") or item_id,
                                    item.get("input") or "",
                                )
                            output_items.append(item)
                            yield response_output_item_done(item)
                        yield response_completed_event(
                            response_id,
                            model=model,
                            output=output_items,
                            end_turn=False,
                            created=created,
                        )
                        return

                    if finish_reason is not None:
                        if message_item_id is None:
                            message_item_id = f"msg_{uuid.uuid4().hex[:24]}"
                            yield response_output_item_added(
                                make_message_added_item(message_item_id)
                            )
                        item = make_message_item(output_text, message_item_id)
                        output_items.append(item)
                        yield response_output_item_done(item)
                        yield response_completed_event(
                            response_id,
                            model=model,
                            output=output_items,
                            end_turn=True,
                            created=created,
                        )
                        return

            yield response_failed_event(
                response_id,
                "Responses stream ended without response.completed",
            )
        except ProxyError as exc:
            yield response_failed_event(response_id, exc.message, code=exc.code)

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
        collected_tool_calls = []
        finish_reason = "stop"
        stream_error_message = None

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
        model_record = self._model_manager.get_model_record(model)
        tool_adapter = select_tool_adapter(model, model_record)
        reasoning_config = parse_reasoning_config(req_data)
        reasoning_config = normalize_reasoning_for_adapter(tool_adapter, reasoning_config)

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
        elif tool_adapter == GLM_5_2_ADAPTER:
            messages = inject_glm_reasoning_prompt(messages, reasoning_config)
        elif tool_adapter in DEEPSEEK_V4_ADAPTERS:
            messages = inject_deepseek_reasoning_prompt(
                messages,
                reasoning_config,
                adapter=tool_adapter,
            )

        if not self._extract_last_user_message(messages):
            raise ProxyError("No user message found in 'messages'")

        return PreparedChatRequest(
            messages=messages,
            model=model,
            root_model_name=(model_record or {}).get("rootModelName"),
            max_tokens=max_tokens,
            has_tools=has_tools,
            tools=tools if has_tools else [],
            tool_choice=tool_choice if has_tools else None,
            tool_adapter=tool_adapter,
            prompt_tokens=estimate_openai_request_tokens(messages, model, tools if has_tools else None),
        )

    def _get_genai_headers(self, token: str | None = None):
        headers = dict(GENAI_BASE_HEADERS)
        headers["X-Access-Token"] = token if token is not None else self._token_manager.token
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
                self._billing_user_id = cached_user_id or self._fetch_current_user_id(token)
            return self._billing_user_id

    def _with_token_auth_retry(self, reason: str, fetch):
        token = self._token_manager.token
        try:
            return fetch(token)
        except ProxyError as exc:
            if exc.code != "upstream_auth_failed":
                raise
            if not self._token_manager.refresh_after_auth_failure(reason, rejected_token=token):
                raise
            return fetch(self._token_manager.token)

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
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502)

        try:
            payload = response.json()
        except ValueError as exc:
            self._logger.warning("Failed to decode billing response JSON: %s", exc)
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502) from exc

        if is_genai_auth_failure(payload):
            raise ProxyError(
                "Upstream GenAI token is invalid or expired",
                error_type="authentication_error",
                code="upstream_auth_failed",
                status=502,
            )

        if payload.get("code", 200) >= 400 or payload.get("success") is False:
            self._logger.warning("GenAI billing business error: %s", payload)
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502)

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
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502) from exc

        if response.status_code == 401:
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
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502)

        try:
            payload = response.json()
        except ValueError as exc:
            self._logger.warning("Failed to decode current user JSON: %s", exc)
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502) from exc

        if is_genai_auth_failure(payload):
            raise ProxyError(
                "Upstream GenAI token is invalid or expired",
                error_type="authentication_error",
                code="upstream_auth_failed",
                status=502,
            )

        if payload.get("code", 200) >= 400 or payload.get("success") is False:
            self._logger.warning("GenAI current user business error: %s", payload)
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502)

        result = payload.get("result")
        user_info = result.get("userInfo") if isinstance(result, dict) else {}
        user_id = user_info.get("id")
        if not user_id:
            self._logger.warning("Current user response missing id: %s", payload)
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502)

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
            raise ProxyError("Failed to fetch subscription quota", error_type="upstream_error", status=502)

    def _stream_genai_response(
        self,
        prepared: PreparedChatRequest,
        messages: list | None = None,
    ):
        root_ai_type = self._model_manager.root_ai_type_for(prepared.model)
        messages = messages if messages is not None else prepared.messages
        genai_data = {
            "chatInfo": "",
            "messages": messages,
            "type": "3",
            "stream": True,
            "aiType": prepared.model,
            "aiSecType": "1",
            "promptTokens": 0,
            "rootAiType": root_ai_type,
            "maxToken": prepared.max_tokens or 30000,
        }
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
        self._logger.debug("Messages count: %d", len(messages))
        for index, message in enumerate(messages):
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
                    if (
                        response.status_code in GENAI_RETRYABLE_STATUS_CODES
                        and self._schedule_chat_retry(
                            network_retry_count,
                            f"HTTP {response.status_code}",
                            sent_any_chunk=sent_any_chunk,
                        )
                    ):
                        network_retry_count += 1
                        continue
                    if response.status_code in (401, 403):
                        if (
                            not auth_retry_used
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
                        yield make_error_chunk("Upstream authentication failed", prepared.model)
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

                    if is_genai_auth_failure(genai_json):
                        err_msg = genai_json.get("message", "Unknown upstream error")
                        self._logger.warning("GenAI authentication business error: %s", err_msg)
                        if (
                            not auth_retry_used
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
                        yield make_error_chunk("Upstream authentication failed", prepared.model)
                        return

                    if isinstance(genai_json, dict) and genai_json.get("code", 200) >= 400:
                        err_msg = genai_json.get("message", "Unknown upstream error")
                        err_code = genai_json.get("code", 500)
                        self._logger.warning(
                            "GenAI business error (code=%s): %s",
                            err_code,
                            err_msg,
                        )
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

                    content, reasoning, tool_calls = self._extract_delta_from_genai(genai_json)
                    delta = {}
                    if content:
                        delta["content"] = content
                    if reasoning:
                        delta["reasoning_content"] = reasoning
                    if tool_calls:
                        delta["tool_calls"] = tool_calls

                    if delta:
                        sent_any_chunk = True
                        yield self._make_chunk(prepared.model, delta)

                    if finish_reason is not None:
                        finished = True
                        sent_any_chunk = True
                        yield self._make_chunk(prepared.model, {}, finish_reason=finish_reason)
                        yield "data: [DONE]\n\n"
                        break

                if retry_after_refresh:
                    continue

                self._logger.debug("Total lines received: %d, finished: %s", line_count, finished)

                if not finished:
                    self._logger.warning("Stream ended without finish_reason from GenAI")
                    if self._schedule_chat_retry(
                        network_retry_count,
                        "stream ended before any response data",
                        sent_any_chunk=sent_any_chunk,
                    ):
                        network_retry_count += 1
                        continue
                    yield _error_chunk_or_raise(
                        sent_any_chunk,
                        "Stream ended unexpectedly without completion",
                        prepared.model,
                    )
                return
            except requests.RequestException as exc:
                self._logger.warning("GenAI chat request failed: %s", exc)
                if self._schedule_chat_retry(
                    network_retry_count,
                    str(exc),
                    sent_any_chunk=sent_any_chunk,
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
                    response.close()

    def _schedule_chat_retry(
        self,
        retry_count: int,
        reason: str,
        *,
        sent_any_chunk: bool,
    ) -> bool:
        if sent_any_chunk or retry_count >= self._max_retries:
            return False

        delay = self._retry_backoff * (2**retry_count)
        self._logger.warning(
            "Retrying GenAI chat request (%d/%d) in %.2f seconds: %s",
            retry_count + 1,
            self._max_retries,
            delay,
            reason,
        )
        if delay:
            time.sleep(delay)
        return True

    def _stream_genai_response_with_tools(self, prepared: PreparedChatRequest):
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
                                "id": tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                                "type": tool_call.get("type", "function"),
                                "function": {
                                    "name": tool_call.get("function", {}).get("name"),
                                    "arguments": tool_call.get("function", {}).get("arguments", ""),
                                },
                            }
                        ]
                    }
                )

        attempt = self._collect_tool_attempt(
            prepared,
            prepared.messages,
        )

        tool_calls = attempt.get("tool_calls") or []
        content = attempt.get("content") or ""
        remaining = content
        if not tool_calls:
            tool_calls, remaining = extract_tool_calls(
                content,
                self._logger,
                tools=prepared.tools,
                model=prepared.model,
                adapter=prepared.tool_adapter,
            )
            tool_calls = tool_calls or []

        sent_role = False
        if tool_calls:
            clean_remaining = _strip_visible_tool_syntax(remaining or "", open_tags).strip()
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
            yield "data: [DONE]\n\n"
            return

        clean_content = _strip_visible_tool_syntax(content, open_tags)
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
        yield "data: [DONE]\n\n"

    def _collect_tool_attempt(
        self,
        prepared: PreparedChatRequest,
        messages: list,
    ) -> dict:
        complete_content = ""
        collected_tool_calls = []
        stream = self._stream_genai_response(prepared, messages=messages)
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
                            _strip_error_prefix(delta.get("content") or "Upstream error"),
                            error_type="upstream_error",
                            status=502,
                        )

                    content = delta.get("content") or ""
                    if content:
                        complete_content += content

                    for tool_call in delta.get("tool_calls", []) or []:
                        collected_tool_calls.append(_normalize_stream_tool_call(tool_call))

        finally:
            close = getattr(stream, "close", None)
            if close:
                close()

        return {
            "content": complete_content,
            "tool_calls": _merge_tool_call_deltas(collected_tool_calls) if collected_tool_calls else [],
        }

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


def _upstream_auth_error() -> ProxyError:
    return ProxyError(
        "Upstream GenAI token is invalid or expired",
        error_type="authentication_error",
        code="upstream_auth_failed",
        status=502,
    )


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


def _tool_start_tags_for_request(adapter: str, tools: list | None) -> tuple[str, ...]:
    tags = list(tool_start_tags(adapter))
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
            parameters = function_data.get("parameters", {}) or tool.get("input_schema", {})
            properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
            specs[name] = list(properties.keys()) if isinstance(properties, dict) else []
    return specs


def _tool_choice_is_none(tool_choice) -> bool:
    return tool_choice == "none" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "none"
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
