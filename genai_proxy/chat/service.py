"""Shared chat orchestration and upstream account helpers."""

import threading
import time

import requests

from genai_proxy.chat.preparation import (
    KIMI_EMPTY_CURRENT_INPUT,
    ChatPreparationMixin,
)
from genai_proxy.chat.streaming import (
    GENAI_TIMEOUT_MAX_RETRIES,
    ChatStreamingMixin,
)
from genai_proxy.chat.tool_loop import (
    KIMI_TOOL_ATTEMPTS,
    REQUIRED_TOOL_ATTEMPTS,
    ToolLoopMixin,
)
from genai_proxy.chat.types import PreparedChatRequest, ResolvedModelContext
from genai_proxy.chat.usage import ChatUsageMixin
from genai_proxy.errors import ProxyError
from genai_proxy.retry import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF,
    TRANSIENT_UPSTREAM_ERROR_CODE,
    is_retryable_business_error,
    is_retryable_status,
    schedule_retry,
    transient_upstream_error,
)
from genai_proxy.upstream import transport as upstream_transport
from genai_proxy.upstream.auth import is_genai_auth_failure, parse_jwt_payload
from genai_proxy.upstream.kimi_history import KimiHistoryCleanupMixin

# Keep the former service-level constants available to existing callers.
GENAI_BASE_HEADERS = upstream_transport.GENAI_BASE_HEADERS
GENAI_CURRENT_USER_URL = upstream_transport.GENAI_CURRENT_USER_URL
GENAI_HISTORY_DELETE_URL = upstream_transport.GENAI_HISTORY_DELETE_URL
GENAI_HISTORY_LIST_URL = upstream_transport.GENAI_HISTORY_LIST_URL
GENAI_HISTORY_TIMEOUT = upstream_transport.GENAI_HISTORY_TIMEOUT
GENAI_STREAM_TIMEOUT = upstream_transport.GENAI_STREAM_TIMEOUT
GENAI_URL = upstream_transport.GENAI_URL
GENAI_USER_INFO_URL = upstream_transport.GENAI_USER_INFO_URL

__all__ = [
    "GENAI_BASE_HEADERS",
    "GENAI_CURRENT_USER_URL",
    "GENAI_HISTORY_DELETE_URL",
    "GENAI_HISTORY_LIST_URL",
    "GENAI_HISTORY_TIMEOUT",
    "GENAI_STREAM_TIMEOUT",
    "GENAI_TIMEOUT_MAX_RETRIES",
    "GENAI_URL",
    "GENAI_USER_INFO_URL",
    "ChatService",
    "KIMI_EMPTY_CURRENT_INPUT",
    "KIMI_TOOL_ATTEMPTS",
    "PreparedChatRequest",
    "REQUIRED_TOOL_ATTEMPTS",
    "ResolvedModelContext",
]


class ChatService(
    ChatPreparationMixin,
    ChatUsageMixin,
    ChatStreamingMixin,
    ToolLoopMixin,
    KimiHistoryCleanupMixin,
):
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

    def _fetch_user_info_record(self, user_token: str, user_id: str):
        try:
            response = upstream_transport.fetch_user_info(user_token, user_id)
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
            response = upstream_transport.fetch_current_user(user_token)
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
