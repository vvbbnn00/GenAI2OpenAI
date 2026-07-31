"""Safe cleanup for Kimi K3 history records created by the upstream API."""

import hashlib
import threading
import time
from dataclasses import dataclass

import requests

from genai_proxy.errors import ProxyError
from genai_proxy.retry import (
    is_retryable_business_error,
    is_retryable_status,
    transient_upstream_error,
)
from genai_proxy.upstream import transport as upstream_transport
from genai_proxy.upstream.auth import is_genai_auth_failure

KIMI_HISTORY_PAGE_SIZE = 200
KIMI_HISTORY_MAX_PAGES = 50
KIMI_HISTORY_POLL_ATTEMPTS = 20
KIMI_HISTORY_POLL_INTERVAL = 0.25


@dataclass(frozen=True, slots=True)
class _KimiHistoryCleanup:
    question: str
    user_id: str
    existing_group_ids: frozenset[str]


class KimiHistoryCleanupMixin:
    """Remove only the unambiguous K3 history record created by one request."""

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
                "Kimi K3 history cleanup disabled for this request (%s)",
                type(exc).__name__,
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
            self._logger.warning(
                "Failed to delete Kimi K3 history record (%s)",
                type(exc).__name__,
            )

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
                response = upstream_transport.fetch_history_page(
                    token,
                    user_id,
                    page_number,
                    KIMI_HISTORY_PAGE_SIZE,
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
            response = upstream_transport.delete_history_group(token, group_id)
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


def _history_group_ids(records: list[dict]) -> set[str]:
    return {
        str(record["chatGroupId"])
        for record in records
        if isinstance(record, dict) and record.get("chatGroupId")
    }


__all__ = ["KimiHistoryCleanupMixin"]
