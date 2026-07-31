import time

from genai_proxy.errors import ProxyError

DEFAULT_MAX_RETRIES = 10
DEFAULT_RETRY_BACKOFF = 0.5
MAX_RETRY_DELAY = 5.0

RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
TRANSIENT_UPSTREAM_ERROR_CODE = "upstream_transient_error"
NON_RETRYABLE_BUSINESS_ERROR_MARKERS = (
    "模型不存在",
    "未找到对应节点",
    "节点信息不存在",
    "参数不合法",
    "参数错误",
    "invalid model",
    "model not found",
    "node not found",
    "no corresponding node",
    "unsupported model",
    "invalid parameter",
    "invalid request",
)


def retry_delay(backoff: float, retry_count: int) -> float:
    return min(max(0.0, float(backoff)) * (2**retry_count), MAX_RETRY_DELAY)


def is_retryable_status(status_code) -> bool:
    try:
        return int(status_code) in RETRYABLE_STATUS_CODES
    except (TypeError, ValueError):
        return False


def is_retryable_business_error(status_code, message: str = "") -> bool:
    if not is_retryable_status(status_code):
        return False

    try:
        is_generic_server_error = int(status_code) == 500
    except (TypeError, ValueError):
        return False
    if not is_generic_server_error:
        return True

    normalized_message = str(message or "").lower()
    return not any(
        marker in normalized_message for marker in NON_RETRYABLE_BUSINESS_ERROR_MARKERS
    )


def schedule_retry(
    logger,
    *,
    max_retries: int,
    backoff: float,
    retry_count: int,
    operation: str,
    reason: str,
) -> bool:
    if retry_count >= max_retries:
        return False

    delay = retry_delay(backoff, retry_count)
    logger.warning(
        "Retrying %s (%d/%d) in %.2f seconds: %s",
        operation,
        retry_count + 1,
        max_retries,
        delay,
        reason,
    )
    if delay:
        time.sleep(delay)
    return True


def transient_upstream_error(message: str) -> ProxyError:
    return ProxyError(
        message,
        error_type="upstream_error",
        code=TRANSIENT_UPSTREAM_ERROR_CODE,
        status=502,
    )
