import time

from genai_proxy.errors import ProxyError

DEFAULT_MAX_RETRIES = 10
DEFAULT_RETRY_BACKOFF = 0.5
MAX_RETRY_DELAY = 5.0
OPAQUE_BUSINESS_ERROR_MAX_RETRIES = 1

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
TRANSIENT_BUSINESS_ERROR_MARKERS = (
    "temporary",
    "temporarily",
    "transient",
    "try again",
    "please retry",
    "retry later",
    "retryable",
    "busy",
    "overloaded",
    "timeout",
    "timed out",
    "service unavailable",
    "gateway timeout",
    "暂时",
    "临时",
    "稍后重试",
    "请重试",
    "繁忙",
    "超时",
    "服务不可用",
)


def retry_delay(backoff: float, retry_count: int) -> float:
    return min(max(0.0, float(backoff)) * (2**retry_count), MAX_RETRY_DELAY)


def is_retryable_status(status_code) -> bool:
    try:
        return int(status_code) in RETRYABLE_STATUS_CODES
    except (TypeError, ValueError):
        return False


def is_retryable_business_error(status_code, message: str = "") -> bool:
    return business_error_retry_limit(status_code, message, 1) > 0


def business_error_retry_limit(
    status_code,
    message: str,
    configured_max_retries: int,
) -> int:
    retry_limit = max(0, int(configured_max_retries))
    if retry_limit == 0 or not is_retryable_status(status_code):
        return 0

    try:
        is_generic_server_error = int(status_code) == 500
    except (TypeError, ValueError):
        return 0
    if not is_generic_server_error:
        return retry_limit

    normalized_message = str(message or "").lower()
    if any(
        marker in normalized_message for marker in NON_RETRYABLE_BUSINESS_ERROR_MARKERS
    ):
        return 0
    if any(marker in normalized_message for marker in TRANSIENT_BUSINESS_ERROR_MARKERS):
        return retry_limit
    return min(retry_limit, OPAQUE_BUSINESS_ERROR_MAX_RETRIES)


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
