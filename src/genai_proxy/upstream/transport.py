"""Low-level HTTP and SSE operations for the ShanghaiTech GenAI service."""

import json
import time

import requests

GENAI_URL = "https://genai.shanghaitech.edu.cn/htk/chat/start/chat"
GENAI_USER_INFO_URL = "https://genai.shanghaitech.edu.cn/htk/ai-user-info/list"
GENAI_CURRENT_USER_URL = "https://genai.shanghaitech.edu.cn/htk/user/info/{token}"
GENAI_HISTORY_LIST_URL = (
    "https://genai.shanghaitech.edu.cn/htk/ai/history/listByContentGroup"
)
GENAI_HISTORY_DELETE_URL = (
    "https://genai.shanghaitech.edu.cn/htk/ai/history/delete/groupId"
)
GENAI_STREAM_TIMEOUT = (10, 90)
GENAI_DEEPSEEK_STREAM_TIMEOUT = (10, 60)
GENAI_HISTORY_TIMEOUT = (5, 15)

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


def genai_headers(token: str | None) -> dict[str, str | None]:
    headers = dict(GENAI_BASE_HEADERS)
    headers["X-Access-Token"] = token
    return headers


def user_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "X-Access-Token": token,
    }


def post_chat(
    token: str | None,
    payload: dict,
    *,
    timeout=GENAI_STREAM_TIMEOUT,
):
    return requests.post(
        GENAI_URL,
        headers=genai_headers(token),
        json=payload,
        stream=True,
        timeout=timeout,
    )


def fetch_user_info(token: str, user_id: str):
    return requests.get(
        GENAI_USER_INFO_URL,
        params={
            "_t": int(time.time()),
            "pageNo": 1,
            "pageSize": 1,
            "userId": user_id,
        },
        headers=user_headers(token),
        timeout=30,
    )


def fetch_current_user(token: str):
    return requests.get(
        GENAI_CURRENT_USER_URL.format(token=token),
        params={"_t": int(time.time() * 1000)},
        headers=user_headers(token),
        timeout=30,
    )


def fetch_history_page(
    token: str,
    user_id: str,
    page_number: int,
    page_size: int,
):
    return requests.get(
        GENAI_HISTORY_LIST_URL,
        params={
            "_t": int(time.time() * 1000),
            "pageNo": page_number,
            "pageSize": page_size,
            "userId": user_id,
            "question": "",
        },
        headers=user_headers(token),
        timeout=GENAI_HISTORY_TIMEOUT,
    )


def delete_history_group(token: str, group_id: str):
    return requests.get(
        GENAI_HISTORY_DELETE_URL,
        params={
            "_t": int(time.time() * 1000),
            "id": group_id,
        },
        headers=user_headers(token),
        timeout=GENAI_HISTORY_TIMEOUT,
    )


def iter_sse_json(response, logger):
    line_count = 0
    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
        if not line:
            continue

        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
        if line_count < 5:
            logger.debug("Raw line [%d]: %s", line_count, line_str[:300])
        line_count += 1

        if line_str.startswith("data:"):
            line_str = line_str[5:].strip()
        if not line_str:
            continue

        try:
            payload = json.loads(line_str)
        except json.JSONDecodeError as exc:
            logger.debug("JSON decode error: %s, line: %s", exc, line_str[:200])
            continue
        yield line_count, payload


__all__ = [
    "GENAI_BASE_HEADERS",
    "GENAI_CURRENT_USER_URL",
    "GENAI_HISTORY_DELETE_URL",
    "GENAI_HISTORY_LIST_URL",
    "GENAI_HISTORY_TIMEOUT",
    "GENAI_DEEPSEEK_STREAM_TIMEOUT",
    "GENAI_STREAM_TIMEOUT",
    "GENAI_URL",
    "GENAI_USER_INFO_URL",
    "delete_history_group",
    "fetch_current_user",
    "fetch_history_page",
    "fetch_user_info",
    "genai_headers",
    "iter_sse_json",
    "post_chat",
    "user_headers",
]
