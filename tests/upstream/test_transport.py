import logging
from unittest.mock import call, patch

from genai_proxy.services.models import ModelManager as LegacyModelManager
from genai_proxy.services.token_manager import TokenManager as LegacyTokenManager
from genai_proxy.upstream import transport
from genai_proxy.upstream.auth import TokenManager
from genai_proxy.upstream.catalog import ModelManager


class _LineResponse:
    def __init__(self, lines):
        self._lines = lines
        self.iter_kwargs = None

    def iter_lines(self, **kwargs):
        self.iter_kwargs = kwargs
        yield from self._lines


def test_chat_transport_preserves_payload_and_stream_contract():
    payload = {
        "chatInfo": "hello",
        "messages": [],
        "stream": True,
    }
    response = object()

    with patch.object(transport.requests, "post", return_value=response) as post:
        assert transport.post_chat("token", payload) is response

    post.assert_called_once_with(
        transport.GENAI_URL,
        headers=transport.genai_headers("token"),
        json=payload,
        stream=True,
        timeout=transport.GENAI_STREAM_TIMEOUT,
    )
    assert post.call_args.kwargs["json"] is payload
    assert "chatGroupId" not in post.call_args.kwargs["json"]


def test_chat_transport_accepts_an_explicit_model_timeout():
    payload = {
        "chatInfo": "hello",
        "messages": [],
        "stream": True,
    }
    response = object()

    with patch.object(transport.requests, "post", return_value=response) as post:
        assert (
            transport.post_chat(
                "token",
                payload,
                timeout=transport.GENAI_DEEPSEEK_STREAM_TIMEOUT,
            )
            is response
        )

    post.assert_called_once_with(
        transport.GENAI_URL,
        headers=transport.genai_headers("token"),
        json=payload,
        stream=True,
        timeout=transport.GENAI_DEEPSEEK_STREAM_TIMEOUT,
    )
    assert "chatGroupId" not in post.call_args.kwargs["json"]


def test_user_and_history_transports_keep_exact_query_contracts():
    responses = [object(), object(), object(), object()]
    with (
        patch.object(transport.time, "time", return_value=1234.5),
        patch.object(transport.requests, "get", side_effect=responses) as get,
    ):
        assert transport.fetch_user_info("token", "user") is responses[0]
        assert transport.fetch_current_user("token") is responses[1]
        assert transport.fetch_history_page("token", "user", 2, 200) is responses[2]
        assert transport.delete_history_group("token", "group") is responses[3]

    assert get.call_args_list == [
        call(
            transport.GENAI_USER_INFO_URL,
            params={"_t": 1234, "pageNo": 1, "pageSize": 1, "userId": "user"},
            headers=transport.user_headers("token"),
            timeout=30,
        ),
        call(
            transport.GENAI_CURRENT_USER_URL.format(token="token"),
            params={"_t": 1234500},
            headers=transport.user_headers("token"),
            timeout=30,
        ),
        call(
            transport.GENAI_HISTORY_LIST_URL,
            params={
                "_t": 1234500,
                "pageNo": 2,
                "pageSize": 200,
                "userId": "user",
                "question": "",
            },
            headers=transport.user_headers("token"),
            timeout=transport.GENAI_HISTORY_TIMEOUT,
        ),
        call(
            transport.GENAI_HISTORY_DELETE_URL,
            params={"_t": 1234500, "id": "group"},
            headers=transport.user_headers("token"),
            timeout=transport.GENAI_HISTORY_TIMEOUT,
        ),
    ]


def test_sse_transport_parses_incrementally_and_skips_invalid_frames(caplog):
    response = _LineResponse(
        [
            b"",
            b"data: {not-json}",
            b'data: {"choices": [{"delta": {"content": "a"}}]}',
            'data: {"choices": [{"finish_reason": "stop"}]}',
        ]
    )

    with caplog.at_level(logging.DEBUG):
        parsed = list(transport.iter_sse_json(response, logging.getLogger(__name__)))

    assert response.iter_kwargs == {"chunk_size": 1, "decode_unicode": True}
    assert parsed == [
        (2, {"choices": [{"delta": {"content": "a"}}]}),
        (3, {"choices": [{"finish_reason": "stop"}]}),
    ]
    assert "JSON decode error" in caplog.text


def test_sse_transport_yields_each_frame_before_reading_the_next_one():
    class GuardedResponse:
        def iter_lines(self, **kwargs):
            assert kwargs == {"chunk_size": 1, "decode_unicode": True}
            yield b'data: {"choices": [{"delta": {"content": "first"}}]}'
            raise AssertionError("transport read past the first complete frame")

    stream = transport.iter_sse_json(GuardedResponse(), logging.getLogger(__name__))
    assert next(stream) == (
        1,
        {"choices": [{"delta": {"content": "first"}}]},
    )
    stream.close()


def test_legacy_upstream_imports_preserve_class_identity():
    assert LegacyModelManager is ModelManager
    assert LegacyTokenManager is TokenManager
