import json
import logging
from typing import ClassVar
from unittest.mock import patch

import pytest

from genai_proxy.errors import ProxyError
from genai_proxy.services.genai import (
    GENAI_HISTORY_DELETE_URL,
    GENAI_HISTORY_LIST_URL,
    GenAIService,
)


class FakeTokenManager:
    token = "token"
    billing_user_id = "42"

    def refresh_after_auth_failure(self, *_args, **_kwargs):
        return False


class FakeModelManager:
    records: ClassVar = {
        "kimi-k3": {
            "aiType": "kimi-k3",
            "aiName": "Kimi-K3",
            "rootAiType": "xinference",
            "rootModelName": "Xinference",
        },
        "chatglm": {
            "aiType": "chatglm",
            "aiName": "GLM-5.2",
            "rootAiType": "xinference",
            "rootModelName": "Xinference",
        },
    }

    def resolve_model(self, model):
        return model

    def get_model_record(self, model):
        return self.records.get(model)

    def root_ai_type_for(self, model):
        return self.records[model]["rootAiType"]


class FakeStreamResponse:
    status_code = 200
    text = ""

    def __init__(self, events):
        self.events = events
        self.closed = False

    def iter_lines(self, *args, **kwargs):
        for event in self.events:
            yield json.dumps(event).encode()

    def close(self):
        self.closed = True


class FakeJsonResponse:
    text = ""

    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


def history_response(records, *, pages=None):
    result = {"records": records}
    if pages is not None:
        result["pages"] = pages
    return FakeJsonResponse(
        {
            "success": True,
            "code": 200,
            "result": result,
        }
    )


def delete_response():
    return FakeJsonResponse({"success": True, "code": 200})


def completion_events(*, error=False):
    return [
        {
            "id": "upstream-completion-id",
            "choices": [
                {
                    "delta": {"content": "answer"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "upstream-completion-id",
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "error" if error else "stop",
                }
            ],
        },
    ]


def service(*, cleanup=True):
    return GenAIService(
        logging.getLogger("test_kimi_history_cleanup"),
        FakeTokenManager(),
        FakeModelManager(),
        max_retries=0,
        cleanup_kimi_history=cleanup,
    )


def request(model="kimi-k3", content="same question"):
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
    }


def test_kimi_deletes_only_the_new_history_group_before_terminal_chunk():
    old_record = {
        "chatGroupId": "old-group",
        "question": "same question",
    }
    new_record = {
        "chatGroupId": "new-group",
        "question": "same question",
    }
    history_reads = iter(
        [
            history_response([old_record]),
            history_response([new_record, old_record]),
        ]
    )
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if url == GENAI_HISTORY_LIST_URL:
            return next(history_reads)
        assert url == GENAI_HISTORY_DELETE_URL
        return delete_response()

    upstream = FakeStreamResponse(completion_events())
    with (
        patch(
            "genai_proxy.services.genai.requests.post",
            return_value=upstream,
        ) as post,
        patch("genai_proxy.services.genai.requests.get", side_effect=fake_get),
    ):
        stream = service().stream_openai_completion(request())
        content_chunk = next(stream)
        terminal_chunk = next(stream)
        remainder = list(stream)

    assert '"content": "answer"' in content_chunk
    assert '"finish_reason": "stop"' in terminal_chunk
    assert remainder == ["data: [DONE]\n\n"]
    assert upstream.closed
    assert "chatGroupId" not in post.call_args.kwargs["json"]
    assert [url for url, _ in calls] == [
        GENAI_HISTORY_LIST_URL,
        GENAI_HISTORY_LIST_URL,
        GENAI_HISTORY_DELETE_URL,
    ]
    assert calls[0][1]["params"]["question"] == ""
    assert "same question" not in calls[0][1]["params"].values()
    assert calls[-1][1]["params"]["id"] == "new-group"


def test_non_kimi_never_reads_or_deletes_history():
    upstream = FakeStreamResponse(completion_events())
    with (
        patch("genai_proxy.services.genai.requests.post", return_value=upstream),
        patch("genai_proxy.services.genai.requests.get") as get,
    ):
        chunks = list(
            service().stream_openai_completion(
                request(model="chatglm"),
            )
        )

    assert any('"finish_reason": "stop"' in chunk for chunk in chunks)
    get.assert_not_called()


def test_kimi_cleanup_can_be_disabled_without_changing_transport():
    upstream = FakeStreamResponse(completion_events())
    with (
        patch(
            "genai_proxy.services.genai.requests.post",
            return_value=upstream,
        ) as post,
        patch("genai_proxy.services.genai.requests.get") as get,
    ):
        chunks = list(service(cleanup=False).stream_openai_completion(request()))

    assert any('"finish_reason": "stop"' in chunk for chunk in chunks)
    assert post.call_args.kwargs["json"]["chatInfo"] == "same question"
    assert "chatGroupId" not in post.call_args.kwargs["json"]
    get.assert_not_called()


def test_kimi_does_not_delete_before_successful_completion():
    upstream = FakeStreamResponse(completion_events(error=True))
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return history_response([])

    with (
        patch("genai_proxy.services.genai.requests.post", return_value=upstream),
        patch("genai_proxy.services.genai.requests.get", side_effect=fake_get),
    ):
        chunks = list(service().stream_openai_completion(request()))

    assert any('"finish_reason": "error"' in chunk for chunk in chunks)
    assert [url for url, _ in calls] == [GENAI_HISTORY_LIST_URL]


def test_kimi_client_disconnect_releases_lock_without_deleting():
    upstream = FakeStreamResponse(completion_events())
    instance = service()
    with (
        patch("genai_proxy.services.genai.requests.post", return_value=upstream),
        patch(
            "genai_proxy.services.genai.requests.get",
            return_value=history_response([]),
        ) as get,
    ):
        stream = instance.stream_openai_completion(request())
        next(stream)
        stream.close()

    assert upstream.closed
    assert instance._kimi_history_locks == {}
    assert get.call_count == 1


def test_kimi_waits_for_delayed_history_visibility():
    new_record = {
        "chatGroupId": "new-group",
        "question": "same question",
    }
    history_reads = iter(
        [
            history_response([]),
            history_response([]),
            history_response([new_record]),
        ]
    )
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if url == GENAI_HISTORY_LIST_URL:
            return next(history_reads)
        return delete_response()

    with (
        patch(
            "genai_proxy.services.genai.requests.post",
            return_value=FakeStreamResponse(completion_events()),
        ),
        patch("genai_proxy.services.genai.requests.get", side_effect=fake_get),
        patch("genai_proxy.services.genai.time.sleep") as sleep,
    ):
        chunks = list(service().stream_openai_completion(request()))

    assert any('"finish_reason": "stop"' in chunk for chunk in chunks)
    sleep.assert_called_once()
    assert calls[-1][0] == GENAI_HISTORY_DELETE_URL
    assert calls[-1][1]["params"]["id"] == "new-group"


def test_kimi_scans_all_history_pages_before_deleting_new_group():
    old_first = {"chatGroupId": "old-first", "question": "same question"}
    old_second = {"chatGroupId": "old-second", "question": "same question"}
    new_record = {"chatGroupId": "new-group", "question": "same question"}
    history_reads = iter(
        [
            history_response([old_first], pages=2),
            history_response([old_second], pages=2),
            history_response([new_record, old_first], pages=2),
            history_response([old_second], pages=2),
        ]
    )
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if url == GENAI_HISTORY_LIST_URL:
            return next(history_reads)
        assert url == GENAI_HISTORY_DELETE_URL
        return delete_response()

    with (
        patch(
            "genai_proxy.services.genai.requests.post",
            return_value=FakeStreamResponse(completion_events()),
        ),
        patch("genai_proxy.services.genai.requests.get", side_effect=fake_get),
    ):
        chunks = list(service().stream_openai_completion(request()))

    assert any('"finish_reason": "stop"' in chunk for chunk in chunks)
    list_calls = [kwargs for url, kwargs in calls if url == GENAI_HISTORY_LIST_URL]
    assert [call["params"]["pageNo"] for call in list_calls] == [1, 2, 1, 2]
    assert all(call["params"]["pageSize"] == 200 for call in list_calls)
    assert calls[-1][0] == GENAI_HISTORY_DELETE_URL
    assert calls[-1][1]["params"]["id"] == "new-group"


def test_kimi_skips_ambiguous_history_deletion():
    records = [
        {"chatGroupId": "new-1", "question": "same question"},
        {"chatGroupId": "new-2", "question": "same question"},
    ]
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            return history_response([])
        return history_response(records)

    with (
        patch(
            "genai_proxy.services.genai.requests.post",
            return_value=FakeStreamResponse(completion_events()),
        ),
        patch("genai_proxy.services.genai.requests.get", side_effect=fake_get),
    ):
        chunks = list(service().stream_openai_completion(request()))

    assert any('"finish_reason": "stop"' in chunk for chunk in chunks)
    assert [url for url, _ in calls] == [
        GENAI_HISTORY_LIST_URL,
        GENAI_HISTORY_LIST_URL,
    ]


def test_kimi_delete_failure_does_not_corrupt_completed_response():
    new_record = {
        "chatGroupId": "new-group",
        "question": "same question",
    }
    history_reads = iter(
        [
            history_response([]),
            history_response([new_record]),
        ]
    )

    def fake_get(url, **_kwargs):
        if url == GENAI_HISTORY_LIST_URL:
            return next(history_reads)
        return FakeJsonResponse(
            {"success": False, "code": 400, "message": "delete failed"},
        )

    with (
        patch(
            "genai_proxy.services.genai.requests.post",
            return_value=FakeStreamResponse(completion_events()),
        ),
        patch("genai_proxy.services.genai.requests.get", side_effect=fake_get),
    ):
        chunks = list(service().stream_openai_completion(request()))

    assert any('"finish_reason": "stop"' in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


def test_kimi_snapshot_failure_does_not_attempt_unsafe_deletion():
    with (
        patch(
            "genai_proxy.services.genai.requests.post",
            return_value=FakeStreamResponse(completion_events()),
        ),
        patch(
            "genai_proxy.services.genai.requests.get",
            return_value=FakeJsonResponse({}, status_code=400),
        ) as get,
    ):
        chunks = list(service().stream_openai_completion(request()))

    assert any('"finish_reason": "stop"' in chunk for chunk in chunks)
    assert get.call_count == 1
    assert get.call_args.args[0] == GENAI_HISTORY_LIST_URL


def test_kimi_malformed_snapshot_records_never_delete_an_old_matching_group():
    old_record = {
        "chatGroupId": "old-group",
        "question": "same question",
    }
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if url == GENAI_HISTORY_LIST_URL:
            return FakeJsonResponse(
                {
                    "success": True,
                    "code": 200,
                    "result": {"records": {"old": old_record}},
                }
            )
        raise AssertionError("history deletion must stay disabled")

    with (
        patch(
            "genai_proxy.services.genai.requests.post",
            return_value=FakeStreamResponse(completion_events()),
        ),
        patch("genai_proxy.services.genai.requests.get", side_effect=fake_get),
    ):
        chunks = list(service().stream_openai_completion(request()))

    assert any('"finish_reason": "stop"' in chunk for chunk in chunks)
    assert [url for url, _ in calls] == [GENAI_HISTORY_LIST_URL]


def test_kimi_ignores_malformed_history_entries_and_response_codes():
    new_record = {
        "chatGroupId": "new-group",
        "question": "same question",
    }
    history_reads = iter(
        [
            history_response([]),
            FakeJsonResponse(
                {
                    "success": True,
                    "code": "200",
                    "result": {"records": [None, "invalid", new_record]},
                }
            ),
        ]
    )
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if url == GENAI_HISTORY_LIST_URL:
            return next(history_reads)
        return delete_response()

    with (
        patch(
            "genai_proxy.services.genai.requests.post",
            return_value=FakeStreamResponse(completion_events()),
        ),
        patch("genai_proxy.services.genai.requests.get", side_effect=fake_get),
    ):
        chunks = list(service().stream_openai_completion(request()))

    assert any('"finish_reason": "stop"' in chunk for chunk in chunks)
    assert calls[-1][0] == GENAI_HISTORY_DELETE_URL
    assert calls[-1][1]["params"]["id"] == "new-group"


def test_kimi_malformed_history_payload_is_nonfatal_and_not_deleted():
    with (
        patch(
            "genai_proxy.services.genai.requests.post",
            return_value=FakeStreamResponse(completion_events()),
        ),
        patch(
            "genai_proxy.services.genai.requests.get",
            return_value=FakeJsonResponse([]),
        ) as get,
    ):
        chunks = list(service().stream_openai_completion(request()))

    assert any('"finish_reason": "stop"' in chunk for chunk in chunks)
    assert get.call_count == 1


def test_kimi_history_string_auth_code_is_classified_for_token_refresh():
    instance = service()

    with pytest.raises(ProxyError) as raised:
        instance._decode_kimi_history_response(
            FakeJsonResponse(
                {
                    "success": False,
                    "code": "401",
                    "message": "expired",
                }
            ),
            "fetch Kimi K3 history",
        )

    assert raised.value.code == "upstream_auth_failed"
