import base64
import io
import logging
import signal
import time
from types import SimpleNamespace

import pytest
from PIL import Image

import genai_proxy.upstream.transport as upstream_transport
from genai_proxy.upstream.transport import (
    GENAI_HISTORY_DELETE_URL,
    GENAI_HISTORY_LIST_URL,
    GENAI_URL,
)
from tests.live import allowed_models as live_models
from tests.live.allowed_models import (
    ALLOWED_MODELS,
    VISION_IMAGE_SIZE,
    VISION_RGB,
    LiveCaseTimeout,
    LiveTransportAudit,
    _deepseek_max_prefixes,
    assert_optional_responses_reasoning_preserved,
    assert_red_vision_result,
    live_case_deadline,
    quiet_integration_logger,
    red_image_url,
)


@pytest.mark.parametrize(
    "text",
    (
        "Red",
        "The dominant color is crimson.",
        "scarlet",
        "#ff0000",
        "RGB(255, 0, 0)",
        "rgba(255, 0, 0, 1.0)",
        "It is red, not green.",
    ),
)
def test_live_vision_result_accepts_unambiguous_red_answers(text):
    assert_red_vision_result(
        text,
        {"input_tokens": 10, "output_tokens": 2},
        input_token_key="input_tokens",
        output_token_key="output_tokens",
    )


@pytest.mark.parametrize(
    ("text", "expected_error"),
    (
        ("", "has no final text"),
        ("green", "did not identify red"),
        ("pink", "did not identify red"),
        ("The image is not red.", "negative or uncertain"),
        ("It is not #ff0000.", "negative or uncertain"),
        ("It is not RGB(255, 0, 0).", "negative or uncertain"),
        ("I cannot tell whether it is red.", "negative or uncertain"),
    ),
)
def test_live_vision_result_rejects_empty_wrong_or_uncertain_answers(
    text,
    expected_error,
):
    with pytest.raises(AssertionError, match=expected_error):
        assert_red_vision_result(
            text,
            {"input_tokens": 10, "output_tokens": 2},
            input_token_key="input_tokens",
            output_token_key="output_tokens",
        )


@pytest.mark.parametrize(
    ("usage", "expected_error"),
    (
        ({"input_tokens": 0, "output_tokens": 2}, "invalid input token usage"),
        ({"input_tokens": 10, "output_tokens": 0}, "invalid output token usage"),
        ({"input_tokens": "10", "output_tokens": 2}, "invalid input token usage"),
        ({"input_tokens": 10, "output_tokens": True}, "invalid output token usage"),
    ),
)
def test_live_vision_result_rejects_invalid_usage_without_leaking_other_fields(
    usage,
    expected_error,
):
    usage = {**usage, "secret": "must-not-leak"}

    with pytest.raises(AssertionError, match=expected_error) as exc_info:
        assert_red_vision_result(
            "red",
            usage,
            input_token_key="input_tokens",
            output_token_key="output_tokens",
        )

    assert "must-not-leak" not in str(exc_info.value)


def test_live_red_image_uses_a_standard_size_and_exact_rgb_value():
    prefix, encoded = red_image_url().split(",", 1)

    assert prefix == "data:image/png;base64"
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.mode == "RGB"
        assert image.size == VISION_IMAGE_SIZE
        assert image.getextrema() == tuple((value, value) for value in VISION_RGB)


def test_live_model_matrix_covers_protocols_capabilities_and_continuations():
    common = {
        "openai_text",
        "openai_stream_text",
        "responses_text",
        "responses_stream_text",
        "openai_tool_call",
        "openai_stream_tool_call",
        "openai_tool_result_turn",
        "responses_multiturn_tool_call",
        "claude_text",
        "claude_tool_use",
        "claude_stream_tool_use",
        "claude_tool_result_turn",
    }
    non_kimi = {
        "openai_bash_tool_call",
        "openai_stream_bash_tool_call",
        "openai_no_tool_needed",
        "claude_bash_tool_use",
        "claude_stream_bash_tool_use",
    }
    reasoning = {
        "openai_stream_reasoning",
        "responses_stream_reasoning",
        "claude_stream_reasoning",
    }
    vision = {"openai_vision", "responses_vision", "claude_vision"}

    for model in ALLOWED_MODELS:
        names = [name for name, _function in live_models.tests_for_model(model)]
        assert len(names) == len(set(names)), f"duplicate live tests for {model}"

        expected = set(common)
        if model != "kimi-k3":
            expected.update(non_kimi)
            expected.update(reasoning)
        if model in {"qwen-instruct", "kimi-k3"}:
            expected.update(vision)
        else:
            expected.add("nonvisual_vision_rejection")
        if model in {"deepseek-chat", "deepseek-pro"}:
            expected.add("deepseek_thinking_modes")

        assert set(names) == expected


def test_live_runner_uses_a_quiet_non_propagating_logger():
    logger = quiet_integration_logger()

    assert logger.propagate is False
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.NullHandler)


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires setitimer")
def test_live_case_deadline_interrupts_and_restores_the_alarm_handler():
    previous_handler = signal.getsignal(signal.SIGALRM)

    with pytest.raises(LiveCaseTimeout, match="exceeded"):
        with live_case_deadline(0.01):
            time.sleep(0.1)

    assert signal.getsignal(signal.SIGALRM) is previous_handler
    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    assert remaining == 0
    assert interval == 0


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires setitimer")
def test_live_case_deadline_preserves_an_outer_deadline_without_extending_it():
    with live_case_deadline(0.5):
        outer_handler = signal.getsignal(signal.SIGALRM)
        outer_remaining, outer_interval = signal.getitimer(signal.ITIMER_REAL)

        with pytest.raises(LiveCaseTimeout, match="exceeded"):
            with live_case_deadline(0.03):
                time.sleep(0.2)

        restored_remaining, restored_interval = signal.getitimer(signal.ITIMER_REAL)
        assert signal.getsignal(signal.SIGALRM) is outer_handler
        assert restored_remaining < outer_remaining - 0.015
        assert restored_interval == outer_interval


def test_optional_responses_reasoning_must_match_streamed_deltas():
    events = [
        {"type": "response.reasoning_text.delta", "delta": "think "},
        {"type": "response.reasoning_text.delta", "delta": "again"},
    ]
    completed = {
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "think again"}],
            }
        ]
    }

    assert_optional_responses_reasoning_preserved(events, completed)
    assert_optional_responses_reasoning_preserved([], {"output": []})

    completed["output"][0]["content"][0]["text"] = "different"
    with pytest.raises(AssertionError):
        assert_optional_responses_reasoning_preserved(events, completed)


def test_live_audit_records_sanitized_transport_metadata():
    upstream = object()
    audit = LiveTransportAudit()
    audit._original_post = lambda *_args, **_kwargs: upstream
    prefix = next(prefix for prefix in _deepseek_max_prefixes() if prefix)

    result = audit._post(
        GENAI_URL,
        json={
            "aiType": "deepseek-pro",
            "chatInfo": "",
            "messages": [{"role": "system", "content": prefix + "instructions"}],
            "stream": True,
            "thinking": True,
        },
    )

    assert result is upstream
    assert audit.chat_requests_since((0, 0, 0)) == [
        {
            "model": "deepseek-pro",
            "thinking_present": True,
            "thinking": True,
            "has_max_prefix": True,
            "image_count": 0,
            "image_url_present": False,
            "image_urls_present": False,
            "image_width": None,
            "image_height": None,
        }
    ]


@pytest.mark.parametrize("model", ("qwen-instruct", "kimi-k3"))
def test_live_audit_records_only_safe_image_metadata(model):
    upstream = object()
    audit = LiveTransportAudit()
    audit._original_post = lambda *_args, **_kwargs: upstream
    checkpoint = audit.checkpoint()
    secret_image = "data:image/png;base64,MUST_NOT_BE_RECORDED"
    payload = {
        "aiType": model,
        "chatInfo": "identify the image" if model == "kimi-k3" else "",
        "messages": [],
        "stream": True,
    }
    if model == "kimi-k3":
        payload.update(
            {
                "imageUrl": secret_image,
                "imageUrls": [secret_image],
                "width": VISION_IMAGE_SIZE[0],
                "height": VISION_IMAGE_SIZE[1],
            }
        )
    else:
        payload["messages"] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": secret_image},
                    }
                ],
            }
        ]

    assert audit._post(GENAI_URL, json=payload) is upstream
    audit.assert_vision_transport(model, checkpoint)

    observations = audit.chat_requests_since(checkpoint)
    assert "MUST_NOT_BE_RECORDED" not in repr(observations)
    assert observations[0]["image_count"] == 1
    if model == "kimi-k3":
        assert (
            observations[0]["image_width"],
            observations[0]["image_height"],
        ) == VISION_IMAGE_SIZE


def test_live_audit_rejects_empty_kimi_image_fields():
    audit = LiveTransportAudit()
    audit._original_post = lambda *_args, **_kwargs: object()
    checkpoint = audit.checkpoint()

    audit._post(
        GENAI_URL,
        json={
            "aiType": "kimi-k3",
            "chatInfo": "identify the image",
            "messages": [],
            "stream": True,
            "imageUrl": "",
            "imageUrls": [None],
            "width": VISION_IMAGE_SIZE[0],
            "height": VISION_IMAGE_SIZE[1],
        },
    )

    with pytest.raises(AssertionError, match="exactly one image"):
        audit.assert_vision_transport("kimi-k3", checkpoint)


def test_live_audit_installs_and_restores_request_wrappers():
    original_post = upstream_transport.requests.post
    original_get = upstream_transport.requests.get
    post_response = object()
    get_response = SimpleNamespace(
        status_code=200,
        json=lambda: {"code": 200, "success": True},
    )
    audit = LiveTransportAudit()
    audit._original_post = lambda *_args, **_kwargs: post_response
    audit._original_get = lambda *_args, **_kwargs: get_response

    with audit.installed():
        assert (
            upstream_transport.requests.post(
                GENAI_URL,
                json={
                    "aiType": "chatglm",
                    "chatInfo": "",
                    "messages": [],
                    "stream": True,
                },
            )
            is post_response
        )
        assert upstream_transport.requests.get(GENAI_HISTORY_LIST_URL) is get_response

    assert upstream_transport.requests.post is original_post
    assert upstream_transport.requests.get is original_get


@pytest.mark.parametrize(
    "payload, expected_message",
    (
        (
            {
                "aiType": "kimi-k3",
                "chatInfo": "question",
                "messages": [],
                "stream": True,
                "chatGroupId": "forbidden",
            },
            "chatGroupId",
        ),
        (
            {
                "aiType": "kimi-k3",
                "chatInfo": "",
                "messages": [],
                "stream": True,
            },
            "empty chatInfo",
        ),
        (
            {
                "aiType": "chatglm",
                "chatInfo": "",
                "messages": [],
                "stream": False,
            },
            "upstream SSE",
        ),
    ),
)
def test_live_audit_rejects_transport_contract_violations(payload, expected_message):
    audit = LiveTransportAudit()
    audit._original_post = lambda *_args, **_kwargs: pytest.fail(
        "invalid request reached the network"
    )

    with pytest.raises(AssertionError, match=expected_message):
        audit._post(GENAI_URL, json=payload)


def test_live_audit_requires_kimi_history_lookup_and_successful_delete():
    success = SimpleNamespace(
        status_code=200,
        json=lambda: {"code": 200, "success": True},
    )
    audit = LiveTransportAudit()
    audit._original_get = lambda *_args, **_kwargs: success
    checkpoint = audit.checkpoint()

    audit._get(GENAI_HISTORY_LIST_URL)
    audit._get(GENAI_HISTORY_LIST_URL)
    audit._get(GENAI_HISTORY_DELETE_URL)

    audit.assert_model_side_effects("kimi-k3", checkpoint)
    assert audit.summary() == {
        "chat_requests": 0,
        "history_list_requests": 2,
        "successful_history_deletes": 1,
    }


def test_live_audit_does_not_count_failed_history_deletion():
    failed = SimpleNamespace(
        status_code=200,
        json=lambda: {"code": 500, "success": False},
    )
    audit = LiveTransportAudit()
    audit._original_get = lambda *_args, **_kwargs: failed

    audit._get(GENAI_HISTORY_DELETE_URL)

    assert audit.summary()["successful_history_deletes"] == 0
