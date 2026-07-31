import logging
import signal
import time
from types import SimpleNamespace

import pytest

import genai_proxy.services.genai as genai_module
from genai_proxy.services.genai import (
    GENAI_HISTORY_DELETE_URL,
    GENAI_HISTORY_LIST_URL,
    GENAI_URL,
)
from test_allowed_models_integration import (
    ALLOWED_MODELS,
    LiveTransportAudit,
    LiveCaseTimeout,
    _deepseek_max_prefixes,
    live_case_deadline,
    quiet_integration_logger,
    tests_for_model as _tests_for_model,
)


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
        names = [name for name, _function in _tests_for_model(model)]
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

        restored_remaining, restored_interval = signal.getitimer(
            signal.ITIMER_REAL
        )
        assert signal.getsignal(signal.SIGALRM) is outer_handler
        assert restored_remaining < outer_remaining - 0.015
        assert restored_interval == outer_interval


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
        }
    ]


def test_live_audit_installs_and_restores_request_wrappers():
    original_post = genai_module.requests.post
    original_get = genai_module.requests.get
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
            genai_module.requests.post(
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
        assert genai_module.requests.get(GENAI_HISTORY_LIST_URL) is get_response

    assert genai_module.requests.post is original_post
    assert genai_module.requests.get is original_get


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
