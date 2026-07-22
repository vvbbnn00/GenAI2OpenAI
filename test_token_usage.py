import hashlib
import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import requests

from genai_proxy.app import create_app
from genai_proxy.optimizations.deepseek import inject_deepseek_reasoning_prompt
from genai_proxy.services.genai import GenAIService
from genai_proxy.token_usage import (
    Artifact,
    QWEN_3_5_SPEC,
    TokenizerSpec,
    _artifact_path,
    _count_encoded,
    _serialized_completion,
    count_openai_completion_tokens,
    count_openai_reasoning_tokens,
    count_openai_request_tokens,
    render_chat_prompt,
    tokenizer_family_for_model,
)


class FakeTokenManager:
    token = "token"
    billing_user_id = None


class FakeModelManager:
    records = {
        "chatglm": {
            "aiType": "chatglm",
            "aiName": "GLM-5.2",
            "rootModelName": "Xinference",
        },
        "deepseek-chat": {
            "aiType": "deepseek-chat",
            "aiName": "DeepSeek-V4-Flash",
            "rootModelName": "Xinference",
        },
        "deepseek-pro": {
            "aiType": "deepseek-pro",
            "aiName": "DeepSeek-V4-Pro",
            "rootModelName": "Xinference",
        },
        "qwen3.5": {
            "aiType": "qwen3.5",
            "aiName": "Qwen3.5-397B-A17B",
            "rootModelName": "Xinference",
        },
    }

    def resolve_model(self, model):
        return model

    def get_model_record(self, model):
        return self.records.get(model)

    def list_genai_models(self):
        return list(self.records.values())

    def root_ai_type_for(self, _model):
        return "xinference"


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, lines):
        self.lines = lines

    def iter_lines(self):
        return iter(self.lines)

    def close(self):
        pass


def _app():
    logger = logging.getLogger("test_token_usage")
    model_manager = FakeModelManager()
    service = GenAIService(logger, FakeTokenManager(), model_manager)
    app = create_app(
        SimpleNamespace(
            token=None,
            keystore=None,
            token_check_interval=0,
            api_key=None,
            claude_haiku_model="deepseek-chat",
            claude_sonnet_model="chatglm",
            claude_opus_model="chatglm",
        ),
        logger,
    )
    app.extensions["model_manager"] = model_manager
    app.extensions["genai_service"] = service
    return app


def test_qwen_uses_full_model_as_revision_authority():
    assert QWEN_3_5_SPEC.repository == "Qwen/Qwen3.5-397B-A17B"
    assert QWEN_3_5_SPEC.revision == "8472618112abcbd45acbcdc58436aff4233c23f7"


def test_supported_model_families_are_resolved_from_alias_and_record():
    manager = FakeModelManager()
    assert (
        tokenizer_family_for_model("chatglm", manager.get_model_record("chatglm"))
        == "glm_5_2"
    )
    assert (
        tokenizer_family_for_model(
            "deepseek-chat", manager.get_model_record("deepseek-chat")
        )
        == "deepseek_v4_flash"
    )
    assert (
        tokenizer_family_for_model(
            "deepseek-pro", manager.get_model_record("deepseek-pro")
        )
        == "deepseek_v4_pro"
    )
    assert (
        tokenizer_family_for_model("qwen3.5", manager.get_model_record("qwen3.5"))
        == "qwen_3_5"
    )


def test_model_family_version_matching_does_not_accept_longer_minor_version():
    assert tokenizer_family_for_model("qwen3.50") is None
    assert tokenizer_family_for_model("glm5.20") is None


def test_tokenizer_artifact_download_retries_transient_failure(tmp_path):
    content = b"tokenizer"
    digest = hashlib.sha256(content).hexdigest()
    spec = TokenizerSpec(
        family="test",
        repository="example/model",
        revision="revision",
        tokenizer=Artifact("tokenizer.json", digest),
    )
    attempts = 0

    def download(_url, _cache_dir, destination, _expected_sha256):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise requests.ConnectionError("connection reset")
        destination.write_bytes(content)

    with (
        patch.dict("os.environ", {"GENAI_TOKENIZER_CACHE": str(tmp_path)}),
        patch("genai_proxy.token_usage._download_artifact", side_effect=download),
        patch("genai_proxy.token_usage.schedule_retry", return_value=True) as retry,
    ):
        path = _artifact_path(spec, spec.tokenizer)

    assert path.read_bytes() == content
    assert attempts == 2
    retry.assert_called_once()


def test_official_prompt_templates_have_stable_reference_counts():
    messages = [{"role": "user", "content": "Hello, 世界"}]
    cases = [
        ("chatglm", "glm_5_2", "glm_5_2", 15, "[gMASK]<sop>"),
        (
            "deepseek-pro",
            "deepseek_v4_pro",
            "deepseek_v4_pro",
            8,
            "<｜begin▁of▁sentence｜>",
        ),
        (
            "deepseek-chat",
            "deepseek_v4_flash",
            "deepseek_v4_flash",
            8,
            "<｜begin▁of▁sentence｜>",
        ),
        ("qwen3.5", "qwen_3_5", None, 14, "<|im_start|>user\n"),
    ]
    for model, family, adapter, expected_count, prompt_prefix in cases:
        prompt = render_chat_prompt(messages, family, add_generation_prompt=True)
        assert prompt.startswith(prompt_prefix)
        assert (
            count_openai_request_tokens(messages, model, tool_adapter=adapter)
            == expected_count
        )


def test_deepseek_max_injection_matches_official_encoder_prompt_exactly():
    messages = [{"role": "user", "content": "Hello"}]
    official_prompt = render_chat_prompt(
        messages,
        "deepseek_v4_pro",
        add_generation_prompt=True,
        reasoning_config={"effort": "max"},
    )
    injected_messages = inject_deepseek_reasoning_prompt(
        messages,
        {"effort": "max"},
        adapter="deepseek_v4_pro",
    )
    transported_prompt = render_chat_prompt(
        injected_messages,
        "deepseek_v4_pro",
        add_generation_prompt=True,
    )
    assert transported_prompt == official_prompt


def test_openai_responses_input_tokens_route_uses_official_qwen_template():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/responses/input_tokens",
            json={"model": "qwen3.5", "input": "Hello, 世界"},
        )
    )
    assert response.status_code == 200
    assert response.get_json() == {
        "object": "response.input_tokens",
        "input_tokens": 14,
    }


def test_qwen_merges_openai_system_and_developer_messages_before_counting():
    service = _app().extensions["genai_service"]
    prepared = service._prepare_chat_request(
        {
            "model": "qwen3.5",
            "messages": [
                {"role": "user", "content": "Earlier"},
                {"role": "developer", "content": "Developer instruction"},
                {"role": "system", "content": "System instruction"},
                {"role": "user", "content": "Hello"},
            ],
        }
    )

    assert prepared.messages == [
        {
            "role": "system",
            "content": "Developer instruction\n\nSystem instruction",
        },
        {"role": "user", "content": "Earlier"},
        {"role": "user", "content": "Hello"},
    ]
    assert prepared.prompt_tokens == count_openai_request_tokens(
        prepared.messages,
        "qwen3.5",
    )


def test_responses_count_tokens_handles_instructions_plus_developer_input():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/responses/input_tokens",
            json={
                "model": "qwen3.5",
                "instructions": "System instruction",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": "Developer instruction",
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": "Hello",
                    },
                ],
            },
        )
    )
    assert response.status_code == 200
    assert response.get_json()["input_tokens"] > 0


def test_glm_maps_developer_role_to_supported_system_role():
    service = _app().extensions["genai_service"]
    prepared = service._prepare_chat_request(
        {
            "model": "chatglm",
            "messages": [
                {"role": "developer", "content": "Developer instruction"},
                {"role": "user", "content": "Hello"},
            ],
        }
    )
    assert prepared.messages[0] == {
        "role": "system",
        "content": "Developer instruction\n\nReasoning Effort: Max",
    }


def test_openai_input_token_route_covers_all_supported_model_families():
    client = _app().test_client()
    expected = {
        "chatglm": 20,
        "deepseek-pro": 5,
        "deepseek-chat": 5,
        "qwen3.5": 11,
    }
    for model, input_tokens in expected.items():
        response = client.post(
            "/v1/responses/input_tokens",
            json={"model": model, "input": "Hello"},
        )
        assert response.status_code == 200
        assert response.get_json()["input_tokens"] == input_tokens


def test_completion_serialization_matches_official_templates():
    user = {"role": "user", "content": "Hello"}
    assistant = {
        "role": "assistant",
        "reasoning_content": "think",
        "content": " answer",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": (
                        '{"city":"上海","active":true,"value":null,'
                        '"items":[1],"metadata":{"unit":"c"}}'
                    ),
                },
            }
        ],
    }
    for family in ("glm_5_2", "deepseek_v4_pro", "qwen_3_5"):
        generation_prompt = render_chat_prompt(
            [user], family, add_generation_prompt=True
        )
        completed_prompt = render_chat_prompt(
            [user, assistant], family, add_generation_prompt=False
        )
        assert completed_prompt == generation_prompt + _serialized_completion(
            assistant, family
        )


def test_completion_serialization_matches_official_templates_for_sparse_outputs():
    user = {"role": "user", "content": "Hello"}
    assistants = (
        {"role": "assistant", "content": "answer"},
        {"role": "assistant", "reasoning_content": "think", "content": ""},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "ping", "arguments": "{}"},
                }
            ],
        },
    )
    for family in ("glm_5_2", "deepseek_v4_pro", "qwen_3_5"):
        generation_prompt = render_chat_prompt(
            [user], family, add_generation_prompt=True
        )
        for assistant in assistants:
            completed_prompt = render_chat_prompt(
                [user, assistant], family, add_generation_prompt=False
            )
            assert completed_prompt == generation_prompt + _serialized_completion(
                assistant, family
            )


def test_qwen_completion_count_uses_full_sequence_boundary():
    messages = [{"role": "user", "content": "Hello"}]
    assistant = {"role": "assistant", "content": "answer"}
    prompt = render_chat_prompt(messages, "qwen_3_5", add_generation_prompt=True)
    completed_prompt = render_chat_prompt(
        [*messages, assistant],
        "qwen_3_5",
        add_generation_prompt=False,
    )
    expected = _count_encoded("qwen_3_5", completed_prompt) - _count_encoded(
        "qwen_3_5", prompt
    )

    assert expected == 5
    assert (
        count_openai_completion_tokens(
            assistant,
            "qwen3.5",
            prompt_messages=messages,
        )
        == expected
    )


def test_qwen_reasoning_count_uses_prompt_boundary_and_template_trimming():
    messages = [{"role": "user", "content": "Hello"}]
    assert (
        count_openai_reasoning_tokens(
            "\nhello",
            "qwen3.5",
            prompt_messages=messages,
        )
        == 1
    )


def test_length_completion_does_not_count_unproduced_end_tokens():
    messages = [{"role": "user", "content": "Hello"}]
    assistant = {
        "role": "assistant",
        "reasoning_content": "think",
        "content": "answer",
    }
    cases = [
        ("deepseek-pro", "deepseek_v4_pro", "deepseek_v4_pro", 3),
        ("qwen3.5", "qwen_3_5", "generic", 5),
    ]
    for model, family, adapter, expected in cases:
        assert (
            count_openai_completion_tokens(
                assistant,
                model,
                tool_adapter=adapter,
                prompt_messages=messages,
                finish_reason="length",
            )
            == expected
        )
        assert not _serialized_completion(
            assistant,
            family,
            finish_reason="length",
        ).endswith(("<｜end▁of▁sentence｜>", "<|im_end|>\n"))


def test_length_during_reasoning_does_not_add_thinking_close_marker():
    messages = [{"role": "user", "content": "Hello"}]
    assistant = {"role": "assistant", "reasoning_content": "unfinished"}
    for model, family, adapter in (
        ("deepseek-pro", "deepseek_v4_pro", "deepseek_v4_pro"),
        ("qwen3.5", "qwen_3_5", "generic"),
    ):
        prompt = render_chat_prompt(messages, family, add_generation_prompt=True)
        expected = _count_encoded(family, prompt + "unfinished") - _count_encoded(
            family, prompt
        )
        actual = count_openai_completion_tokens(
            assistant,
            model,
            tool_adapter=adapter,
            prompt_messages=messages,
            finish_reason="length",
        )
        assert actual == expected
        assert "</think>" not in _serialized_completion(
            assistant,
            family,
            finish_reason="length",
        )


def test_anthropic_count_tokens_route_uses_mapped_official_glm_template():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/messages/count_tokens",
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello, 世界"}],
            },
        )
    )
    assert response.status_code == 200
    assert response.get_json() == {"input_tokens": 22}


def test_anthropic_count_token_route_covers_all_supported_model_families():
    client = _app().test_client()
    expected = {
        "chatglm": 20,
        "deepseek-pro": 5,
        "deepseek-chat": 5,
        "qwen3.5": 11,
    }
    for model, input_tokens in expected.items():
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 200
        assert response.get_json() == {"input_tokens": input_tokens}


def test_nonstream_usage_covers_all_supported_model_families():
    expected = {
        "chatglm": (20, 2),
        "deepseek-pro": (5, 3),
        "deepseek-chat": (5, 3),
        "qwen3.5": (11, 5),
    }
    for model, (prompt_tokens, completion_tokens) in expected.items():
        service = _app().extensions["genai_service"]
        upstream = FakeResponse(
            [
                json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {"content": "answer"},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ).encode()
            ]
        )
        with patch(
            "genai_proxy.services.genai.requests.post",
            return_value=upstream,
        ):
            response = service.build_openai_completion(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )

        assert response["usage"]["prompt_tokens"] == prompt_tokens
        assert response["usage"]["completion_tokens"] == completion_tokens
        assert response["usage"]["total_tokens"] == prompt_tokens + completion_tokens


def test_responses_usage_uses_exact_qwen_counts():
    service = _app().extensions["genai_service"]
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "answer"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]
    )
    with patch("genai_proxy.services.genai.requests.post", return_value=upstream):
        response = service.build_response({"model": "qwen3.5", "input": "Hello"})

    assert response["usage"] == {
        "input_tokens": 11,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 5,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 16,
    }


def test_anthropic_nonstream_usage_uses_exact_qwen_counts():
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "answer"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]
    )
    with patch("genai_proxy.services.genai.requests.post", return_value=upstream):
        response = (
            _app()
            .test_client()
            .post(
                "/v1/messages",
                json={
                    "model": "qwen3.5",
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        )

    assert response.status_code == 200
    assert response.get_json()["usage"] == {
        "input_tokens": 11,
        "output_tokens": 5,
    }


def test_anthropic_stream_usage_uses_exact_qwen_counts():
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "answer"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]
    )
    with patch("genai_proxy.services.genai.requests.post", return_value=upstream):
        response = (
            _app()
            .test_client()
            .post(
                "/v1/messages",
                json={
                    "model": "qwen3.5",
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )
        )
        body = response.get_data(as_text=True)

    events = []
    for block in body.split("\n\n"):
        data_line = next(
            (line for line in block.splitlines() if line.startswith("data: ")),
            None,
        )
        if data_line:
            events.append(json.loads(data_line[6:]))

    start = next(event for event in events if event["type"] == "message_start")
    delta = next(event for event in events if event["type"] == "message_delta")
    assert start["message"]["usage"] == {
        "input_tokens": 11,
        "output_tokens": 0,
    }
    assert delta["usage"] == {"input_tokens": 11, "output_tokens": 5}


def test_anthropic_count_tokens_rejects_non_object_json():
    response = _app().test_client().post("/v1/messages/count_tokens", json=[])
    assert response.status_code == 400
    assert (
        response.get_json()["error"]["message"] == "Request body must be a JSON object"
    )


def test_anthropic_messages_rejects_non_object_or_invalid_json():
    client = _app().test_client()
    responses = (
        client.post("/v1/messages", json=[]),
        client.post("/v1/messages", data="{", content_type="application/json"),
    )
    for response in responses:
        assert response.status_code == 400
        assert response.get_json()["error"]["message"] == (
            "Request body must be a JSON object"
        )


def test_token_count_routes_reject_invalid_json_as_client_error():
    client = _app().test_client()
    for path in ("/v1/messages/count_tokens", "/v1/responses/input_tokens"):
        response = client.post(path, data="{", content_type="application/json")
        assert response.status_code == 400


def test_anthropic_count_tokens_rejects_invalid_messages_shape():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/messages/count_tokens",
            json={"model": "claude-sonnet-4-5", "messages": "hello"},
        )
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["message"] == (
        "'messages' must be a list of objects"
    )


def test_anthropic_messages_rejects_invalid_messages_shape():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 10,
                "messages": 1,
            },
        )
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["message"] == (
        "'messages' must be a list of objects"
    )


def test_openai_chat_rejects_non_object_message_items():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/chat/completions",
            json={"model": "qwen3.5", "messages": [1]},
        )
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["message"] == (
        "'messages' must be a list of objects"
    )


def test_openai_chat_rejects_non_object_or_invalid_json():
    client = _app().test_client()
    responses = (
        client.post("/v1/chat/completions", json=[]),
        client.post("/v1/chat/completions", data="{", content_type="application/json"),
    )
    for response in responses:
        assert response.status_code == 400
        assert response.get_json()["error"]["message"] == (
            "Request body must be a JSON object"
        )


def test_token_interfaces_reject_invalid_tool_shapes():
    client = _app().test_client()
    openai_response = client.post(
        "/v1/responses/input_tokens",
        json={"model": "qwen3.5", "input": "Hello", "tools": "weather"},
    )
    claude_response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "qwen3.5",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [1],
        },
    )

    assert openai_response.status_code == 400
    assert claude_response.status_code == 400
    assert claude_response.get_json()["error"]["message"] == (
        "'tools' must be a list of objects"
    )


def test_openai_token_interface_rejects_non_string_model():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/responses/input_tokens",
            json={"model": 35, "input": "Hello"},
        )
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["message"] == "'model' must be a string"


def test_anthropic_interfaces_reject_zero_model_as_non_string():
    client = _app().test_client()
    payload = {
        "model": 0,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    responses = (
        client.post("/v1/messages/count_tokens", json=payload),
        client.post("/v1/messages", json={**payload, "max_tokens": 10}),
    )
    for response in responses:
        assert response.status_code == 400
        assert response.get_json()["error"]["message"] == "'model' must be a string"


def test_token_interfaces_reject_malformed_nested_content_and_tool_calls():
    client = _app().test_client()
    cases = (
        (
            "/v1/messages/count_tokens",
            {
                "model": "qwen3.5",
                "messages": [{"role": "user", "content": [1]}],
            },
        ),
        (
            "/v1/messages/count_tokens",
            {
                "model": "qwen3.5",
                "system": [None],
                "messages": [{"role": "user", "content": "Hello"}],
            },
        ),
        (
            "/v1/responses/input_tokens",
            {"model": "qwen3.5", "input": [1]},
        ),
        (
            "/v1/chat/completions",
            {
                "model": "qwen3.5",
                "messages": [{"role": "user", "content": "Hello", "tool_calls": [1]}],
            },
        ),
    )
    for path, payload in cases:
        response = client.post(path, json=payload)
        assert response.status_code == 400


def test_openai_chat_rejects_malformed_function_tool_definitions():
    client = _app().test_client()
    invalid_functions = (
        {},
        {"name": 1, "parameters": {}},
        {"name": "weather", "parameters": 1},
    )
    for function in invalid_functions:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.5",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [{"type": "function", "function": function}],
            },
        )
        assert response.status_code == 400


def test_anthropic_rejects_non_object_tool_choice():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/messages/count_tokens",
            json={
                "model": "qwen3.5",
                "messages": [{"role": "user", "content": "Hello"}],
                "tool_choice": "auto",
            },
        )
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["message"] == (
        "'tool_choice' must be an object"
    )


def test_chat_stream_include_usage_counts_reasoning_and_content():
    service = _app().extensions["genai_service"]
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"reasoning_content": "think", "content": "Hi"},
                            "finish_reason": None,
                        }
                    ]
                }
            ).encode(),
            json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode(),
        ]
    )
    with patch("genai_proxy.services.genai.requests.post", return_value=upstream):
        body = "".join(
            service.stream_openai_completion(
                {
                    "model": "chatglm",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            )
        )

    chunks = [
        json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")
    ]
    usage = next(chunk["usage"] for chunk in chunks if not chunk["choices"])
    assert usage == {
        "prompt_tokens": 20,
        "completion_tokens": 3,
        "total_tokens": 23,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 1},
    }


def test_chat_stream_without_usage_skips_tokenizer_counting():
    service = _app().extensions["genai_service"]
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "answer"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]
    )
    with (
        patch("genai_proxy.services.genai.requests.post", return_value=upstream),
        patch(
            "genai_proxy.services.genai.count_openai_request_tokens"
        ) as count_request,
    ):
        body = "".join(
            service.stream_openai_completion(
                {
                    "model": "qwen3.5",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                }
            )
        )

    count_request.assert_not_called()
    chunks = [
        json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")
    ]
    assert not any(not chunk["choices"] for chunk in chunks)


def test_chat_usage_propagates_length_finish_reason():
    service = _app().extensions["genai_service"]
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": "think",
                                "content": "answer",
                            },
                            "finish_reason": "length",
                        }
                    ]
                }
            ).encode()
        ]
    )
    request = {
        "model": "qwen3.5",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    with patch("genai_proxy.services.genai.requests.post", return_value=upstream):
        response = service.build_openai_completion(request)

    assert response["choices"][0]["finish_reason"] == "length"
    assert response["usage"]["completion_tokens"] == 5


def test_parsed_qwen_tool_call_usage_counts_raw_generated_syntax():
    service = _app().extensions["genai_service"]
    raw_content = (
        '<tool_call>{"name":"weather","arguments":{"city":"上海"}}</tool_call>'
    )
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {"delta": {"content": raw_content}, "finish_reason": None}
                    ]
                }
            ).encode(),
            json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode(),
        ]
    )
    request = {
        "model": "qwen3.5",
        "messages": [{"role": "user", "content": "Check weather"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }
    with patch("genai_proxy.services.genai.requests.post", return_value=upstream):
        response = service.build_openai_completion(request)

    assert response["choices"][0]["finish_reason"] == "tool_calls"
    assert (
        response["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        == "weather"
    )
    expected = count_openai_completion_tokens(
        {"role": "assistant", "content": raw_content},
        "qwen3.5",
        model_record=FakeModelManager.records["qwen3.5"],
        tool_adapter="generic",
        prompt_messages=service._prepare_chat_request(request).messages,
    )
    assert response["usage"]["completion_tokens"] == expected
