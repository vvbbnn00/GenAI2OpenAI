import json
import logging
from unittest.mock import patch

import pytest

from genai_proxy.api.openai.service import GenAIService
from genai_proxy.errors import ProxyError
from genai_proxy.messages import normalize_message_contents
from genai_proxy.models.registry import (
    DEEPSEEK_V4_FLASH_ADAPTER,
    DEEPSEEK_V4_PRO_ADAPTER,
    GLM_5_2_ADAPTER,
    KIMI_K3_ADAPTER,
    QWEN_3_5_ADAPTER,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

MODEL_CASES = (
    (
        "chatglm",
        GLM_5_2_ADAPTER,
        {
            "aiType": "chatglm",
            "aiName": "GLM-5.2",
            "rootModelName": "Xinference",
            "rootAiType": "xinference",
        },
    ),
    (
        "deepseek-chat",
        DEEPSEEK_V4_FLASH_ADAPTER,
        {
            "aiType": "deepseek-chat",
            "aiName": "DeepSeek-V4-Flash",
            "rootModelName": "Xinference",
            "rootAiType": "xinference",
        },
    ),
    (
        "deepseek-pro",
        DEEPSEEK_V4_PRO_ADAPTER,
        {
            "aiType": "deepseek-pro",
            "aiName": "DeepSeek-V4-Pro",
            "rootModelName": "Xinference",
            "rootAiType": "xinference",
        },
    ),
    (
        "qwen3.5",
        QWEN_3_5_ADAPTER,
        {
            "aiType": "qwen3.5",
            "aiName": "Qwen3.5-397B-A17B",
            "rootModelName": "Xinference",
            "rootAiType": "xinference",
        },
    ),
    (
        "kimi-k3",
        KIMI_K3_ADAPTER,
        {
            "aiType": "kimi-k3",
            "aiName": "Kimi-K3",
            "rootModelName": "Xinference",
            "rootAiType": "xinference",
        },
    ),
)


class FakeTokenManager:
    token = "token"
    billing_user_id = None

    def refresh_after_auth_failure(self, *_args, **_kwargs):
        return False


class FakeModelManager:
    def __init__(self, record):
        self.record = record

    def resolve_model(self, model):
        return model

    def get_model_record(self, model):
        if model == self.record.get("aiType"):
            return self.record
        return None

    def root_ai_type_for(self, _model):
        return self.record.get("rootAiType") or "xinference"


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self):
        self.lines = [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "Done."},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]

    def iter_lines(self, *args, **kwargs):
        return iter(self.lines)

    def close(self):
        pass


def make_service(record):
    return GenAIService(
        logging.getLogger("test_messages"),
        FakeTokenManager(),
        FakeModelManager(record),
    )


def text_parts(*parts):
    return [{"type": "text", "text": part} for part in parts]


@pytest.mark.parametrize(
    "adapter",
    (
        GLM_5_2_ADAPTER,
        DEEPSEEK_V4_FLASH_ADAPTER,
        DEEPSEEK_V4_PRO_ADAPTER,
        QWEN_3_5_ADAPTER,
        KIMI_K3_ADAPTER,
    ),
)
def test_text_parts_are_canonicalized_for_every_message_role(adapter):
    messages = [
        {"role": "system", "content": text_parts("sys", "tem")},
        {"role": "developer", "content": text_parts("dev", "eloper")},
        {"role": "user", "content": text_parts("user", " input")},
        {"role": "assistant", "content": text_parts("assistant", " output")},
        {"role": "tool", "tool_call_id": "call_1", "content": text_parts("to", "ol")},
    ]

    normalized = normalize_message_contents(messages, adapter=adapter)

    assert [message["content"] for message in normalized] == [
        "system",
        "developer",
        "user input",
        "assistant output",
        "tool",
    ]
    assert messages[0]["content"] == text_parts("sys", "tem")


@pytest.mark.parametrize("adapter", (QWEN_3_5_ADAPTER, KIMI_K3_ADAPTER))
def test_visual_adapters_preserve_user_image_parts(adapter):
    image = {
        "type": "image_url",
        "image_url": {"url": "https://example.test/image.png", "detail": "high"},
    }
    normalized = normalize_message_contents(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect it."},
                    image,
                ],
            }
        ],
        adapter=adapter,
    )

    assert normalized == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect it."},
                image,
            ],
        }
    ]


@pytest.mark.parametrize(
    "adapter",
    (GLM_5_2_ADAPTER, DEEPSEEK_V4_FLASH_ADAPTER, DEEPSEEK_V4_PRO_ADAPTER),
)
def test_non_visual_adapters_reject_images_before_transport(adapter):
    with pytest.raises(ProxyError) as exc_info:
        normalize_message_contents(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.test/image.png"},
                        }
                    ],
                }
            ],
            adapter=adapter,
        )

    assert exc_info.value.status == 400
    assert exc_info.value.code == "unsupported_content_type"


@pytest.mark.parametrize(
    "content",
    (
        [{"type": "text"}],
        [{"type": "input_text", "text": "Responses-only part"}],
        [{"type": "image_url", "image_url": {"url": "relative.png"}}],
        [{"type": "image_url", "image_url": {"url": "http://["}}],
        [{"type": "image_url", "image_url": {"url": "data:image/png,raw"}}],
    ),
)
def test_invalid_canonical_parts_are_rejected(content):
    with pytest.raises(ProxyError) as exc_info:
        normalize_message_contents(
            [{"role": "user", "content": content}],
            adapter=QWEN_3_5_ADAPTER,
        )

    assert exc_info.value.status == 400
    assert exc_info.value.code in {"unsupported_content_type", "invalid_image"}


@pytest.mark.parametrize("adapter", (QWEN_3_5_ADAPTER, KIMI_K3_ADAPTER))
def test_visual_adapters_reject_images_outside_user_messages(adapter):
    with pytest.raises(ProxyError) as exc_info:
        normalize_message_contents(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.test/image.png"},
                        }
                    ],
                }
            ],
            adapter=adapter,
        )

    assert exc_info.value.status == 400
    assert exc_info.value.code == "invalid_image"


def test_missing_assistant_content_is_not_added_during_normalization():
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ],
    }

    assert normalize_message_contents(
        [message],
        adapter=GLM_5_2_ADAPTER,
    ) == [message]


@pytest.mark.parametrize("model,adapter,record", MODEL_CASES)
@pytest.mark.parametrize("with_tools", (False, True))
def test_string_and_text_parts_prepare_identical_official_prompts(
    model,
    adapter,
    record,
    with_tools,
):
    service = make_service(record)
    common = {
        "model": model,
        "tools": [WEATHER_TOOL] if with_tools else [],
    }
    plain = service._prepare_chat_request(
        {
            **common,
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "File:\ncontents"},
            ],
        }
    )
    chunked = service._prepare_chat_request(
        {
            **common,
            "messages": [
                {"role": "system", "content": text_parts("Be ", "concise.")},
                {"role": "user", "content": text_parts("File:\n", "contents")},
            ],
        }
    )

    assert plain.tool_adapter == adapter
    assert chunked.messages == plain.messages
    assert chunked.prompt_tokens == plain.prompt_tokens


@pytest.mark.parametrize("model,_,record", MODEL_CASES[1:3])
@pytest.mark.parametrize("with_tools", (False, True))
@pytest.mark.parametrize("stream", (False, True))
def test_deepseek_text_parts_reach_upstream_identically(
    model,
    _,
    record,
    with_tools,
    stream,
):
    service = make_service(record)
    captured = []

    def fake_post(_url, **kwargs):
        captured.append(kwargs["json"])
        return FakeResponse()

    request = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": text_parts("Attached file contents.\n", "Answer it."),
            }
        ],
    }
    if with_tools:
        request["tools"] = [WEATHER_TOOL]

    with patch("genai_proxy.upstream.transport.requests.post", fake_post):
        if stream:
            list(service.stream_openai_completion(request))
        else:
            service.build_openai_completion(request)

    assert captured
    user_messages = [
        message
        for message in captured[0]["messages"]
        if message.get("role") == "user"
    ]
    assert user_messages[-1]["content"] == (
        "Attached file contents.\nAnswer it."
    )
    assert "chatGroupId" not in captured[0]


def test_qwen_tool_prompt_accepts_system_and_developer_text_parts():
    model, _, record = MODEL_CASES[3]
    prepared = make_service(record)._prepare_chat_request(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": text_parts("System ", "instruction.")},
                {
                    "role": "developer",
                    "content": text_parts("Developer ", "instruction."),
                },
                {"role": "user", "content": text_parts("Read ", "the file.")},
            ],
            "tools": [WEATHER_TOOL],
        },
        count_usage=False,
    )

    assert prepared.messages[0]["role"] == "system"
    assert prepared.messages[0]["content"].endswith(
        "System instruction.\n\nDeveloper instruction."
    )
    assert prepared.messages[-1] == {"role": "user", "content": "Read the file."}


@pytest.mark.parametrize("stream", (False, True))
def test_invalid_content_is_rejected_consistently_before_upstream(stream):
    model, _, record = MODEL_CASES[2]
    service = make_service(record)
    request = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.test/image.png"},
                    }
                ],
            }
        ],
    }

    with patch("genai_proxy.upstream.transport.requests.post") as post:
        with pytest.raises(ProxyError) as exc_info:
            if stream:
                service.stream_openai_completion(request)
            else:
                service.build_openai_completion(request)

    assert exc_info.value.status == 400
    assert exc_info.value.code == "unsupported_content_type"
    post.assert_not_called()
