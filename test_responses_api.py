import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from genai_proxy.app import create_app
from genai_proxy.compat.responses import convert_responses_to_openai_request
from genai_proxy.errors import ProxyError
from genai_proxy.services.genai import GenAIService

RESPONSES_WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}

RESPONSES_APPLY_PATCH_TOOL = {
    "type": "custom",
    "name": "apply_patch",
    "description": "Use apply_patch to edit files.",
    "format": {
        "type": "grammar",
        "syntax": "lark",
        "definition": "start: /.+/",
    },
}

RESPONSES_NAMESPACE_TOOL = {
    "type": "namespace",
    "name": "codex_app",
    "description": "Codex app tools.",
    "tools": [
        {
            "type": "function",
            "name": "view_image",
            "description": "View a local image.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ],
}

RESPONSES_WEB_SEARCH_TOOL = {
    "type": "web_search",
    "search_context_size": "medium",
}


class FakeTokenManager:
    token = "token"
    billing_user_id = None

    def refresh_after_auth_failure(self, *_args, **_kwargs):
        return False


class FakeModelManager:
    def __init__(self, record=None):
        self.record = record or {
            "aiType": "chatglm",
            "aiName": "GLM",
            "descInfo": "GLM 5.2",
            "rootModelName": "Xinference",
            "rootAiType": "xinference",
        }
        self.resolve_calls = 0

    def resolve_model(self, model):
        self.resolve_calls += 1
        return model or "chatglm"

    def get_model_record(self, model):
        if model == self.record["aiType"]:
            return self.record
        return None

    def root_ai_type_for(self, model):
        record = self.get_model_record(model) or {}
        return record.get("rootAiType") or "xinference"


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self, *args, **kwargs):
        for line in self._lines:
            yield line.encode("utf-8")

    def close(self):
        pass


def make_service(lines, *, record=None):
    captured = []

    def fake_post(_url, **kwargs):
        captured.append(kwargs["json"])
        return FakeResponse(lines)

    service = GenAIService(
        logging.getLogger("test_responses_api"),
        FakeTokenManager(),
        FakeModelManager(record),
    )
    return service, captured, fake_post


def genai_sse(delta=None, finish_reason=None):
    return "data: " + json.dumps(
        {"choices": [{"delta": delta or {}, "finish_reason": finish_reason}]},
        ensure_ascii=False,
    )


def parse_response_events(chunks):
    events = []
    for chunk in chunks:
        event_name = None
        data = None
        for line in str(chunk).splitlines():
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event_name and data:
            events.append({"event": event_name, "data": data})
    return events


def output_items(events):
    return [
        event["data"]["item"]
        for event in events
        if event["event"] == "response.output_item.done"
    ]


def completed_event(events):
    matches = [event for event in events if event["event"] == "response.completed"]
    assert matches, json.dumps(events, ensure_ascii=False)
    return matches[-1]["data"]["response"]


def test_responses_text_stream_emits_codex_events_and_reasoning_delta():
    service, captured, fake_post = make_service(
        [
            genai_sse({"reasoning_content": "thinking", "content": "Hello"}),
            genai_sse({"content": " world"}),
            genai_sse({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "instructions": "You are Codex.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hello."}],
            }
        ],
        "stream": True,
        "reasoning": {"effort": "high"},
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        events = parse_response_events(service.stream_responses(request))

    assert [event["event"] for event in events][:4] == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.reasoning_text.delta",
    ]
    reasoning_added = events[1]["data"]
    reasoning_part = events[2]["data"]
    reasoning_delta = events[3]["data"]
    assert reasoning_added["item"]["type"] == "reasoning"
    assert reasoning_added["output_index"] == 0
    assert reasoning_part["item_id"] == reasoning_added["item"]["id"]
    assert reasoning_delta["item_id"] == reasoning_added["item"]["id"]
    assert reasoning_delta["output_index"] == 0
    assert reasoning_delta["content_index"] == 0
    assert reasoning_delta["delta"] == "thinking"
    assert [event["data"]["sequence_number"] for event in events] == list(
        range(len(events))
    )
    reasoning_items = [
        item for item in output_items(events) if item["type"] == "reasoning"
    ]
    assert reasoning_items[-1]["content"] == [
        {"type": "reasoning_text", "text": "thinking"}
    ]
    assert any(
        event["event"] == "response.output_text.delta"
        and event["data"]["delta"] == " world"
        for event in events
    )
    message_items = [item for item in output_items(events) if item["type"] == "message"]
    assert message_items[-1]["content"] == [
        {
            "type": "output_text",
            "text": "Hello world",
            "annotations": [],
        }
    ]
    assert completed_event(events)["end_turn"] is True
    assert captured[0]["messages"][0] == {
        "role": "system",
        "content": "You are Codex.",
    }
    assert captured[0]["messages"][1] == {"role": "user", "content": "Say hello."}


def test_responses_created_event_precedes_prompt_token_counting():
    service, _captured, fake_post = make_service(
        [
            genai_sse({"reasoning_content": "thinking"}),
            genai_sse({"content": "Hello"}),
            genai_sse({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "input": "Hello",
        "stream": True,
    }

    with (
        patch("genai_proxy.services.genai.requests.post", fake_post),
        patch(
            "genai_proxy.services.genai.count_openai_request_tokens",
            return_value=7,
        ) as count_tokens,
    ):
        stream = service.stream_responses(request)
        first_event = parse_response_events([next(stream)])
        count_tokens.assert_not_called()
        remaining_events = parse_response_events(stream)

    assert first_event[0]["event"] == "response.created"
    assert any(
        event["event"] == "response.reasoning_text.delta" for event in remaining_events
    )
    count_tokens.assert_called_once()


def test_responses_input_accepts_easy_message_without_type():
    service, captured, fake_post = make_service(
        [
            genai_sse({"content": "Hello"}),
            genai_sse({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "input": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        events = parse_response_events(service.stream_responses(request))

    assert completed_event(events)["end_turn"] is True
    assert captured[0]["messages"][-1] == {"role": "user", "content": "hi"}


def test_responses_input_image_is_preserved_as_openai_vision_content():
    image_url = "https://example.test/image.png"
    context = convert_responses_to_openai_request(
        {
            "model": "kimi-k3",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this image."},
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": "high",
                        },
                    ],
                }
            ],
        }
    )

    assert context.openai_request["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"},
                },
            ],
        }
    ]


def test_responses_converter_preserves_images_until_model_validation():
    context = convert_responses_to_openai_request(
        {
            "model": "GPT-4.1",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "https://example.test/image.png",
                        }
                    ],
                }
            ],
        }
    )

    assert context.openai_request["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/image.png"},
                }
            ],
        }
    ]


def test_responses_text_parts_keep_protocol_order_without_added_newlines():
    context = convert_responses_to_openai_request(
        {
            "model": "chatglm",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "first"},
                        {"type": "input_text", "text": "second"},
                    ],
                }
            ],
        }
    )

    assert context.openai_request["messages"] == [
        {"role": "user", "content": "firstsecond"}
    ]


def test_responses_uses_resolved_qwen_record_for_visual_transport():
    data_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    record = {
        "aiType": "campus-vision",
        "aiName": "Qwen3.5-397B-A17B",
        "rootModelName": "Xinference",
        "rootAiType": "xinference",
    }
    service, captured, fake_post = make_service(
        [genai_sse({"content": "Visible."}), genai_sse({}, "stop")],
        record=record,
    )
    request = {
        "model": "campus-vision",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe it."},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        "stream": True,
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        list(service.stream_responses(request))

    assert captured[0]["messages"][-1]["content"] == [
        {"type": "text", "text": "Describe it."},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    assert service._model_manager.resolve_calls == 1
    assert "chatGroupId" not in captured[0]


def test_responses_non_visual_record_rejects_image_before_upstream():
    service, _, _ = make_service([])
    request = {
        "model": "chatglm",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "https://example.test/image.png",
                    }
                ],
            }
        ],
        "stream": True,
    }

    with patch("genai_proxy.services.genai.requests.post") as post:
        with pytest.raises(ProxyError) as exc_info:
            list(service.stream_responses(request))

    assert exc_info.value.status == 400
    assert exc_info.value.code == "unsupported_content_type"
    post.assert_not_called()


def test_responses_resolved_kimi_adapter_keeps_tools_after_tool_output():
    record = {
        "aiType": "campus-assistant",
        "aiName": "Kimi-K3",
        "rootModelName": "Xinference",
        "rootAiType": "xinference",
    }
    service, _, _ = make_service([], record=record)
    context, model_context = service._convert_responses_request(
        {
            "model": "campus-assistant",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "done",
                }
            ],
            "tools": [RESPONSES_WEATHER_TOOL],
        }
    )

    assert model_context.tool_adapter == "kimi_k3"
    assert context.openai_request.get("tool_choice") != "none"


def test_responses_qwen_image_is_preserved_for_official_visual_template():
    image_url = "https://example.test/image.png"
    context = convert_responses_to_openai_request(
        {
            "model": "qwen-instruct",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this image."},
                        {
                            "type": "input_image",
                            "image_url": image_url,
                        },
                    ],
                }
            ],
        }
    )

    assert context.openai_request["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ]


def test_responses_local_shell_call_preserves_command_argv():
    service, captured, fake_post = make_service(
        [
            genai_sse({"content": "Done"}),
            genai_sse({}, "stop"),
        ]
    )
    command = ["python", "-c", "print('quoted arg')", "path with spaces"]
    request = {
        "model": "chatglm",
        "input": [
            {
                "type": "local_shell_call",
                "call_id": "call_shell",
                "action": {
                    "type": "exec",
                    "command": command,
                    "timeout_ms": 1000,
                    "working_directory": "/tmp/work dir",
                },
            },
            {"type": "message", "role": "user", "content": "continue"},
        ],
        "stream": True,
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        events = parse_response_events(service.stream_responses(request))

    assert completed_event(events)["end_turn"] is True
    tool_message = next(
        message for message in captured[0]["messages"] if message.get("tool_calls")
    )
    arguments = json.loads(tool_message["tool_calls"][0]["function"]["arguments"])
    assert arguments == {
        "command": command,
        "timeout_ms": 1000,
        "working_directory": "/tmp/work dir",
    }

    context = convert_responses_to_openai_request(
        {
            "model": "chatglm",
            "input": [
                {
                    "type": "local_shell_call",
                    "call_id": "call_empty",
                    "action": {"type": "exec", "command": ""},
                },
                {"type": "message", "role": "user", "content": "continue"},
            ],
        }
    )
    empty_tool_message = next(
        message
        for message in context.openai_request["messages"]
        if message.get("tool_calls")
    )
    empty_arguments = json.loads(
        empty_tool_message["tool_calls"][0]["function"]["arguments"]
    )
    assert empty_arguments["command"] == ""

    malformed_context = convert_responses_to_openai_request(
        {
            "model": "chatglm",
            "input": [
                {
                    "type": "local_shell_call",
                    "call_id": "call_malformed",
                    "action": "not an object",
                },
                {"type": "message", "role": "user", "content": "continue"},
            ],
        }
    )
    malformed_tool_message = next(
        message
        for message in malformed_context.openai_request["messages"]
        if message.get("tool_calls")
    )
    malformed_arguments = json.loads(
        malformed_tool_message["tool_calls"][0]["function"]["arguments"]
    )
    assert malformed_arguments["command"] == []


def test_responses_function_tool_call_from_glm_xml_is_codex_function_call_item():
    service, captured, fake_post = make_service(
        [
            genai_sse({"reasoning_content": "I should call weather."}),
            genai_sse(
                {
                    "content": (
                        "<tool_call>get_weather"
                        "<arg_key>location</arg_key><arg_value>Shanghai</arg_value>"
                        "</tool_call>"
                    )
                }
            ),
            genai_sse({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Use weather."}],
            }
        ],
        "tools": [RESPONSES_WEATHER_TOOL],
        "tool_choice": "auto",
        "stream": True,
        "reasoning": {"effort": "high"},
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        events = parse_response_events(service.stream_responses(request))

    function_items = [
        item for item in output_items(events) if item["type"] == "function_call"
    ]
    function_added = next(
        event["data"]["item"]
        for event in events
        if event["event"] == "response.output_item.added"
        and event["data"]["item"].get("type") == "function_call"
    )
    reasoning_events = [
        event for event in events if event["event"] == "response.reasoning_text.delta"
    ]
    assert reasoning_events
    assert reasoning_events[0]["data"]["delta"] == "I should call weather."
    assert function_items
    assert function_items[0]["name"] == "get_weather"
    assert function_items[0]["id"].startswith("fc_")
    assert function_items[0]["status"] == "completed"
    assert json.loads(function_items[0]["arguments"]) == {"location": "Shanghai"}
    assert function_items[0]["call_id"].startswith("call_")
    assert function_added == {
        **function_items[0],
        "arguments": "",
        "status": "in_progress",
    }
    assert completed_event(events)["end_turn"] is False
    function_done_index = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "response.output_item.done"
        and event["data"]["item"].get("type") == "function_call"
    )
    assert events.index(reasoning_events[0]) < function_done_index
    prompt = captured[0]["messages"][0]["content"]
    assert prompt.startswith("\n# Tools\n\n")
    assert "Reasoning Effort:" not in prompt
    assert '"name": "get_weather"' in prompt


def test_responses_object_tool_choice_selects_named_function():
    context = convert_responses_to_openai_request(
        {
            "model": "chatglm",
            "input": "Use weather.",
            "tools": [RESPONSES_WEATHER_TOOL],
            "tool_choice": {"type": "function", "name": "get_weather"},
        }
    )

    assert context.openai_request["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_weather"},
    }


def test_responses_function_call_output_turn_returns_final_message():
    service, captured, fake_post = make_service(
        [
            genai_sse({"content": "Shanghai is sunny."}),
            genai_sse({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Use weather."}],
            },
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"location":"Shanghai"}',
                "call_id": "call_weather",
            },
            {
                "type": "function_call_output",
                "call_id": "call_weather",
                "output": "Shanghai is sunny.",
            },
        ],
        "tools": [RESPONSES_WEATHER_TOOL],
        "stream": True,
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        events = parse_response_events(service.stream_responses(request))

    message_items = [item for item in output_items(events) if item["type"] == "message"]
    assert message_items[-1]["content"] == [
        {
            "type": "output_text",
            "text": "Shanghai is sunny.",
            "annotations": [],
        }
    ]
    assert completed_event(events)["end_turn"] is True
    assert (
        "This turn must end with final assistant text only"
        not in captured[0]["messages"][0]["content"]
    )
    assert captured[0]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_weather",
        "content": "Shanghai is sunny.",
    }
    assert captured[0]["messages"][-2]["role"] == "assistant"
    assert captured[0]["messages"][-2]["tool_calls"][0]["function"]["name"] == (
        "get_weather"
    )


def test_responses_kimi_function_output_keeps_tools_available_by_default():
    request = {
        "model": "kimi-k3",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Inspect the project."}],
            },
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"location":"Shanghai"}',
                "call_id": "call_weather",
            },
            {
                "type": "function_call_output",
                "call_id": "call_weather",
                "output": "Continue with another check.",
            },
        ],
        "tools": [RESPONSES_WEATHER_TOOL],
    }

    context = convert_responses_to_openai_request(
        request,
        keep_tools_after_output=True,
    )

    assert context.openai_request["tool_choice"] == "auto"
    service, _captured, _fake_post = make_service([])
    prepared = service._prepare_chat_request(
        context.openai_request,
        count_usage=False,
    )
    assert prepared.has_tools
    assert prepared.tool_choice == "auto"
    assert prepared.messages[-2]["content"].startswith("# Client response protocol\n")
    assert prepared.messages[-1]["role"] == "user"
    assert prepared.messages[-1]["content"].startswith(
        "Completed client action result: "
    )
    assert "Continue the current user task" not in prepared.messages[-1]["content"]
    assert not any(
        message.get("role") == "system"
        and str(message.get("content", "")).startswith(
            "Completed client action result: "
        )
        for message in prepared.messages
    )

    request["tool_choice"] = "none"
    explicit_none = convert_responses_to_openai_request(
        request,
        keep_tools_after_output=True,
    )
    assert explicit_none.openai_request["tool_choice"] == "none"


def test_responses_custom_apply_patch_tool_becomes_custom_tool_call_with_input_delta():
    service, _captured, fake_post = make_service(
        [
            genai_sse(
                {
                    "content": (
                        "<tool_call>apply_patch"
                        "<arg_key>input</arg_key><arg_value>*** Begin Patch\n"
                        "*** Add File: hello.txt\n"
                        "+hello\n"
                        "*** End Patch</arg_value>"
                        "</tool_call>"
                    )
                }
            ),
            genai_sse({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Patch a file."}],
            }
        ],
        "tools": [RESPONSES_APPLY_PATCH_TOOL],
        "stream": True,
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        events = parse_response_events(service.stream_responses(request))

    delta_events = [
        event
        for event in events
        if event["event"] == "response.custom_tool_call_input.delta"
    ]
    assert delta_events
    assert delta_events[0]["data"]["delta"].startswith("*** Begin Patch")
    custom_items = [
        item for item in output_items(events) if item["type"] == "custom_tool_call"
    ]
    custom_added = next(
        event["data"]["item"]
        for event in events
        if event["event"] == "response.output_item.added"
        and event["data"]["item"].get("type") == "custom_tool_call"
    )
    assert custom_items
    assert custom_items[0]["id"].startswith("ctc_")
    assert custom_items[0]["name"] == "apply_patch"
    assert custom_items[0]["input"].startswith("*** Begin Patch")
    assert custom_added == {**custom_items[0], "input": ""}


def test_responses_namespace_tool_flattens_for_model_and_restores_namespace():
    service, captured, fake_post = make_service(
        [
            genai_sse(
                {
                    "content": (
                        "<tool_call>codex_app__view_image"
                        "<arg_key>path</arg_key><arg_value>/tmp/a.png</arg_value>"
                        "</tool_call>"
                    )
                }
            ),
            genai_sse({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "View image."}],
            }
        ],
        "tools": [RESPONSES_NAMESPACE_TOOL],
        "stream": True,
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        events = parse_response_events(service.stream_responses(request))

    function_items = [
        item for item in output_items(events) if item["type"] == "function_call"
    ]
    assert function_items[0]["name"] == "view_image"
    assert function_items[0]["namespace"] == "codex_app"
    assert json.loads(function_items[0]["arguments"]) == {"path": "/tmp/a.png"}
    assert '"name": "codex_app__view_image"' in captured[0]["messages"][0]["content"]


def test_responses_route_streams_sse():
    service, _captured, fake_post = make_service(
        [
            genai_sse({"content": "OK"}),
            genai_sse({}, "stop"),
        ]
    )
    app = create_app(
        SimpleNamespace(
            token=None,
            keystore=None,
            token_check_interval=0,
            api_key=None,
        ),
        logging.getLogger("test_responses_api_route"),
    )
    app.extensions["genai_service"] = service

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        response = app.test_client().post(
            "/v1/responses",
            json={
                "model": "chatglm",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "ok"}],
                    }
                ],
                "stream": True,
            },
        )
        body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    assert "event: response.created" in body
    assert "event: response.completed" in body


def test_responses_route_defaults_to_non_stream_json():
    service, _captured, fake_post = make_service(
        [
            genai_sse({"content": "OK"}),
            genai_sse({}, "stop"),
        ]
    )
    app = create_app(
        SimpleNamespace(
            token=None,
            keystore=None,
            token_check_interval=0,
            api_key=None,
        ),
        logging.getLogger("test_responses_api_route_non_stream"),
    )
    app.extensions["genai_service"] = service

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        response = app.test_client().post(
            "/v1/responses",
            json={
                "model": "chatglm",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "ok"}],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert response.is_json
    assert response.get_json()["output_text"] == "OK"


def test_responses_ignores_hosted_tools_codex_may_send_by_default():
    service, captured, fake_post = make_service(
        [
            genai_sse({"content": "No search needed."}),
            genai_sse({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Answer locally."}],
            }
        ],
        "tools": [RESPONSES_WEB_SEARCH_TOOL],
        "stream": True,
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        events = parse_response_events(service.stream_responses(request))

    message_items = [item for item in output_items(events) if item["type"] == "message"]
    assert message_items[-1]["content"] == [
        {
            "type": "output_text",
            "text": "No search needed.",
            "annotations": [],
        }
    ]
    assert "web_search" not in json.dumps(captured[0]["messages"], ensure_ascii=False)


if __name__ == "__main__":
    test_responses_text_stream_emits_codex_events_and_reasoning_delta()
    test_responses_created_event_precedes_prompt_token_counting()
    test_responses_input_accepts_easy_message_without_type()
    test_responses_local_shell_call_preserves_command_argv()
    test_responses_function_tool_call_from_glm_xml_is_codex_function_call_item()
    test_responses_function_call_output_turn_returns_final_message()
    test_responses_kimi_function_output_keeps_tools_available_by_default()
    test_responses_custom_apply_patch_tool_becomes_custom_tool_call_with_input_delta()
    test_responses_namespace_tool_flattens_for_model_and_restores_namespace()
    test_responses_route_streams_sse()
    test_responses_route_defaults_to_non_stream_json()
    test_responses_ignores_hosted_tools_codex_may_send_by_default()
    print("responses api tests passed")
