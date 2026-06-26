import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

from genai_proxy.app import create_app
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
    def __init__(self):
        self.record = {
            "aiType": "chatglm",
            "aiName": "GLM",
            "descInfo": "GLM 5.2",
            "rootModelName": "Xinference",
            "rootAiType": "xinference",
        }

    def resolve_model(self, model):
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

    def iter_lines(self):
        for line in self._lines:
            yield line.encode("utf-8")

    def close(self):
        pass


def make_service(lines):
    captured = []

    def fake_post(_url, **kwargs):
        captured.append(kwargs["json"])
        return FakeResponse(lines)

    service = GenAIService(
        logging.getLogger("test_responses_api"),
        FakeTokenManager(),
        FakeModelManager(),
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
        "response.reasoning_text.delta",
        "response.output_item.added",
        "response.output_text.delta",
    ]
    assert events[2]["data"]["item"]["type"] == "message"
    assert any(
        event["event"] == "response.output_text.delta"
        and event["data"]["delta"] == " world"
        for event in events
    )
    message_items = [item for item in output_items(events) if item["type"] == "message"]
    assert message_items[-1]["content"] == [{"type": "output_text", "text": "Hello world"}]
    assert completed_event(events)["end_turn"] is True
    assert captured[0]["messages"][0] == {
        "role": "system",
        "content": "You are Codex.\n\nReasoning Effort: High",
    }
    assert captured[0]["messages"][1] == {"role": "user", "content": "Say hello."}


def test_responses_function_tool_call_from_glm_xml_is_codex_function_call_item():
    service, captured, fake_post = make_service(
        [
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

    function_items = [item for item in output_items(events) if item["type"] == "function_call"]
    assert function_items
    assert function_items[0]["name"] == "get_weather"
    assert json.loads(function_items[0]["arguments"]) == {"location": "Shanghai"}
    assert function_items[0]["call_id"].startswith("call_")
    assert completed_event(events)["end_turn"] is False
    prompt = captured[0]["messages"][0]["content"]
    assert prompt.startswith("Reasoning Effort: High\n\n# Tools\n\n")
    assert '"name": "get_weather"' in prompt


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
                "arguments": "{\"location\":\"Shanghai\"}",
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
        {"type": "output_text", "text": "Shanghai is sunny."}
    ]
    assert completed_event(events)["end_turn"] is True
    assert "This turn must end with final assistant text only" in captured[0]["messages"][0]["content"]


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
    custom_items = [item for item in output_items(events) if item["type"] == "custom_tool_call"]
    assert custom_items
    assert custom_items[0]["name"] == "apply_patch"
    assert custom_items[0]["input"].startswith("*** Begin Patch")


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

    function_items = [item for item in output_items(events) if item["type"] == "function_call"]
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
        {"type": "output_text", "text": "No search needed."}
    ]
    assert "web_search" not in json.dumps(captured[0]["messages"], ensure_ascii=False)


if __name__ == "__main__":
    test_responses_text_stream_emits_codex_events_and_reasoning_delta()
    test_responses_function_tool_call_from_glm_xml_is_codex_function_call_item()
    test_responses_function_call_output_turn_returns_final_message()
    test_responses_custom_apply_patch_tool_becomes_custom_tool_call_with_input_delta()
    test_responses_namespace_tool_flattens_for_model_and_restores_namespace()
    test_responses_route_streams_sse()
    test_responses_route_defaults_to_non_stream_json()
    test_responses_ignores_hosted_tools_codex_may_send_by_default()
    print("responses api tests passed")
