import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

from genai_proxy.compat.claude import (
    convert_claude_to_openai,
    convert_openai_to_claude_response,
    stream_openai_to_claude,
)
from genai_proxy.errors import ProxyError
from genai_proxy.optimizations.deepseek import DEEPSEEK_V4_REASONING_EFFORT_MAX
from genai_proxy.routes.claude import map_claude_model_alias
from genai_proxy.services.genai import GenAIService


OPENAI_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}

OPENAI_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "Bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "description": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
    },
}

CLAUDE_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get current weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}

CLAUDE_BASH_TOOL = {
    "name": "Bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "description": {"type": "string"},
            "timeout": {"type": "integer"},
        },
        "required": ["command"],
    },
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

    def resolve_model(self, model):
        return model or "chatglm"

    def get_model_record(self, model):
        if model == self.record.get("aiType"):
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


def make_service(lines, record=None):
    captured = []

    def fake_post(_url, **kwargs):
        captured.append(kwargs["json"])
        return FakeResponse(lines)

    model_manager = FakeModelManager(record)
    service = GenAIService(
        logging.getLogger("test_glm52_service"),
        FakeTokenManager(),
        model_manager,
    )
    return service, captured, fake_post


def sse_line(delta=None, finish_reason=None):
    payload = {
        "choices": [
            {
                "delta": delta or {},
                "finish_reason": finish_reason,
            }
        ]
    }
    return "data: " + json.dumps(payload, ensure_ascii=False)


def parse_openai_events(chunks):
    events = []
    for chunk in chunks:
        for line in str(chunk).splitlines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                continue
            events.append(json.loads(data))
    return events


def parse_claude_events(chunks):
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


def first_tool_call(response):
    message = response["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    assert tool_calls, json.dumps(response, ensure_ascii=False)
    return tool_calls[0]


def claude_tool_input_from_events(events, tool_name):
    tool_indices = set()
    for event in events:
        if event["event"] != "content_block_start":
            continue
        block = event["data"].get("content_block", {})
        if block.get("type") == "tool_use" and block.get("name") == tool_name:
            tool_indices.add(event["data"]["index"])

    for event in events:
        if event["event"] != "content_block_delta":
            continue
        if event["data"].get("index") not in tool_indices:
            continue
        delta = event["data"].get("delta", {})
        if delta.get("type") == "input_json_delta":
            return json.loads(delta.get("partial_json") or "{}")
    raise AssertionError(json.dumps(events, ensure_ascii=False))


def test_glm52_maps_lower_reasoning_effort_to_max():
    service, captured, fake_post = make_service(
        [
            sse_line({"content": "Done."}),
            sse_line({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "messages": [{"role": "user", "content": "Use get_weather for Shanghai."}],
        "tools": [OPENAI_WEATHER_TOOL],
        "reasoning": {"effort": "low"},
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        response = service.build_openai_completion(request)

    assert response["choices"][0]["message"]["content"] == "Done."
    system_prompt = captured[0]["messages"][0]["content"]
    assert system_prompt.startswith("Reasoning Effort: Max\n\n# Tools\n\n"), (
        system_prompt
    )


def test_openai_reasoning_effort_max_is_preserved_for_glm52():
    service, captured, fake_post = make_service(
        [
            sse_line({"content": "Done."}),
            sse_line({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "messages": [{"role": "user", "content": "Answer directly."}],
        "reasoning": {"effort": "max"},
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        response = service.build_openai_completion(request)

    assert response["choices"][0]["message"]["content"] == "Done."
    assert captured[0]["messages"][0] == {
        "role": "system",
        "content": "Reasoning Effort: Max",
    }


def test_glm52_openai_reasoning_effort_alias_injects_prompt_without_tools():
    service, captured, fake_post = make_service(
        [
            sse_line({"content": "Done."}),
            sse_line({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "messages": [{"role": "user", "content": "Answer directly."}],
        "reasoning_effort": "high",
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        response = service.build_openai_completion(request)

    assert response["choices"][0]["message"]["content"] == "Done."
    assert captured[0]["messages"][0] == {
        "role": "system",
        "content": "Reasoning Effort: High",
    }


def test_deepseek_v4_reasoning_effort_is_normalized_before_prompt_injection():
    record = {
        "aiType": "deepseek-pro",
        "aiName": "DeepSeek V4 Pro",
        "descInfo": "DeepSeek V4",
        "rootModelName": "Xinference",
        "rootAiType": "xinference",
    }
    service, captured, fake_post = make_service(
        [
            sse_line({"content": "Done."}),
            sse_line({}, "stop"),
        ],
        record=record,
    )
    request = {
        "model": "deepseek-pro",
        "messages": [{"role": "user", "content": "Answer directly."}],
        "reasoning": {"effort": "xhigh"},
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        response = service.build_openai_completion(request)

    assert response["choices"][0]["message"]["content"] == "Done."
    assert captured[0]["messages"][0] == {
        "role": "system",
        "content": DEEPSEEK_V4_REASONING_EFFORT_MAX,
    }
    assert captured[0]["messages"][1] == {
        "role": "user",
        "content": "Answer directly.",
    }


def test_glm52_openai_non_stream_xml_tool_call_uses_reasoning_high():
    service, captured, fake_post = make_service(
        [
            sse_line(
                {
                    "content": (
                        "<tool_call>get_weather"
                        "<arg_key>location</arg_key><arg_value>Shanghai</arg_value>"
                        "</tool_call>"
                    )
                }
            ),
            sse_line({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "messages": [{"role": "user", "content": "Use get_weather for Shanghai."}],
        "tools": [OPENAI_WEATHER_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "reasoning": {"effort": "high"},
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        response = service.build_openai_completion(request)

    tool_call = first_tool_call(response)
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"location": "Shanghai"}
    prompt = captured[0]["messages"][0]["content"]
    assert prompt.startswith("Reasoning Effort: High\n\n# Tools\n\n")
    assert "<|system|>" not in prompt


def test_glm52_openai_stream_bash_tool_call_uses_default_reasoning_max():
    service, captured, fake_post = make_service(
        [
            sse_line(
                {
                    "content": (
                        "<tool_call>Bash"
                        "<arg_key>command</arg_key><arg_value>ls -la</arg_value>"
                        "<arg_key>timeout</arg_key><arg_value>60000</arg_value>"
                        "</tool_call>"
                    )
                }
            ),
            sse_line({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "messages": [{"role": "user", "content": "Run ls with Bash."}],
        "tools": [OPENAI_BASH_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "Bash"}},
        "stream": True,
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        events = parse_openai_events(service.stream_openai_completion(request))

    tool_chunks = [
        tool_call
        for event in events
        for choice in event.get("choices", [])
        for tool_call in choice.get("delta", {}).get("tool_calls", []) or []
    ]
    assert tool_chunks
    assert tool_chunks[0]["function"]["name"] == "Bash"
    assert any(
        choice.get("finish_reason") == "tool_calls"
        for event in events
        for choice in event["choices"]
    )
    assert captured[0]["messages"][0]["content"].startswith(
        "Reasoning Effort: Max\n\n# Tools\n\n"
    )


def test_glm52_required_tool_choice_accepts_xml_tool_call():
    service, _captured, fake_post = make_service(
        [
            sse_line(
                {
                    "content": (
                        "<tool_call>get_weather"
                        "<arg_key>location</arg_key><arg_value>Beijing</arg_value>"
                        "</tool_call>"
                    )
                }
            ),
            sse_line({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "messages": [{"role": "user", "content": "Use a weather tool."}],
        "tools": [OPENAI_WEATHER_TOOL],
        "tool_choice": "required",
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        response = service.build_openai_completion(request)

    assert first_tool_call(response)["function"]["name"] == "get_weather"


def test_glm52_tool_choice_none_does_not_return_tool_calls():
    service, captured, fake_post = make_service(
        [
            sse_line({"content": "Paris."}),
            sse_line({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "tools": [OPENAI_WEATHER_TOOL],
        "tool_choice": "none",
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        response = service.build_openai_completion(request)

    message = response["choices"][0]["message"]
    assert not message.get("tool_calls")
    assert message["content"] == "Paris."
    assert (
        "For this turn, do not call any tool" in captured[0]["messages"][0]["content"]
    )


def test_glm52_tool_result_turn_returns_final_text():
    service, captured, fake_post = make_service(
        [
            sse_line({"content": "Shanghai is sunny."}),
            sse_line({}, "stop"),
        ]
    )
    tool_call = {
        "id": "call_weather",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": json.dumps({"location": "Shanghai"}),
        },
    }
    request = {
        "model": "chatglm",
        "messages": [
            {"role": "user", "content": "Use get_weather for Shanghai."},
            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            {
                "role": "tool",
                "tool_call_id": "call_weather",
                "content": "Shanghai is sunny.",
            },
        ],
        "tools": [OPENAI_WEATHER_TOOL],
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        response = service.build_openai_completion(request)

    message = response["choices"][0]["message"]
    assert not message.get("tool_calls")
    assert message["content"] == "Shanghai is sunny."
    assert (
        "This turn must end with final assistant text only"
        not in captured[0]["messages"][0]["content"]
    )
    assert "<tools>" in captured[0]["messages"][0]["content"]
    assert captured[0]["messages"][-1]["content"].startswith("<|observation|>")
    assert "Return the final answer only" not in captured[0]["messages"][-1]["content"]
    assert (
        "Do not emit <tool_call>, <arg_key>, or <arg_value> tags"
        not in captured[0]["messages"][-1]["content"]
    )


def test_glm52_native_upstream_tool_call_deltas_are_preserved():
    service, _captured, fake_post = make_service(
        [
            sse_line(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_native",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ]
                }
            ),
            sse_line(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"location":"Shanghai"}'},
                        }
                    ]
                }
            ),
            sse_line({}, "tool_calls"),
        ]
    )
    request = {
        "model": "chatglm",
        "messages": [{"role": "user", "content": "Use get_weather for Shanghai."}],
        "tools": [OPENAI_WEATHER_TOOL],
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        response = service.build_openai_completion(request)

    tool_call = first_tool_call(response)
    assert tool_call["id"] == "call_native"
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"location": "Shanghai"}


def test_glm52_openai_stream_reasoning_content_passes_through_without_tools():
    service, _captured, fake_post = make_service(
        [
            sse_line({"reasoning_content": "thinking", "content": "Answer"}),
            sse_line({}, "stop"),
        ]
    )
    request = {
        "model": "chatglm",
        "messages": [{"role": "user", "content": "Answer directly."}],
        "stream": True,
    }

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        events = parse_openai_events(service.stream_openai_completion(request))

    assert any(
        choice.get("delta", {}).get("reasoning_content") == "thinking"
        for event in events
        for choice in event["choices"]
    )
    assert any(
        choice.get("delta", {}).get("content") == "Answer"
        for event in events
        for choice in event["choices"]
    )


def test_claude_output_config_effort_maps_to_glm52_openai_reasoning_and_tool_use():
    model_manager = FakeModelManager()
    service, _captured, fake_post = make_service(
        [
            sse_line(
                {
                    "content": (
                        "<tool_call>get_weather"
                        "<arg_key>location</arg_key><arg_value>Shanghai</arg_value>"
                        "</tool_call>"
                    )
                }
            ),
            sse_line({}, "stop"),
        ]
    )
    claude_request = {
        "model": "chatglm",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Use get_weather for Shanghai."}],
        "tools": [CLAUDE_WEATHER_TOOL],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "output_config": {"effort": "high"},
    }
    openai_request = convert_claude_to_openai(claude_request, model_manager)
    assert openai_request["reasoning"] == {"effort": "high"}

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        openai_response = service.build_openai_completion(openai_request)

    response = convert_openai_to_claude_response(
        openai_response,
        {**claude_request, "_estimator_model": "chatglm"},
    )
    tool_blocks = [
        block for block in response["content"] if block.get("type") == "tool_use"
    ]
    assert tool_blocks
    assert tool_blocks[0]["name"] == "get_weather"
    assert tool_blocks[0]["input"] == {"location": "Shanghai"}
    assert response["stop_reason"] == "tool_use"


def test_claude_streaming_tool_use_survives_glm52_reasoning_effort():
    model_manager = FakeModelManager()
    service, _captured, fake_post = make_service(
        [
            sse_line(
                {
                    "content": (
                        "<tool_call>get_weather"
                        "<arg_key>location</arg_key><arg_value>Beijing</arg_value>"
                        "</tool_call>"
                    )
                }
            ),
            sse_line({}, "stop"),
        ]
    )
    claude_request = {
        "model": "chatglm",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Use get_weather for Beijing."}],
        "tools": [CLAUDE_WEATHER_TOOL],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "output_config": {"effort": "max"},
        "stream": True,
    }
    openai_request = convert_claude_to_openai(claude_request, model_manager)
    assert openai_request["reasoning"] == {"effort": "max"}

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        events = parse_claude_events(
            stream_openai_to_claude(
                service.stream_openai_completion(openai_request),
                {**claude_request, "_estimator_model": "chatglm"},
                logging.getLogger("test_glm52_service"),
            )
        )

    tool_starts = [
        event["data"]["content_block"]
        for event in events
        if event["event"] == "content_block_start"
        and event["data"].get("content_block", {}).get("type") == "tool_use"
    ]
    assert tool_starts
    assert tool_starts[0]["name"] == "get_weather"
    assert any(
        event["event"] == "message_delta"
        and event["data"].get("delta", {}).get("stop_reason") == "tool_use"
        for event in events
    )


def test_claude_streaming_bare_required_tool_name_is_forwarded_as_tool_use():
    model_manager = FakeModelManager()
    service, _captured, fake_post = make_service(
        [
            sse_line({"content": "<tool_call>get_weather</tool_call>"}),
            sse_line({}, "stop"),
        ]
    )
    claude_request = {
        "model": "chatglm",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Use get_weather."}],
        "tools": [CLAUDE_WEATHER_TOOL],
        "stream": True,
    }
    openai_request = convert_claude_to_openai(claude_request, model_manager)

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        events = parse_claude_events(
            stream_openai_to_claude(
                service.stream_openai_completion(openai_request),
                {**claude_request, "_estimator_model": "chatglm"},
                logging.getLogger("test_glm52_service"),
            )
        )

    tool_starts = [
        event["data"]["content_block"]
        for event in events
        if event["event"] == "content_block_start"
        and event["data"].get("content_block", {}).get("type") == "tool_use"
    ]
    assert tool_starts
    assert tool_starts[0]["name"] == "get_weather"
    assert tool_starts[0]["input"] == {}
    assert any(
        event["event"] == "message_delta"
        and event["data"].get("delta", {}).get("stop_reason") == "tool_use"
        for event in events
    )


def test_claude_non_stream_bash_tool_preserves_shell_string_arguments():
    model_manager = FakeModelManager()
    commands = (
        "true",
        "printf '%s\\n' \"hi\"",
        '"./script with spaces.sh" --flag',
    )

    for command in commands:
        service, _captured, fake_post = make_service(
            [
                sse_line(
                    {
                        "content": (
                            "<tool_call>Bash"
                            f"<arg_key>command</arg_key><arg_value>{command}</arg_value>"
                            "</tool_call>"
                        )
                    }
                ),
                sse_line({}, "stop"),
            ]
        )
        claude_request = {
            "model": "chatglm",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Use Bash."}],
            "tools": [CLAUDE_BASH_TOOL],
            "tool_choice": {"type": "tool", "name": "Bash"},
        }
        openai_request = convert_claude_to_openai(claude_request, model_manager)

        with patch("genai_proxy.services.genai.requests.post", fake_post):
            openai_response = service.build_openai_completion(openai_request)

        response = convert_openai_to_claude_response(
            openai_response,
            {**claude_request, "_estimator_model": "chatglm"},
        )
        tool_blocks = [
            block for block in response["content"] if block.get("type") == "tool_use"
        ]
        assert tool_blocks
        assert tool_blocks[0]["name"] == "Bash"
        assert tool_blocks[0]["input"]["command"] == command
        assert isinstance(tool_blocks[0]["input"]["command"], str)
        assert response["stop_reason"] == "tool_use"


def test_claude_streaming_bash_tool_preserves_heredoc_input_json_delta():
    model_manager = FakeModelManager()
    command = "python - <<'PY'\nprint(\"hi\")\nPY"
    service, _captured, fake_post = make_service(
        [
            sse_line(
                {
                    "content": (
                        "<tool_call>Bash"
                        f"<arg_key>command</arg_key><arg_value>{command}</arg_value>"
                        "</tool_call>"
                    )
                }
            ),
            sse_line({}, "stop"),
        ]
    )
    claude_request = {
        "model": "chatglm",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Use Bash."}],
        "tools": [CLAUDE_BASH_TOOL],
        "tool_choice": {"type": "tool", "name": "Bash"},
        "stream": True,
    }
    openai_request = convert_claude_to_openai(claude_request, model_manager)

    with patch("genai_proxy.services.genai.requests.post", fake_post):
        events = parse_claude_events(
            stream_openai_to_claude(
                service.stream_openai_completion(openai_request),
                {**claude_request, "_estimator_model": "chatglm"},
                logging.getLogger("test_glm52_service"),
            )
        )

    assert claude_tool_input_from_events(events, "Bash") == {"command": command}
    assert any(
        event["event"] == "message_delta"
        and event["data"].get("delta", {}).get("stop_reason") == "tool_use"
        for event in events
    )


def test_claude_messages_accepts_system_role_message_from_harness():
    model_manager = FakeModelManager()
    claude_request = {
        "model": "chatglm",
        "max_tokens": 1024,
        "system": [{"type": "text", "text": "Top-level system."}],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Inspect the prompt."}],
            },
            {
                "role": "system",
                "content": "Available agent types for the Agent tool.",
            },
        ],
    }

    openai_request = convert_claude_to_openai(claude_request, model_manager)

    assert openai_request["messages"] == [
        {"role": "system", "content": "Top-level system."},
        {"role": "user", "content": "Inspect the prompt."},
        {"role": "system", "content": "Available agent types for the Agent tool."},
    ]


def test_claude_rejects_non_official_output_config_effort():
    model_manager = FakeModelManager()
    claude_request = {
        "model": "chatglm",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Answer directly."}],
        "output_config": {"effort": "minimal"},
    }

    try:
        convert_claude_to_openai(claude_request, model_manager)
    except ProxyError as exc:
        assert exc.status == 400
        assert "Claude output_config.effort" in exc.message
        assert "minimal" in exc.message
    else:
        raise AssertionError("non-Claude output_config effort did not fail")


def test_claude_model_alias_route_uses_configured_genai_models():
    config = SimpleNamespace(
        claude_haiku_model="deepseek-chat",
        claude_sonnet_model="chatglm",
        claude_opus_model="MiniMax-M1",
    )

    assert map_claude_model_alias("claude-3-haiku", config) == "deepseek-chat"
    assert map_claude_model_alias("claude-3-5-sonnet", config) == "chatglm"
    assert map_claude_model_alias("claude-3-opus", config) == "MiniMax-M1"
    assert map_claude_model_alias("chatglm", config) == "chatglm"


def test_claude_route_preserves_shell_strings_across_target_adapters():
    command = "printf '%s\\n' \"hi\""
    cases = (
        (
            {
                "aiType": "chatglm",
                "aiName": "GLM",
                "descInfo": "GLM 5.2",
                "rootModelName": "Xinference",
                "rootAiType": "xinference",
            },
            (
                "<tool_call>Bash"
                f"<arg_key>command</arg_key><arg_value>{command}</arg_value>"
                "</tool_call>"
            ),
        ),
        (
            {
                "aiType": "MiniMax-M1",
                "aiName": "MiniMax",
                "descInfo": "MiniMax 2.7",
                "rootModelName": "Xinference",
                "rootAiType": "xinference",
            },
            (
                "<minimax:tool_call>"
                '<invoke name="Bash">'
                f'<parameter name="command">{command}</parameter>'
                "</invoke>"
                "</minimax:tool_call>"
            ),
        ),
        (
            {
                "aiType": "deepseek-pro",
                "aiName": "DeepSeek-V4-Pro",
                "descInfo": "DeepSeek V4 Pro",
                "rootModelName": "Xinference",
                "rootAiType": "xinference",
            },
            "<tool_call>"
            + json.dumps(
                {"name": "Bash", "arguments": {"command": command}},
                ensure_ascii=False,
            )
            + "</tool_call>",
        ),
        (
            {
                "aiType": "azure-model",
                "aiName": "Azure Model",
                "descInfo": "Generic OpenAI-compatible model",
                "rootModelName": "Azure",
                "rootAiType": "azure",
            },
            (
                "<tool_call>Bash"
                f"<arg_key>command</arg_key><arg_value>{command}</arg_value>"
                "</tool_call>"
            ),
        ),
    )

    for record, upstream_content in cases:
        model_manager = FakeModelManager(record)
        service, _captured, fake_post = make_service(
            [
                sse_line({"content": upstream_content}),
                sse_line({}, "stop"),
            ],
            record=record,
        )
        claude_request = {
            "model": record["aiType"],
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Use Bash."}],
            "tools": [CLAUDE_BASH_TOOL],
            "tool_choice": {"type": "tool", "name": "Bash"},
        }
        openai_request = convert_claude_to_openai(claude_request, model_manager)

        with patch("genai_proxy.services.genai.requests.post", fake_post):
            openai_response = service.build_openai_completion(openai_request)

        response = convert_openai_to_claude_response(
            openai_response,
            {**claude_request, "_estimator_model": record["aiType"]},
        )
        tool_blocks = [
            block for block in response["content"] if block.get("type") == "tool_use"
        ]
        assert tool_blocks
        assert tool_blocks[0]["name"] == "Bash"
        assert tool_blocks[0]["input"]["command"] == command
        assert isinstance(tool_blocks[0]["input"]["command"], str)


if __name__ == "__main__":
    test_glm52_maps_lower_reasoning_effort_to_max()
    test_openai_reasoning_effort_max_is_preserved_for_glm52()
    test_glm52_openai_reasoning_effort_alias_injects_prompt_without_tools()
    test_deepseek_v4_reasoning_effort_is_normalized_before_prompt_injection()
    test_glm52_openai_non_stream_xml_tool_call_uses_reasoning_high()
    test_glm52_openai_stream_bash_tool_call_uses_default_reasoning_max()
    test_glm52_required_tool_choice_accepts_xml_tool_call()
    test_glm52_tool_choice_none_does_not_return_tool_calls()
    test_glm52_tool_result_turn_returns_final_text()
    test_glm52_native_upstream_tool_call_deltas_are_preserved()
    test_glm52_openai_stream_reasoning_content_passes_through_without_tools()
    test_claude_output_config_effort_maps_to_glm52_openai_reasoning_and_tool_use()
    test_claude_streaming_tool_use_survives_glm52_reasoning_effort()
    test_claude_streaming_bare_required_tool_name_is_forwarded_as_tool_use()
    test_claude_non_stream_bash_tool_preserves_shell_string_arguments()
    test_claude_streaming_bash_tool_preserves_heredoc_input_json_delta()
    test_claude_messages_accepts_system_role_message_from_harness()
    test_claude_rejects_non_official_output_config_effort()
    test_claude_model_alias_route_uses_configured_genai_models()
    test_claude_route_preserves_shell_strings_across_target_adapters()
    print("glm52 service tests passed")
