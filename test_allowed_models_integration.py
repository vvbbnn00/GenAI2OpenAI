import argparse
import base64
import io
import json
import sys

from PIL import Image

from genai_proxy.app import create_app
from genai_proxy.config import AppConfig
from genai_proxy.logging_utils import setup_logging

# This is an opt-in keystore-backed integration runner, not an offline pytest module.
__test__ = False

ALLOWED_MODELS = (
    "deepseek-chat",
    "deepseek-pro",
    "chatglm",
    "qwen-instruct",
    "kimi-k3",
)
REASONING_STREAM_MODELS = {
    "deepseek-chat",
    "deepseek-pro",
    "chatglm",
    "qwen-instruct",
}
CITIES = (
    "Shanghai",
    "Beijing",
    "Tokyo",
    "Hangzhou",
    "Guangzhou",
    "Shenzhen",
    "Nanjing",
    "Chengdu",
    "Suzhou",
    "Wuhan",
)
CONDITIONS = ("sunny", "cloudy", "rainy", "windy", "clear")
BASH_TASKS = (
    "inspect package.json and list dependency names",
    "run npm audit in JSON mode and show the first 200 lines",
    "print the first 80 lines of package.json",
    "list package lock files in the app directory",
    "show the npm scripts from package.json",
    "check installed Next.js dependency metadata",
    "count direct dependencies in package.json",
    "show overridden dependency versions",
    "inspect TypeScript and ESLint versions",
    "check whether sharp is pinned",
)

OPENAI_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

CLAUDE_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get current weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
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
                "command": {"type": "string", "description": "Command to run"},
                "description": {"type": "string", "description": "Short description"},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds",
                },
            },
            "required": ["command"],
        },
    },
}

CLAUDE_BASH_TOOL = {
    "name": "Bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command to run"},
            "description": {"type": "string", "description": "Short description"},
            "timeout": {"type": "integer", "description": "Timeout in milliseconds"},
        },
        "required": ["command"],
    },
}

RESPONSES_STAGE_TOOLS = [
    {
        "type": "function",
        "name": "get_stage_one",
        "description": "Run the mandatory first stage of the current task.",
        "parameters": {
            "type": "object",
            "properties": {
                "marker": {
                    "type": "string",
                    "description": "The exact run marker from the user request",
                }
            },
            "required": ["marker"],
        },
    },
    {
        "type": "function",
        "name": "get_stage_two",
        "description": "Run the second stage after stage one returns its value.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "The value returned by stage one",
                }
            },
            "required": ["value"],
        },
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keystore", default="docker-deploy.keystore")
    parser.add_argument("--models", nargs="+", default=list(ALLOWED_MODELS))
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--kimi-repeat",
        type=int,
        default=1,
        help="Kimi-K3 repetitions; kept separate to limit live test traffic",
    )
    args = parser.parse_args()

    disallowed = [model for model in args.models if model not in ALLOWED_MODELS]
    if disallowed:
        raise SystemExit(
            f"Refusing to test disallowed model(s): {', '.join(disallowed)}"
        )
    if args.repeat < 1 or args.kimi_repeat < 1:
        raise SystemExit("Repeat counts must be positive")

    logger = setup_logging(False)
    app = create_app(
        AppConfig(
            token=None,
            keystore=args.keystore,
            port=0,
            debug=False,
            api_key=None,
            token_check_interval=60,
            claude_haiku_model="chatglm",
            claude_sonnet_model="deepseek-pro",
            claude_opus_model="deepseek-chat",
        ),
        logger,
    )

    failures = []
    try:
        with app.test_client() as client:
            for model in args.models:
                if model.casefold() == "kimi-k3":
                    tests = [
                        ("openai_text", test_openai_text),
                        ("openai_stream_text", test_openai_stream_text),
                        ("openai_tool_call", test_openai_tool_call),
                        ("openai_stream_tool_call", test_openai_stream_tool_call),
                        ("openai_vision", test_openai_vision),
                        ("responses_vision", test_responses_vision),
                        (
                            "responses_multiturn_tool_call",
                            test_responses_multiturn_tool_call,
                        ),
                        ("claude_text", test_claude_text),
                        ("claude_stream_tool_use", test_claude_stream_tool_use),
                        ("claude_vision", test_claude_vision),
                    ]
                else:
                    tests = [
                        ("openai_tool_call", test_openai_tool_call),
                        ("openai_stream_tool_call", test_openai_stream_tool_call),
                        ("openai_bash_tool_call", test_openai_bash_tool_call),
                        (
                            "openai_stream_bash_tool_call",
                            test_openai_stream_bash_tool_call,
                        ),
                        ("openai_tool_result_turn", test_openai_tool_result_turn),
                        ("openai_no_tool_needed", test_openai_no_tool_needed),
                        ("claude_tool_use", test_claude_tool_use),
                        ("claude_stream_tool_use", test_claude_stream_tool_use),
                        ("claude_bash_tool_use", test_claude_bash_tool_use),
                        (
                            "claude_stream_bash_tool_use",
                            test_claude_stream_bash_tool_use,
                        ),
                        ("claude_tool_result_turn", test_claude_tool_result_turn),
                    ]
                    if model.casefold() in REASONING_STREAM_MODELS:
                        tests.extend(
                            (
                                (
                                    "openai_stream_reasoning",
                                    test_openai_stream_reasoning,
                                ),
                                (
                                    "responses_stream_reasoning",
                                    test_responses_stream_reasoning,
                                ),
                            )
                        )
                repeat = (
                    args.kimi_repeat if model.casefold() == "kimi-k3" else args.repeat
                )
                for name, fn in tests:
                    for iteration in range(repeat):
                        label = f"{model}:{name}:run{iteration + 1:02d}"
                        try:
                            fn(client, model, iteration)
                        except Exception as exc:
                            failures.append((label, exc))
                            print(f"[FAIL] {label}: {exc}")
                        else:
                            print(f"[PASS] {label}")
    finally:
        app.extensions["token_manager"].shutdown()

    if failures:
        print("\nFailures:")
        for label, exc in failures:
            print(f"  - {label}: {exc}")
        return 1
    return 0


def test_openai_tool_call(client, model, iteration):
    city = city_for(iteration)
    data = post_json(
        client,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Run {iteration}: use get_weather for {city}.",
                }
            ],
            "tools": [OPENAI_WEATHER_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "max_tokens": 1024,
        },
    )
    tool_call = first_openai_tool_call(data)
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]).get("location")


def test_openai_stream_tool_call(client, model, iteration):
    city = city_for(iteration + 1)
    events = post_stream(
        client,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Stream run {iteration}: use get_weather for {city}.",
                }
            ],
            "tools": [OPENAI_WEATHER_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "stream": True,
            "max_tokens": 1024,
        },
    )
    tool_chunks = [
        tc
        for event in events
        for choice in event.get("choices", [])
        for tc in choice.get("delta", {}).get("tool_calls", []) or []
    ]
    assert tool_chunks, (
        "stream did not contain tool_calls: "
        + json.dumps(events, ensure_ascii=False)[:800]
    )
    assert any(
        choice.get("finish_reason") == "tool_calls"
        for event in events
        for choice in event.get("choices", [])
    )


def test_openai_bash_tool_call(client, model, iteration):
    data = post_json(
        client,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": bash_prompt(iteration)}],
            "tools": [OPENAI_BASH_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "Bash"}},
            "max_tokens": 1024,
        },
    )
    tool_call = first_openai_tool_call(data)
    assert_bash_tool_call(tool_call)


def test_openai_stream_bash_tool_call(client, model, iteration):
    events = post_stream(
        client,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": bash_prompt(iteration + 1)}],
            "tools": [OPENAI_BASH_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "Bash"}},
            "stream": True,
            "max_tokens": 1024,
        },
    )
    tool_chunks = [
        tc
        for event in events
        for choice in event.get("choices", [])
        for tc in choice.get("delta", {}).get("tool_calls", []) or []
    ]
    assert tool_chunks, (
        "stream did not contain Bash tool_calls: "
        + json.dumps(events, ensure_ascii=False)[:800]
    )
    assert any(
        choice.get("finish_reason") == "tool_calls"
        for event in events
        for choice in event.get("choices", [])
    )


def test_openai_tool_result_turn(client, model, iteration):
    city = city_for(iteration + 2)
    condition = CONDITIONS[iteration % len(CONDITIONS)]
    temperature = 18 + (iteration % 11)
    first = post_json(
        client,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Use get_weather for {city}, run {iteration}.",
                }
            ],
            "tools": [OPENAI_WEATHER_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "max_tokens": 1024,
        },
    )
    tool_call = first_openai_tool_call(first)
    second = post_json(
        client,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Use get_weather for {city}, run {iteration}.",
                },
                {
                    "role": "assistant",
                    "content": first["choices"][0]["message"].get("content"),
                    "tool_calls": [tool_call],
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(
                        {
                            "location": city,
                            "temperature": temperature,
                            "condition": condition,
                        }
                    ),
                },
            ],
            "tools": [OPENAI_WEATHER_TOOL],
            "max_tokens": 1024,
        },
    )
    message = second["choices"][0]["message"]
    assert not message.get("tool_calls"), (
        "model called another tool after receiving result"
    )
    assert (message.get("content") or "").strip(), "final answer was empty"


def test_openai_no_tool_needed(client, model, iteration):
    data = post_json(
        client,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Run {iteration}: what is the capital city of France? Answer briefly.",
                }
            ],
            "tools": [OPENAI_WEATHER_TOOL],
            "max_tokens": 1024,
        },
    )
    message = data["choices"][0]["message"]
    assert not message.get("tool_calls"), "unexpected tool call"
    assert "paris" in (message.get("content") or "").lower()


def test_openai_text(client, model, iteration):
    marker = f"K3_OPENAI_TEXT_{iteration}"
    data = post_json(
        client,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": f"Reply with exactly {marker}"}],
            "max_tokens": 128,
        },
    )

    assert (data["choices"][0]["message"].get("content") or "").strip() == marker
    assert (data.get("usage") or {}).get("prompt_tokens", 0) > 0


def test_openai_stream_text(client, model, iteration):
    marker = f"K3_OPENAI_STREAM_{iteration}"
    events = post_stream(
        client,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": f"Reply with exactly {marker}"}],
            "stream": True,
            "max_tokens": 128,
        },
    )
    content = "".join(
        str(choice.get("delta", {}).get("content") or "")
        for event in events
        for choice in event.get("choices", [])
    )

    assert content.strip() == marker


def test_openai_stream_reasoning(client, model, iteration):
    events = post_stream(
        client,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Reasoning stream run {iteration}: calculate 127 * 389 "
                        "and give the final number."
                    ),
                }
            ],
            "reasoning": {"effort": "high"},
            "stream": True,
            "max_tokens": 512,
        },
    )
    reasoning_indices = []
    content = []
    finish_indices = []
    for event_index, event in enumerate(events):
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("reasoning_content"):
                reasoning_indices.append(event_index)
            if delta.get("content"):
                content.append(str(delta["content"]))
            if choice.get("finish_reason") is not None:
                finish_indices.append(event_index)

    assert reasoning_indices, (
        "stream did not contain reasoning_content: "
        + json.dumps(events, ensure_ascii=False)[:800]
    )
    assert finish_indices and reasoning_indices[0] < finish_indices[-1]
    assert "".join(content).strip(), "stream did not contain final answer content"


def test_responses_stream_reasoning(client, model, iteration):
    events = post_stream(
        client,
        "/v1/responses",
        {
            "model": model,
            "input": (
                f"Responses reasoning run {iteration}: calculate 127 * 389 "
                "and give the final number."
            ),
            "reasoning": {"effort": "high"},
            "stream": True,
            "max_output_tokens": 512,
        },
    )
    reasoning_deltas = [
        event
        for event in events
        if event.get("type") == "response.reasoning_text.delta"
    ]
    assert reasoning_deltas, (
        "Responses stream did not contain reasoning_text.delta: "
        + json.dumps(events, ensure_ascii=False)[:800]
    )
    reasoning_item_id = reasoning_deltas[0].get("item_id")
    reasoning_output_index = reasoning_deltas[0].get("output_index")
    assert reasoning_item_id
    assert isinstance(reasoning_output_index, int)
    assert all(
        event.get("item_id") == reasoning_item_id
        and event.get("output_index") == reasoning_output_index
        and event.get("content_index") == 0
        for event in reasoning_deltas
    )
    sequence_numbers = [event.get("sequence_number") for event in events]
    assert sequence_numbers == list(range(len(events)))

    completed = completed_responses_stream(events)
    reasoning_items = [
        item for item in completed.get("output", []) if item.get("type") == "reasoning"
    ]
    assert reasoning_items
    streamed_reasoning = "".join(event.get("delta", "") for event in reasoning_deltas)
    assert reasoning_items[0].get("content") == [
        {"type": "reasoning_text", "text": streamed_reasoning}
    ]
    assert any(
        item.get("type") == "message"
        and any(
            part.get("type") == "output_text" and part.get("text", "").strip()
            for part in item.get("content", [])
        )
        for item in completed.get("output", [])
    )


def test_openai_vision(client, model, iteration):
    image_url = red_image_url()
    data = post_json(
        client,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Vision run {iteration}: what is the dominant color? "
                                "Answer with one English color word."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            "max_tokens": 128,
        },
    )

    message = data["choices"][0]["message"]
    assert "red" in (message.get("content") or "").lower()
    usage = data.get("usage") or {}
    assert usage.get("prompt_tokens", 0) > 0
    assert usage.get("completion_tokens", 0) > 0


def test_responses_vision(client, model, iteration):
    data = post_json(
        client,
        "/v1/responses",
        {
            "model": model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Responses vision run {iteration}: what is the "
                                "dominant color? Answer with one English color word."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": red_image_url(),
                        },
                    ],
                }
            ],
            "max_output_tokens": 128,
        },
    )

    assert "red" in (data.get("output_text") or "").lower()
    assert (data.get("usage") or {}).get("input_tokens", 0) > 0
    assert (data.get("usage") or {}).get("output_tokens", 0) > 0


def test_responses_multiturn_tool_call(client, model, iteration):
    marker = f"K3_MULTITURN_DONE_{iteration}"
    history = []
    for index in range(8):
        history.extend(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Earlier project note {index}.",
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"Recorded note {index}.",
                        }
                    ],
                },
            ]
        )
    task = {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": (
                    "Complete both mandatory stages of this sequential task "
                    f"using the available operations. Start with marker {marker}; "
                    "pass the first stage's returned value into the second stage. "
                    f"After both results, reply with exactly {marker}."
                ),
            }
        ],
    }
    initial_input = [*history, task]
    common = {
        "model": model,
        "instructions": (
            "You are a careful tool-using agent. Continue using available tools "
            "after each result until the current task is complete."
        ),
        "tools": RESPONSES_STAGE_TOOLS,
        "max_output_tokens": 512,
    }

    first_events = post_stream(
        client,
        "/v1/responses",
        {
            **common,
            "input": initial_input,
            "tool_choice": {"type": "function", "name": "get_stage_one"},
            "stream": True,
        },
    )
    first = completed_responses_stream(first_events)
    assert any(
        event.get("type") == "response.reasoning_text.delta" for event in first_events
    ), json.dumps(first_events, ensure_ascii=False)[:800]
    first_call = first_responses_function_call(first)
    assert first_call["name"] == "get_stage_one"

    second_input = [
        *initial_input,
        first_call,
        {
            "type": "function_call_output",
            "call_id": first_call["call_id"],
            "output": "STAGE_ONE_OK. The returned value is beta.",
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "\u200b"}],
        },
    ]
    second_events = post_stream(
        client,
        "/v1/responses",
        {**common, "input": second_input, "stream": True},
    )
    second = completed_responses_stream(second_events)
    assert any(
        event.get("type") == "response.reasoning_text.delta" for event in second_events
    ), json.dumps(second_events, ensure_ascii=False)[:800]
    second_call = first_responses_function_call(second)
    assert second_call["name"] == "get_stage_two", json.dumps(
        second, ensure_ascii=False
    )[:800]
    assert json.loads(second_call["arguments"]).get("value") == "beta", json.dumps(
        second, ensure_ascii=False
    )[:800]

    third = post_json(
        client,
        "/v1/responses",
        {
            **common,
            "input": [
                *second_input,
                second_call,
                {
                    "type": "function_call_output",
                    "call_id": second_call["call_id"],
                    "output": (
                        "STAGE_TWO_OK. Both stages are complete. "
                        f"Reply with exactly {marker}."
                    ),
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "\u200b"}],
                },
            ],
        },
    )

    assert not any(
        item.get("type") == "function_call" for item in third.get("output", [])
    ), json.dumps(third, ensure_ascii=False)[:800]
    assert (third.get("output_text") or "").strip() == marker


def test_claude_vision(client, model, iteration):
    image_url = red_image_url()
    data = post_json(
        client,
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Claude vision run {iteration}: what is the "
                                "dominant color? Answer with one English color word."
                            ),
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_url.partition(",")[2],
                            },
                        },
                    ],
                }
            ],
        },
    )

    text = "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    )
    assert "red" in text.lower()
    assert (data.get("usage") or {}).get("input_tokens", 0) > 0
    assert (data.get("usage") or {}).get("output_tokens", 0) > 0


def test_claude_text(client, model, iteration):
    marker = f"K3_CLAUDE_TEXT_{iteration}"
    data = post_json(
        client,
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": f"Reply with exactly {marker}"}],
        },
    )
    text = "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    )

    assert text.strip() == marker
    assert (data.get("usage") or {}).get("input_tokens", 0) > 0


def test_claude_tool_use(client, model, iteration):
    city = city_for(iteration + 3)
    data = post_json(
        client,
        "/v1/messages",
        claude_tool_request(model, f"Use get_weather for {city}."),
    )
    block = first_claude_tool_use(data)
    assert block["name"] == "get_weather"
    assert block["input"]
    assert data["stop_reason"] == "tool_use"


def test_claude_stream_tool_use(client, model, iteration):
    city = city_for(iteration + 4)
    events = post_claude_stream(
        client,
        "/v1/messages",
        {
            **claude_tool_request(
                model, f"Stream this request: use get_weather for {city}."
            ),
            "stream": True,
        },
    )
    tool_starts = [
        event["data"]["content_block"]
        for event in events
        if event["event"] == "content_block_start"
        and event["data"].get("content_block", {}).get("type") == "tool_use"
    ]
    assert tool_starts, (
        "Claude stream did not start a tool_use block: "
        + json.dumps(events, ensure_ascii=False)[:800]
    )
    assert any(
        event["event"] == "message_delta"
        and event["data"].get("delta", {}).get("stop_reason") == "tool_use"
        for event in events
    )


def test_claude_bash_tool_use(client, model, iteration):
    data = post_json(
        client,
        "/v1/messages",
        claude_bash_tool_request(model, bash_prompt(iteration + 2)),
    )
    block = first_claude_tool_use(data)
    assert_bash_tool_use(block)
    assert data["stop_reason"] == "tool_use"


def test_claude_stream_bash_tool_use(client, model, iteration):
    events = post_claude_stream(
        client,
        "/v1/messages",
        {
            **claude_bash_tool_request(model, bash_prompt(iteration + 3)),
            "stream": True,
        },
    )
    tool_starts = [
        event["data"]["content_block"]
        for event in events
        if event["event"] == "content_block_start"
        and event["data"].get("content_block", {}).get("type") == "tool_use"
    ]
    assert tool_starts, (
        "Claude stream did not start a Bash tool_use block: "
        + json.dumps(events, ensure_ascii=False)[:800]
    )
    assert any(block.get("name") == "Bash" for block in tool_starts)
    assert any(
        event["event"] == "message_delta"
        and event["data"].get("delta", {}).get("stop_reason") == "tool_use"
        for event in events
    )


def test_claude_tool_result_turn(client, model, iteration):
    city = city_for(iteration + 5)
    condition = CONDITIONS[(iteration + 2) % len(CONDITIONS)]
    temperature = 17 + (iteration % 13)
    first = post_json(
        client,
        "/v1/messages",
        claude_tool_request(model, f"Use get_weather for {city}."),
    )
    tool_use = first_claude_tool_use(first)
    second = post_json(
        client,
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": f"Use get_weather for {city}."},
                {"role": "assistant", "content": [tool_use]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": f"{city} is {condition} and {temperature} degrees Celsius.",
                        }
                    ],
                },
            ],
            "tools": [CLAUDE_WEATHER_TOOL],
        },
    )
    assert second["stop_reason"] == "end_turn", json.dumps(second, ensure_ascii=False)[
        :500
    ]
    assert any(
        block.get("type") == "text" and block.get("text", "").strip()
        for block in second["content"]
    ), json.dumps(second, ensure_ascii=False)[:500]
    assert not any(block.get("type") == "tool_use" for block in second["content"]), (
        json.dumps(second, ensure_ascii=False)[:500]
    )


def claude_tool_request(model, text):
    return {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": text}],
        "tools": [CLAUDE_WEATHER_TOOL],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }


def claude_bash_tool_request(model, text):
    return {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": text}],
        "tools": [CLAUDE_BASH_TOOL],
        "tool_choice": {"type": "tool", "name": "Bash"},
    }


def bash_prompt(iteration):
    task = BASH_TASKS[iteration % len(BASH_TASKS)]
    return (
        f"Run {iteration}: use the Bash tool for this Claude Code style request. "
        f"Target project path is f:/onedrive-vercel/app. Task: {task}. "
        "Return a tool call only."
    )


def red_image_url():
    buffer = io.BytesIO()
    Image.new("RGB", (56, 28), (255, 0, 0)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def city_for(iteration):
    return CITIES[iteration % len(CITIES)]


def post_json(client, path, payload):
    response = client.post(path, json=payload)
    if response.status_code != 200:
        raise AssertionError(
            f"HTTP {response.status_code}: {response.get_data(as_text=True)[:500]}"
        )
    data = response.get_json()
    if not data:
        raise AssertionError("empty JSON response")
    assert_no_error_payload(data)
    return data


def post_stream(client, path, payload):
    response = client.post(path, json=payload, buffered=False)
    if response.status_code != 200:
        raise AssertionError(
            f"HTTP {response.status_code}: {response.get_data(as_text=True)[:500]}"
        )
    events = []
    for raw in response.response:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        for line in text.splitlines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                return events
            event = json.loads(data)
            assert_no_error_payload(event)
            events.append(event)
    return events


def post_claude_stream(client, path, payload):
    response = client.post(path, json=payload, buffered=False)
    if response.status_code != 200:
        raise AssertionError(
            f"HTTP {response.status_code}: {response.get_data(as_text=True)[:500]}"
        )

    events = []
    current_event = None
    for raw in response.response:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        for line in text.splitlines():
            if line.startswith("event: "):
                current_event = line[7:].strip()
            elif line.startswith("data: ") and current_event:
                payload = json.loads(line[6:].strip())
                if current_event == "error":
                    raise AssertionError(json.dumps(payload, ensure_ascii=False)[:500])
                assert_no_error_payload(payload)
                events.append({"event": current_event, "data": payload})
                current_event = None
    return events


def first_openai_tool_call(data):
    message = data["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    assert tool_calls, (
        f"no tool_calls in response: {json.dumps(data, ensure_ascii=False)[:500]}"
    )
    return tool_calls[0]


def first_responses_function_call(data):
    for item in data.get("output", []):
        if item.get("type") == "function_call":
            return item
    raise AssertionError(
        "no function_call in response: " + json.dumps(data, ensure_ascii=False)[:800]
    )


def completed_responses_stream(events):
    failures = [event for event in events if event.get("type") == "response.failed"]
    assert not failures, json.dumps(failures, ensure_ascii=False)[:800]
    for event in reversed(events):
        if event.get("type") == "response.completed":
            return event["response"]
    raise AssertionError(
        "no response.completed event: " + json.dumps(events, ensure_ascii=False)[:800]
    )


def first_claude_tool_use(data):
    for block in data.get("content", []):
        if block.get("type") == "tool_use":
            return block
    raise AssertionError(
        f"no tool_use block: {json.dumps(data, ensure_ascii=False)[:500]}"
    )


def assert_bash_tool_call(tool_call):
    assert tool_call["function"]["name"] == "Bash"
    arguments = json.loads(tool_call["function"]["arguments"])
    assert isinstance(arguments.get("command"), str) and arguments["command"].strip()


def assert_bash_tool_use(block):
    assert block["name"] == "Bash"
    assert isinstance(block.get("input"), dict)
    assert (
        isinstance(block["input"].get("command"), str)
        and block["input"]["command"].strip()
    )


def assert_no_error_payload(data):
    if isinstance(data, dict) and data.get("error"):
        raise AssertionError(json.dumps(data, ensure_ascii=False)[:500])

    for choice in data.get("choices", []) if isinstance(data, dict) else []:
        if choice.get("finish_reason") == "error":
            raise AssertionError(json.dumps(data, ensure_ascii=False)[:500])
        delta = choice.get("delta", {})
        message = choice.get("message", {})
        for text in (delta.get("content"), message.get("content")):
            if isinstance(text, str) and text.startswith("[Error]"):
                raise AssertionError(json.dumps(data, ensure_ascii=False)[:500])

    for block in data.get("content", []) if isinstance(data, dict) else []:
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str) and text.startswith("[Error]"):
                raise AssertionError(json.dumps(data, ensure_ascii=False)[:500])


if __name__ == "__main__":
    sys.exit(main())
