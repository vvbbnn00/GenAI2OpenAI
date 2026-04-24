import argparse
import json
import sys

from genai_proxy.app import create_app
from genai_proxy.config import AppConfig
from genai_proxy.logging_utils import setup_logging


ALLOWED_MODELS = ("deepseek-chat", "MiniMax-M1", "chatglm")
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
                "timeout": {"type": "integer", "description": "Timeout in milliseconds"},
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keystore", default="docker-deploy.keystore")
    parser.add_argument("--models", nargs="+", default=list(ALLOWED_MODELS))
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    disallowed = [model for model in args.models if model not in ALLOWED_MODELS]
    if disallowed:
        raise SystemExit(f"Refusing to test disallowed model(s): {', '.join(disallowed)}")

    logger = setup_logging(False)
    app = create_app(
        AppConfig(
            token=None,
            keystore=args.keystore,
            port=0,
            debug=False,
            api_key=None,
            claude_haiku_model="chatglm",
            claude_sonnet_model="MiniMax-M1",
            claude_opus_model="deepseek-chat",
        ),
        logger,
    )

    failures = []
    try:
        with app.test_client() as client:
            for model in args.models:
                for name, fn in (
                    ("openai_tool_call", test_openai_tool_call),
                    ("openai_stream_tool_call", test_openai_stream_tool_call),
                    ("openai_bash_tool_call", test_openai_bash_tool_call),
                    ("openai_stream_bash_tool_call", test_openai_stream_bash_tool_call),
                    ("openai_tool_result_turn", test_openai_tool_result_turn),
                    ("openai_no_tool_needed", test_openai_no_tool_needed),
                    ("claude_tool_use", test_claude_tool_use),
                    ("claude_stream_tool_use", test_claude_stream_tool_use),
                    ("claude_bash_tool_use", test_claude_bash_tool_use),
                    ("claude_stream_bash_tool_use", test_claude_stream_bash_tool_use),
                    ("claude_tool_result_turn", test_claude_tool_result_turn),
                ):
                    for iteration in range(args.repeat):
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
            "messages": [{"role": "user", "content": f"Run {iteration}: use get_weather for {city}."}],
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
            "messages": [{"role": "user", "content": f"Stream run {iteration}: use get_weather for {city}."}],
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
    assert tool_chunks, "stream did not contain tool_calls"
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
    assert tool_chunks, "stream did not contain Bash tool_calls"
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
            "messages": [{"role": "user", "content": f"Use get_weather for {city}, run {iteration}."}],
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
                {"role": "user", "content": f"Use get_weather for {city}, run {iteration}."},
                {
                    "role": "assistant",
                    "content": first["choices"][0]["message"].get("content"),
                    "tool_calls": [tool_call],
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(
                        {"location": city, "temperature": temperature, "condition": condition}
                    ),
                },
            ],
            "tools": [OPENAI_WEATHER_TOOL],
            "max_tokens": 1024,
        },
    )
    message = second["choices"][0]["message"]
    assert not message.get("tool_calls"), "model called another tool after receiving result"
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


def test_claude_tool_use(client, model, iteration):
    city = city_for(iteration + 3)
    data = post_json(client, "/v1/messages", claude_tool_request(model, f"Use get_weather for {city}."))
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
            **claude_tool_request(model, f"Stream this request: use get_weather for {city}."),
            "stream": True,
        },
    )
    tool_starts = [
        event["data"]["content_block"]
        for event in events
        if event["event"] == "content_block_start"
        and event["data"].get("content_block", {}).get("type") == "tool_use"
    ]
    assert tool_starts, "Claude stream did not start a tool_use block"
    assert any(
        event["event"] == "message_delta"
        and event["data"].get("delta", {}).get("stop_reason") == "tool_use"
        for event in events
    )


def test_claude_bash_tool_use(client, model, iteration):
    data = post_json(client, "/v1/messages", claude_bash_tool_request(model, bash_prompt(iteration + 2)))
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
    assert tool_starts, "Claude stream did not start a Bash tool_use block"
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
    first = post_json(client, "/v1/messages", claude_tool_request(model, f"Use get_weather for {city}."))
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
    assert second["stop_reason"] == "end_turn", json.dumps(second, ensure_ascii=False)[:500]
    assert any(
        block.get("type") == "text" and block.get("text", "").strip()
        for block in second["content"]
    ), json.dumps(second, ensure_ascii=False)[:500]
    assert not any(
        block.get("type") == "tool_use" for block in second["content"]
    ), json.dumps(second, ensure_ascii=False)[:500]


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


def city_for(iteration):
    return CITIES[iteration % len(CITIES)]


def post_json(client, path, payload):
    response = client.post(path, json=payload)
    if response.status_code != 200:
        raise AssertionError(f"HTTP {response.status_code}: {response.get_data(as_text=True)[:500]}")
    data = response.get_json()
    if not data:
        raise AssertionError("empty JSON response")
    assert_no_error_payload(data)
    return data


def post_stream(client, path, payload):
    response = client.post(path, json=payload, buffered=False)
    if response.status_code != 200:
        raise AssertionError(f"HTTP {response.status_code}: {response.get_data(as_text=True)[:500]}")
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
        raise AssertionError(f"HTTP {response.status_code}: {response.get_data(as_text=True)[:500]}")

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
    assert tool_calls, f"no tool_calls in response: {json.dumps(data, ensure_ascii=False)[:500]}"
    return tool_calls[0]


def first_claude_tool_use(data):
    for block in data.get("content", []):
        if block.get("type") == "tool_use":
            return block
    raise AssertionError(f"no tool_use block: {json.dumps(data, ensure_ascii=False)[:500]}")


def assert_bash_tool_call(tool_call):
    assert tool_call["function"]["name"] == "Bash"
    arguments = json.loads(tool_call["function"]["arguments"])
    assert isinstance(arguments.get("command"), str) and arguments["command"].strip()


def assert_bash_tool_use(block):
    assert block["name"] == "Bash"
    assert isinstance(block.get("input"), dict)
    assert isinstance(block["input"].get("command"), str) and block["input"]["command"].strip()


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
