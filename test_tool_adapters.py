import json

from genai_proxy.compat.openai import extract_tool_calls, tag_prefix_len
from genai_proxy.errors import ProxyError
from genai_proxy.optimizations import (
    DEEPSEEK_LEGACY_ADAPTER,
    DEEPSEEK_V4_FLASH_ADAPTER,
    DEEPSEEK_V4_PRO_ADAPTER,
    GENERIC_ADAPTER,
    GLM_5_1_ADAPTER,
    GLM_5_2_ADAPTER,
    GLM_ADAPTER,
    KIMI_K3_ADAPTER,
    MINIMAX_ADAPTER,
    select_tool_adapter,
)
from genai_proxy.optimizations.deepseek import (
    DEEPSEEK_V4_REASONING_EFFORT_MAX,
    inject_deepseek_reasoning_prompt,
    inject_deepseek_tool_prompt,
)
from genai_proxy.optimizations.glm import inject_glm_tool_prompt
from genai_proxy.optimizations.kimi import inject_kimi_tool_prompt
from genai_proxy.optimizations.minimax import inject_minimax_tool_prompt
from genai_proxy.reasoning import normalize_reasoning_for_adapter
from genai_proxy.services.genai import _tool_start_tags_for_request

WEATHER_TOOL = {
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

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                },
            },
            "required": ["query"],
        },
    },
}

BASH_TOOL = {
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

GLOB_TOOL = {
    "type": "function",
    "function": {
        "name": "Glob",
        "description": "Find files by glob pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"],
        },
    },
}

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read a file from the local filesystem.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["file_path"],
        },
    },
}

EXEC_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "exec_command",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
}

LIST_MCP_RESOURCES_TOOL = {
    "type": "function",
    "function": {
        "name": "list_mcp_resources",
        "description": "List MCP resources.",
        "parameters": {"type": "object", "properties": {}},
    },
}

GENERIC_STRING_TOOL = {
    "type": "function",
    "function": {
        "name": "run_anything",
        "description": "A generic string argument tool.",
        "parameters": {
            "type": "object",
            "properties": {"payload": {"type": "string"}},
            "required": ["payload"],
        },
    },
}

GENERIC_TRANSCRIPT_TOOL = {
    "type": "function",
    "function": {
        "name": "generic_runner",
        "description": "A generic transcript-style runner.",
        "parameters": {
            "type": "object",
            "properties": {
                "payload": {"type": "string"},
                "description": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["payload"],
        },
    },
}

UNTYPED_ARGUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "untyped_runner",
        "description": "A tool with an untyped argument.",
        "parameters": {
            "type": "object",
            "properties": {"payload": {}},
            "required": ["payload"],
        },
    },
}

UNION_SCHEMA_TOOL = {
    "type": "function",
    "function": {
        "name": "union_runner",
        "description": "A tool with nullable schema types.",
        "parameters": {
            "type": "object",
            "properties": {
                "payload": {"type": ["string", "null"]},
                "count": {"type": ["integer", "null"]},
                "flag": {"type": ["boolean", "null"]},
            },
            "required": ["payload"],
        },
    },
}

ANYOF_SCHEMA_TOOL = {
    "type": "function",
    "function": {
        "name": "anyof_runner",
        "description": "A tool with anyOf nullable schema types.",
        "parameters": {
            "type": "object",
            "properties": {
                "payload": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "count": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            },
            "required": ["payload"],
        },
    },
}


def test_genai_model_record_mapping():
    assert (
        select_tool_adapter(
            "deepseek-chat",
            {"aiName": "DeepSeek-V4-Flash", "descInfo": "最新DeepSeek V4 284B模型"},
        )
        == DEEPSEEK_V4_FLASH_ADAPTER
    )
    assert (
        select_tool_adapter(
            "deepseek-pro",
            {
                "aiName": "DeepSeek-V4-Pro",
                "descInfoEn": "Local deployment of the latest DeepSeeK V4 trillion-parameter model",
                "rootModelName": "Xinference",
            },
        )
        == DEEPSEEK_V4_PRO_ADAPTER
    )
    assert (
        select_tool_adapter(
            "deepseek-v3:671b",
            {"aiName": "DeepSeek V3", "descInfo": "legacy DeepSeek model"},
        )
        == DEEPSEEK_LEGACY_ADAPTER
    )
    assert (
        select_tool_adapter(
            "deepseek-chat",
            {"aiName": "DeepSeek", "rootModelName": "Azure"},
        )
        == GENERIC_ADAPTER
    )
    assert (
        select_tool_adapter(
            "MiniMax-M1",
            {"aiName": "MiniMax", "descInfo": "MiniMax 2.7"},
        )
        == MINIMAX_ADAPTER
    )
    assert (
        select_tool_adapter(
            "chatglm",
            {"aiName": "GLM", "descInfo": "GLM 5.1适合长任务执行"},
        )
        == GLM_5_1_ADAPTER
    )
    assert (
        select_tool_adapter(
            "chatglm",
            {"aiName": "GLM", "descInfo": "GLM 5.2适合长任务执行"},
        )
        == GLM_5_2_ADAPTER
    )
    assert select_tool_adapter("chatglm", None) == GLM_5_2_ADAPTER
    assert (
        select_tool_adapter(
            "chatglm",
            {"aiName": "GLM 5.2", "rootModelName": "Azure"},
        )
        == GENERIC_ADAPTER
    )
    assert (
        select_tool_adapter(
            "kimi-k3",
            {"aiName": "Kimi-K3", "rootModelName": "Xinference"},
        )
        == KIMI_K3_ADAPTER
    )
    assert (
        select_tool_adapter(
            "kimi-k3",
            {"aiName": "Kimi-K3", "rootModelName": "Azure"},
        )
        == GENERIC_ADAPTER
    )
    assert select_tool_adapter("kimi-k3.1") == GENERIC_ADAPTER
    assert GLM_ADAPTER == GLM_5_1_ADAPTER


def test_kimi_official_xtml_tool_call_is_recovered():
    content = (
        "Searching now."
        "<|open|>tools<|sep|>"
        '<|open|>call tool="SEARCH" index="1"<|sep|>'
        '<|open|>argument key="query" type="string"<|sep|>'
        "Kimi K3"
        "<|close|>argument<|sep|>"
        '<|open|>argument key="num_results" type="integer"<|sep|>'
        "3"
        "<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
        "<|close|>tools<|sep|>"
    )

    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[SEARCH_TOOL],
        model="kimi-k3",
        adapter=KIMI_K3_ADAPTER,
    )

    assert remaining == "Searching now."
    assert tool_calls[0]["function"]["name"] == "search"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "query": "Kimi K3",
        "num_results": 3,
    }


def test_kimi_official_xtml_rejects_partial_or_mistyped_calls():
    valid_call = (
        '<|open|>call tool="search" index="1"<|sep|>'
        '<|open|>argument key="query" type="string"<|sep|>'
        "Kimi K3"
        "<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
    )
    invalid_calls = [
        (
            '<|open|>call tool="search" index="2"<|sep|>'
            '<|open|>json type="object"<|sep|>{"query":'
            "<|close|>json<|sep|>"
            "<|close|>call<|sep|>"
        ),
        (
            '<|open|>call tool="search" index="2"<|sep|>'
            '<|open|>argument key="num_results" type="number"<|sep|>'
            "three"
            "<|close|>argument<|sep|>"
            "<|close|>call<|sep|>"
        ),
    ]

    for invalid_call in invalid_calls:
        content = (
            "<|open|>tools<|sep|>"
            f"{valid_call}{invalid_call}"
            "<|close|>tools<|sep|>"
        )
        tool_calls, remaining = extract_tool_calls(
            content,
            tools=[SEARCH_TOOL],
            model="kimi-k3",
            adapter=KIMI_K3_ADAPTER,
        )
        assert tool_calls is None
        assert remaining == content

    content = (
        "<|open|>tools<|sep|>"
        f"{valid_call}"
        "<|close|>tools<|sep|>"
        "<|open|>tools<|sep|>"
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[SEARCH_TOOL],
        model="kimi-k3",
        adapter=KIMI_K3_ADAPTER,
    )
    assert tool_calls is None
    assert remaining == content


def test_kimi_nonofficial_function_expression_is_not_recovered():
    content = 'search(query="Kimi K3")'
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[SEARCH_TOOL],
        model="kimi-k3",
        adapter=KIMI_K3_ADAPTER,
    )

    assert tool_calls is None
    assert remaining == content


def test_kimi_active_tools_use_external_operation_bridge():
    messages = [
        {"role": "user", "content": "Search for Kimi K3."}
    ]

    bridged = inject_kimi_tool_prompt(
        messages, [SEARCH_TOOL], tool_choice="required"
    )

    assert bridged[-1] == messages[-1]
    prompt = bridged[0]["content"]
    assert bridged[0]["role"] == "system"
    assert prompt.startswith("# External operation request")
    assert "<k3_operations>" in prompt
    assert "<k3_action>" in prompt
    assert '"name":"search"' in prompt
    assert "at least one operation request is mandatory" in prompt
    assert "Call-expression schemas" not in prompt
    assert "User request:" not in prompt
    assert "<|open|>" not in prompt


def test_kimi_non_function_tools_are_rejected_before_transport():
    cases = [
        [{"type": "custom", "name": "runner"}],
        [SEARCH_TOOL, {"type": "custom", "name": "runner"}],
    ]
    for tools in cases:
        try:
            inject_kimi_tool_prompt(
                [{"role": "user", "content": "Run the tool."}],
                tools,
            )
        except ProxyError as exc:
            assert exc.code == "unsupported_tool_type"
        else:
            raise AssertionError("Kimi K3 non-function tool was not rejected")


def test_kimi_tool_choice_none_does_not_inject_or_reject():
    messages = [{"role": "user", "content": "Answer directly."}]

    assert (
        inject_kimi_tool_prompt(messages, [SEARCH_TOOL], tool_choice="none")
        == messages
    )


def test_kimi_named_tool_choice_must_match_supplied_tool():
    try:
        inject_kimi_tool_prompt(
            [{"role": "user", "content": "Search."}],
            [SEARCH_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "missing"},
            },
        )
    except ProxyError as exc:
        assert exc.code == "invalid_tool_choice"
    else:
        raise AssertionError("Kimi K3 accepted an unknown named tool choice")


def test_kimi_external_operation_bridge_output_is_recovered():
    cases = [
        (
            (
                "Checking.\n"
                '<k3_action>{"name":"search","arguments":'
                '{"query":"Kimi K3","num_results":3}}</k3_action>'
            ),
            "Checking.",
        ),
        (
            (
                '{"name":"search","arguments":'
                '{"query":"Kimi K3","num_results":3}}'
            ),
            None,
        ),
        (
            (
                "```json\n"
                '{"name":"search","arguments":'
                '{"query":"Kimi K3","num_results":3}}\n'
                "```"
            ),
            None,
        ),
    ]

    for content, expected_remaining in cases:
        tool_calls, remaining = extract_tool_calls(
            content,
            tools=[SEARCH_TOOL],
            model="kimi-k3",
            adapter=KIMI_K3_ADAPTER,
        )

        assert remaining == expected_remaining
        assert tool_calls[0]["function"]["name"] == "search"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {
            "query": "Kimi K3",
            "num_results": 3,
        }


def test_kimi_external_operation_bridge_rejects_unknown_or_invalid_calls():
    cases = [
        '<k3_action>{"name":"unknown","arguments":{}}</k3_action>',
        '<k3_action>{"name":"search","arguments":[]}</k3_action>',
        '<k3_action>{"name":"search","arguments":bad}</k3_action>',
        (
            '<k3_action>{"name":"search","arguments":{},'
            '"extra":true}</k3_action>'
        ),
        (
            '<k3_action>{"name":"search","arguments":{"query":"ok"}}</k3_action>'
            '<k3_action>{"name":"search","arguments":[]}</k3_action>'
        ),
        (
            '<k3_action>{"name":"search","arguments":{"query":"ok"}}</k3_action>'
            '<k3_action>{"name":"search","arguments":'
        ),
    ]

    for content in cases:
        tool_calls, remaining = extract_tool_calls(
            content,
            tools=[SEARCH_TOOL],
            model="kimi-k3",
            adapter=KIMI_K3_ADAPTER,
        )
        assert tool_calls is None
        assert remaining == content


def test_kimi_external_operation_bridge_handles_close_tag_inside_argument():
    content = (
        '<k3_action>{"name":"search","arguments":'
        '{"query":"literal </k3_action> text"}}</k3_action>'
    )

    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[SEARCH_TOOL],
        model="kimi-k3",
        adapter=KIMI_K3_ADAPTER,
    )

    assert remaining is None
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "query": "literal </k3_action> text"
    }


def test_kimi_tool_history_is_serialized_outside_native_tool_fields():
    messages = [
        {"role": "user", "content": "Search for Kimi K3."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query":"Kimi K3","num_results":3}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_search",
            "content": '{"results":["Kimi-K3"]}',
        },
    ]

    bridged = inject_kimi_tool_prompt(messages, [SEARCH_TOOL])

    assert all(message.get("role") != "tool" for message in bridged)
    assert all(not message.get("tool_calls") for message in bridged)
    assistant = next(
        message for message in bridged if message.get("role") == "assistant"
    )
    result = bridged[-1]
    assert "<k3_action>" in assistant["content"]
    assert '"name":"search"' in assistant["content"]
    assert result["role"] == "system"
    assert result["content"].startswith("<k3_result>")
    assert '"id":"call_search"' in result["content"]
    assert '"name":"search"' in result["content"]


def test_kimi_tool_choice_none_history_omits_operation_schemas():
    messages = [
        {"role": "user", "content": "Search."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query":"Kimi K3"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_search",
            "content": "Done.",
        },
    ]

    bridged = inject_kimi_tool_prompt(
        messages,
        [SEARCH_TOOL],
        tool_choice="none",
    )
    prompt = next(
        message["content"]
        for message in bridged
        if str(message.get("content", "")).startswith(
            "# External operation results"
        )
    )

    assert "<k3_operations>" not in prompt
    assert '"name":"search"' not in prompt
    assert "Do not output <k3_action>" in prompt


def test_kimi_reasoning_effort_uses_upstream_default_max():
    for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        assert normalize_reasoning_for_adapter(
            KIMI_K3_ADAPTER,
            {"effort": effort},
        ) == {"effort": "max"}


def test_glm_malformed_tool_call_close_tag():
    content = '<tool_call>{"name": "get_weather", "arguments": {"location": "Shanghai"}}</arg_value>'
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[WEATHER_TOOL],
        model="chatglm",
        adapter=GLM_ADAPTER,
    )
    assert remaining is None
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "location": "Shanghai"
    }


def test_minimax_think_block_is_not_returned_as_content():
    content = (
        "<think>I should use the weather tool.</think>\n"
        '<tool_call>{"name":"get_weather","arguments":{"location":"Shanghai"}}</tool_call>'
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[WEATHER_TOOL],
        model="MiniMax-M1",
        adapter=MINIMAX_ADAPTER,
    )
    assert remaining is None
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "location": "Shanghai"
    }


def test_claude_code_arg_key_tool_call_body_is_recovered():
    content = (
        '<tool_call>Bash<arg_key>command": '
        '"cat f:/onedrive-vercel/app/package.json | head -80", '
        '"description": "Read package.json to understand project structure"</tool_call>'
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="chatglm",
        adapter=GLM_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert tool_calls[0]["function"]["name"] == "Bash"
    assert arguments == {
        "command": "cat f:/onedrive-vercel/app/package.json | head -80",
        "description": "Read package.json to understand project structure",
    }


def test_claude_code_arg_key_tool_call_with_windows_backslashes_is_recovered():
    content = (
        '<tool_call>Bash<arg_key>command": '
        "\"cd 'f:\\onedrive-vercel\\app' && npm audit --json 2>&1 | head -2000\", "
        '"timeout": 60000</tool_call>'
    )
    tool_calls, _ = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="MiniMax-M1",
        adapter=MINIMAX_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert (
        arguments["command"]
        == "cd 'f:\\onedrive-vercel\\app' && npm audit --json 2>&1 | head -2000"
    )
    assert arguments["timeout"] == 60000


def test_claude_code_arg_value_tool_call_body_is_recovered():
    content = (
        "<tool_call>Bash"
        "<arg_key>command<arg_value>npm audit 2>&1</arg_value>"
        "<arg_key>description<arg_value>Check dependencies</arg_value>"
        "<arg_key>timeout<arg_value>60000</arg_value>"
        "</tool_call>"
    )
    tool_calls, _ = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="chatglm",
        adapter=GLM_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert arguments == {
        "command": "npm audit 2>&1",
        "description": "Check dependencies",
        "timeout": 60000,
    }


def test_claude_code_close_only_arg_value_tool_call_body_is_recovered():
    content = (
        "<tool_call>Bash<arg_key>"
        "command head -n 80 f:/onedrive-vercel/app/package.json</arg_value>"
        "description Print first 80 lines of package.json</arg_value>"
        "timeout 5000</arg_value>"
        "</tool_call>"
    )
    tool_calls, _ = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="chatglm",
        adapter=GLM_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert arguments == {
        "command": "head -n 80 f:/onedrive-vercel/app/package.json",
        "description": "Print first 80 lines of package.json",
        "timeout": 5000,
    }


def test_glm_official_arg_key_tool_call_is_recovered():
    content = (
        "<tool_call>Bash"
        "<arg_key>command</arg_key><arg_value>ls -la</arg_value>"
        "<arg_key>timeout</arg_key><arg_value>60000</arg_value>"
        "</tool_call>"
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="chatglm",
        adapter=GLM_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments == {"command": "ls -la", "timeout": 60000}


def test_glm_arg_value_shell_command_preserves_trailing_quote():
    content = (
        "<tool_call>Bash"
        "<arg_key>command</arg_key><arg_value>printf '%s\\n' \"hi\"</arg_value>"
        "</tool_call>"
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="chatglm",
        adapter=GLM_5_2_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments["command"] == "printf '%s\\n' \"hi\""


def test_glm_malformed_reversed_arg_value_tool_call_is_recovered():
    content = (
        "<tool_call>exec_command"
        "<arg_value>cmd</arg_key><arg_value>cd /tmp && rg --files | head</arg_value>"
        "</tool_call>"
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[EXEC_COMMAND_TOOL],
        model="chatglm",
        adapter=GLM_5_2_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert tool_calls[0]["function"]["name"] == "exec_command"
    assert arguments == {"cmd": "cd /tmp && rg --files | head"}


def test_glm_malformed_reversed_shell_command_preserves_trailing_quote():
    content = (
        "<tool_call>exec_command"
        '<arg_value>cmd</arg_key><arg_value>python -c "print(\\"hi\\")"</arg_value>'
        "</tool_call>"
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[EXEC_COMMAND_TOOL],
        model="chatglm",
        adapter=GLM_5_2_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments["cmd"] == 'python -c "print(\\"hi\\")"'


def test_glm_string_typed_shell_arguments_preserve_literal_values():
    commands = (
        "true",
        "123",
        "null",
        '"./script with spaces.sh" --flag',
    )
    cases = (
        (BASH_TOOL, "Bash", "command"),
        (EXEC_COMMAND_TOOL, "exec_command", "cmd"),
    )

    for tool, tool_name, argument_name in cases:
        for command in commands:
            contents = (
                (
                    f"<tool_call>{tool_name}"
                    f"<arg_key>{argument_name}</arg_key><arg_value>{command}</arg_value>"
                    "</tool_call>"
                ),
                (
                    f"<tool_call>{tool_name}"
                    f"<arg_value>{argument_name}</arg_key><arg_value>{command}</arg_value>"
                    "</tool_call>"
                ),
            )
            for content in contents:
                tool_calls, remaining = extract_tool_calls(
                    content,
                    tools=[tool],
                    model="chatglm",
                    adapter=GLM_5_2_ADAPTER,
                )
                arguments = json.loads(tool_calls[0]["function"]["arguments"])
                assert remaining is None
                assert arguments[argument_name] == command
                assert isinstance(arguments[argument_name], str)


def test_glm_close_only_shell_command_preserves_leading_quote():
    content = (
        '<tool_call>Bash<arg_key>command "./script with spaces.sh" --flag</arg_value>'
        "</tool_call>"
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="chatglm",
        adapter=GLM_5_2_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments["command"] == '"./script with spaces.sh" --flag'


def test_string_arguments_are_preserved_for_unknown_tool_names():
    payloads = (
        "true",
        "123",
        "null",
        '"./script with spaces.sh" --flag',
        "printf '%s\\n' \"hi\"",
    )

    for payload in payloads:
        cases = (
            (
                "<tool_call>run_anything"
                f"<arg_key>payload</arg_key><arg_value>{payload}</arg_value>"
                "</tool_call>"
            ),
            (
                "<tool_call>run_anything"
                f"<arg_value>payload</arg_key><arg_value>{payload}</arg_value>"
                "</tool_call>"
            ),
            (
                "<minimax:tool_call>"
                '<invoke name="run_anything">'
                f'<parameter name="payload">{payload}</parameter>'
                "</invoke>"
                "</minimax:tool_call>"
            ),
            f"run_anything payload: {payload}",
        )

        for content in cases:
            tool_calls, remaining = extract_tool_calls(
                content,
                tools=[GENERIC_STRING_TOOL],
                model="chatglm",
                adapter=GLM_5_2_ADAPTER,
            )
            arguments = json.loads(tool_calls[0]["function"]["arguments"])
            assert remaining is None
            assert arguments["payload"] == payload
            assert isinstance(arguments["payload"], str)


def test_inline_heredoc_arguments_are_preserved_for_unknown_tool_names():
    expected = "python - <<'PY'\nprint(\"hi\")\nPY"
    content = "run_anything payload: " + expected
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[GENERIC_STRING_TOOL],
        model="chatglm",
        adapter=GLM_5_2_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments["payload"] == expected


def test_transcript_input_maps_to_single_required_argument_without_tool_name_special_case():
    content = "generic_runner Run arbitrary payload\nIN\nprintf '%s\\n' \"hi\"\nOUT\n"
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[GENERIC_TRANSCRIPT_TOOL],
        model="chatglm",
        adapter=GLM_5_2_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments == {
        "payload": "printf '%s\\n' \"hi\"",
        "description": "Run arbitrary payload",
    }


def test_xml_untyped_arguments_default_to_literal_strings():
    payloads = ("true", "123", '"./script with spaces.sh" --flag')
    for payload in payloads:
        cases = (
            (
                "<tool_call>untyped_runner"
                f"<arg_key>payload</arg_key><arg_value>{payload}</arg_value>"
                "</tool_call>"
            ),
            (
                "<minimax:tool_call>"
                '<invoke name="untyped_runner">'
                f'<parameter name="payload">{payload}</parameter>'
                "</invoke>"
                "</minimax:tool_call>"
            ),
        )

        for content in cases:
            tool_calls, remaining = extract_tool_calls(
                content,
                tools=[UNTYPED_ARGUMENT_TOOL],
                model="chatglm",
                adapter=GLM_5_2_ADAPTER,
            )
            arguments = json.loads(tool_calls[0]["function"]["arguments"])
            assert remaining is None
            assert arguments["payload"] == payload
            assert isinstance(arguments["payload"], str)


def test_nullable_schema_types_are_normalized_before_parsing():
    content = (
        "<tool_call>union_runner"
        "<arg_key>payload</arg_key><arg_value>true</arg_value>"
        "<arg_key>count</arg_key><arg_value>123</arg_value>"
        "<arg_key>flag</arg_key><arg_value>true</arg_value>"
        "</tool_call>"
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[UNION_SCHEMA_TOOL],
        model="chatglm",
        adapter=GLM_5_2_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments == {"payload": "true", "count": 123, "flag": True}


def test_json_tool_call_paths_preserve_explicit_json_argument_types():
    payload = '{"payload": true, "count": "123", "flag": "true"}'
    cases = (
        f'<tool_call>{{"name": "union_runner", "arguments": {payload}}}</tool_call>',
        f'<tool_call>prefix {{"name": "union_runner", "arguments": {payload}}} suffix</tool_call>',
        f"<tool_call><name>union_runner</name><arguments>{payload}</arguments></tool_call>",
    )

    for content in cases:
        tool_calls, remaining = extract_tool_calls(
            content,
            tools=[UNION_SCHEMA_TOOL],
            model="chatglm",
            adapter=GLM_5_2_ADAPTER,
        )
        arguments = json.loads(tool_calls[0]["function"]["arguments"])
        assert remaining is None
        assert arguments == {"payload": True, "count": "123", "flag": "true"}


def test_anyof_schema_types_are_used_for_text_arguments_only():
    text_content = (
        "<tool_call>anyof_runner"
        "<arg_key>payload</arg_key><arg_value>true</arg_value>"
        "<arg_key>count</arg_key><arg_value>123</arg_value>"
        "</tool_call>"
    )
    tool_calls, remaining = extract_tool_calls(
        text_content,
        tools=[ANYOF_SCHEMA_TOOL],
        model="chatglm",
        adapter=GLM_5_2_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments == {"payload": "true", "count": 123}

    json_content = (
        '<tool_call>{"name": "anyof_runner", "arguments": '
        '{"payload": true, "count": "123"}}</tool_call>'
    )
    tool_calls, remaining = extract_tool_calls(
        json_content,
        tools=[ANYOF_SCHEMA_TOOL],
        model="chatglm",
        adapter=GLM_5_2_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments == {"payload": True, "count": "123"}


def test_glm_bare_malformed_reversed_arg_value_tool_call_is_recovered():
    content = "exec_command<arg_value>cmd</arg_key><arg_value>pwd && git status --short"
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[EXEC_COMMAND_TOOL],
        model="chatglm",
        adapter=GLM_5_2_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert tool_calls[0]["function"]["name"] == "exec_command"
    assert arguments == {"cmd": "pwd && git status --short"}


def test_glm_empty_argument_tool_call_is_recovered():
    tool_calls, remaining = extract_tool_calls(
        "<tool_call>list_mcp_resources</tool_call>",
        tools=[LIST_MCP_RESOURCES_TOOL],
        model="chatglm",
        adapter=GLM_5_2_ADAPTER,
    )

    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert tool_calls[0]["function"]["name"] == "list_mcp_resources"
    assert arguments == {}


def test_glm_bare_required_tool_call_is_forwarded_with_empty_arguments():
    tool_calls, remaining = extract_tool_calls(
        "<tool_call>exec_command</tool_call>",
        tools=[EXEC_COMMAND_TOOL],
        model="chatglm",
        adapter=GLM_5_2_ADAPTER,
    )

    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert tool_calls[0]["function"]["name"] == "exec_command"
    assert arguments == {}


def test_minimax_official_tool_call_is_recovered():
    content = (
        "<think>\nNeed a shell check.\n</think>\n"
        "<minimax:tool_call>\n"
        '<invoke name="Bash">\n'
        '<parameter name="command">ls -la</parameter>\n'
        '<parameter name="timeout">60000</parameter>\n'
        "</invoke>\n"
        "</minimax:tool_call>"
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="MiniMax-M1",
        adapter=MINIMAX_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments == {"command": "ls -la", "timeout": 60000}


def test_minimax_xml_shell_command_preserves_trailing_quote():
    content = (
        "<minimax:tool_call>\n"
        '<invoke name="Bash">\n'
        '<parameter name="command">printf \'%s\\n\' "hi"</parameter>\n'
        "</invoke>\n"
        "</minimax:tool_call>"
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="MiniMax-M1",
        adapter=MINIMAX_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments["command"] == "printf '%s\\n' \"hi\""


def test_minimax_string_typed_shell_arguments_preserve_literal_values():
    commands = (
        "true",
        "123",
        "null",
        '"./script with spaces.sh" --flag',
    )

    for command in commands:
        content = (
            "<minimax:tool_call>\n"
            '<invoke name="Bash">\n'
            f'<parameter name="command">{command}</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        tool_calls, remaining = extract_tool_calls(
            content,
            tools=[BASH_TOOL],
            model="MiniMax-M1",
            adapter=MINIMAX_ADAPTER,
        )
        arguments = json.loads(tool_calls[0]["function"]["arguments"])
        assert remaining is None
        assert arguments["command"] == command
        assert isinstance(arguments["command"], str)


def test_deepseek_v4_dsml_tool_calls_are_recovered():
    content = (
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="Read">\n'
        '<｜DSML｜parameter name="file_path" string="true">f:\\onedrive-vercel\\app\\package.json</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="limit" string="false">80</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>"
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[READ_TOOL],
        model="deepseek-pro",
        adapter=DEEPSEEK_V4_PRO_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert tool_calls[0]["function"]["name"] == "Read"
    assert arguments == {
        "file_path": "f:\\onedrive-vercel\\app\\package.json",
        "limit": 80,
    }


def test_deepseek_fallback_json_shell_command_preserves_backslash_escapes():
    content = (
        r"""<tool_call>{"name": "Bash", "arguments": {"command": """
        r""""printf '%s\\n' \"hi\""}}}</tool_call>"""
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="deepseek-pro",
        adapter=DEEPSEEK_V4_PRO_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments["command"] == "printf '%s\\n' \"hi\""


def test_deepseek_fallback_valid_json_preserves_explicit_json_argument_types():
    content = (
        '<tool_call>{"name": "union_runner", "arguments": '
        '{"payload": true, "count": "123", "flag": "true"}}</tool_call>'
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[UNION_SCHEMA_TOOL],
        model="deepseek-pro",
        adapter=DEEPSEEK_V4_PRO_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments == {"payload": True, "count": "123", "flag": "true"}


def test_deepseek_fallback_valid_json_uses_anyof_schema_name_but_preserves_types():
    content = (
        '<tool_call>{"name": "ANYOF_RUNNER", "arguments": '
        '{"payload": true, "count": "123"}}</tool_call>'
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[ANYOF_SCHEMA_TOOL],
        model="deepseek-pro",
        adapter=DEEPSEEK_V4_PRO_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert tool_calls[0]["function"]["name"] == "anyof_runner"
    assert arguments == {"payload": True, "count": "123"}


def test_deepseek_fallback_xml_arguments_preserve_explicit_json_types_and_canonical_name():
    content = (
        "<tool_call>"
        "<name>ANYOF_RUNNER</name>"
        '<arguments>{"payload": true, "count": "123"}</arguments>'
        "</tool_call>"
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[ANYOF_SCHEMA_TOOL],
        model="deepseek-pro",
        adapter=DEEPSEEK_V4_PRO_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert tool_calls[0]["function"]["name"] == "anyof_runner"
    assert arguments == {"payload": True, "count": "123"}


def test_json_tool_call_with_shell_regex_backslashes_is_recovered():
    content = (
        r"""<tool_call>{"name": "Bash", "arguments": {"command": """
        r""""cat \"f:/onedrive-vercel/app/package.json\" | grep -E '^\s+\"[^\"]+\":' | wc -l", """
        r""""description": "Count direct dependencies"}}</tool_call>"""
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="MiniMax-M1",
        adapter=MINIMAX_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments["command"] == (
        'cat "f:/onedrive-vercel/app/package.json" | grep -E \'^\\s+"[^"]+":\' | wc -l'
    )
    assert arguments["description"] == "Count direct dependencies"


def test_jsonish_tool_call_with_unescaped_command_quotes_is_recovered():
    content = (
        '<tool_call>{"name": "Bash", "arguments": {"command": '
        '"find f:/onedrive-vercel/app -maxdepth 2 -name "package-lock.json" '
        '-o -name "yarn.lock" -o -name "pnpm-lock.yaml" -o -name "bun.lockb""'
        "}}</arg_value></tool_call>"
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="chatglm",
        adapter=GLM_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments["command"] == (
        'find f:/onedrive-vercel/app -maxdepth 2 -name "package-lock.json" '
        '-o -name "yarn.lock" -o -name "pnpm-lock.yaml" -o -name "bun.lockb"'
    )


def test_bare_claude_code_arg_key_tool_call_is_recovered_for_non_streaming():
    content = 'I will inspect it.\nBash<arg_key>command": "npm audit 2>&1"'
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="chatglm",
        adapter=GLM_ADAPTER,
    )
    assert remaining == "I will inspect it."
    assert tool_calls[0]["function"]["name"] == "Bash"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "command": "npm audit 2>&1"
    }


def test_inline_shell_command_uses_schema_and_preserves_shell_quotes():
    cases = (
        ("Bash command: true", "command", "true"),
        (
            'Bash command: "./script with spaces.sh" --flag',
            "command",
            '"./script with spaces.sh" --flag',
        ),
        (
            'Bash command: "./script with spaces.sh" "arg value"',
            "command",
            '"./script with spaces.sh" "arg value"',
        ),
    )

    for content, key, expected in cases:
        tool_calls, remaining = extract_tool_calls(
            content,
            tools=[BASH_TOOL],
            model="chatglm",
            adapter=GLM_5_2_ADAPTER,
        )
        arguments = json.loads(tool_calls[0]["function"]["arguments"])
        assert remaining is None
        assert arguments[key] == expected
        assert isinstance(arguments[key], str)


def test_inline_non_shell_string_arguments_still_unquote_jsonish_values():
    content = 'Glob pattern: "**/package.json" path: "src"'
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[GLOB_TOOL],
        model="chatglm",
        adapter=GLM_5_2_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments == {"pattern": "**/package.json", "path": "src"}


def test_inline_shell_heredoc_command_collects_body():
    expected = "python - <<'PY'\nprint(\"hi\")\nPY"
    content = "Bash command: " + expected
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="chatglm",
        adapter=GLM_5_2_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments["command"] == expected


def test_claude_code_transcript_out_marker_without_trailing_newline_is_removed():
    content = "Bash\nIN\nprintf '%s\\n' \"hi\"\nOUT\n"
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="chatglm",
        adapter=GLM_5_2_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert remaining is None
    assert arguments["command"] == "printf '%s\\n' \"hi\""


def test_mixed_claude_code_transcript_and_xml_tool_calls_are_recovered():
    content = (
        "我将检查项目依赖是否存在安全漏洞。\n\n"
        "Bash\n"
        "IN\n"
        "cd f:/onedrive-vercel && ls app/package.json package.json 2>/dev/null"
        '<tool_call>description": "Find package.json files\n\n'
        "OUT\n"
        "Exit code 1\n"
        '<tool_call>Bash<arg_key>command": "cd f:/onedrive-vercel && ls -la</arg_value>'
        'description": "List project root files</arg_value></tool_call>'
        '<tool_call>Glob<arg_key>pattern": "**/package.json</arg_value>'
        '<arg_key>path": "f:/onedrive-vercel</arg_value></tool_call>'
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[BASH_TOOL, GLOB_TOOL],
        model="chatglm",
        adapter=GLM_ADAPTER,
    )

    assert remaining == "我将检查项目依赖是否存在安全漏洞。"
    assert [call["function"]["name"] for call in tool_calls] == ["Bash", "Bash", "Glob"]
    first_args = json.loads(tool_calls[0]["function"]["arguments"])
    second_args = json.loads(tool_calls[1]["function"]["arguments"])
    third_args = json.loads(tool_calls[2]["function"]["arguments"])
    assert first_args == {
        "command": "cd f:/onedrive-vercel && ls app/package.json package.json 2>/dev/null"
    }
    assert second_args["command"] == "cd f:/onedrive-vercel && ls -la"
    assert second_args["description"] == "List project root files"
    assert third_args == {"pattern": "**/package.json", "path": "f:/onedrive-vercel"}


def test_deepseek_mixed_transcript_and_dsml_prefers_dsml_calls():
    content = (
        "我来检查这个项目的依赖安全问题。\n\n"
        "Bash List project root files\n"
        "IN\n"
        "ls -la\n\n"
        "OUT\n"
        "total 192\n"
        'Globpattern: "**/package.json"\n'
        "Found 101 files\n"
        "<｜DSML｜function_calls>\n"
        '<｜DSML｜invoke name="Read">\n'
        '<｜DSML｜parameter name="file_path" string="true">f:\\onedrive-vercel\\app\\package.json</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        '<｜DSML｜invoke name="Bash">\n'
        '<｜DSML｜parameter name="command" string="true">cd f:/onedrive-vercel/app && npm audit --json 2>&1</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="timeout" string="false">60000</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜function_calls>"
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[BASH_TOOL, GLOB_TOOL, READ_TOOL],
        model="deepseek-chat",
        adapter=DEEPSEEK_V4_FLASH_ADAPTER,
    )
    assert [call["function"]["name"] for call in tool_calls] == ["Read", "Bash"]
    read_args = json.loads(tool_calls[0]["function"]["arguments"])
    bash_args = json.loads(tool_calls[1]["function"]["arguments"])
    assert read_args == {"file_path": "f:\\onedrive-vercel\\app\\package.json"}
    assert bash_args == {
        "command": "cd f:/onedrive-vercel/app && npm audit --json 2>&1",
        "timeout": 60000,
    }
    assert "<｜DSML｜function_calls>" not in (remaining or "")


def test_streaming_detection_keeps_claude_code_tool_name_prefix():
    tags = _tool_start_tags_for_request(GLM_ADAPTER, [BASH_TOOL])
    assert "Bash<arg_key>" in tags
    assert "Bash <arg_key>" in tags
    assert max(tag_prefix_len("Bash", tag) for tag in tags) == len("Bash")
    assert max(tag_prefix_len("Bash ", tag) for tag in tags) == len("Bash ")


def test_tool_result_turn_allows_additional_tools_by_default():
    messages = [
        {"role": "user", "content": "Use get_weather for Shanghai."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "Shanghai"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather",
            "content": "Shanghai is sunny.",
        },
    ]

    minimax_messages = inject_minimax_tool_prompt(messages, [WEATHER_TOOL])
    assert minimax_messages[-1]["content"].startswith(
        "<response>Shanghai is sunny.</response>"
    )
    assert (
        "Only call another tool if the current result is genuinely insufficient"
        in minimax_messages[-1]["content"]
    )

    deepseek_messages = inject_deepseek_tool_prompt(
        messages,
        [WEATHER_TOOL],
        adapter=DEEPSEEK_V4_FLASH_ADAPTER,
    )
    assert (
        deepseek_messages[-1]["content"]
        == "<tool_result>Shanghai is sunny.</tool_result>"
    )

    glm52_messages = inject_glm_tool_prompt(
        messages,
        [WEATHER_TOOL],
        adapter=GLM_5_2_ADAPTER,
    )
    assert "# Tools" in glm52_messages[0]["content"]
    assert "<tools>" in glm52_messages[0]["content"]
    assert (
        "This turn must end with final assistant text only"
        not in glm52_messages[0]["content"]
    )
    assert "Do not call any tool again" not in glm52_messages[-1]["content"]


def test_official_prompt_shapes_are_not_mixed_between_model_versions():
    messages = [{"role": "user", "content": "What's the weather in Beijing?"}]

    deepseek_v4_messages = inject_deepseek_tool_prompt(
        messages,
        [WEATHER_TOOL],
        adapter=DEEPSEEK_V4_PRO_ADAPTER,
    )
    deepseek_v4_prompt = deepseek_v4_messages[0]["content"]
    assert "### Available Tool Schemas" in deepseek_v4_prompt
    assert "<｜DSML｜tool_calls>" in deepseek_v4_prompt
    assert "<｜DSML｜function_calls>" not in deepseek_v4_prompt
    assert "<functions>" not in deepseek_v4_prompt

    deepseek_legacy_messages = inject_deepseek_tool_prompt(
        messages,
        [WEATHER_TOOL],
        adapter=DEEPSEEK_LEGACY_ADAPTER,
    )
    deepseek_legacy_prompt = deepseek_legacy_messages[0]["content"]
    assert "<｜DSML｜function_calls>" in deepseek_legacy_prompt
    assert "### Available Tool Schemas" not in deepseek_legacy_prompt

    minimax_prompt = inject_minimax_tool_prompt(messages, [WEATHER_TOOL])[0]["content"]
    assert minimax_prompt.startswith("# Tools\nYou may call one or more tools")
    assert "<minimax:tool_call>" in minimax_prompt
    assert '<tool>{"name": "get_weather"' in minimax_prompt
    assert "Rules:" not in minimax_prompt

    glm51_prompt = inject_glm_tool_prompt(
        messages,
        [WEATHER_TOOL],
        adapter=GLM_5_1_ADAPTER,
    )[0]["content"]
    assert glm51_prompt.startswith("# Tools\n\nYou may call one or more functions")
    assert "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>" in glm51_prompt
    assert "Reasoning Effort:" not in glm51_prompt
    assert "<minimax:tool_call>" not in glm51_prompt
    assert "Rules:" not in glm51_prompt

    glm52_prompt = inject_glm_tool_prompt(
        messages,
        [WEATHER_TOOL],
        adapter=GLM_5_2_ADAPTER,
    )[0]["content"]
    assert glm52_prompt.startswith("Reasoning Effort: Max\n\n# Tools\n\n")
    assert "<tools>" in glm52_prompt
    assert "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>" in glm52_prompt
    assert "<|system|>" not in glm52_prompt
    assert "<minimax:tool_call>" not in glm52_prompt
    assert "<｜DSML｜tool_calls>" not in glm52_prompt
    assert "Rules:" not in glm52_prompt


def test_glm52_reasoning_effort_prompt_mapping():
    messages = [{"role": "user", "content": "What's the weather in Beijing?"}]

    high_prompt = inject_glm_tool_prompt(
        messages,
        [WEATHER_TOOL],
        adapter=GLM_5_2_ADAPTER,
        reasoning_config={"effort": "high"},
    )[0]["content"]
    assert high_prompt.startswith("Reasoning Effort: High\n\n# Tools\n\n")

    max_prompt = inject_glm_tool_prompt(
        messages,
        [WEATHER_TOOL],
        adapter=GLM_5_2_ADAPTER,
        reasoning_config={"effort": "max"},
    )[0]["content"]
    assert max_prompt.startswith("Reasoning Effort: Max\n\n# Tools\n\n")

    other_prompt = inject_glm_tool_prompt(
        messages,
        [WEATHER_TOOL],
        adapter=GLM_5_2_ADAPTER,
        reasoning_config={"effort": "none"},
    )[0]["content"]
    assert other_prompt.startswith("Reasoning Effort: Max\n\n# Tools\n\n")


def test_glm52_prompt_filters_template_internal_tool_fields():
    messages = [{"role": "user", "content": "Run a command."}]
    strict_tool = {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Run a shell command.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }
    deferred_tool = {
        "type": "function",
        "function": {
            "name": "deferred_tool",
            "description": "Should not be rendered yet.",
            "defer_loading": True,
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }

    glm52_prompt = inject_glm_tool_prompt(
        messages,
        [strict_tool, deferred_tool],
        adapter=GLM_5_2_ADAPTER,
    )[0]["content"]
    assert '"name": "exec_command"' in glm52_prompt
    assert '"parameters"' in glm52_prompt
    assert '"name": "deferred_tool"' not in glm52_prompt
    assert "strict" not in glm52_prompt
    assert "defer_loading" not in glm52_prompt


def test_deepseek_v4_prompt_matches_hf_encoding_test_shape():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Beijing?"},
    ]
    prompt = inject_deepseek_tool_prompt(
        messages,
        [WEATHER_TOOL, SEARCH_TOOL],
        adapter=DEEPSEEK_V4_PRO_ADAPTER,
    )[0]["content"]

    assert prompt.startswith("You are a helpful assistant.\n\n## Tools\n\n")
    assert (
        'You have access to a set of tools to help answer the user\'s question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:'
        in prompt
    )
    assert (
        "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response."
        in prompt
    )
    assert (
        "Otherwise, output directly after </think> with tool calls or final response."
        in prompt
    )
    assert "### Available Tool Schemas" in prompt
    assert '"name": "get_weather"' in prompt
    assert '"name": "search"' in prompt
    assert (
        "You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls."
        in prompt
    )
    assert "<functions>" not in prompt
    assert "function_calls" not in prompt


def test_deepseek_v4_reasoning_effort_matches_official_prefix_placement():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Solve this carefully."},
    ]

    max_prompt = inject_deepseek_tool_prompt(
        messages,
        [WEATHER_TOOL],
        adapter=DEEPSEEK_V4_PRO_ADAPTER,
        reasoning_config={"effort": "max"},
    )[0]["content"]
    assert max_prompt.startswith(
        DEEPSEEK_V4_REASONING_EFFORT_MAX
        + "You are a helpful assistant.\n\n## Tools\n\n"
    )

    high_messages = inject_deepseek_reasoning_prompt(
        [{"role": "user", "content": "Solve this carefully."}],
        {"effort": "high"},
        adapter=DEEPSEEK_V4_PRO_ADAPTER,
    )
    assert high_messages == [{"role": "user", "content": "Solve this carefully."}]

    max_messages = inject_deepseek_reasoning_prompt(
        [{"role": "user", "content": "Solve this carefully."}],
        {"effort": "max"},
        adapter=DEEPSEEK_V4_PRO_ADAPTER,
    )
    assert max_messages[0] == {
        "role": "system",
        "content": DEEPSEEK_V4_REASONING_EFFORT_MAX,
    }
    assert max_messages[1] == {"role": "user", "content": "Solve this carefully."}


def test_history_tool_calls_render_in_each_official_adapter_format():
    messages = [
        {"role": "user", "content": "Use get_weather for Shanghai."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "Shanghai"}),
                    },
                }
            ],
        },
    ]

    deepseek_messages = inject_deepseek_tool_prompt(
        messages,
        [WEATHER_TOOL],
        adapter=DEEPSEEK_V4_FLASH_ADAPTER,
    )
    assert "<｜DSML｜tool_calls>" in deepseek_messages[-1]["content"]
    assert "<｜DSML｜function_calls>" not in deepseek_messages[-1]["content"]

    minimax_messages = inject_minimax_tool_prompt(messages, [WEATHER_TOOL])
    assert "<minimax:tool_call>" in minimax_messages[-1]["content"]
    assert '<invoke name="get_weather">' in minimax_messages[-1]["content"]

    glm_messages = inject_glm_tool_prompt(messages, [WEATHER_TOOL])
    assert (
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Shanghai</arg_value></tool_call>"
        in glm_messages[-1]["content"]
    )


def test_required_tool_choice_still_allows_additional_tool_calls():
    messages = [
        {"role": "user", "content": "Use get_weather for Shanghai."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "Shanghai"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather",
            "content": "Shanghai is sunny.",
        },
    ]
    tool_choice = {"type": "function", "function": {"name": "get_weather"}}
    minimax_messages = inject_minimax_tool_prompt(
        messages, [WEATHER_TOOL], tool_choice=tool_choice
    )
    assert minimax_messages[-1]["content"].startswith(
        "<response>Shanghai is sunny.</response>"
    )
    assert (
        "Only call another tool if the current result is genuinely insufficient"
        in minimax_messages[-1]["content"]
    )


if __name__ == "__main__":
    test_genai_model_record_mapping()
    test_kimi_official_xtml_tool_call_is_recovered()
    test_kimi_official_xtml_rejects_partial_or_mistyped_calls()
    test_kimi_nonofficial_function_expression_is_not_recovered()
    test_kimi_active_tools_use_external_operation_bridge()
    test_kimi_non_function_tools_are_rejected_before_transport()
    test_kimi_tool_choice_none_does_not_inject_or_reject()
    test_kimi_named_tool_choice_must_match_supplied_tool()
    test_kimi_external_operation_bridge_output_is_recovered()
    test_kimi_external_operation_bridge_rejects_unknown_or_invalid_calls()
    test_kimi_external_operation_bridge_handles_close_tag_inside_argument()
    test_kimi_tool_history_is_serialized_outside_native_tool_fields()
    test_kimi_tool_choice_none_history_omits_operation_schemas()
    test_kimi_reasoning_effort_uses_upstream_default_max()
    test_glm_malformed_tool_call_close_tag()
    test_minimax_think_block_is_not_returned_as_content()
    test_claude_code_arg_key_tool_call_body_is_recovered()
    test_claude_code_arg_key_tool_call_with_windows_backslashes_is_recovered()
    test_claude_code_arg_value_tool_call_body_is_recovered()
    test_claude_code_close_only_arg_value_tool_call_body_is_recovered()
    test_glm_official_arg_key_tool_call_is_recovered()
    test_glm_arg_value_shell_command_preserves_trailing_quote()
    test_glm_malformed_reversed_arg_value_tool_call_is_recovered()
    test_glm_malformed_reversed_shell_command_preserves_trailing_quote()
    test_glm_string_typed_shell_arguments_preserve_literal_values()
    test_glm_close_only_shell_command_preserves_leading_quote()
    test_string_arguments_are_preserved_for_unknown_tool_names()
    test_inline_heredoc_arguments_are_preserved_for_unknown_tool_names()
    test_transcript_input_maps_to_single_required_argument_without_tool_name_special_case()
    test_xml_untyped_arguments_default_to_literal_strings()
    test_nullable_schema_types_are_normalized_before_parsing()
    test_json_tool_call_paths_preserve_explicit_json_argument_types()
    test_anyof_schema_types_are_used_for_text_arguments_only()
    test_glm_bare_malformed_reversed_arg_value_tool_call_is_recovered()
    test_glm_empty_argument_tool_call_is_recovered()
    test_glm_bare_required_tool_call_is_forwarded_with_empty_arguments()
    test_minimax_official_tool_call_is_recovered()
    test_minimax_xml_shell_command_preserves_trailing_quote()
    test_minimax_string_typed_shell_arguments_preserve_literal_values()
    test_deepseek_v4_dsml_tool_calls_are_recovered()
    test_deepseek_fallback_json_shell_command_preserves_backslash_escapes()
    test_deepseek_fallback_valid_json_preserves_explicit_json_argument_types()
    test_deepseek_fallback_valid_json_uses_anyof_schema_name_but_preserves_types()
    test_deepseek_fallback_xml_arguments_preserve_explicit_json_types_and_canonical_name()
    test_json_tool_call_with_shell_regex_backslashes_is_recovered()
    test_jsonish_tool_call_with_unescaped_command_quotes_is_recovered()
    test_bare_claude_code_arg_key_tool_call_is_recovered_for_non_streaming()
    test_inline_shell_command_uses_schema_and_preserves_shell_quotes()
    test_inline_non_shell_string_arguments_still_unquote_jsonish_values()
    test_inline_shell_heredoc_command_collects_body()
    test_claude_code_transcript_out_marker_without_trailing_newline_is_removed()
    test_mixed_claude_code_transcript_and_xml_tool_calls_are_recovered()
    test_deepseek_mixed_transcript_and_dsml_prefers_dsml_calls()
    test_streaming_detection_keeps_claude_code_tool_name_prefix()
    test_tool_result_turn_allows_additional_tools_by_default()
    test_official_prompt_shapes_are_not_mixed_between_model_versions()
    test_glm52_reasoning_effort_prompt_mapping()
    test_glm52_prompt_filters_template_internal_tool_fields()
    test_deepseek_v4_prompt_matches_hf_encoding_test_shape()
    test_deepseek_v4_reasoning_effort_matches_official_prefix_placement()
    test_history_tool_calls_render_in_each_official_adapter_format()
    test_required_tool_choice_still_allows_additional_tool_calls()
    print("tool adapter tests passed")
