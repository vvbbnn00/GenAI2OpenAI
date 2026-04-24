import json

from genai_proxy.compat.openai import extract_tool_calls, tag_prefix_len
from genai_proxy.optimizations import GLM_ADAPTER, MINIMAX_ADAPTER, select_tool_adapter
from genai_proxy.optimizations.deepseek import inject_deepseek_tool_prompt
from genai_proxy.optimizations.minimax import inject_minimax_tool_prompt
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


def test_genai_model_record_mapping():
    assert (
        select_tool_adapter(
            "deepseek-chat",
            {"aiName": "DeepSeek-V4-Flash", "descInfo": "最新DeepSeek V4 284B模型"},
        )
        == "deepseek"
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
        == GLM_ADAPTER
    )


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
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"location": "Shanghai"}


def test_minimax_think_block_is_not_returned_as_content():
    content = (
        "<think>I should use the weather tool.</think>\n"
        "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Shanghai\"}}</tool_call>"
    )
    tool_calls, remaining = extract_tool_calls(
        content,
        tools=[WEATHER_TOOL],
        model="MiniMax-M1",
        adapter=MINIMAX_ADAPTER,
    )
    assert remaining is None
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"location": "Shanghai"}


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
        '"cd \'f:\\onedrive-vercel\\app\' && npm audit --json 2>&1 | head -2000", '
        '"timeout": 60000</tool_call>'
    )
    tool_calls, _ = extract_tool_calls(
        content,
        tools=[BASH_TOOL],
        model="MiniMax-M1",
        adapter=MINIMAX_ADAPTER,
    )
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert arguments["command"] == "cd 'f:\\onedrive-vercel\\app' && npm audit --json 2>&1 | head -2000"
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


def test_json_tool_call_with_shell_regex_backslashes_is_recovered():
    content = (
        r'''<tool_call>{"name": "Bash", "arguments": {"command": '''
        r'''"cat \"f:/onedrive-vercel/app/package.json\" | grep -E '^\s+\"[^\"]+\":' | wc -l", '''
        r'''"description": "Count direct dependencies"}}</tool_call>'''
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
        "cat \"f:/onedrive-vercel/app/package.json\" | grep -E '^\\s+\"[^\"]+\":' | wc -l"
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
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"command": "npm audit 2>&1"}


def test_streaming_detection_keeps_claude_code_tool_name_prefix():
    tags = _tool_start_tags_for_request(GLM_ADAPTER, [BASH_TOOL])
    assert "Bash<arg_key>" in tags
    assert "Bash <arg_key>" in tags
    assert max(tag_prefix_len("Bash", tag) for tag in tags) == len("Bash")
    assert max(tag_prefix_len("Bash ", tag) for tag in tags) == len("Bash ")


def test_tool_result_turn_defaults_to_final_answer_prompt():
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
        {"role": "tool", "tool_call_id": "call_weather", "content": "Shanghai is sunny."},
    ]

    minimax_messages = inject_minimax_tool_prompt(messages, [WEATHER_TOOL])
    assert "The tool results are sufficient and final for this turn" in minimax_messages[-1]["content"]
    assert "Do not emit <tool_call> tags" in minimax_messages[-1]["content"]

    deepseek_messages = inject_deepseek_tool_prompt(messages, [WEATHER_TOOL])
    assert "The function results are sufficient and final for this turn" in deepseek_messages[-1]["content"]
    assert "Do not emit DSML function_calls" in deepseek_messages[-1]["content"]


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
        {"role": "tool", "tool_call_id": "call_weather", "content": "Shanghai is sunny."},
    ]
    tool_choice = {"type": "function", "function": {"name": "get_weather"}}
    minimax_messages = inject_minimax_tool_prompt(messages, [WEATHER_TOOL], tool_choice=tool_choice)
    assert "Only call another tool if the current result is genuinely insufficient" in minimax_messages[-1]["content"]


if __name__ == "__main__":
    test_genai_model_record_mapping()
    test_glm_malformed_tool_call_close_tag()
    test_minimax_think_block_is_not_returned_as_content()
    test_claude_code_arg_key_tool_call_body_is_recovered()
    test_claude_code_arg_key_tool_call_with_windows_backslashes_is_recovered()
    test_claude_code_arg_value_tool_call_body_is_recovered()
    test_claude_code_close_only_arg_value_tool_call_body_is_recovered()
    test_json_tool_call_with_shell_regex_backslashes_is_recovered()
    test_jsonish_tool_call_with_unescaped_command_quotes_is_recovered()
    test_bare_claude_code_arg_key_tool_call_is_recovered_for_non_streaming()
    test_streaming_detection_keeps_claude_code_tool_name_prefix()
    test_tool_result_turn_defaults_to_final_answer_prompt()
    test_required_tool_choice_still_allows_additional_tool_calls()
    print("tool adapter tests passed")
