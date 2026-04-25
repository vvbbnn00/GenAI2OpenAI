import json

from genai_proxy.compat.openai import extract_tool_calls, tag_prefix_len
from genai_proxy.optimizations import (
    DEEPSEEK_LEGACY_ADAPTER,
    DEEPSEEK_V4_FLASH_ADAPTER,
    DEEPSEEK_V4_PRO_ADAPTER,
    GENERIC_ADAPTER,
    GLM_ADAPTER,
    MINIMAX_ADAPTER,
    select_tool_adapter,
)
from genai_proxy.optimizations.deepseek import inject_deepseek_tool_prompt
from genai_proxy.optimizations.glm import inject_glm_tool_prompt
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

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results to return"},
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
    assert arguments == {"file_path": "f:\\onedrive-vercel\\app\\package.json", "limit": 80}


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
        {"role": "tool", "tool_call_id": "call_weather", "content": "Shanghai is sunny."},
    ]

    minimax_messages = inject_minimax_tool_prompt(messages, [WEATHER_TOOL])
    assert minimax_messages[-1]["content"].startswith("<response>Shanghai is sunny.</response>")
    assert "Only call another tool if the current result is genuinely insufficient" in minimax_messages[-1]["content"]

    deepseek_messages = inject_deepseek_tool_prompt(
        messages,
        [WEATHER_TOOL],
        adapter=DEEPSEEK_V4_FLASH_ADAPTER,
    )
    assert deepseek_messages[-1]["content"] == "<tool_result>Shanghai is sunny.</tool_result>"


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

    glm_prompt = inject_glm_tool_prompt(messages, [WEATHER_TOOL])[0]["content"]
    assert glm_prompt.startswith("# Tools\n\nYou may call one or more functions")
    assert "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>" in glm_prompt
    assert "<minimax:tool_call>" not in glm_prompt
    assert "Rules:" not in glm_prompt


def test_deepseek_v4_prompt_matches_hf_encoding_test_shape():
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What's the weather in Beijing?"}]
    prompt = inject_deepseek_tool_prompt(
        messages,
        [WEATHER_TOOL, SEARCH_TOOL],
        adapter=DEEPSEEK_V4_PRO_ADAPTER,
    )[0]["content"]

    assert prompt.startswith("You are a helpful assistant.\n\n## Tools\n\n")
    assert 'You have access to a set of tools to help answer the user\'s question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:' in prompt
    assert "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response." in prompt
    assert "Otherwise, output directly after </think> with tool calls or final response." in prompt
    assert "### Available Tool Schemas" in prompt
    assert '"name": "get_weather"' in prompt
    assert '"name": "search"' in prompt
    assert "You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls." in prompt
    assert "<functions>" not in prompt
    assert "function_calls" not in prompt


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
    assert "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Shanghai</arg_value></tool_call>" in glm_messages[-1]["content"]


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
    assert minimax_messages[-1]["content"].startswith("<response>Shanghai is sunny.</response>")
    assert "Only call another tool if the current result is genuinely insufficient" in minimax_messages[-1]["content"]


if __name__ == "__main__":
    test_genai_model_record_mapping()
    test_glm_malformed_tool_call_close_tag()
    test_minimax_think_block_is_not_returned_as_content()
    test_claude_code_arg_key_tool_call_body_is_recovered()
    test_claude_code_arg_key_tool_call_with_windows_backslashes_is_recovered()
    test_claude_code_arg_value_tool_call_body_is_recovered()
    test_claude_code_close_only_arg_value_tool_call_body_is_recovered()
    test_glm_official_arg_key_tool_call_is_recovered()
    test_minimax_official_tool_call_is_recovered()
    test_deepseek_v4_dsml_tool_calls_are_recovered()
    test_json_tool_call_with_shell_regex_backslashes_is_recovered()
    test_jsonish_tool_call_with_unescaped_command_quotes_is_recovered()
    test_bare_claude_code_arg_key_tool_call_is_recovered_for_non_streaming()
    test_mixed_claude_code_transcript_and_xml_tool_calls_are_recovered()
    test_deepseek_mixed_transcript_and_dsml_prefers_dsml_calls()
    test_streaming_detection_keeps_claude_code_tool_name_prefix()
    test_tool_result_turn_allows_additional_tools_by_default()
    test_official_prompt_shapes_are_not_mixed_between_model_versions()
    test_deepseek_v4_prompt_matches_hf_encoding_test_shape()
    test_history_tool_calls_render_in_each_official_adapter_format()
    test_required_tool_choice_still_allows_additional_tool_calls()
    print("tool adapter tests passed")
