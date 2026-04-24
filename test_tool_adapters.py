import json

from genai_proxy.compat.openai import extract_tool_calls
from genai_proxy.optimizations import GLM_ADAPTER, MINIMAX_ADAPTER, select_tool_adapter
from genai_proxy.optimizations.deepseek import inject_deepseek_tool_prompt
from genai_proxy.optimizations.minimax import inject_minimax_tool_prompt


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
    test_tool_result_turn_defaults_to_final_answer_prompt()
    test_required_tool_choice_still_allows_additional_tool_calls()
    print("tool adapter tests passed")
