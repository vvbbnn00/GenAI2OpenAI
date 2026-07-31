import json

from test_omp_long_context_integration import _consume_omp_line


def _state():
    return {
        "terminal": False,
        "tool_starts": {},
        "tool_executions": [],
        "tool_turns": 0,
        "reasoning_deltas": 0,
        "toolcall_deltas": 0,
        "text_deltas": 0,
        "input_tokens": [],
        "final_text": "",
        "dynamic_notices": [],
        "diagnostics": [],
        "deepseek_tool_turns_without_reasoning": 0,
    }


def test_omp_runner_detects_top_level_tool_execution_error(capsys):
    state = _state()
    _consume_omp_line(
        json.dumps(
            {
                "type": "tool_execution_start",
                "toolCallId": "call_1",
                "toolName": "read",
                "args": {"path": "missing.txt"},
            }
        ),
        state,
        "deepseek-chat",
    )
    _consume_omp_line(
        json.dumps(
            {
                "type": "tool_execution_end",
                "toolCallId": "call_1",
                "toolName": "read",
                "result": {"content": [{"type": "text", "text": "missing"}]},
                "isError": True,
            }
        ),
        state,
        "deepseek-chat",
    )

    assert state["tool_executions"] == [
        {
            "name": "read",
            "args": {"path": "missing.txt"},
            "is_error": True,
        }
    ]
    assert "missing.txt" in capsys.readouterr().out
