import json

from test_omp_long_context_integration import (
    PASS_MARKER,
    _consume_omp_line,
    _write_workspace,
)


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


def test_omp_workspace_makes_chain_and_terminal_gates_explicit(tmp_path):
    workspace = tmp_path / "workspace"
    expected = _write_workspace(workspace, "small context", stages=8)
    task = (workspace / "TASK.md").read_text(encoding="utf-8")

    assert "NEXT=<filename> proves that the chain is still incomplete" in task
    assert "literal NEXT=NONE" in task
    assert "exactly 8 CHAIN_TOKEN values" in task
    assert "do not add Markdown, a table, an explanation" in task
    assert "A successful write result is not a readback" in task
    assert "the next action must be a read of that same file" in task
    assert f"finish with exactly {PASS_MARKER}" in task

    for index, stage_name in enumerate(expected["stage_names"]):
        stage = (workspace / stage_name).read_text(encoding="utf-8")
        expected_next = (
            expected["stage_names"][index + 1]
            if index + 1 < len(expected["stage_names"])
            else "NONE"
        )
        assert f"CHAIN_INDEX={index + 1:02d}" in stage
        assert f"CHAIN_TOKEN={expected['stage_tokens'][index]}" in stage
        assert f"NEXT={expected_next}" in stage
