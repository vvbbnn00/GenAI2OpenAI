import argparse
import hashlib
import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from werkzeug.serving import WSGIRequestHandler, make_server

from genai_proxy.app import create_app
from genai_proxy.config import AppConfig
from genai_proxy.optimizations.registry import select_tool_adapter
from genai_proxy.token_usage import count_openai_request_tokens

# This is an opt-in keystore-backed OMP integration runner, not an offline test.
__test__ = False

MODELS = ("chatglm", "deepseek-chat", "deepseek-pro")
MODEL_RECORDS = {
    "chatglm": {
        "aiType": "chatglm",
        "aiName": "GLM-5.2",
        "descInfo": "GLM-5.2",
        "rootModelName": "Xinference",
        "rootAiType": "xinference",
    },
    "deepseek-chat": {
        "aiType": "deepseek-chat",
        "aiName": "DeepSeek-V4-Flash",
        "descInfo": "DeepSeek V4 Flash",
        "rootModelName": "Xinference",
        "rootAiType": "xinference",
        "enableDeepThink": 1,
    },
    "deepseek-pro": {
        "aiType": "deepseek-pro",
        "aiName": "DeepSeek-V4-Pro",
        "descInfo": "DeepSeek V4 Pro",
        "rootModelName": "Xinference",
        "rootAiType": "xinference",
        "enableDeepThink": 1,
    },
}
SENTINELS = (
    "BEGIN-7B19E2A4-KEEP-EXACT",
    "MIDDLE-41D6C903-KEEP-EXACT",
    "END-AC82F751-KEEP-EXACT",
)
PASS_MARKER = "LONG_TOOL_CHAIN_PASS"


class _QuietRequestHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-") -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated OMP long-context, multi-turn tool-call tests through "
            "a keystore-backed local GenAI2OpenAI server."
        )
    )
    parser.add_argument("--keystore", default="docker-deploy.keystore")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--omp-bin", default=shutil.which("omp"))
    parser.add_argument("--min-context-tokens", type=int, default=120_000)
    parser.add_argument("--context-window", type=int, default=400_000)
    parser.add_argument("--stages", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1_800)
    args = parser.parse_args()

    invalid = [model for model in args.models if model not in MODELS]
    if invalid:
        raise SystemExit(f"Refusing unsupported model(s): {', '.join(invalid)}")
    if not args.omp_bin:
        raise SystemExit("OMP executable not found; pass --omp-bin")
    if not Path(args.omp_bin).is_file():
        raise SystemExit(f"OMP executable does not exist: {args.omp_bin}")
    if not Path(args.keystore).is_file():
        raise SystemExit(f"Keystore does not exist: {args.keystore}")
    if args.min_context_tokens < 8_000:
        raise SystemExit("--min-context-tokens must be at least 8000")
    if args.context_window <= args.min_context_tokens + 32_768:
        raise SystemExit("--context-window leaves too little room for the agent loop")
    if args.stages < 8:
        raise SystemExit("--stages must be at least 8")
    if args.repeat < 1 or args.timeout < 1:
        raise SystemExit("--repeat and --timeout must be positive")

    context, context_counts = _build_context(args.min_context_tokens)
    if max(context_counts.values()) + 32_768 >= args.context_window:
        raise SystemExit(
            "Generated context leaves less than 32768 tokens for the agent loop; "
            "increase --context-window or lower --min-context-tokens"
        )
    print(
        "[context] "
        + " ".join(f"{model}={count}" for model, count in context_counts.items()),
        flush=True,
    )

    failures = []
    summaries = []
    with tempfile.TemporaryDirectory(prefix="genai2openai-omp-long-") as temp:
        root = Path(temp)
        app = _make_app(args.keystore, root / "models-cache.json")
        server = make_server(
            "127.0.0.1",
            0,
            app,
            threaded=True,
            request_handler=_QuietRequestHandler,
        )
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            agent_dir = root / "agent"
            _write_omp_config(agent_dir, server.server_port, args.context_window)
            for model in args.models:
                for iteration in range(1, args.repeat + 1):
                    label = f"{model}:run{iteration:02d}"
                    workspace = root / "workspaces" / label.replace(":", "-")
                    expected = _write_workspace(
                        workspace,
                        context,
                        stages=args.stages,
                    )
                    try:
                        summary = _run_omp(
                            omp_bin=args.omp_bin,
                            agent_dir=agent_dir,
                            workspace=workspace,
                            model=model,
                            timeout=args.timeout,
                            min_context_tokens=args.min_context_tokens,
                            expected=expected,
                        )
                    except Exception as exc:
                        failures.append((label, str(exc)))
                        print(f"[FAIL] {label}: {exc}", flush=True)
                    else:
                        summary["run"] = iteration
                        summaries.append(summary)
                        print(
                            f"[PASS] {label} tools={summary['tool_executions']} "
                            f"tool_turns={summary['tool_turns']} "
                            f"first_input_tokens={summary['first_input_tokens']} "
                            f"max_input_tokens={summary['max_input_tokens']} "
                            f"reasoning_deltas={summary['reasoning_deltas']} "
                            f"duration={summary['duration_seconds']:.1f}s",
                            flush=True,
                        )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)
            app.extensions["token_manager"].shutdown()

    print(json.dumps({"runs": summaries}, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(
            "OMP long-context integration failures: "
            + "; ".join(f"{label}: {error}" for label, error in failures)
        )


def _make_app(keystore: str, model_cache: Path):
    logger = logging.getLogger("omp_long_context_integration")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return create_app(
        AppConfig(
            token=None,
            keystore=keystore,
            port=0,
            debug=False,
            api_key=None,
            token_check_interval=0,
            claude_haiku_model="deepseek-chat",
            claude_sonnet_model="chatglm",
            claude_opus_model="deepseek-pro",
            genai_model_cache=str(model_cache),
        ),
        logger,
    )


def _build_context(min_tokens: int) -> tuple[str, dict[str, int]]:
    records = []
    next_index = 1
    while True:
        records.extend(
            _archive_record(index) for index in range(next_index, next_index + 1_000)
        )
        next_index += 1_000
        middle = len(records) // 2
        context = "".join(
            [
                "LONG CONTEXT ARCHIVE\n",
                f"TARGET_SENTINEL={SENTINELS[0]}\n",
                *records[:middle],
                f"TARGET_SENTINEL={SENTINELS[1]}\n",
                *records[middle:],
                f"TARGET_SENTINEL={SENTINELS[2]}\n",
                "END LONG CONTEXT ARCHIVE\n",
            ]
        )
        counts = {model: _count_context_tokens(model, context) for model in MODELS}
        if min(counts.values()) >= min_tokens:
            return context, counts


def _archive_record(index: int) -> str:
    digest = hashlib.sha256(f"archive-record-{index}".encode()).hexdigest()[:20]
    lane = ("amber", "cobalt", "jade", "violet")[index % 4]
    return (
        f"ARCHIVE_RECORD={index:06d} lane={lane} digest={digest} "
        "status=reference-only action=retain-explicit-target-sentinels\n"
    )


def _count_context_tokens(model: str, context: str) -> int:
    record = MODEL_RECORDS[model]
    return count_openai_request_tokens(
        [{"role": "user", "content": context}],
        model,
        model_record=record,
        tool_adapter=select_tool_adapter(model, record),
    )


def _write_omp_config(agent_dir: Path, port: int, context_window: int) -> None:
    agent_dir.mkdir(parents=True)
    (agent_dir / "config.yml").write_text(
        """disabledProviders:
  - native
  - claude
  - codex
  - gemini
  - github
  - opencode
  - cursor
  - agents-md
mcp:
  enableProjectConfig: false
tools:
  xdev: false
""",
        encoding="utf-8",
    )
    (agent_dir / "models.yml").write_text(
        f"""providers:
  genai-long:
    baseUrl: http://127.0.0.1:{port}/v1
    api: openai-completions
    auth: none
    disableStrictTools: false
    models:
      - id: chatglm
        name: GLM-5.2 isolated long-context test
        reasoning: true
        input: [text]
        contextWindow: {context_window}
        maxTokens: 16384
        compat:
          supportsReasoningEffort: true
          supportsReasoningParams: true
          thinkingFormat: openai
          reasoningContentField: reasoning_content
          supportsUsageInStreaming: true
          supportsForcedToolChoice: true
      - id: deepseek-chat
        name: DeepSeek V4 Flash isolated long-context test
        reasoning: true
        input: [text]
        contextWindow: {context_window}
        maxTokens: 16384
        compat:
          supportsReasoningEffort: true
          supportsReasoningParams: true
          thinkingFormat: openai
          reasoningContentField: reasoning_content
          requiresReasoningContentForToolCalls: true
          allowsSyntheticReasoningContentForToolCalls: false
          supportsUsageInStreaming: true
          supportsForcedToolChoice: true
      - id: deepseek-pro
        name: DeepSeek V4 Pro isolated long-context test
        reasoning: true
        input: [text]
        contextWindow: {context_window}
        maxTokens: 16384
        compat:
          supportsReasoningEffort: true
          supportsReasoningParams: true
          thinkingFormat: openai
          reasoningContentField: reasoning_content
          requiresReasoningContentForToolCalls: true
          allowsSyntheticReasoningContentForToolCalls: false
          supportsUsageInStreaming: true
          supportsForcedToolChoice: true
""",
        encoding="utf-8",
    )


def _write_workspace(workspace: Path, context: str, *, stages: int) -> dict:
    workspace.mkdir(parents=True)
    stage_names = []
    stage_tokens = []
    for index in range(1, stages + 1):
        suffix = hashlib.sha256(f"stage-name-{index}".encode()).hexdigest()[:6]
        stage_names.append(f"stage_{index:02d}_{suffix}.txt")
        token = hashlib.sha256(f"stage-token-{index}".encode()).hexdigest()[:12]
        stage_tokens.append(f"S{index:02d}-{token}")

    for index, (name, token) in enumerate(zip(stage_names, stage_tokens)):
        next_name = stage_names[index + 1] if index + 1 < len(stage_names) else "NONE"
        (workspace / name).write_text(
            f"CHAIN_INDEX={index + 1:02d}\nCHAIN_TOKEN={token}\nNEXT={next_name}\n",
            encoding="utf-8",
        )

    (workspace / "CONTEXT.md").write_text(context, encoding="utf-8")
    (workspace / "TASK.md").write_text(
        f"""This is a deterministic long-context tool transport test.

The attached CONTEXT.md contains exactly three TARGET_SENTINEL lines. Preserve
their values in archive order. Do not substitute or shorten them.

Use the read tool on {stage_names[0]}. Each stage contains one CHAIN_INDEX,
CHAIN_TOKEN, and the next filename in NEXT. Follow the chain one file at a time
until NEXT=NONE. Wait for each read result before using its NEXT value. Do not
infer filenames from the directory listing, guess unseen tokens, or skip a stage.
Order the output by CHAIN_INDEX.

A read result with NEXT=<filename> proves that the chain is still incomplete,
and the next action must be a read of that exact filename. Do not use the write
tool until a read result contains the literal NEXT=NONE and you have collected
exactly {stages} CHAIN_TOKEN values. If either condition is false, continue the
chain instead of summarizing or writing a partial result.

After reading all {stages} stages, use the write tool to create result.json as
valid JSON with exactly these keys:
{{"sentinels": [three values in archive order], "stage_tokens": [all chain tokens in stage order]}}

Read result.json back with the read tool. If it is not exact, fix it and read it
again. Then write COMPLETE.txt containing only {PASS_MARKER}, read COMPLETE.txt
back, and finish with exactly {PASS_MARKER}. The final assistant message is
checked by exact string equality: do not add Markdown, a table, an explanation,
punctuation, a prefix, or a suffix.

A successful write result is not a readback and does not verify file contents.
Immediately after each write, the next action must be a read of that same file.
Do not finish from memory of the value passed to write.
""",
        encoding="utf-8",
    )
    return {
        "stage_names": stage_names,
        "stage_tokens": stage_tokens,
        "sentinels": list(SENTINELS),
    }


def _run_omp(
    *,
    omp_bin: str,
    agent_dir: Path,
    workspace: Path,
    model: str,
    timeout: int,
    min_context_tokens: int,
    expected: dict,
) -> dict:
    thinking = "high" if model == "chatglm" else "max"
    command = [
        omp_bin,
        "--model",
        f"genai-long/{model}",
        "--cwd",
        str(workspace),
        "--mode",
        "json",
        "--no-session",
        "--no-extensions",
        "--no-skills",
        "--no-rules",
        "--no-title",
        "--no-lsp",
        "--no-pty",
        "--tools=read,write",
        "--thinking",
        thinking,
        "--auto-approve",
        "--approval-mode",
        "yolo",
        "--max-time",
        f"{timeout}s",
        "-p",
        "@CONTEXT.md",
        "@TASK.md",
    ]
    environment = os.environ.copy()
    for key in (
        "OMP_PROFILE",
        "PI_PROFILE",
        "PI_CONFIG_FILES",
        "PI_SMOL_MODEL",
        "PI_SLOW_MODEL",
        "PI_PLAN_MODEL",
    ):
        environment.pop(key, None)
    environment["PI_CODING_AGENT_DIR"] = str(agent_dir)
    environment["PI_NO_PTY"] = "1"

    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=workspace,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    lines = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    state = {
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
    try:
        while True:
            elapsed = time.monotonic() - started
            if elapsed > timeout + 15:
                raise TimeoutError(f"OMP exceeded {timeout}s timeout")
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                if process.poll() is not None and not reader.is_alive():
                    break
                continue
            if line is None:
                break
            _consume_omp_line(line, state, model)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        raise
    finally:
        reader.join(timeout=5)

    return_code = process.wait(timeout=5)
    duration = time.monotonic() - started
    _validate_omp_run(
        return_code=return_code,
        state=state,
        workspace=workspace,
        model=model,
        min_context_tokens=min_context_tokens,
        expected=expected,
    )
    return {
        "model": model,
        "tool_executions": len(state["tool_executions"]),
        "tool_turns": state["tool_turns"],
        "first_input_tokens": state["input_tokens"][0],
        "max_input_tokens": max(state["input_tokens"]),
        "reasoning_deltas": state["reasoning_deltas"],
        "toolcall_deltas": state["toolcall_deltas"],
        "text_deltas": state["text_deltas"],
        "duration_seconds": round(duration, 3),
    }


def _consume_omp_line(line: str, state: dict, model: str) -> None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        text = line.strip()
        if text:
            state["diagnostics"].append(text)
            del state["diagnostics"][:-20]
        return

    event_type = event.get("type")
    if event_type == "agent_end":
        state["terminal"] = bool(event.get("isTerminal"))
        return
    if event_type == "notice":
        message = str(event.get("message") or "")
        if "xd://" in message or "mcp__" in message:
            state["dynamic_notices"].append(message[:300])
        return
    if event_type == "message_update":
        update_type = (event.get("assistantMessageEvent") or {}).get("type")
        if update_type == "thinking_delta":
            state["reasoning_deltas"] += 1
        elif update_type == "toolcall_delta":
            state["toolcall_deltas"] += 1
        elif update_type == "text_delta":
            state["text_deltas"] += 1
        return
    if event_type == "message_end":
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            return
        usage = message.get("usage") or {}
        input_tokens = usage.get("input")
        if isinstance(input_tokens, int) and input_tokens > 0:
            state["input_tokens"].append(input_tokens)
        content = message.get("content") or []
        if message.get("stopReason") == "toolUse":
            state["tool_turns"] += 1
            has_reasoning = any(block.get("type") == "thinking" for block in content)
            if model.startswith("deepseek") and not has_reasoning:
                state["deepseek_tool_turns_without_reasoning"] += 1
        elif message.get("stopReason") == "stop":
            state["final_text"] = "".join(
                str(block.get("text") or "")
                for block in content
                if block.get("type") == "text"
            )
        return
    if event_type == "tool_execution_start":
        state["tool_starts"][event.get("toolCallId")] = {
            "name": event.get("toolName"),
            "args": event.get("args") or {},
        }
        return
    if event_type == "tool_execution_end":
        call_id = event.get("toolCallId")
        started = state["tool_starts"].pop(call_id, {})
        execution = {
            "name": event.get("toolName") or started.get("name"),
            "args": started.get("args") or {},
            "is_error": bool(
                event.get("isError") or (event.get("result") or {}).get("isError")
            ),
        }
        state["tool_executions"].append(execution)
        path = execution["args"].get("path")
        display_path = Path(path).name if isinstance(path, str) else "?"
        print(
            f"[{model}] tool#{len(state['tool_executions']):02d} "
            f"{execution['name']} {display_path}",
            flush=True,
        )


def _validate_omp_run(
    *,
    return_code: int,
    state: dict,
    workspace: Path,
    model: str,
    min_context_tokens: int,
    expected: dict,
) -> None:
    errors = []
    if return_code != 0:
        errors.append(f"OMP exited with {return_code}: {state['diagnostics'][-5:]}")
    if not state["terminal"]:
        errors.append("OMP did not emit a terminal agent_end event")
    if state["dynamic_notices"]:
        errors.append(
            "external MCP/xdev tools leaked into the isolated run: "
            + state["dynamic_notices"][0]
        )
    if any(call["is_error"] for call in state["tool_executions"]):
        errors.append("at least one OMP tool execution failed")
    if state["reasoning_deltas"] == 0:
        errors.append("no streamed reasoning deltas were observed")
    if state["toolcall_deltas"] == 0:
        errors.append("no streamed tool-call argument deltas were observed")
    if state["text_deltas"] == 0:
        errors.append("no streamed final text deltas were observed")
    if not state["input_tokens"]:
        errors.append("OMP reported no non-zero input token usage")
    elif state["input_tokens"][0] < min_context_tokens:
        errors.append(
            f"first request used only {state['input_tokens'][0]} input tokens"
        )
    if model.startswith("deepseek") and state["deepseek_tool_turns_without_reasoning"]:
        errors.append("a DeepSeek tool turn lost its reasoning_content history")
    if state["final_text"].strip() != PASS_MARKER:
        errors.append(f"unexpected final text: {state['final_text']!r}")

    calls = state["tool_executions"]
    read_paths = [
        Path(call["args"]["path"]).name
        for call in calls
        if call["name"] == "read" and isinstance(call["args"].get("path"), str)
    ]
    missing_stages = [
        path for path in expected["stage_names"] if path not in read_paths
    ]
    if missing_stages:
        errors.append(f"stage chain skipped: {', '.join(missing_stages)}")
    for required_read in ("result.json", "COMPLETE.txt"):
        if required_read not in read_paths:
            errors.append(f"OMP did not read back {required_read}")

    result_path = workspace / "result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"result.json is missing or invalid: {exc}")
    else:
        expected_result = {
            "sentinels": expected["sentinels"],
            "stage_tokens": expected["stage_tokens"],
        }
        if result != expected_result:
            errors.append("result.json does not match the long-context ground truth")
    try:
        complete = (workspace / "COMPLETE.txt").read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"COMPLETE.txt is missing: {exc}")
    else:
        if complete != PASS_MARKER:
            errors.append(f"COMPLETE.txt has unexpected content: {complete!r}")

    if errors:
        raise AssertionError("; ".join(errors))


if __name__ == "__main__":
    main()
