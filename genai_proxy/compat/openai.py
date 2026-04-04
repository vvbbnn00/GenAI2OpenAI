import json
import re
import uuid
from datetime import datetime

from flask import jsonify


TOOL_SYSTEM_PROMPT = """\
You have access to the following tools:

<tools>
{tool_definitions}
</tools>

When you need to call a tool, you MUST use the following XML format. Do NOT use markdown code blocks.

<tool_call>
{{"name": "<function-name>", "arguments": {{<arguments-as-json>}}}}
</tool_call>

Rules:
1. You can call multiple tools by using multiple <tool_call> blocks.
2. If you don't need any tool, just respond normally in plain text without any <tool_call> tags.
3. After receiving tool results, analyze them and either call more tools or give a final answer in plain text.
4. The "arguments" field MUST be a valid JSON object matching the tool's parameter schema.
5. NEVER wrap <tool_call> in markdown code blocks like ```xml or ```json."""

TOOL_CHOICE_REQUIRED_PROMPT = (
    "\nYou MUST call at least one tool in your response. Do NOT respond with plain text only."
)
TOOL_CHOICE_SPECIFIC_PROMPT = (
    '\nYou MUST call the tool named "{name}" in your response.'
)


def openai_error(message, error_type="invalid_request_error", code=None, status=400):
    return (
        jsonify(
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "code": code,
                }
            }
        ),
        status,
    )


def make_error_chunk(message, model="unknown", completion_id=None):
    cid = completion_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    error_chunk = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": f"[Error] {message}"},
                "finish_reason": "error",
            }
        ],
    }
    return f"data: {json.dumps(error_chunk)}\n\ndata: [DONE]\n\n"


def format_tool_definitions(tools):
    definitions = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool["function"]
        params = func.get("parameters", {})
        params_json = json.dumps(params, ensure_ascii=False, indent=2)
        definitions.append(
            f"<tool_definition>\n"
            f"  <name>{func['name']}</name>\n"
            f"  <description>{func.get('description', '')}</description>\n"
            f"  <parameters>\n{params_json}\n  </parameters>\n"
            f"</tool_definition>"
        )
    return "\n".join(definitions)


def inject_tool_prompt(messages, tools, tool_choice=None):
    tool_defs = format_tool_definitions(tools)
    tool_prompt = TOOL_SYSTEM_PROMPT.format(tool_definitions=tool_defs)

    if tool_choice == "required":
        tool_prompt += TOOL_CHOICE_REQUIRED_PROMPT
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice["function"]["name"]
        tool_prompt += TOOL_CHOICE_SPECIFIC_PROMPT.format(name=name)

    new_messages = []
    has_system = False

    for msg in messages:
        role = msg.get("role")

        if role == "system":
            new_messages.append(
                {
                    "role": "system",
                    "content": msg.get("content", "") + "\n\n" + tool_prompt,
                }
            )
            has_system = True
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "unknown")
            new_messages.append(
                {
                    "role": "user",
                    "content": (
                        "<tool_result>\n"
                        f"  <tool_call_id>{tool_call_id}</tool_call_id>\n"
                        f"  <result>\n{msg.get('content', '')}\n  </result>\n"
                        "</tool_result>"
                    ),
                }
            )
        elif role == "assistant" and msg.get("tool_calls"):
            tc_text = msg.get("content") or ""
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                call_obj = {
                    "name": func.get("name", ""),
                    "arguments": json.loads(func.get("arguments", "{}")),
                }
                tc_text += (
                    f"\n<tool_call>\n{json.dumps(call_obj, ensure_ascii=False)}\n</tool_call>"
                )
            new_messages.append({"role": "assistant", "content": tc_text.strip()})
        else:
            new_messages.append(msg)

    if not has_system:
        new_messages.insert(0, {"role": "system", "content": tool_prompt})

    return new_messages


def strip_think_blocks(content):
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _parse_tool_call_body(raw):
    raw = raw.strip()

    try:
        call = json.loads(raw)
        if "name" in call:
            return call
    except (json.JSONDecodeError, ValueError):
        pass

    name_m = re.search(r"<name>\s*(.*?)\s*</name>", raw, re.DOTALL)
    args_m = re.search(r"<arguments>\s*(.*?)\s*</arguments>", raw, re.DOTALL)
    if name_m:
        name = name_m.group(1).strip()
        arguments = {}
        if args_m:
            args_str = args_m.group(1).strip()
            try:
                arguments = json.loads(args_str)
            except (json.JSONDecodeError, ValueError):
                arguments = {"raw": args_str}
        return {"name": name, "arguments": arguments}

    return None


def extract_tool_calls(content, logger=None):
    cleaned = strip_think_blocks(content)
    cleaned = re.sub(
        r"```(?:xml|json|plaintext|text)?\s*\n?\s*(<tool_call>.*?</tool_call>)\s*\n?\s*```",
        r"\1",
        cleaned,
        flags=re.DOTALL,
    )

    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, cleaned, re.DOTALL)

    if not matches:
        if logger:
            logger.debug(
                "No <tool_call> tags found in content (%d chars): %s",
                len(content),
                content[:500],
            )
        return None, content

    if logger:
        logger.debug("Found %d <tool_call> match(es)", len(matches))

    tool_calls = []
    for index, match in enumerate(matches):
        call = _parse_tool_call_body(match)
        if not call:
            if logger:
                logger.warning(
                    "Failed to parse tool_call[%d] — raw: %s",
                    index,
                    match[:300],
                )
            continue

        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(
                        call.get("arguments", {}),
                        ensure_ascii=False,
                    ),
                },
            }
        )

    if not tool_calls:
        return None, content

    remaining = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL).strip()
    return tool_calls, remaining or None


def tag_prefix_len(text, tag):
    max_len = min(len(tag) - 1, len(text))
    for length in range(max_len, 0, -1):
        if text[-length:] == tag[:length]:
            return length
    return 0

