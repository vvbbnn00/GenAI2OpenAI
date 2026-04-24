import json


def inject_xml_tool_prompt(messages, tool_prompt, allow_additional_tool_calls=False):
    new_messages = []
    has_system = False
    index = 0

    while index < len(messages):
        msg = messages[index]
        role = msg.get("role")

        if role == "system":
            new_messages.append(
                {
                    "role": "system",
                    "content": msg.get("content", "") + "\n\n" + tool_prompt,
                }
            )
            has_system = True
            index += 1
            continue

        if role == "assistant" and msg.get("tool_calls"):
            new_messages.append(
                {
                    "role": "assistant",
                    "content": _render_tool_call_message(msg),
                }
            )
            index += 1
            continue

        if role == "tool":
            tool_messages = []
            while index < len(messages) and messages[index].get("role") == "tool":
                tool_messages.append(messages[index])
                index += 1
            new_messages.append(
                {
                    "role": "user",
                    "content": _render_tool_results(
                        tool_messages,
                        allow_additional_tool_calls=allow_additional_tool_calls,
                    ),
                }
            )
            continue

        new_messages.append(msg)
        index += 1

    if not has_system:
        new_messages.insert(0, {"role": "system", "content": tool_prompt})
    return new_messages


def _render_tool_call_message(message):
    parts = []
    if message.get("content"):
        parts.append(str(message["content"]))

    for tool_call in message.get("tool_calls") or []:
        function_data = tool_call.get("function", {})
        call_obj = {
            "name": function_data.get("name", ""),
            "arguments": _safe_json_loads(function_data.get("arguments", "{}")),
        }
        parts.append(
            "<tool_call>\n"
            f"{json.dumps(call_obj, ensure_ascii=False)}\n"
            "</tool_call>"
        )
    return "\n\n".join(part for part in parts if part).strip()


def _render_tool_results(tool_messages, allow_additional_tool_calls=False):
    lines = [
        "<tool_results>",
    ]
    if allow_additional_tool_calls:
        lines.append(
            "Use these tool results to answer the user. Only call another tool if the current result is genuinely insufficient."
        )
    else:
        lines.append(
            "The tool results are sufficient and final for this turn. Answer the user normally using them. Do not call any tool. Do not emit <tool_call> tags."
        )
    for msg in tool_messages:
        lines.extend(
            [
                "<tool_result>",
                f"<tool_call_id>{msg.get('tool_call_id', 'unknown')}</tool_call_id>",
                f"<content>{_normalize_content(msg.get('content'))}</content>",
                "</tool_result>",
            ]
        )
    lines.append("</tool_results>")
    return "\n".join(lines)


def _normalize_content(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _safe_json_loads(raw):
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
