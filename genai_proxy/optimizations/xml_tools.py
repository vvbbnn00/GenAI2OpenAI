def inject_xml_tool_prompt(
    messages,
    tool_prompt,
    prepend_tool_prompt=False,
):
    new_messages = (
        [{"role": "system", "content": tool_prompt}] if prepend_tool_prompt else []
    )
    injected_tool_prompt = prepend_tool_prompt
    index = 0

    while index < len(messages):
        msg = messages[index]
        role = msg.get("role")

        if role == "system" and not injected_tool_prompt:
            new_messages.append(
                {
                    **msg,
                    "content": msg.get("content", "") + "\n\n" + tool_prompt,
                }
            )
            injected_tool_prompt = True
            index += 1
            continue

        new_messages.append(msg)
        index += 1

    if not injected_tool_prompt:
        new_messages.insert(0, {"role": "system", "content": tool_prompt})
    return new_messages
