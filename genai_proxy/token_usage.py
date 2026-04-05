import json
import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Multipliers:
    word: float
    number: float
    cjk: float
    symbol: float
    math_symbol: float
    url_delim: float
    at_sign: float
    emoji: float
    newline: float
    space: float
    base_pad: int = 0


OPENAI_MULTIPLIERS = Multipliers(
    word=1.02,
    number=1.55,
    cjk=0.85,
    symbol=0.4,
    math_symbol=2.68,
    url_delim=1.0,
    at_sign=2.0,
    emoji=2.12,
    newline=0.5,
    space=0.42,
)

CLAUDE_MULTIPLIERS = Multipliers(
    word=1.13,
    number=1.63,
    cjk=1.21,
    symbol=0.4,
    math_symbol=4.52,
    url_delim=1.26,
    at_sign=2.82,
    emoji=2.6,
    newline=0.89,
    space=0.39,
)

GEMINI_MULTIPLIERS = Multipliers(
    word=1.15,
    number=2.8,
    cjk=0.68,
    symbol=0.38,
    math_symbol=1.05,
    url_delim=1.2,
    at_sign=2.5,
    emoji=1.08,
    newline=1.15,
    space=0.2,
)

MATH_SYMBOLS = set(
    "∑∫∂√∞≤≥≠≈±×÷∈∉∋∌⊂⊃⊆⊇∪∩∧∨¬∀∃∄∅∆∇∝∟∠∡∢°′″‴⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎²³¹⁴⁵⁶⁷⁸⁹⁰"
)
URL_DELIMS = set("/:?&=;#%")


def estimate_token_by_model(model: str | None, text: str) -> int:
    if not text:
        return 0

    lowered = (model or "").lower()
    if "gemini" in lowered:
        multipliers = GEMINI_MULTIPLIERS
    elif "claude" in lowered:
        multipliers = CLAUDE_MULTIPLIERS
    else:
        multipliers = OPENAI_MULTIPLIERS

    return _estimate_tokens(text, multipliers)


def estimate_openai_request_tokens(messages, model: str | None, tools=None) -> int:
    texts = []
    message_count = 0
    name_count = 0
    tool_count = 0

    for message in messages or []:
        message_count += 1
        role = message.get("role")
        if role:
            texts.append(str(role))

        name = message.get("name")
        if name:
            name_count += 1
            texts.append(str(name))

        texts.extend(_extract_message_texts(message))

    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        tool_count += 1
        function = tool.get("function", {})
        if function.get("name"):
            texts.append(str(function["name"]))
        if function.get("description"):
            texts.append(str(function["description"]))
        if function.get("parameters") is not None:
            texts.append(json.dumps(function["parameters"], ensure_ascii=False, sort_keys=True))

    token_count = estimate_token_by_model(model, "\n".join(texts))
    token_count += tool_count * 8
    token_count += message_count * 3
    token_count += name_count * 3
    if message_count:
        token_count += 3
    return token_count


def estimate_claude_request_tokens(system, messages, model: str | None, tools=None) -> int:
    texts = []

    if isinstance(system, str):
        texts.append(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))

    for message in messages or []:
        role = message.get("role")
        if role:
            texts.append(str(role))
        texts.extend(_extract_claude_content_texts(message.get("content")))

    for tool in tools or []:
        if tool.get("name"):
            texts.append(str(tool["name"]))
        if tool.get("description"):
            texts.append(str(tool["description"]))
        if tool.get("input_schema") is not None:
            texts.append(json.dumps(tool["input_schema"], ensure_ascii=False, sort_keys=True))

    return estimate_token_by_model(model, "\n".join(texts))


def _extract_message_texts(message) -> list[str]:
    content = message.get("content")
    if isinstance(content, str):
        return [content]

    texts = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                texts.append(str(part))
                continue
            part_type = part.get("type")
            if part_type == "text":
                texts.append(part.get("text", ""))
            elif part_type == "image_url":
                image_url = part.get("image_url", {})
                if isinstance(image_url, dict) and image_url.get("url"):
                    texts.append("[image]")
            elif part_type == "input_audio":
                texts.append("[audio]")
            elif part_type == "file":
                texts.append("[file]")
            else:
                texts.append(json.dumps(part, ensure_ascii=False, sort_keys=True))
        return texts

    if content is not None:
        texts.append(json.dumps(content, ensure_ascii=False, sort_keys=True))

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function", {})
        if function.get("name"):
            texts.append(str(function["name"]))
        if function.get("arguments"):
            texts.append(str(function["arguments"]))

    return texts


def _extract_claude_content_texts(content) -> list[str]:
    if content is None:
        return []
    if isinstance(content, str):
        return [content]

    texts = []
    for block in content:
        if not isinstance(block, dict):
            texts.append(str(block))
            continue
        block_type = block.get("type")
        if block_type == "text":
            texts.append(block.get("text", ""))
        elif block_type == "tool_use":
            if block.get("name"):
                texts.append(str(block["name"]))
            if block.get("input") is not None:
                texts.append(json.dumps(block["input"], ensure_ascii=False, sort_keys=True))
        elif block_type == "tool_result":
            texts.append(_normalize_text(block.get("content")))
        else:
            texts.append(json.dumps(block, ensure_ascii=False, sort_keys=True))
    return texts


def _normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_normalize_text(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") == "text":
            return value.get("text", "")
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _estimate_tokens(text: str, multipliers: Multipliers) -> int:
    count = 0.0
    current_word_type = None

    for char in text:
        if char.isspace():
            current_word_type = None
            if char in "\n\t":
                count += multipliers.newline
            else:
                count += multipliers.space
            continue

        if _is_cjk(char):
            current_word_type = None
            count += multipliers.cjk
            continue

        if _is_emoji(char):
            current_word_type = None
            count += multipliers.emoji
            continue

        if char.isalpha() or char.isnumeric():
            new_type = "number" if char.isnumeric() else "latin"
            if current_word_type != new_type:
                count += multipliers.number if new_type == "number" else multipliers.word
                current_word_type = new_type
            continue

        current_word_type = None
        if _is_math_symbol(char):
            count += multipliers.math_symbol
        elif char == "@":
            count += multipliers.at_sign
        elif char in URL_DELIMS:
            count += multipliers.url_delim
        else:
            count += multipliers.symbol

    return int(math.ceil(count)) + multipliers.base_pad


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7A3
    )


def _is_emoji(char: str) -> bool:
    code = ord(char)
    return (
        0x1F300 <= code <= 0x1F9FF
        or 0x2600 <= code <= 0x26FF
        or 0x2700 <= code <= 0x27BF
        or 0x1FA00 <= code <= 0x1FAFF
    )


def _is_math_symbol(char: str) -> bool:
    code = ord(char)
    return (
        char in MATH_SYMBOLS
        or 0x2200 <= code <= 0x22FF
        or 0x2A00 <= code <= 0x2AFF
        or 0x1D400 <= code <= 0x1D7FF
    )
