"""Normalization and assembly of streamed tool-call deltas."""

import uuid


def normalize_stream_tool_call(tool_call: dict) -> dict:
    function_data = tool_call.get("function") or {}
    return {
        "index": tool_call.get("index", 0),
        "id": tool_call.get("id"),
        "type": tool_call.get("type", "function"),
        "function": {
            "name": function_data.get("name"),
            "arguments": function_data.get("arguments", ""),
        },
    }


def merge_tool_call_deltas(tool_call_deltas: list[dict]) -> list[dict]:
    merged = {}
    order = []

    for delta in tool_call_deltas:
        index = delta.get("index", len(order))
        if index not in merged:
            merged[index] = {
                "id": delta.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": delta.get("type", "function"),
                "function": {"name": "", "arguments": ""},
            }
            order.append(index)

        current = merged[index]
        if delta.get("id"):
            current["id"] = delta["id"]
        if delta.get("type"):
            current["type"] = delta["type"]

        function_data = delta.get("function") or {}
        if function_data.get("name"):
            current["function"]["name"] = function_data["name"]
        if function_data.get("arguments") is not None:
            current["function"]["arguments"] += function_data["arguments"]

    return [merged[index] for index in order]


__all__ = ["merge_tool_call_deltas", "normalize_stream_tool_call"]
