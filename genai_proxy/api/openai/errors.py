"""OpenAI-compatible HTTP and streaming error serialization."""

import json
import uuid
from datetime import datetime

from flask import jsonify


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


def make_error_chunk(
    message,
    model="unknown",
    completion_id=None,
    created=None,
):
    cid = completion_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_at = int(created if created is not None else datetime.now().timestamp())
    error_chunk = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created_at,
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


__all__ = ["make_error_chunk", "openai_error"]
