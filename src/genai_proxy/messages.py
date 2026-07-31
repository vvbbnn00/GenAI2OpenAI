from urllib.parse import urlsplit

from genai_proxy.errors import ProxyError
from genai_proxy.models.registry import KIMI_K3_ADAPTER, QWEN_3_5_ADAPTER

_VISUAL_ADAPTERS = frozenset({KIMI_K3_ADAPTER, QWEN_3_5_ADAPTER})


def adapter_supports_vision(adapter: str) -> bool:
    return adapter in _VISUAL_ADAPTERS


def normalize_message_contents(messages: list[dict], *, adapter: str) -> list[dict]:
    """Return copied messages with OpenAI content parts in canonical form."""
    return [_normalize_message(message, adapter=adapter) for message in messages]


def _normalize_message(message: dict, *, adapter: str) -> dict:
    copied = dict(message)
    if "content" in copied:
        copied["content"] = _normalize_content(
            copied["content"],
            role=copied.get("role"),
            adapter=adapter,
        )
    return copied


def _normalize_content(content, *, role, adapter: str):
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _unsupported_content(
            "Message content must be a string, null, or an array of content parts"
        )

    text_parts = []
    canonical_parts = []
    has_image = False
    for part in content:
        if not isinstance(part, dict):
            raise _unsupported_content(
                "Message content arrays must contain objects"
            )
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise _unsupported_content(
                    "Text content parts require a string 'text' field"
                )
            text_parts.append(text)
            canonical_parts.append({"type": "text", "text": text})
            continue

        if part_type not in {"image", "image_url"}:
            raise _unsupported_content(
                f"Unsupported message content part type: {part_type!r}"
            )
        if not adapter_supports_vision(adapter):
            raise _unsupported_content(
                "This model accepts only text message content"
            )
        if role != "user":
            raise _invalid_image("Image content is allowed only in user messages")
        _validate_image_source(part)
        canonical_parts.append(dict(part))
        has_image = True

    if not has_image:
        return "".join(text_parts)
    return canonical_parts


def _validate_image_source(part: dict) -> None:
    part_type = part.get("type")
    source = part.get(part_type)
    if source is None and part_type == "image":
        source = part.get("url")
    if isinstance(source, dict):
        source = source.get("url", source.get("data"))
    if not isinstance(source, str) or not source:
        raise _invalid_image("Image content is missing its URL")

    if source.startswith("data:"):
        header, separator, payload = source.partition(",")
        if (
            not separator
            or not payload
            or not header.lower().startswith("data:image/")
            or ";base64" not in header.lower()
        ):
            raise _invalid_image(
                "Image data URLs must contain non-empty base64 image data"
            )
        return

    try:
        parsed = urlsplit(source)
    except ValueError as exc:
        raise _invalid_image("Image URL is malformed") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise _invalid_image("Image URLs must use http, https, or an image data URL")


def _unsupported_content(message: str) -> ProxyError:
    return ProxyError(
        message,
        error_type="invalid_request_error",
        code="unsupported_content_type",
        status=400,
    )


def _invalid_image(message: str) -> ProxyError:
    return ProxyError(
        message,
        error_type="invalid_request_error",
        code="invalid_image",
        status=400,
    )
