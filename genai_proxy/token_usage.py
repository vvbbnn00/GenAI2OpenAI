import base64
import hashlib
import io
import ipaddress
import json
import logging
import math
import os
import re
import socket
import sys
import tempfile
import threading
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType, ModuleType

import requests
import tiktoken
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from PIL import Image, ImageFile, UnidentifiedImageError
from tiktoken.load import load_tiktoken_bpe
from tokenizers import Tokenizer
from urllib3 import HTTPConnectionPool, HTTPSConnectionPool, Timeout
from urllib3.exceptions import HTTPError

from genai_proxy.errors import ProxyError
from genai_proxy.optimizations.registry import (
    DEEPSEEK_V4_FLASH_ADAPTER,
    DEEPSEEK_V4_PRO_ADAPTER,
    GLM_5_2_ADAPTER,
    KIMI_K3_ADAPTER,
)
from genai_proxy.retry import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF,
    is_retryable_status,
    schedule_retry,
)

HF_BASE_URL = "https://huggingface.co"
TOKENIZER_CACHE_ENV = "GENAI_TOKENIZER_CACHE"
KIMI_IMAGE_MAX_BYTES = 50 * 1024 * 1024
# Values from preprocessor_config.json at KIMI_K3_SPEC.revision.
KIMI_IMAGE_PATCH_SIZE = 14
KIMI_IMAGE_MERGE_KERNEL_SIZE = 2
KIMI_IMAGE_PATCH_LIMIT = 512
KIMI_IMAGE_IN_PATCH_LIMIT = 65536
KIMI_IMAGE_MAX_REDIRECTS = 5
# Common fake-IP range used by transparent DNS proxies such as Mihomo.
KIMI_IMAGE_TRANSPARENT_PROXY_NETWORKS = (ipaddress.ip_network("198.18.0.0/15"),)
_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Artifact:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class TokenizerSpec:
    family: str
    repository: str
    revision: str
    tokenizer: Artifact
    template: Artifact | None = None
    encoder: Artifact | None = None


GLM_5_2_SPEC = TokenizerSpec(
    family="glm_5_2",
    repository="zai-org/GLM-5.2",
    revision="b4734de4facf877f85769a911abafc5283eab3d9",
    tokenizer=Artifact(
        "tokenizer.json",
        "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    ),
    template=Artifact(
        "chat_template.jinja",
        "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679",
    ),
)

DEEPSEEK_V4_PRO_SPEC = TokenizerSpec(
    family="deepseek_v4_pro",
    repository="deepseek-ai/DeepSeek-V4-Pro",
    revision="b5968e9190ef611bbf34a7229255be88a0e937c1",
    tokenizer=Artifact(
        "tokenizer.json",
        "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
    ),
    encoder=Artifact(
        "encoding/encoding_dsv4.py",
        "bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0",
    ),
)

DEEPSEEK_V4_FLASH_SPEC = TokenizerSpec(
    family="deepseek_v4_flash",
    repository="deepseek-ai/DeepSeek-V4-Flash",
    revision="60d8d70770c6776ff598c94bb586a859a38244f1",
    tokenizer=DEEPSEEK_V4_PRO_SPEC.tokenizer,
    encoder=DEEPSEEK_V4_PRO_SPEC.encoder,
)

# The full MoE checkpoint is deliberately the canonical Qwen3.5 source. The
# smaller checkpoints currently publish identical tokenizer assets, but they
# are not used as the revision authority here.
QWEN_3_5_SPEC = TokenizerSpec(
    family="qwen_3_5",
    repository="Qwen/Qwen3.5-397B-A17B",
    revision="8472618112abcbd45acbcdc58436aff4233c23f7",
    tokenizer=Artifact(
        "tokenizer.json",
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    template=Artifact(
        "chat_template.jinja",
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    ),
)

KIMI_K3_SPEC = TokenizerSpec(
    family="kimi_k3",
    repository="moonshotai/Kimi-K3",
    revision="9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
    tokenizer=Artifact(
        "tiktoken.model",
        "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
    ),
    encoder=Artifact(
        "encoding_k3.py",
        "b9cb7ae100fed34b9337f80dacee5abbf7e261fe9b74bc0e76366701d46f5333",
    ),
)

KIMI_PAT_STR = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)
KIMI_SPECIAL_TOKEN_OVERRIDES = {
    163584: "[BOS]",
    163585: "[EOS]",
    163586: "<|end_of_msg|>",
    163587: "<|open|>",
    163588: "<|close|>",
    163589: "<|sep|>",
    163590: "[start_header_id]",
    163591: "[end_header_id]",
    163593: "[EOT]",
    163602: "<|media_begin|>",
    163603: "<|media_content|>",
    163604: "<|media_end|>",
    163605: "<|media_pad|>",
    163649: "<osagent_mode>",
    163838: "[UNK]",
    163839: "[PAD]",
}

SPECS = MappingProxyType(
    {
        spec.family: spec
        for spec in (
            GLM_5_2_SPEC,
            DEEPSEEK_V4_PRO_SPEC,
            DEEPSEEK_V4_FLASH_SPEC,
            QWEN_3_5_SPEC,
            KIMI_K3_SPEC,
        )
    }
)

_cache_lock = threading.RLock()
_tokenizers = {}
_templates = {}
_encoders = {}


def tokenizer_family_for_model(
    model: str | None,
    model_record: dict | None = None,
    tool_adapter: str | None = None,
) -> str | None:
    if tool_adapter == GLM_5_2_ADAPTER:
        return GLM_5_2_SPEC.family
    if tool_adapter == DEEPSEEK_V4_PRO_ADAPTER:
        return DEEPSEEK_V4_PRO_SPEC.family
    if tool_adapter == DEEPSEEK_V4_FLASH_ADAPTER:
        return DEEPSEEK_V4_FLASH_SPEC.family
    if tool_adapter == KIMI_K3_ADAPTER:
        return KIMI_K3_SPEC.family

    text = _model_text(model, model_record)
    if _contains_kimi_k3_version(text):
        return KIMI_K3_SPEC.family
    if _contains_version(text, "qwen", "3", "5"):
        return QWEN_3_5_SPEC.family
    if _contains_version(text, "glm", "5", "2"):
        return GLM_5_2_SPEC.family
    if "deepseek-pro" in text or "deepseek v4 pro" in text or "deepseek-v4-pro" in text:
        return DEEPSEEK_V4_PRO_SPEC.family
    if (
        "deepseek-chat" in text
        or "deepseek v4 flash" in text
        or "deepseek-v4-flash" in text
    ):
        return DEEPSEEK_V4_FLASH_SPEC.family
    return None


def count_openai_request_tokens(
    messages,
    model: str | None,
    *,
    model_record: dict | None = None,
    tool_adapter: str | None = None,
    reasoning_config: dict | None = None,
    image_sizes=None,
) -> int:
    family = tokenizer_family_for_model(model, model_record, tool_adapter)
    if family is None:
        return _estimate_openai_request_tokens(messages, model)

    return _count_chat_prompt(
        messages,
        family,
        add_generation_prompt=True,
        reasoning_config=reasoning_config,
        image_sizes=image_sizes,
    )


def count_openai_completion_tokens(
    message: dict,
    model: str | None,
    *,
    model_record: dict | None = None,
    tool_adapter: str | None = None,
    prompt_messages=None,
    reasoning_config: dict | None = None,
    finish_reason: str = "stop",
    image_sizes=None,
) -> int:
    family = tokenizer_family_for_model(model, model_record, tool_adapter)
    if family is None:
        return estimate_token_by_model(model, _completion_text(message))

    if family == KIMI_K3_SPEC.family:
        if finish_reason == "length":
            return _count_kimi_segments(_kimi_partial_completion_segments(message))
        prompt_messages = prompt_messages or []
        if image_sizes is None:
            image_sizes = kimi_image_sizes_for_messages(prompt_messages)
        prompt_count = _count_chat_prompt(
            prompt_messages,
            family,
            add_generation_prompt=True,
            reasoning_config=reasoning_config,
            image_sizes=image_sizes,
        )
        completed_count = _count_chat_prompt(
            [*prompt_messages, message],
            family,
            add_generation_prompt=False,
            reasoning_config=reasoning_config,
            image_sizes=image_sizes,
        )
        return completed_count - prompt_count

    completion = _serialized_completion(
        message,
        family,
        finish_reason=finish_reason,
    )
    if prompt_messages is None:
        return _count_encoded(family, completion)

    prompt = render_chat_prompt(
        prompt_messages,
        family,
        add_generation_prompt=True,
        reasoning_config=reasoning_config,
    )
    return _count_encoded(family, prompt + completion) - _count_encoded(family, prompt)


def count_text_tokens(
    text: str,
    model: str | None,
    *,
    model_record: dict | None = None,
    tool_adapter: str | None = None,
) -> int:
    if not text:
        return 0
    family = tokenizer_family_for_model(model, model_record, tool_adapter)
    if family is None:
        return _estimate_tokens(text, _multipliers_for_model(model))
    return _count_encoded(family, text)


def count_openai_reasoning_tokens(
    reasoning: str,
    model: str | None,
    *,
    model_record: dict | None = None,
    tool_adapter: str | None = None,
    prompt_messages=None,
    reasoning_config: dict | None = None,
    image_sizes=None,
) -> int:
    if not reasoning:
        return 0
    family = tokenizer_family_for_model(model, model_record, tool_adapter)
    if family == KIMI_K3_SPEC.family:
        return _count_kimi_text(reasoning)
    if family is None or prompt_messages is None:
        return count_text_tokens(
            reasoning,
            model,
            model_record=model_record,
            tool_adapter=tool_adapter,
        )

    prompt = render_chat_prompt(
        prompt_messages,
        family,
        add_generation_prompt=True,
        reasoning_config=reasoning_config,
    )
    rendered_reasoning = (
        reasoning.strip() if family == QWEN_3_5_SPEC.family else reasoning
    )
    return _count_encoded(family, prompt + rendered_reasoning) - _count_encoded(
        family, prompt
    )


def render_chat_prompt(
    messages,
    family: str,
    *,
    add_generation_prompt: bool,
    reasoning_config: dict | None = None,
    image_sizes=None,
) -> str:
    spec = SPECS[family]
    if family == KIMI_K3_SPEC.family:
        segments = _kimi_chat_segments(
            messages,
            add_generation_prompt=add_generation_prompt,
            reasoning_config=reasoning_config,
            image_sizes=image_sizes,
        )
        return "".join(segment.text for segment in segments)

    normalized_messages = _normalize_messages(
        messages,
        parse_tool_arguments=spec.encoder is None,
    )

    if spec.encoder is not None:
        encode_messages = _load_python_encoder(spec)["encode_messages"]
        effort = (reasoning_config or {}).get("effort")
        return encode_messages(
            normalized_messages,
            thinking_mode="thinking",
            drop_thinking=True,
            add_default_bos_token=True,
            reasoning_effort=effort,
        )

    template = _load_template(spec)
    effort = (reasoning_config or {}).get("effort")
    context = {
        "messages": normalized_messages,
        "tools": None,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": True,
        "clear_thinking": True,
        "add_vision_id": False,
    }
    if effort:
        context["reasoning_effort"] = effort
    return template.render(**context)


def kimi_image_sizes_for_messages(messages) -> tuple[tuple[int, int], ...]:
    sizes = []
    for message in messages or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {
                "image",
                "image_url",
            }:
                continue
            if message.get("role") != "user":
                raise ProxyError(
                    "Kimi K3 accepts image content only in user messages",
                    error_type="invalid_request_error",
                    code="invalid_image",
                    status=400,
                )
            sizes.append(_image_size(_image_source(part)))
    return tuple(sizes)


def _count_chat_prompt(
    messages,
    family: str,
    *,
    add_generation_prompt: bool,
    reasoning_config: dict | None = None,
    image_sizes=None,
) -> int:
    if family != KIMI_K3_SPEC.family:
        prompt = render_chat_prompt(
            messages,
            family,
            add_generation_prompt=add_generation_prompt,
            reasoning_config=reasoning_config,
        )
        return _count_encoded(family, prompt)

    if image_sizes is None:
        image_sizes = kimi_image_sizes_for_messages(messages)
    segments = _kimi_chat_segments(
        messages,
        add_generation_prompt=add_generation_prompt,
        reasoning_config=reasoning_config,
        image_sizes=image_sizes,
    )
    return _count_kimi_segments(segments) + sum(
        _kimi_image_token_count(width, height) - 1 for width, height in image_sizes
    )


def _kimi_chat_segments(
    messages,
    *,
    add_generation_prompt: bool,
    reasoning_config: dict | None,
    image_sizes,
):
    encoder = _load_python_encoder(KIMI_K3_SPEC)
    image_prompts = (
        None
        if image_sizes is None
        else [
            (
                f"<|media_begin|>image {width}x{height}"
                "<|media_content|><|media_pad|><|media_end|>"
            )
            for width, height in image_sizes
        ]
    )
    effort = (reasoning_config or {}).get("effort") or "max"
    return encoder["build_chat_segments"](
        _normalize_messages(messages, parse_tool_arguments=False),
        tools=None,
        add_generation_prompt=add_generation_prompt,
        thinking=True,
        image_prompts=image_prompts,
        thinking_effort=effort,
    )


def _kimi_partial_completion_segments(message: dict):
    encoder = _load_python_encoder(KIMI_K3_SPEC)
    segments = []
    reasoning = str(message.get("reasoning_content") or "")
    content = str(message.get("content") or "")
    if reasoning:
        segments.extend(encoder["_text"](reasoning))
    if content or message.get("tool_calls"):
        segments.extend(encoder["_close_tag"]("think"))
        segments.extend(encoder["_open_tag"]("response"))
        if content:
            segments.extend(encoder["_text"](content))
    return segments


def _image_source(part: dict):
    part_type = part.get("type")
    source = part.get(part_type)
    if source is None and part_type == "image":
        source = part.get("url")
    if isinstance(source, dict):
        source = source.get("url", source.get("data"))
    if source is None:
        raise _invalid_image("Image content is missing its source")
    return source


def _image_size(source) -> tuple[int, int]:
    try:
        if isinstance(source, bytes):
            return _image_size_from_bytes(source)
        if not isinstance(source, str):
            raise TypeError(f"unsupported image source type {type(source).__name__}")
        if source.startswith("data:"):
            return _image_size_from_bytes(_decode_data_url(source))
        parsed = urllib.parse.urlparse(source)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("image URL must use http, https, or a data URL")
        return _remote_image_size(source)
    except ProxyError:
        raise
    except Exception as exc:
        raise _invalid_image(f"Unable to read image dimensions: {exc}") from exc


def _decode_data_url(source: str) -> bytes:
    header, separator, payload = source.partition(",")
    if not separator:
        raise ValueError("malformed image data URL")
    if not header.lower().startswith("data:image/") or ";base64" not in header.lower():
        raise ValueError("image data URL must contain base64-encoded image data")
    maximum_encoded_size = 4 * math.ceil(KIMI_IMAGE_MAX_BYTES / 3)
    if len(payload) > maximum_encoded_size:
        raise ValueError("image exceeds the 50 MiB limit")
    data = base64.b64decode(payload, validate=True)
    if len(data) > KIMI_IMAGE_MAX_BYTES:
        raise ValueError("image exceeds the 50 MiB limit")
    return data


def _image_size_from_bytes(data: bytes) -> tuple[int, int]:
    if len(data) > KIMI_IMAGE_MAX_BYTES:
        raise ValueError("image exceeds the 50 MiB limit")
    try:
        with Image.open(io.BytesIO(data)) as image:
            return _validated_image_size(image.size)
    except UnidentifiedImageError as exc:
        raise ValueError("unsupported or corrupt image") from exc


def _remote_image_size(url: str) -> tuple[int, int]:
    response = None
    pool = None
    try:
        current_url = url
        for redirect_count in range(KIMI_IMAGE_MAX_REDIRECTS + 1):
            response, pool = _request_public_image(current_url)
            if response.status in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("image redirect is missing a location")
                if redirect_count == KIMI_IMAGE_MAX_REDIRECTS:
                    raise ValueError("image URL has too many redirects")
                current_url = urllib.parse.urljoin(current_url, location)
                response.close()
                pool.close()
                response = None
                pool = None
                continue

            if not 200 <= response.status < 300:
                raise ValueError(f"image URL returned HTTP {response.status}")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    content_length = int(content_length)
                except (TypeError, ValueError):
                    content_length = None
                if (
                    content_length is not None
                    and content_length > KIMI_IMAGE_MAX_BYTES
                ):
                    raise ValueError("image exceeds the 50 MiB limit")
            parser = ImageFile.Parser()
            total = 0
            for chunk in response.stream(amt=64 * 1024, decode_content=True):
                if not chunk:
                    continue
                total += len(chunk)
                if total > KIMI_IMAGE_MAX_BYTES:
                    raise ValueError("image exceeds the 50 MiB limit")
                parser.feed(chunk)
                if parser.image is not None:
                    return _validated_image_size(parser.image.size)
            raise ValueError("unsupported or corrupt image")
        raise ValueError("image URL has too many redirects")
    finally:
        if response is not None:
            response.close()
        if pool is not None:
            pool.close()


def _request_public_image(url: str):
    parsed, addresses = _resolve_public_image_url(url)
    hostname = parsed.hostname.encode("idna").decode("ascii")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    host_header = f"[{hostname}]" if ":" in hostname else hostname
    if parsed.port is not None:
        host_header = f"{host_header}:{parsed.port}"
    target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    last_error = None

    for address in addresses:
        pool = (
            HTTPSConnectionPool(
                address,
                port=port,
                maxsize=1,
                block=True,
                cert_reqs="CERT_REQUIRED",
                assert_hostname=hostname,
                server_hostname=hostname,
            )
            if parsed.scheme == "https"
            else HTTPConnectionPool(
                address,
                port=port,
                maxsize=1,
                block=True,
            )
        )
        try:
            response = pool.urlopen(
                "GET",
                target,
                headers={
                    "Accept": "image/*",
                    "Host": host_header,
                },
                redirect=False,
                retries=False,
                preload_content=False,
                timeout=Timeout(connect=10, read=30),
            )
            return response, pool
        except HTTPError as exc:
            last_error = exc
            pool.close()

    raise ValueError("unable to fetch image URL") from last_error


def _resolve_public_image_url(
    url: str,
) -> tuple[urllib.parse.SplitResult, tuple[str, ...]]:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("image URL must be a public HTTP(S) URL")

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = socket.getaddrinfo(
            parsed.hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except (OSError, ValueError) as exc:
        raise ValueError("unable to resolve image URL host") from exc

    if not addresses:
        raise ValueError("unable to resolve image URL host")
    try:
        literal_address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal_address = None
    public_addresses = []
    for address in addresses:
        host = address[4][0].split("%", 1)[0]
        parsed_address = ipaddress.ip_address(host)
        if not parsed_address.is_global and not any(
            literal_address is None and parsed_address in network
            for network in KIMI_IMAGE_TRANSPARENT_PROXY_NETWORKS
        ):
            raise ValueError("image URL must not resolve to a private address")
        if host not in public_addresses:
            public_addresses.append(host)
    return parsed, tuple(public_addresses)


def _validated_image_size(size) -> tuple[int, int]:
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return int(width), int(height)


def _invalid_image(message: str) -> ProxyError:
    return ProxyError(
        message,
        error_type="invalid_request_error",
        code="invalid_image",
        status=400,
    )


def _kimi_image_token_count(width: int, height: int) -> int:
    scale = min(
        1.0,
        math.sqrt(
            KIMI_IMAGE_IN_PATCH_LIMIT
            / (
                max(1.0, width // KIMI_IMAGE_PATCH_SIZE)
                * max(1.0, height // KIMI_IMAGE_PATCH_SIZE)
            )
        ),
        KIMI_IMAGE_PATCH_LIMIT * KIMI_IMAGE_PATCH_SIZE / width,
        KIMI_IMAGE_PATCH_LIMIT * KIMI_IMAGE_PATCH_SIZE / height,
    )
    new_width = min(
        max(1, int(width * scale)),
        KIMI_IMAGE_PATCH_LIMIT * KIMI_IMAGE_PATCH_SIZE,
    )
    new_height = min(
        max(1, int(height * scale)),
        KIMI_IMAGE_PATCH_LIMIT * KIMI_IMAGE_PATCH_SIZE,
    )
    factor = KIMI_IMAGE_MERGE_KERNEL_SIZE * KIMI_IMAGE_PATCH_SIZE
    return math.ceil(new_width / factor) * math.ceil(new_height / factor)


def estimate_token_by_model(model: str | None, text: str) -> int:
    return count_text_tokens(text, model)


def estimate_openai_request_tokens(messages, model: str | None, tools=None) -> int:
    # Kept for callers outside the service. Exact supported-model counting is
    # available when the actual model record/adapter is supplied above.
    family = tokenizer_family_for_model(model)
    if family is not None and not tools:
        return count_openai_request_tokens(messages, model)
    return _estimate_openai_request_tokens(messages, model, tools)


def estimate_claude_request_tokens(
    system, messages, model: str | None, tools=None
) -> int:
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
            texts.append(
                json.dumps(tool["input_schema"], ensure_ascii=False, sort_keys=True)
            )
    return estimate_token_by_model(model, "\n".join(texts))


def _load_tokenizer(spec: TokenizerSpec):
    with _cache_lock:
        tokenizer = _tokenizers.get(spec.tokenizer.sha256)
        if tokenizer is None:
            path = _artifact_path(spec, spec.tokenizer)
            try:
                if spec.family == KIMI_K3_SPEC.family:
                    mergeable_ranks = load_tiktoken_bpe(str(path))
                    base_tokens = len(mergeable_ranks)
                    special_tokens = {
                        KIMI_SPECIAL_TOKEN_OVERRIDES.get(
                            token_id, f"<|reserved_token_{token_id}|>"
                        ): token_id
                        for token_id in range(base_tokens, base_tokens + 256)
                    }
                    tokenizer = tiktoken.Encoding(
                        name=path.name,
                        pat_str=KIMI_PAT_STR,
                        mergeable_ranks=mergeable_ranks,
                        special_tokens=special_tokens,
                    )
                else:
                    tokenizer = Tokenizer.from_file(str(path))
            except Exception as exc:
                raise _tokenizer_error(spec, "load tokenizer", exc) from exc
            _tokenizers[spec.tokenizer.sha256] = tokenizer
        return tokenizer


def _load_template(spec: TokenizerSpec):
    with _cache_lock:
        template = _templates.get(spec.family)
        if template is None:
            if spec.template is None:
                raise _tokenizer_error(spec, "load missing chat template")
            source = _artifact_path(spec, spec.template).read_text(encoding="utf-8")
            environment = ImmutableSandboxedEnvironment(
                trim_blocks=True,
                lstrip_blocks=True,
                autoescape=False,
                extensions=["jinja2.ext.loopcontrols"],
            )
            environment.filters["tojson"] = _tojson
            environment.globals["raise_exception"] = _raise_template_exception
            environment.globals["strftime_now"] = _strftime_now
            template = environment.from_string(source)
            _templates[spec.family] = template
        return template


def _load_python_encoder(spec: TokenizerSpec):
    with _cache_lock:
        cache_key = spec.encoder.sha256 if spec.encoder else spec.family
        encoder = _encoders.get(cache_key)
        if encoder is None:
            if spec.encoder is None:
                raise _tokenizer_error(spec, "load missing message encoder")
            path = _artifact_path(spec, spec.encoder)
            module_name = f"_genai_{spec.family}_encoding"
            module = ModuleType(module_name)
            module.__file__ = str(path)
            namespace = module.__dict__
            try:
                sys.modules[module_name] = module
                source = path.read_text(encoding="utf-8")
                exec(compile(source, str(path), "exec"), namespace)
                encoder = namespace
            except Exception as exc:
                sys.modules.pop(module_name, None)
                raise _tokenizer_error(spec, "load message encoder", exc) from exc
            _encoders[cache_key] = encoder
        return encoder


def _artifact_path(spec: TokenizerSpec, artifact: Artifact) -> Path:
    cache_dir = Path(
        os.environ.get(TOKENIZER_CACHE_ENV)
        or Path.home() / ".cache" / "genai2openai" / "tokenizers"
    )
    filename = f"{artifact.sha256[:12]}-{Path(artifact.path).name}"
    destination = cache_dir / filename

    with _cache_lock:
        if destination.is_file() and _sha256(destination) == artifact.sha256:
            return destination
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = f"{HF_BASE_URL}/{spec.repository}/resolve/{spec.revision}/{artifact.path}"
        retry_count = 0
        while True:
            try:
                _download_artifact(url, cache_dir, destination, artifact.sha256)
                return destination
            except Exception as exc:
                status_code = getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                retryable = (
                    isinstance(exc, requests.RequestException)
                    and (status_code is None or is_retryable_status(status_code))
                ) or isinstance(exc, ArtifactChecksumError)
                if retryable and schedule_retry(
                    _logger,
                    max_retries=DEFAULT_MAX_RETRIES,
                    backoff=DEFAULT_RETRY_BACKOFF,
                    retry_count=retry_count,
                    operation=f"tokenizer artifact download for {spec.repository}",
                    reason=str(exc),
                ):
                    retry_count += 1
                    continue
                raise _tokenizer_error(spec, f"download {artifact.path}", exc) from exc


class ArtifactChecksumError(ValueError):
    pass


def _download_artifact(
    url: str,
    cache_dir: Path,
    destination: Path,
    expected_sha256: str,
) -> None:
    temporary_path = None
    response = None
    try:
        response = requests.get(url, stream=True, timeout=(10, 120))
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    temporary.write(chunk)
        if _sha256(temporary_path) != expected_sha256:
            raise ArtifactChecksumError("downloaded artifact checksum mismatch")
        temporary_path.replace(destination)
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _count_encoded(family: str, text: str) -> int:
    spec = SPECS[family]
    try:
        if family == KIMI_K3_SPEC.family:
            return _count_kimi_text(text)
        return len(_load_tokenizer(spec).encode(text, add_special_tokens=False).ids)
    except ProxyError:
        raise
    except Exception as exc:
        raise _tokenizer_error(spec, "encode prompt", exc) from exc


def _count_kimi_segments(segments) -> int:
    return sum(
        _count_kimi_text(segment.text, allow_special=segment.allow_special)
        for segment in segments
    )


def _count_kimi_text(text: str, *, allow_special: bool = False) -> int:
    if not text:
        return 0
    try:
        tokenizer = _load_tokenizer(KIMI_K3_SPEC)
        count = 0
        for offset in range(0, len(text), 400_000):
            piece = text[offset : offset + 400_000]
            for substring in _split_whitespace_runs(piece, 25_000):
                if allow_special:
                    count += len(tokenizer.encode(substring, allowed_special="all"))
                else:
                    count += len(tokenizer.encode(substring, disallowed_special=()))
        return count
    except ProxyError:
        raise
    except Exception as exc:
        raise _tokenizer_error(KIMI_K3_SPEC, "encode prompt", exc) from exc


def _split_whitespace_runs(text: str, maximum: int):
    start = 0
    run_length = 0
    current_is_space = text[0].isspace() if text else False
    for index, character in enumerate(text):
        is_space = character.isspace()
        if current_is_space != is_space:
            run_length = 1
            current_is_space = is_space
        else:
            run_length += 1
            if run_length > maximum:
                yield text[start:index]
                start = index
                run_length = 1
    yield text[start:]


def _tokenizer_error(
    spec: TokenizerSpec, operation: str, exc: Exception | None = None
) -> ProxyError:
    detail = f": {exc}" if exc else ""
    return ProxyError(
        f"Unable to {operation} for {spec.repository}{detail}",
        error_type="api_error",
        code="tokenizer_unavailable",
        status=503,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tojson(
    value,
    ensure_ascii=False,
    indent=None,
    separators=None,
    sort_keys=False,
):
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def _raise_template_exception(message):
    raise TemplateError(message)


def _strftime_now(format_string):
    return datetime.now().strftime(format_string)


def _normalize_messages(messages, *, parse_tool_arguments: bool) -> list[dict]:
    normalized = []
    for original in messages or []:
        message = dict(original)
        tool_calls = []
        for original_call in message.get("tool_calls") or []:
            tool_call = dict(original_call)
            function = dict(tool_call.get("function") or {})
            arguments = function.get("arguments", {})
            if parse_tool_arguments and isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"arguments": arguments}
            function["arguments"] = arguments
            tool_call["function"] = function
            tool_calls.append(tool_call)
        if tool_calls:
            message["tool_calls"] = tool_calls
        normalized.append(message)
    return normalized


def _serialized_completion(
    message: dict,
    family: str,
    *,
    finish_reason: str = "stop",
) -> str:
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []

    if family == KIMI_K3_SPEC.family:
        if finish_reason == "length":
            return "".join(
                segment.text for segment in _kimi_partial_completion_segments(message)
            )
        generation_prompt = render_chat_prompt(
            [],
            family,
            add_generation_prompt=True,
            image_sizes=(),
        )
        completed_prompt = render_chat_prompt(
            [message],
            family,
            add_generation_prompt=False,
            image_sizes=(),
        )
        if not completed_prompt.startswith(generation_prompt):
            raise _tokenizer_error(KIMI_K3_SPEC, "serialize completion")
        return completed_prompt[len(generation_prompt) :]

    if finish_reason == "length" and not content and not tool_calls:
        return (
            str(reasoning).strip() if family == QWEN_3_5_SPEC.family else str(reasoning)
        )

    include_end_token = finish_reason != "length"

    if family.startswith("deepseek_v4"):
        parts = [str(reasoning), "</think>"]
        parts.append(str(content))
        if tool_calls:
            parts.append(_deepseek_tool_calls(tool_calls))
        if include_end_token:
            parts.append("<｜end▁of▁sentence｜>")
        return "".join(parts)

    if family == GLM_5_2_SPEC.family:
        rendered_content = (
            "None"
            if "content" in message and message.get("content") is None
            else str(content).strip()
        )
        parts = [str(reasoning), "</think>", rendered_content]
        parts.extend(_glm_tool_call(call) for call in tool_calls)
        return "".join(parts)

    rendered_content = str(content).strip()
    parts = [str(reasoning).strip(), "\n</think>\n\n", rendered_content]
    for index, call in enumerate(tool_calls):
        if index == 0:
            parts.append("\n\n" if rendered_content else "")
        else:
            parts.append("\n")
        parts.append(_qwen_tool_call(call))
    if include_end_token:
        parts.append("<|im_end|>\n")
    return "".join(parts)


def _deepseek_tool_calls(tool_calls) -> str:
    calls = []
    for call in tool_calls:
        function = call.get("function") or {}
        arguments = _json_arguments(function.get("arguments"))
        parameters = []
        for key, value in arguments.items():
            is_string = isinstance(value, str)
            rendered = value if is_string else json.dumps(value, ensure_ascii=False)
            parameters.append(
                f'<｜DSML｜parameter name="{key}" string="{str(is_string).lower()}">'
                f"{rendered}</｜DSML｜parameter>"
            )
        calls.append(
            f'<｜DSML｜invoke name="{function.get("name", "")}">\n'
            + "\n".join(parameters)
            + "\n</｜DSML｜invoke>"
        )
    return "\n\n<｜DSML｜tool_calls>\n" + "\n".join(calls) + "\n</｜DSML｜tool_calls>"


def _glm_tool_call(call) -> str:
    function = call.get("function") or {}
    arguments = _json_arguments(function.get("arguments"))
    rendered = "".join(
        f"<arg_key>{key}</arg_key><arg_value>"
        f"{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
        "</arg_value>"
        for key, value in arguments.items()
    )
    return f"<tool_call>{function.get('name', '')}{rendered}</tool_call>"


def _qwen_tool_call(call) -> str:
    function = call.get("function") or {}
    arguments = _json_arguments(function.get("arguments"))
    parameters = "".join(
        f"<parameter={key}>\n{_qwen_argument_value(value)}\n</parameter>\n"
        for key, value in arguments.items()
    )
    return (
        f"<tool_call>\n<function={function.get('name', '')}>\n"
        f"{parameters}</function>\n</tool_call>"
    )


def _qwen_argument_value(value) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _json_arguments(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"arguments": str(value or "")}
    return parsed if isinstance(parsed, dict) else {"arguments": parsed}


def _completion_text(message: dict) -> str:
    texts = [
        str(message.get("reasoning_content") or ""),
        str(message.get("content") or ""),
    ]
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        texts.extend(
            (str(function.get("name") or ""), str(function.get("arguments") or ""))
        )
    return "".join(texts)


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


OPENAI_MULTIPLIERS = Multipliers(1.02, 1.55, 0.85, 0.4, 2.68, 1.0, 2.0, 2.12, 0.5, 0.42)
CLAUDE_MULTIPLIERS = Multipliers(
    1.13, 1.63, 1.21, 0.4, 4.52, 1.26, 2.82, 2.6, 0.89, 0.39
)
GEMINI_MULTIPLIERS = Multipliers(1.15, 2.8, 0.68, 0.38, 1.05, 1.2, 2.5, 1.08, 1.15, 0.2)
MATH_SYMBOLS = set(
    "∑∫∂√∞≤≥≠≈±×÷∈∉∋∌⊂⊃⊆⊇∪∩∧∨¬∀∃∄∅∆∇∝∟∠∡∢°′″‴⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎²³¹⁴⁵⁶⁷⁸⁹⁰"
)
URL_DELIMS = set("/:?&=;#%")


def _estimate_openai_request_tokens(messages, model: str | None, tools=None) -> int:
    texts = []
    message_count = 0
    name_count = 0
    tool_count = 0
    for message in messages or []:
        message_count += 1
        if message.get("role"):
            texts.append(str(message["role"]))
        if message.get("name"):
            name_count += 1
            texts.append(str(message["name"]))
        texts.extend(_extract_message_texts(message))
    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        tool_count += 1
        function = tool.get("function", {})
        texts.extend(
            str(function[key]) for key in ("name", "description") if function.get(key)
        )
        if function.get("parameters") is not None:
            texts.append(
                json.dumps(function["parameters"], ensure_ascii=False, sort_keys=True)
            )
    count = _estimate_tokens("\n".join(texts), _multipliers_for_model(model))
    return (
        count
        + tool_count * 8
        + message_count * 3
        + name_count * 3
        + (3 if message_count else 0)
    )


def _multipliers_for_model(model: str | None) -> Multipliers:
    lowered = (model or "").lower()
    if "gemini" in lowered:
        return GEMINI_MULTIPLIERS
    if "claude" in lowered:
        return CLAUDE_MULTIPLIERS
    return OPENAI_MULTIPLIERS


def _extract_message_texts(message) -> list[str]:
    content = message.get("content")
    if isinstance(content, str):
        texts = [content]
    elif isinstance(content, list):
        texts = []
        for part in content:
            if not isinstance(part, dict):
                texts.append(str(part))
            elif part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif part.get("type") in ("image_url", "input_audio", "file"):
                texts.append(f"[{part.get('type')}]")
            else:
                texts.append(json.dumps(part, ensure_ascii=False, sort_keys=True))
    elif content is None:
        texts = []
    else:
        texts = [json.dumps(content, ensure_ascii=False, sort_keys=True)]
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        texts.extend(
            str(function[key]) for key in ("name", "arguments") if function.get(key)
        )
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
        elif block.get("type") == "text":
            texts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            texts.append(str(block.get("name") or ""))
            texts.append(
                json.dumps(block.get("input", {}), ensure_ascii=False, sort_keys=True)
            )
        elif block.get("type") == "tool_result":
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
    if isinstance(value, dict) and value.get("type") == "text":
        return value.get("text", "")
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _estimate_tokens(text: str, multipliers: Multipliers) -> int:
    count = 0.0
    current_word_type = None
    for char in text:
        if char.isspace():
            current_word_type = None
            count += multipliers.newline if char in "\n\t" else multipliers.space
        elif _is_cjk(char):
            current_word_type = None
            count += multipliers.cjk
        elif _is_emoji(char):
            current_word_type = None
            count += multipliers.emoji
        elif char.isalpha() or char.isnumeric():
            new_type = "number" if char.isnumeric() else "latin"
            if current_word_type != new_type:
                count += (
                    multipliers.number if new_type == "number" else multipliers.word
                )
                current_word_type = new_type
        else:
            current_word_type = None
            if _is_math_symbol(char):
                count += multipliers.math_symbol
            elif char == "@":
                count += multipliers.at_sign
            elif char in URL_DELIMS:
                count += multipliers.url_delim
            else:
                count += multipliers.symbol
    return max(1, math.ceil(count)) if text else 0


def _model_text(model: str | None, record: dict | None) -> str:
    values = [model or ""]
    for key in (
        "aiType",
        "aiName",
        "simpleName",
        "descInfo",
        "descInfoEn",
        "rootModelName",
    ):
        if record and record.get(key) is not None:
            values.append(str(record[key]))
    return " ".join(values).lower()


def _contains_version(text: str, name: str, major: str, minor: str) -> bool:
    return bool(re.search(rf"{name}[\s_.-]*{major}(?:[\s_.-]*{minor})(?!\d)", text))


def _contains_kimi_k3_version(text: str) -> bool:
    return bool(re.search(r"(?<![\w])kimi[\s_.-]*k?3(?![\s_.-]*\d)", text))


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7A3
    )


def _is_emoji(char: str) -> bool:
    code = ord(char)
    return 0x1F000 <= code <= 0x1FAFF or 0x2600 <= code <= 0x27BF


def _is_math_symbol(char: str) -> bool:
    code = ord(char)
    return (
        char in MATH_SYMBOLS
        or 0x2200 <= code <= 0x22FF
        or 0x2A00 <= code <= 0x2AFF
        or 0x1D400 <= code <= 0x1D7FF
    )
