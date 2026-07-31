import base64
import io
import ipaddress
import json
import math
import re
import socket
import urllib.parse
from dataclasses import dataclass
from types import MappingProxyType

from PIL import Image, ImageFile, UnidentifiedImageError
from urllib3 import HTTPConnectionPool, HTTPSConnectionPool, Timeout
from urllib3.exceptions import HTTPError

from genai_proxy.errors import ProxyError
from genai_proxy.models import hf_assets as _hf_assets
from genai_proxy.models.deepseek_v4 import codec as _deepseek_codec
from genai_proxy.models.glm52 import codec as _glm_codec
from genai_proxy.models.kimi_k3 import codec as _kimi_codec
from genai_proxy.models.legacy import minimax_codec as _minimax_codec
from genai_proxy.models.qwen35 import codec as _qwen_codec
from genai_proxy.models.registry import (
    DEEPSEEK_V4_FLASH_ADAPTER,
    DEEPSEEK_V4_PRO_ADAPTER,
    GLM_5_1_ADAPTER,
    GLM_5_2_ADAPTER,
    KIMI_K3_ADAPTER,
    MINIMAX_ADAPTER,
    QWEN_3_5_ADAPTER,
)

# Keep the token_usage facade stable while implementations live with their
# model families. These bindings also make every re-export explicit to linters.
DEEPSEEK_V4_FLASH_SPEC = _deepseek_codec.DEEPSEEK_V4_FLASH_SPEC
DEEPSEEK_V4_PRO_SPEC = _deepseek_codec.DEEPSEEK_V4_PRO_SPEC
_deepseek_reasoning_prefix = _deepseek_codec.official_reasoning_prefix
_deepseek_tool_prompt = _deepseek_codec.official_tool_prompt
_deepseek_transport_messages = _deepseek_codec.official_transport_messages
_serialize_deepseek_completion = _deepseek_codec.serialize_completion

GLM_5_1_SPEC = _glm_codec.GLM_5_1_SPEC
GLM_5_2_SPEC = _glm_codec.GLM_5_2_SPEC
_glm_tool_prompt = _glm_codec.official_tool_prompt
_serialize_glm_completion = _glm_codec.serialize_completion

HF_BASE_URL = _hf_assets.HF_BASE_URL
TOKENIZER_CACHE_ENV = _hf_assets.TOKENIZER_CACHE_ENV
Artifact = _hf_assets.Artifact
ArtifactChecksumError = _hf_assets.ArtifactChecksumError
TokenizerSpec = _hf_assets.TokenizerSpec
_artifact_path = _hf_assets.artifact_path
_download_artifact = _hf_assets.download_artifact
_load_hf_tokenizer = _hf_assets.load_tokenizer
_load_python_encoder = _hf_assets.load_python_encoder
_load_template = _hf_assets.load_template
_sha256 = _hf_assets.sha256
_tokenizer_error = _hf_assets.tokenizer_error

KIMI_K3_SPEC = _kimi_codec.KIMI_K3_SPEC
KIMI_IMAGE_IN_PATCH_LIMIT = _kimi_codec.IMAGE_IN_PATCH_LIMIT
KIMI_IMAGE_MERGE_KERNEL_SIZE = _kimi_codec.IMAGE_MERGE_KERNEL_SIZE
KIMI_IMAGE_PATCH_LIMIT = _kimi_codec.IMAGE_PATCH_LIMIT
KIMI_IMAGE_PATCH_SIZE = _kimi_codec.IMAGE_PATCH_SIZE
KIMI_PAT_STR = _kimi_codec.PATTERN
KIMI_SPECIAL_TOKEN_OVERRIDES = _kimi_codec.SPECIAL_TOKEN_OVERRIDES
_build_kimi_tokenizer = _kimi_codec.build_tokenizer
_kimi_codec_image_token_count = _kimi_codec.image_token_count

MINIMAX_M2_7_SPEC = _minimax_codec.MINIMAX_M2_7_SPEC
_minimax_default_system_prompt = _minimax_codec.official_default_system_prompt
_minimax_tool_prompt = _minimax_codec.official_tool_prompt
_serialize_minimax_completion = _minimax_codec.serialize_completion

QWEN_3_5_SPEC = _qwen_codec.QWEN_3_5_SPEC
QWEN_IMAGE_MAX_ASPECT_RATIO = _qwen_codec.IMAGE_MAX_ASPECT_RATIO
QWEN_IMAGE_MAX_PIXELS = _qwen_codec.IMAGE_MAX_PIXELS
QWEN_IMAGE_MERGE_SIZE = _qwen_codec.IMAGE_MERGE_SIZE
QWEN_IMAGE_MIN_PIXELS = _qwen_codec.IMAGE_MIN_PIXELS
QWEN_IMAGE_PATCH_SIZE = _qwen_codec.IMAGE_PATCH_SIZE
_qwen_codec_image_token_count = _qwen_codec.image_token_count
_qwen_tool_prompt = _qwen_codec.official_tool_prompt
_serialize_qwen_completion = _qwen_codec.serialize_completion

KIMI_IMAGE_MAX_BYTES = 50 * 1024 * 1024
KIMI_IMAGE_MAX_REDIRECTS = 5
# Common fake-IP range used by transparent DNS proxies such as Mihomo.
KIMI_IMAGE_TRANSPARENT_PROXY_NETWORKS = (ipaddress.ip_network("198.18.0.0/15"),)


SPECS = MappingProxyType(
    {
        spec.family: spec
        for spec in (
            GLM_5_1_SPEC,
            GLM_5_2_SPEC,
            DEEPSEEK_V4_PRO_SPEC,
            DEEPSEEK_V4_FLASH_SPEC,
            QWEN_3_5_SPEC,
            MINIMAX_M2_7_SPEC,
            KIMI_K3_SPEC,
        )
    }
)

def tokenizer_family_for_model(
    model: str | None,
    model_record: dict | None = None,
    tool_adapter: str | None = None,
) -> str | None:
    if tool_adapter == GLM_5_1_ADAPTER:
        return GLM_5_1_SPEC.family
    if tool_adapter == GLM_5_2_ADAPTER:
        return GLM_5_2_SPEC.family
    if tool_adapter == DEEPSEEK_V4_PRO_ADAPTER:
        return DEEPSEEK_V4_PRO_SPEC.family
    if tool_adapter == DEEPSEEK_V4_FLASH_ADAPTER:
        return DEEPSEEK_V4_FLASH_SPEC.family
    if tool_adapter == KIMI_K3_ADAPTER:
        return KIMI_K3_SPEC.family
    if tool_adapter == QWEN_3_5_ADAPTER:
        return QWEN_3_5_SPEC.family
    if tool_adapter == MINIMAX_ADAPTER:
        return MINIMAX_M2_7_SPEC.family

    text = _model_text(model, model_record)
    if _contains_kimi_k3_version(text):
        return KIMI_K3_SPEC.family
    if _contains_version(text, "qwen", "3", "5"):
        return QWEN_3_5_SPEC.family
    if _contains_version(text, "glm", "5", "2"):
        return GLM_5_2_SPEC.family
    if _contains_version(text, "glm", "5", "1"):
        return GLM_5_1_SPEC.family
    if (
        "minimax-m1" in text
        or "minimax m1" in text
        or _contains_version(text, "minimax", "2", "7")
    ):
        return MINIMAX_M2_7_SPEC.family
    if "deepseek-pro" in text or "deepseek v4 pro" in text or "deepseek-v4-pro" in text:
        return DEEPSEEK_V4_PRO_SPEC.family
    if (
        "deepseek-chat" in text
        or "deepseek v4 flash" in text
        or "deepseek-v4-flash" in text
    ):
        return DEEPSEEK_V4_FLASH_SPEC.family
    return None


def official_tool_prompt_for_adapter(
    adapter: str | None,
    tools,
) -> str | None:
    function_tools = [
        tool
        for tool in tools or []
        if isinstance(tool, dict) and tool.get("type") == "function"
    ]
    if not function_tools:
        return None

    if adapter in (DEEPSEEK_V4_FLASH_ADAPTER, DEEPSEEK_V4_PRO_ADAPTER):
        spec = (
            DEEPSEEK_V4_PRO_SPEC
            if adapter == DEEPSEEK_V4_PRO_ADAPTER
            else DEEPSEEK_V4_FLASH_SPEC
        )
        return _deepseek_tool_prompt(spec, function_tools)
    if adapter == GLM_5_1_ADAPTER:
        return _glm_tool_prompt(GLM_5_1_SPEC, function_tools)
    if adapter == GLM_5_2_ADAPTER:
        return _glm_tool_prompt(GLM_5_2_SPEC, function_tools)
    if adapter == QWEN_3_5_ADAPTER:
        return _qwen_tool_prompt(function_tools)
    if adapter == MINIMAX_ADAPTER:
        return _minimax_tool_prompt(function_tools)
    return None


def official_default_system_prompt_for_adapter(adapter: str | None) -> str | None:
    if adapter != MINIMAX_ADAPTER:
        return None
    return _minimax_default_system_prompt()


def official_reasoning_prefix_for_adapter(
    adapter: str | None,
    effort: str | None,
) -> str:
    if effort != "max" or adapter not in (
        DEEPSEEK_V4_FLASH_ADAPTER,
        DEEPSEEK_V4_PRO_ADAPTER,
    ):
        return ""
    spec = (
        DEEPSEEK_V4_PRO_SPEC
        if adapter == DEEPSEEK_V4_PRO_ADAPTER
        else DEEPSEEK_V4_FLASH_SPEC
    )
    return _deepseek_reasoning_prefix(spec, effort)


def official_deepseek_transport_messages(
    adapter: str,
    messages,
    tools,
    *,
    reasoning_config: dict | None = None,
    tool_choice_suffix: str = "",
) -> list[dict]:
    if adapter not in (DEEPSEEK_V4_FLASH_ADAPTER, DEEPSEEK_V4_PRO_ADAPTER):
        raise ValueError("DeepSeek V4 transport requires a V4 adapter")

    spec = (
        DEEPSEEK_V4_PRO_SPEC
        if adapter == DEEPSEEK_V4_PRO_ADAPTER
        else DEEPSEEK_V4_FLASH_SPEC
    )
    return _deepseek_transport_messages(
        spec,
        messages,
        tools,
        reasoning_config=reasoning_config,
        tool_choice_suffix=tool_choice_suffix,
    )


def count_openai_request_tokens(
    messages,
    model: str | None,
    *,
    model_record: dict | None = None,
    tool_adapter: str | None = None,
    reasoning_config: dict | None = None,
    thinking: bool | None = None,
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
        thinking=thinking,
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
    thinking: bool | None = None,
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
            thinking=thinking,
            image_sizes=image_sizes,
        )
        completed_count = _count_chat_prompt(
            [*prompt_messages, message],
            family,
            add_generation_prompt=False,
            reasoning_config=reasoning_config,
            thinking=thinking,
            image_sizes=image_sizes,
        )
        return completed_count - prompt_count

    completion = _serialized_completion(
        message,
        family,
        finish_reason=finish_reason,
        thinking=thinking,
    )
    if prompt_messages is None:
        return _count_encoded(family, completion)

    prompt = render_chat_prompt(
        prompt_messages,
        family,
        add_generation_prompt=True,
        reasoning_config=reasoning_config,
        thinking=thinking,
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
    thinking: bool | None = None,
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
        thinking=thinking,
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
    thinking: bool | None = None,
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
        if thinking is False or effort == "none":
            effort = None
        return encode_messages(
            normalized_messages,
            thinking_mode="chat" if thinking is False else "thinking",
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
        "enable_thinking": thinking is not False,
        "clear_thinking": True,
        "add_vision_id": False,
    }
    if effort:
        context["reasoning_effort"] = effort
    return template.render(**context)


def kimi_image_sizes_for_messages(messages) -> tuple[tuple[int, int], ...]:
    return _image_sizes_for_messages(messages, model_name="Kimi K3")


def qwen_image_sizes_for_messages(messages) -> tuple[tuple[int, int], ...]:
    return _image_sizes_for_messages(messages, model_name="Qwen 3.5")


def _image_sizes_for_messages(
    messages,
    *,
    model_name: str,
) -> tuple[tuple[int, int], ...]:
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
                    f"{model_name} accepts image content only in user messages",
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
    thinking: bool | None = None,
    image_sizes=None,
) -> int:
    if family != KIMI_K3_SPEC.family:
        prompt = render_chat_prompt(
            messages,
            family,
            add_generation_prompt=add_generation_prompt,
            reasoning_config=reasoning_config,
            thinking=thinking,
        )
        count = _count_encoded(family, prompt)
        if family == QWEN_3_5_SPEC.family:
            if image_sizes is None:
                image_sizes = qwen_image_sizes_for_messages(messages)
            count += sum(
                _qwen_image_token_count(width, height) - 1
                for width, height in image_sizes
            )
        return count

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
        # The GenAI web transport drops native tool declarations and invokes
        # K3 with its no-tool default. The official encoder renders that
        # upstream default as a final type="tool-choice" system message.
        tool_choice="none",
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
                if content_length is not None and content_length > KIMI_IMAGE_MAX_BYTES:
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
    return _kimi_codec_image_token_count(width, height)


def _qwen_image_token_count(width: int, height: int) -> int:
    width, height = _validated_image_size((width, height))
    if max(width, height) / min(width, height) > QWEN_IMAGE_MAX_ASPECT_RATIO:
        raise _invalid_image("Qwen 3.5 image aspect ratio must not exceed 200:1")

    return _qwen_codec_image_token_count(width, height)


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
    factory = _build_kimi_tokenizer if spec.family == KIMI_K3_SPEC.family else None
    return _load_hf_tokenizer(spec, factory=factory)


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
    thinking: bool | None = None,
) -> str:
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []

    if family == KIMI_K3_SPEC.family:
        if finish_reason == "length":
            return "".join(
                segment.text for segment in _kimi_partial_completion_segments(message)
            )
        encoder = _load_python_encoder(KIMI_K3_SPEC)
        common = {
            "tools": None,
            "thinking": True,
            "image_prompts": None,
            "thinking_effort": "max",
        }
        generation_prompt = "".join(
            segment.text
            for segment in encoder["build_chat_segments"](
                [],
                add_generation_prompt=True,
                **common,
            )
        )
        completed_prompt = "".join(
            segment.text
            for segment in encoder["build_chat_segments"](
                _normalize_messages([message], parse_tool_arguments=False),
                add_generation_prompt=False,
                **common,
            )
        )
        if not completed_prompt.startswith(generation_prompt):
            raise _tokenizer_error(KIMI_K3_SPEC, "serialize completion")
        return completed_prompt[len(generation_prompt) :]

    if finish_reason == "length" and not content and not tool_calls:
        return (
            str(reasoning).strip() if family == QWEN_3_5_SPEC.family else str(reasoning)
        )

    if family.startswith("deepseek_v4"):
        return _serialize_deepseek_completion(
            message,
            finish_reason=finish_reason,
            thinking=thinking,
        )

    if family in (GLM_5_1_SPEC.family, GLM_5_2_SPEC.family):
        return _serialize_glm_completion(message)

    if family == MINIMAX_M2_7_SPEC.family:
        return _serialize_minimax_completion(
            message,
            finish_reason=finish_reason,
        )

    return _serialize_qwen_completion(
        message,
        finish_reason=finish_reason,
    )


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
