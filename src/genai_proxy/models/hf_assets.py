"""Pinned Hugging Face artifact retrieval and integrity checks."""

import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType

import requests
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from genai_proxy.errors import ProxyError
from genai_proxy.retry import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF,
    is_retryable_status,
    schedule_retry,
)

HF_BASE_URL = "https://huggingface.co"
TOKENIZER_CACHE_ENV = "GENAI_TOKENIZER_CACHE"
TOKENIZER_OFFLINE_ENV = "GENAI_TOKENIZER_OFFLINE"
HF_HUB_OFFLINE_ENV = "HF_HUB_OFFLINE"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

_logger = logging.getLogger(__name__)
_artifact_lock = threading.RLock()
_runtime_lock = threading.RLock()
_encoders = {}
_templates = {}
_tokenizers = {}


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


class ArtifactChecksumError(ValueError):
    pass


def artifact_path(
    spec: TokenizerSpec,
    artifact: Artifact,
    *,
    allow_download: bool | None = None,
) -> Path:
    cache_dir = Path(
        os.environ.get(TOKENIZER_CACHE_ENV)
        or Path.home() / ".cache" / "genai2openai" / "tokenizers"
    )
    filename = artifact_cache_name(artifact)
    destination = cache_dir / filename

    with _artifact_lock:
        if destination.is_file() and sha256(destination) == artifact.sha256:
            return destination
        if allow_download is False or (
            allow_download is None and artifacts_are_offline()
        ):
            raise tokenizer_error(
                spec,
                f"load verified cached {artifact.path} while offline",
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = f"{HF_BASE_URL}/{spec.repository}/resolve/{spec.revision}/{artifact.path}"
        retry_count = 0
        while True:
            try:
                download_artifact(url, cache_dir, destination, artifact.sha256)
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
                raise tokenizer_error(
                    spec,
                    f"download {artifact.path}",
                    exc,
                ) from exc


def artifact_cache_name(artifact: Artifact) -> str:
    return f"{artifact.sha256[:12]}-{Path(artifact.path).name}"


def artifacts_are_offline() -> bool:
    return any(
        os.environ.get(name, "").strip().lower() in _TRUE_VALUES
        for name in (TOKENIZER_OFFLINE_ENV, HF_HUB_OFFLINE_ENV)
    )


def download_artifact(
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
        if sha256(temporary_path) != expected_sha256:
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


def load_tokenizer(spec: TokenizerSpec, *, factory=None):
    cache_key = (spec.tokenizer.sha256, factory)
    with _runtime_lock:
        tokenizer = _tokenizers.get(cache_key)
        if tokenizer is None:
            path = artifact_path(spec, spec.tokenizer)
            try:
                tokenizer = (
                    factory(path)
                    if factory is not None
                    else Tokenizer.from_file(str(path))
                )
            except Exception as exc:
                raise tokenizer_error(spec, "load tokenizer", exc) from exc
            _tokenizers[cache_key] = tokenizer
        return tokenizer


def load_template(spec: TokenizerSpec):
    with _runtime_lock:
        template = _templates.get(spec.family)
        if template is None:
            if spec.template is None:
                raise tokenizer_error(spec, "load missing chat template")
            source = artifact_path(spec, spec.template).read_text(encoding="utf-8")
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


def load_python_encoder(spec: TokenizerSpec):
    with _runtime_lock:
        cache_key = spec.encoder.sha256 if spec.encoder else spec.family
        encoder = _encoders.get(cache_key)
        if encoder is None:
            if spec.encoder is None:
                raise tokenizer_error(spec, "load missing message encoder")
            path = artifact_path(spec, spec.encoder)
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
                raise tokenizer_error(spec, "load message encoder", exc) from exc
            _encoders[cache_key] = encoder
        return encoder


def tokenizer_error(
    spec: TokenizerSpec,
    operation: str,
    exc: Exception | None = None,
) -> ProxyError:
    detail = f": {exc}" if exc else ""
    return ProxyError(
        f"Unable to {operation} for {spec.repository}{detail}",
        error_type="api_error",
        code="tokenizer_unavailable",
        status=503,
    )


def sha256(path: Path) -> str:
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


__all__ = [
    "Artifact",
    "ArtifactChecksumError",
    "HF_HUB_OFFLINE_ENV",
    "HF_BASE_URL",
    "TOKENIZER_CACHE_ENV",
    "TOKENIZER_OFFLINE_ENV",
    "TokenizerSpec",
    "artifact_cache_name",
    "artifact_path",
    "artifacts_are_offline",
    "download_artifact",
    "load_python_encoder",
    "load_template",
    "load_tokenizer",
    "sha256",
    "tokenizer_error",
]
