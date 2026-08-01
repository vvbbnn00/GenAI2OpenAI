import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from genai_proxy.errors import ProxyError
from genai_proxy.models.hf_assets import (
    HF_HUB_OFFLINE_ENV,
    TOKENIZER_CACHE_ENV,
    TOKENIZER_OFFLINE_ENV,
    Artifact,
    TokenizerSpec,
    artifact_cache_name,
    artifact_path,
)
from genai_proxy.models.hf_prefetch import (
    ACTIVE_TOKENIZER_SPECS,
    prefetch_active_assets,
    unique_artifacts,
    verify_active_assets,
)
from genai_proxy.models.kimi_k3.codec import KIMI_K3_SPEC, build_tokenizer


def _test_spec(content=b"official tokenizer"):
    return TokenizerSpec(
        family="test",
        repository="example/model",
        revision="a" * 40,
        tokenizer=Artifact(
            "tokenizer.json",
            hashlib.sha256(content).hexdigest(),
        ),
    )


@pytest.mark.parametrize(
    ("offline_variable", "offline_value"),
    (
        (TOKENIZER_OFFLINE_ENV, "1"),
        (TOKENIZER_OFFLINE_ENV, "true"),
        (HF_HUB_OFFLINE_ENV, "yes"),
        (HF_HUB_OFFLINE_ENV, "on"),
    ),
)
def test_offline_mode_never_downloads_a_missing_artifact(
    tmp_path,
    monkeypatch,
    offline_variable,
    offline_value,
):
    spec = _test_spec()
    cache_dir = tmp_path / "missing-cache"
    monkeypatch.setenv(TOKENIZER_CACHE_ENV, str(cache_dir))
    monkeypatch.delenv(TOKENIZER_OFFLINE_ENV, raising=False)
    monkeypatch.delenv(HF_HUB_OFFLINE_ENV, raising=False)
    monkeypatch.setenv(offline_variable, offline_value)

    with (
        patch("genai_proxy.models.hf_assets.download_artifact") as download,
        pytest.raises(ProxyError, match="while offline"),
    ):
        artifact_path(spec, spec.tokenizer)

    download.assert_not_called()
    assert not cache_dir.exists()


def test_offline_mode_rejects_a_corrupt_artifact_without_replacing_it(
    tmp_path,
    monkeypatch,
):
    spec = _test_spec()
    destination = tmp_path / artifact_cache_name(spec.tokenizer)
    destination.write_bytes(b"corrupt")
    monkeypatch.setenv(TOKENIZER_CACHE_ENV, str(tmp_path))
    monkeypatch.setenv(TOKENIZER_OFFLINE_ENV, "1")

    with (
        patch("genai_proxy.models.hf_assets.download_artifact") as download,
        pytest.raises(ProxyError, match="while offline"),
    ):
        artifact_path(spec, spec.tokenizer)

    download.assert_not_called()
    assert destination.read_bytes() == b"corrupt"


def test_verify_active_assets_forces_offline_cache_access(monkeypatch):
    calls = []

    def resolve(_spec, artifact, *, allow_download=None):
        calls.append((artifact_cache_name(artifact), allow_download))
        return Path("/cache") / artifact_cache_name(artifact)

    monkeypatch.setattr("genai_proxy.models.hf_prefetch.artifact_path", resolve)

    paths = verify_active_assets()

    assert len(paths) == 8
    assert len(calls) == 8
    assert all(allow_download is False for _, allow_download in calls)


def test_active_asset_manifest_comes_from_the_five_maintained_codecs():
    assert tuple(spec.family for spec in ACTIVE_TOKENIZER_SPECS) == (
        "glm_5_2",
        "deepseek_v4_flash",
        "deepseek_v4_pro",
        "qwen_3_5",
        "kimi_k3",
    )
    assert len(unique_artifacts()) == 8
    assert all("minimax" not in spec.family for spec in ACTIVE_TOKENIZER_SPECS)
    assert all(spec.family != "glm_5_1" for spec in ACTIVE_TOKENIZER_SPECS)


def test_prefetch_downloads_unique_assets_and_loads_every_active_codec(monkeypatch):
    resolved = []

    def resolve(_spec, artifact):
        resolved.append(artifact_cache_name(artifact))
        return Path("/cache") / artifact_cache_name(artifact)

    monkeypatch.setattr("genai_proxy.models.hf_prefetch.artifact_path", resolve)
    with (
        patch("genai_proxy.models.hf_prefetch.load_tokenizer") as load_tokenizer,
        patch("genai_proxy.models.hf_prefetch.load_template") as load_template,
        patch("genai_proxy.models.hf_prefetch.load_python_encoder") as load_encoder,
    ):
        paths = prefetch_active_assets()

    assert len(paths) == 8
    assert len(resolved) == 8
    assert load_tokenizer.call_count == 5
    assert load_template.call_count == 2
    assert load_encoder.call_count == 3
    load_tokenizer.assert_any_call(KIMI_K3_SPEC, factory=build_tokenizer)
