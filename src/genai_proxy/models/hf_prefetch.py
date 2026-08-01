"""Build-time preparation for pinned official Hugging Face assets."""

import argparse
from collections.abc import Iterable

from genai_proxy.models.deepseek_v4.codec import (
    DEEPSEEK_V4_FLASH_SPEC,
    DEEPSEEK_V4_PRO_SPEC,
)
from genai_proxy.models.glm52.codec import GLM_5_2_SPEC
from genai_proxy.models.hf_assets import (
    Artifact,
    TokenizerSpec,
    artifact_cache_name,
    artifact_path,
    load_python_encoder,
    load_template,
    load_tokenizer,
)
from genai_proxy.models.kimi_k3.codec import KIMI_K3_SPEC, build_tokenizer
from genai_proxy.models.qwen35.codec import QWEN_3_5_SPEC

ACTIVE_TOKENIZER_SPECS = (
    GLM_5_2_SPEC,
    DEEPSEEK_V4_FLASH_SPEC,
    DEEPSEEK_V4_PRO_SPEC,
    QWEN_3_5_SPEC,
    KIMI_K3_SPEC,
)


def iter_spec_artifacts(spec: TokenizerSpec):
    for artifact in (spec.tokenizer, spec.template, spec.encoder):
        if artifact is not None:
            yield artifact


def unique_artifacts(
    specs: Iterable[TokenizerSpec] = ACTIVE_TOKENIZER_SPECS,
) -> tuple[tuple[TokenizerSpec, Artifact], ...]:
    unique = []
    seen = set()
    for spec in specs:
        for artifact in iter_spec_artifacts(spec):
            cache_key = (artifact.sha256, artifact_cache_name(artifact))
            if cache_key in seen:
                continue
            seen.add(cache_key)
            unique.append((spec, artifact))
    return tuple(unique)


def prefetch_active_assets(
    specs: Iterable[TokenizerSpec] = ACTIVE_TOKENIZER_SPECS,
):
    specs = tuple(specs)
    paths = tuple(
        artifact_path(spec, artifact) for spec, artifact in unique_artifacts(specs)
    )
    for spec in specs:
        _validate_spec(spec)
    return paths


def verify_active_assets(
    specs: Iterable[TokenizerSpec] = ACTIVE_TOKENIZER_SPECS,
):
    return tuple(
        artifact_path(spec, artifact, allow_download=False)
        for spec, artifact in unique_artifacts(specs)
    )


def _validate_spec(spec: TokenizerSpec) -> None:
    factory = build_tokenizer if spec.family == KIMI_K3_SPEC.family else None
    load_tokenizer(spec, factory=factory)
    if spec.template is not None:
        load_template(spec)
    if spec.encoder is not None:
        load_python_encoder(spec)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare pinned Hugging Face assets for active models.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the existing cache without downloading",
    )
    args = parser.parse_args(argv)

    if args.verify_only:
        paths = verify_active_assets()
        action = "Verified"
    else:
        paths = prefetch_active_assets()
        action = "Prepared"
    print(
        f"{action} {len(paths)} unique Hugging Face assets for "
        f"{len(ACTIVE_TOKENIZER_SPECS)} active model families."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVE_TOKENIZER_SPECS",
    "iter_spec_artifacts",
    "main",
    "prefetch_active_assets",
    "unique_artifacts",
    "verify_active_assets",
]
