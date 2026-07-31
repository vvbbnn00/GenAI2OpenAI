"""Pinned official tokenizer and visual preprocessing constants for Kimi K3."""

import math

import tiktoken
from tiktoken.load import load_tiktoken_bpe

from genai_proxy.models.hf_assets import Artifact, TokenizerSpec

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

PATTERN = "|".join(
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

SPECIAL_TOKEN_OVERRIDES = {
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

IMAGE_PATCH_SIZE = 14
IMAGE_MERGE_KERNEL_SIZE = 2
IMAGE_PATCH_LIMIT = 512
IMAGE_IN_PATCH_LIMIT = 65536


def build_tokenizer(path):
    mergeable_ranks = load_tiktoken_bpe(str(path))
    base_tokens = len(mergeable_ranks)
    special_tokens = {
        SPECIAL_TOKEN_OVERRIDES.get(token_id, f"<|reserved_token_{token_id}|>"): token_id
        for token_id in range(base_tokens, base_tokens + 256)
    }
    return tiktoken.Encoding(
        name=path.name,
        pat_str=PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )


def image_token_count(width: int, height: int) -> int:
    scale = min(
        1.0,
        math.sqrt(
            IMAGE_IN_PATCH_LIMIT
            / (
                max(1.0, width // IMAGE_PATCH_SIZE)
                * max(1.0, height // IMAGE_PATCH_SIZE)
            )
        ),
        IMAGE_PATCH_LIMIT * IMAGE_PATCH_SIZE / width,
        IMAGE_PATCH_LIMIT * IMAGE_PATCH_SIZE / height,
    )
    new_width = min(
        max(1, int(width * scale)),
        IMAGE_PATCH_LIMIT * IMAGE_PATCH_SIZE,
    )
    new_height = min(
        max(1, int(height * scale)),
        IMAGE_PATCH_LIMIT * IMAGE_PATCH_SIZE,
    )
    factor = IMAGE_MERGE_KERNEL_SIZE * IMAGE_PATCH_SIZE
    return math.ceil(new_width / factor) * math.ceil(new_height / factor)


__all__ = [
    "IMAGE_IN_PATCH_LIMIT",
    "IMAGE_MERGE_KERNEL_SIZE",
    "IMAGE_PATCH_LIMIT",
    "IMAGE_PATCH_SIZE",
    "KIMI_K3_SPEC",
    "PATTERN",
    "SPECIAL_TOKEN_OVERRIDES",
    "build_tokenizer",
    "image_token_count",
]
