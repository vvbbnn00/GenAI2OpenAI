from pathlib import Path


def test_docker_build_prefetches_and_then_enforces_offline_hf_assets():
    project_root = Path(__file__).resolve().parents[2]
    dockerfile = (project_root / "Dockerfile").read_text()

    prefetch = "python -m genai_proxy.models.hf_prefetch"
    offline = "ENV GENAI_TOKENIZER_OFFLINE=1"
    assert "GENAI_TOKENIZER_CACHE=/opt/genai2openai/hf-assets" in dockerfile
    assert prefetch in dockerfile
    assert "python -m genai_proxy.models.hf_prefetch --verify-only" in dockerfile
    assert 'chmod -R a-w "${GENAI_TOKENIZER_CACHE}"' in dockerfile
    assert dockerfile.index(prefetch) < dockerfile.index(offline)
    assert "HF_HUB_OFFLINE=1" in dockerfile


def test_compose_does_not_mask_the_hf_assets_bundled_in_the_image():
    project_root = Path(__file__).resolve().parents[2]
    compose = (project_root / "docker-compose.yml").read_text()

    assert "GENAI_TOKENIZER_CACHE: /opt/genai2openai/hf-assets" in compose
    assert 'GENAI_TOKENIZER_OFFLINE: "1"' in compose
    assert 'HF_HUB_OFFLINE: "1"' in compose
    assert "tokenizer-cache" not in compose
