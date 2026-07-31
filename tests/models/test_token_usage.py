import base64
import hashlib
import io
import json
import logging
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
from PIL import Image

from genai_proxy.api.anthropic.compat import convert_claude_to_openai
from genai_proxy.api.openai.service import GenAIService
from genai_proxy.app import create_app
from genai_proxy.errors import ProxyError
from genai_proxy.models.deepseek_v4.tooling import (
    inject_deepseek_reasoning_prompt,
    inject_deepseek_tool_prompt,
)
from genai_proxy.models.glm52.tooling import inject_glm_tool_prompt
from genai_proxy.models.legacy.minimax import inject_minimax_tool_prompt
from genai_proxy.models.qwen35.tooling import inject_qwen35_tool_prompt
from genai_proxy.token_usage import (
    DEEPSEEK_V4_PRO_SPEC,
    GLM_5_1_SPEC,
    GLM_5_2_SPEC,
    KIMI_K3_SPEC,
    MINIMAX_M2_7_SPEC,
    QWEN_3_5_SPEC,
    Artifact,
    TokenizerSpec,
    _artifact_path,
    _count_encoded,
    _decode_data_url,
    _kimi_image_token_count,
    _load_python_encoder,
    _load_template,
    _normalize_messages,
    _qwen_image_token_count,
    _request_public_image,
    _serialized_completion,
    count_openai_completion_tokens,
    count_openai_reasoning_tokens,
    count_openai_request_tokens,
    render_chat_prompt,
    tokenizer_family_for_model,
)


class FakeTokenManager:
    token = "token"
    billing_user_id = None


class FakeModelManager:
    records = {
        "chatglm": {
            "aiType": "chatglm",
            "aiName": "GLM-5.2",
            "rootModelName": "Xinference",
        },
        "chatglm51": {
            "aiType": "chatglm51",
            "aiName": "GLM-5.1",
            "rootModelName": "Xinference",
        },
        "deepseek-chat": {
            "aiType": "deepseek-chat",
            "aiName": "DeepSeek-V4-Flash",
            "rootModelName": "Xinference",
        },
        "deepseek-pro": {
            "aiType": "deepseek-pro",
            "aiName": "DeepSeek-V4-Pro",
            "rootModelName": "Xinference",
        },
        "qwen3.5": {
            "aiType": "qwen3.5",
            "aiName": "Qwen3.5-397B-A17B",
            "rootModelName": "Xinference",
        },
        "MiniMax-M1": {
            "aiType": "MiniMax-M1",
            "aiName": "MiniMax-M2.7",
            "rootModelName": "Xinference",
        },
        "kimi-k3": {
            "aiType": "kimi-k3",
            "aiName": "Kimi-K3",
            "rootModelName": "Xinference",
        },
    }

    def resolve_model(self, model):
        return model

    def get_model_record(self, model):
        return self.records.get(model)

    def list_genai_models(self):
        return list(self.records.values())

    def root_ai_type_for(self, _model):
        return "xinference"


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, lines):
        self.lines = lines

    def iter_lines(self, *args, **kwargs):
        return iter(self.lines)

    def close(self):
        pass


def fake_completion(content, *, reasoning=None):
    delta = {"content": content}
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": delta,
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]
    )


def _official_multiturn_messages():
    return [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "reasoning_content": "Need the current weather.",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city":"Shanghai"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather",
            "content": "Sunny.",
        },
    ]


def _app():
    logger = logging.getLogger("test_token_usage")
    model_manager = FakeModelManager()
    service = GenAIService(logger, FakeTokenManager(), model_manager)
    app = create_app(
        SimpleNamespace(
            token=None,
            keystore=None,
            token_check_interval=0,
            api_key=None,
            claude_haiku_model="deepseek-chat",
            claude_sonnet_model="chatglm",
            claude_opus_model="chatglm",
        ),
        logger,
    )
    app.extensions["model_manager"] = model_manager
    app.extensions["genai_service"] = service
    return app


def _png_data_url(width: int, height: int) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{payload}"


def _official_kimi_completion(message: dict) -> str:
    encoder = _load_python_encoder(KIMI_K3_SPEC)
    common = {
        "tools": None,
        "thinking": True,
        "image_prompts": None,
        "thinking_effort": "max",
    }
    start = "".join(
        segment.text
        for segment in encoder["build_chat_segments"](
            [],
            add_generation_prompt=True,
            **common,
        )
    )
    completed = "".join(
        segment.text
        for segment in encoder["build_chat_segments"](
            [message],
            add_generation_prompt=False,
            **common,
        )
    )
    assert completed.startswith(start)
    return completed[len(start) :]


def test_qwen_uses_full_model_as_revision_authority():
    assert QWEN_3_5_SPEC.repository == "Qwen/Qwen3.5-397B-A17B"
    assert QWEN_3_5_SPEC.revision == "8472618112abcbd45acbcdc58436aff4233c23f7"


def test_token_usage_facade_reexports_family_codec_specs():
    from genai_proxy.models.deepseek_v4.codec import DEEPSEEK_V4_PRO_SPEC as deepseek
    from genai_proxy.models.glm52.codec import GLM_5_2_SPEC as glm
    from genai_proxy.models.hf_assets import Artifact as artifact_type
    from genai_proxy.models.hf_assets import TokenizerSpec as spec_type
    from genai_proxy.models.kimi_k3.codec import KIMI_K3_SPEC as kimi
    from genai_proxy.models.legacy.minimax_codec import MINIMAX_M2_7_SPEC as minimax
    from genai_proxy.models.qwen35.codec import QWEN_3_5_SPEC as qwen

    assert Artifact is artifact_type
    assert TokenizerSpec is spec_type
    assert DEEPSEEK_V4_PRO_SPEC is deepseek
    assert GLM_5_2_SPEC is glm
    assert KIMI_K3_SPEC is kimi
    assert MINIMAX_M2_7_SPEC is minimax
    assert QWEN_3_5_SPEC is qwen


def test_glm51_and_minimax_use_pinned_official_assets():
    assert GLM_5_1_SPEC.repository == "zai-org/GLM-5.1"
    assert GLM_5_1_SPEC.revision == "26e1bd6e011feb778d25ae34b09b07074139d92d"
    assert MINIMAX_M2_7_SPEC.repository == "MiniMaxAI/MiniMax-M2.7"
    assert MINIMAX_M2_7_SPEC.revision == "d494266a4affc0d2995ba1fa35c8481cbd84294b"


def test_kimi_uses_pinned_official_encoder_and_tiktoken_assets():
    assert KIMI_K3_SPEC.repository == "moonshotai/Kimi-K3"
    assert KIMI_K3_SPEC.revision == "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
    assert KIMI_K3_SPEC.encoder.path == "encoding_k3.py"
    assert KIMI_K3_SPEC.tokenizer.path == "tiktoken.model"


def test_kimi_official_encoder_uses_structural_tool_declaration():
    encoder = _load_python_encoder(KIMI_K3_SPEC)
    segments = encoder["build_chat_segments"](
        [{"role": "user", "content": "Search for Kimi K3."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
        add_generation_prompt=True,
        thinking=True,
        image_prompts=None,
        thinking_effort="max",
    )
    prompt = "".join(segment.text for segment in segments)

    assert prompt.startswith(
        '<|open|>message role="system" type="tool-declare"<|sep|># Tools\n'
        "Here are the available tools, described in JSONSchema.\n\n"
        "```json\n"
    )
    assert '"name":"search"' in prompt
    assert "Call-expression schemas" not in prompt
    assert "User request:" not in prompt


def test_kimi_genai_prompt_matches_official_no_tool_encoding():
    messages = [{"role": "user", "content": "Hello"}]
    encoder = _load_python_encoder(KIMI_K3_SPEC)
    expected = "".join(
        segment.text
        for segment in encoder["build_chat_segments"](
            messages,
            tools=None,
            add_generation_prompt=True,
            thinking=True,
            image_prompts=None,
            thinking_effort="max",
            tool_choice="none",
        )
    )

    actual = render_chat_prompt(
        messages,
        "kimi_k3",
        add_generation_prompt=True,
    )

    assert actual == expected
    assert (
        '<|open|>message role="system" type="tool-choice"<|sep|>'
        "The system is invoked with `tool_choice=none`.\n"
        "You MUST NOT call any tools in the next message."
        "<|close|>message<|sep|><|end_of_msg|>"
    ) in actual


def test_supported_model_families_are_resolved_from_alias_and_record():
    manager = FakeModelManager()
    assert (
        tokenizer_family_for_model("chatglm", manager.get_model_record("chatglm"))
        == "glm_5_2"
    )
    assert (
        tokenizer_family_for_model(
            "deepseek-chat", manager.get_model_record("deepseek-chat")
        )
        == "deepseek_v4_flash"
    )
    assert (
        tokenizer_family_for_model(
            "deepseek-pro", manager.get_model_record("deepseek-pro")
        )
        == "deepseek_v4_pro"
    )
    assert (
        tokenizer_family_for_model("qwen3.5", manager.get_model_record("qwen3.5"))
        == "qwen_3_5"
    )
    assert (
        tokenizer_family_for_model("MiniMax-M1", manager.get_model_record("MiniMax-M1"))
        == "minimax_m2_7"
    )
    assert (
        tokenizer_family_for_model(
            "chatglm",
            {
                "aiType": "chatglm",
                "aiName": "GLM-5.1",
                "rootModelName": "Xinference",
            },
        )
        == "glm_5_1"
    )
    assert (
        tokenizer_family_for_model("kimi-k3", manager.get_model_record("kimi-k3"))
        == "kimi_k3"
    )


def test_model_family_version_matching_does_not_accept_longer_minor_version():
    assert tokenizer_family_for_model("qwen3.50") is None
    assert tokenizer_family_for_model("glm5.20") is None
    assert tokenizer_family_for_model("minimax2.70") is None
    assert tokenizer_family_for_model("kimi-k30") is None
    assert tokenizer_family_for_model("kimi-k3.1") is None


def test_tokenizer_artifact_download_retries_transient_failure(tmp_path):
    content = b"tokenizer"
    digest = hashlib.sha256(content).hexdigest()
    spec = TokenizerSpec(
        family="test",
        repository="example/model",
        revision="revision",
        tokenizer=Artifact("tokenizer.json", digest),
    )
    attempts = 0

    def download(_url, _cache_dir, destination, _expected_sha256):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise requests.ConnectionError("connection reset")
        destination.write_bytes(content)

    with (
        patch.dict("os.environ", {"GENAI_TOKENIZER_CACHE": str(tmp_path)}),
        patch(
            "genai_proxy.models.hf_assets.download_artifact",
            side_effect=download,
        ),
        patch(
            "genai_proxy.models.hf_assets.schedule_retry",
            return_value=True,
        ) as retry,
    ):
        path = _artifact_path(spec, spec.tokenizer)

    assert path.read_bytes() == content
    assert attempts == 2
    retry.assert_called_once()


def test_official_prompt_templates_have_stable_reference_counts():
    messages = [{"role": "user", "content": "Hello, 世界"}]
    cases = [
        ("chatglm51", "glm_5_1", "glm_5_1", 8, "[gMASK]<sop>"),
        ("chatglm", "glm_5_2", "glm_5_2", 15, "[gMASK]<sop>"),
        (
            "deepseek-pro",
            "deepseek_v4_pro",
            "deepseek_v4_pro",
            8,
            "<｜begin▁of▁sentence｜>",
        ),
        (
            "deepseek-chat",
            "deepseek_v4_flash",
            "deepseek_v4_flash",
            8,
            "<｜begin▁of▁sentence｜>",
        ),
        ("qwen3.5", "qwen_3_5", None, 14, "<|im_start|>user\n"),
        ("MiniMax-M1", "minimax_m2_7", "minimax", 41, "]~!b["),
        (
            "kimi-k3",
            "kimi_k3",
            "kimi_k3",
            130,
            '<|open|>message role="system" type="thinking-effort"<|sep|>',
        ),
    ]
    for model, family, adapter, expected_count, prompt_prefix in cases:
        prompt = render_chat_prompt(messages, family, add_generation_prompt=True)
        assert prompt.startswith(prompt_prefix)
        assert (
            count_openai_request_tokens(messages, model, tool_adapter=adapter)
            == expected_count
        )


def test_deepseek_max_injection_matches_official_encoder_prompt_exactly():
    messages = [{"role": "user", "content": "Hello"}]
    official_prompt = render_chat_prompt(
        messages,
        "deepseek_v4_pro",
        add_generation_prompt=True,
        reasoning_config={"effort": "max"},
    )
    injected_messages = inject_deepseek_reasoning_prompt(
        messages,
        {"effort": "max"},
        adapter="deepseek_v4_pro",
    )
    transported_prompt = render_chat_prompt(
        injected_messages,
        "deepseek_v4_pro",
        add_generation_prompt=True,
    )
    assert transported_prompt == official_prompt


def test_deepseek_tool_transport_matches_official_encoder_exactly():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    encoder = _load_python_encoder(DEEPSEEK_V4_PRO_SPEC)
    for messages in (
        [{"role": "user", "content": "Weather?"}],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Weather?"},
        ],
    ):
        if messages[0]["role"] == "system":
            official_messages = [
                {**messages[0], "tools": tools},
                *messages[1:],
            ]
        else:
            official_messages = [
                {"role": "system", "content": "", "tools": tools},
                *messages,
            ]
        official_prompt = encoder["encode_messages"](
            official_messages,
            thinking_mode="thinking",
            reasoning_effort="max",
        )
        transported_prompt = render_chat_prompt(
            inject_deepseek_tool_prompt(
                messages,
                tools,
                adapter="deepseek_v4_pro",
                reasoning_config={"effort": "max"},
            ),
            "deepseek_v4_pro",
            add_generation_prompt=True,
            thinking=True,
        )

        assert transported_prompt == official_prompt


def test_deepseek_chat_and_thinking_modes_match_official_encoder_boundaries():
    messages = [{"role": "user", "content": "Hello"}]
    chat_prompt = render_chat_prompt(
        messages,
        "deepseek_v4_pro",
        add_generation_prompt=True,
        thinking=False,
    )
    thinking_prompt = render_chat_prompt(
        messages,
        "deepseek_v4_pro",
        add_generation_prompt=True,
        reasoning_config={"effort": "high"},
        thinking=True,
    )

    assert chat_prompt.endswith("<｜Assistant｜></think>")
    assert thinking_prompt.endswith("<｜Assistant｜><think>")
    chat_message = {"role": "assistant", "content": "answer"}
    assert render_chat_prompt(
        [*messages, chat_message],
        "deepseek_v4_pro",
        add_generation_prompt=False,
        thinking=False,
    ) == chat_prompt + _serialized_completion(
        chat_message,
        "deepseek_v4_pro",
        thinking=False,
    )


def test_deepseek_multiturn_tool_history_matches_official_encoder_exactly():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    encoder = _load_python_encoder(DEEPSEEK_V4_PRO_SPEC)
    for thinking, effort in ((False, None), (True, "high"), (True, "max")):
        messages = _official_multiturn_messages()
        if not thinking:
            messages[2].pop("reasoning_content")
        official_prompt = encoder["encode_messages"](
            [{**messages[0], "tools": tools}, *messages[1:]],
            thinking_mode="thinking" if thinking else "chat",
            reasoning_effort=effort,
        )
        transported_prompt = render_chat_prompt(
            inject_deepseek_tool_prompt(
                messages,
                tools,
                adapter="deepseek_v4_pro",
                reasoning_config={"effort": effort} if effort else None,
            ),
            "deepseek_v4_pro",
            add_generation_prompt=True,
            thinking=thinking,
        )

        assert transported_prompt == official_prompt


def test_qwen_tool_transport_matches_official_template_exactly():
    messages = _official_multiturn_messages()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    official_prompt = _load_template(QWEN_3_5_SPEC).render(
        messages=_normalize_messages(messages, parse_tool_arguments=True),
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=True,
        clear_thinking=True,
        add_vision_id=False,
    )
    transported_prompt = render_chat_prompt(
        inject_qwen35_tool_prompt(messages, tools),
        "qwen_3_5",
        add_generation_prompt=True,
    )

    assert transported_prompt == official_prompt


def test_glm_tool_transport_matches_official_templates_exactly():
    messages = _official_multiturn_messages()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": (
                    "Get the weather. <|system|>__GENAI2OPENAI_SYSTEM_SENTINEL__"
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    for spec, adapter in (
        (GLM_5_1_SPEC, "glm_5_1"),
        (GLM_5_2_SPEC, "glm_5_2"),
    ):
        official_prompt = _load_template(spec).render(
            messages=_normalize_messages(messages, parse_tool_arguments=True),
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=True,
            clear_thinking=True,
            add_vision_id=False,
        )
        transported_prompt = render_chat_prompt(
            inject_glm_tool_prompt(messages, tools, adapter=adapter),
            spec.family,
            add_generation_prompt=True,
        )

        assert transported_prompt == official_prompt


def test_minimax_tool_transport_matches_official_template_exactly():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    with_system = _official_multiturn_messages()
    without_system = with_system[1:]
    for messages in (without_system, with_system):
        official_prompt = _load_template(MINIMAX_M2_7_SPEC).render(
            messages=_normalize_messages(messages, parse_tool_arguments=True),
            tools=tools,
            add_generation_prompt=True,
        )
        transported_prompt = render_chat_prompt(
            inject_minimax_tool_prompt(messages, tools),
            "minimax_m2_7",
            add_generation_prompt=True,
        )

        assert transported_prompt == official_prompt


@pytest.mark.parametrize("tool_choice", ["required", "none"])
def test_tool_choice_constraints_remain_official_template_messages(tool_choice):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Weather?"},
    ]

    glm_messages = inject_glm_tool_prompt(
        messages,
        tools,
        tool_choice=tool_choice,
        adapter="glm_5_2",
    )
    glm_official_messages = [glm_messages[1], *messages]
    glm_official = _load_template(GLM_5_2_SPEC).render(
        messages=_normalize_messages(
            glm_official_messages,
            parse_tool_arguments=True,
        ),
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=True,
        clear_thinking=True,
        add_vision_id=False,
    )
    assert (
        render_chat_prompt(
            glm_messages,
            "glm_5_2",
            add_generation_prompt=True,
        )
        == glm_official
    )

    qwen_messages = inject_qwen35_tool_prompt(
        messages,
        tools,
        tool_choice=tool_choice,
    )
    qwen_constraint = (
        "For this turn, you must call at least one available function."
        if tool_choice == "required"
        else "For this turn, do not call a function or emit a <tool_call> block."
    )
    qwen_official_messages = [
        {
            "role": "system",
            "content": f"{qwen_constraint}\n\nBe concise.",
        },
        messages[1],
    ]
    qwen_official = _load_template(QWEN_3_5_SPEC).render(
        messages=_normalize_messages(
            qwen_official_messages,
            parse_tool_arguments=True,
        ),
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=True,
        clear_thinking=True,
        add_vision_id=False,
    )
    assert (
        render_chat_prompt(
            qwen_messages,
            "qwen_3_5",
            add_generation_prompt=True,
        )
        == qwen_official
    )

    minimax_messages = inject_minimax_tool_prompt(
        messages,
        tools,
        tool_choice=tool_choice,
    )
    minimax_constraint = (
        "\nFor this turn, you must call at least one tool using a "
        "<minimax:tool_call> block."
        if tool_choice == "required"
        else "\nFor this turn, do not call any tool or emit tool call tags."
    )
    minimax_official_messages = [
        {
            "role": "system",
            "content": f"Be concise.{minimax_constraint}",
        },
        messages[1],
    ]
    minimax_official = _load_template(MINIMAX_M2_7_SPEC).render(
        messages=_normalize_messages(
            minimax_official_messages,
            parse_tool_arguments=True,
        ),
        tools=tools,
        add_generation_prompt=True,
    )
    assert (
        render_chat_prompt(
            minimax_messages,
            "minimax_m2_7",
            add_generation_prompt=True,
        )
        == minimax_official
    )


def test_kimi_visual_prompt_and_patch_tokens_match_official_rules():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/image.png"},
                }
            ],
        }
    ]
    prompt = render_chat_prompt(
        messages,
        "kimi_k3",
        add_generation_prompt=True,
        image_sizes=((56, 28),),
    )

    assert (
        "<|media_begin|>image 56x28<|media_content|><|media_pad|><|media_end|>"
    ) in prompt
    assert _kimi_image_token_count(56, 28) == 2
    assert (
        count_openai_request_tokens(
            messages,
            "kimi-k3",
            image_sizes=((56, 28),),
        )
        == 136
    )


def test_kimi_visual_token_count_matches_official_resize_boundaries():
    cases = {
        (1, 1): 1,
        (27, 28): 1,
        (28, 28): 1,
        (29, 28): 2,
        (7168, 7168): 16384,
        (100000, 10): 256,
        (8000, 4000): 16562,
        (4000, 8000): 16562,
    }

    for dimensions, expected in cases.items():
        assert _kimi_image_token_count(*dimensions) == expected


def test_qwen_visual_token_count_matches_official_resize_boundaries():
    cases = {
        (1, 1): 64,
        (32, 32): 64,
        (256, 256): 64,
        (640, 480): 300,
        (512, 512): 256,
        (4096, 4096): 16384,
        (8192, 8192): 16384,
    }

    for dimensions, expected in cases.items():
        assert _qwen_image_token_count(*dimensions) == expected

    with pytest.raises(ProxyError, match="aspect ratio"):
        _qwen_image_token_count(201, 1)


def test_qwen_visual_request_count_expands_official_image_placeholder():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _png_data_url(32, 32)},
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    prompt = render_chat_prompt(
        messages,
        "qwen_3_5",
        add_generation_prompt=True,
    )
    placeholder_count = _count_encoded("qwen_3_5", prompt)

    assert count_openai_request_tokens(messages, "qwen3.5") == (
        placeholder_count + _qwen_image_token_count(32, 32) - 1
    )


def test_kimi_service_preserves_visual_blocks_and_counts_data_url():
    data_url = _png_data_url(56, 28)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    prepared = (
        _app()
        .extensions["genai_service"]
        ._prepare_chat_request({"model": "kimi-k3", "messages": messages})
    )

    assert prepared.messages[0]["content"] == [
        {"type": "text", "text": "\u200b"},
        messages[0]["content"][0],
    ]
    assert prepared.image_sizes == ((56, 28),)
    assert prepared.prompt_tokens == 137


def test_kimi_service_reads_remote_image_dimensions_once():
    image_bytes = base64.b64decode(_png_data_url(56, 28).partition(",")[2])

    class RemoteImageResponse:
        def __init__(self):
            self.closed = False
            self.status = 200
            self.headers = {}

        def stream(self, amt, decode_content):
            assert amt == 64 * 1024
            assert decode_content is True
            yield image_bytes

        def close(self):
            self.closed = True

    class RemoteImagePool:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    remote = RemoteImageResponse()
    pool = RemoteImagePool()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/image.png"},
                }
            ],
        }
    ]

    with patch(
        "genai_proxy.token_usage._request_public_image",
        return_value=(remote, pool),
    ) as request_image:
        prepared = (
            _app()
            .extensions["genai_service"]
            ._prepare_chat_request({"model": "kimi-k3", "messages": messages})
        )

    request_image.assert_called_once_with("https://example.test/image.png")
    assert remote.closed is True
    assert pool.closed is True
    assert prepared.image_sizes == ((56, 28),)
    assert prepared.prompt_tokens == 137


def test_kimi_remote_image_rejects_private_network_targets():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "http://localhost/image.png"},
                }
            ],
        }
    ]

    with (
        patch(
            "genai_proxy.token_usage.socket.getaddrinfo",
            return_value=[
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("127.0.0.1", 80),
                )
            ],
        ),
        patch("genai_proxy.token_usage.HTTPConnectionPool") as connection_pool,
    ):
        response = (
            _app()
            .test_client()
            .post(
                "/v1/chat/completions",
                json={"model": "kimi-k3", "messages": messages},
            )
        )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_image"
    connection_pool.assert_not_called()


def test_kimi_remote_image_request_pins_the_validated_address():
    response = SimpleNamespace()
    with (
        patch(
            "genai_proxy.token_usage.socket.getaddrinfo",
            return_value=[
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("93.184.216.34", 443),
                )
            ],
        ),
        patch("genai_proxy.token_usage.HTTPSConnectionPool") as pool_type,
    ):
        pool_type.return_value.urlopen.return_value = response
        actual_response, actual_pool = _request_public_image(
            "https://example.test/image.png?size=small"
        )

    assert actual_response is response
    assert actual_pool is pool_type.return_value
    pool_type.assert_called_once_with(
        "93.184.216.34",
        port=443,
        maxsize=1,
        block=True,
        cert_reqs="CERT_REQUIRED",
        assert_hostname="example.test",
        server_hostname="example.test",
    )
    pool_type.return_value.urlopen.assert_called_once()


def test_kimi_oversized_data_url_is_rejected_before_base64_decode():
    with (
        patch("genai_proxy.token_usage.KIMI_IMAGE_MAX_BYTES", 3),
        patch("genai_proxy.token_usage.base64.b64decode") as decode,
    ):
        try:
            _decode_data_url("data:image/png;base64,QUJDRA==")
        except ValueError as exc:
            assert str(exc) == "image exceeds the 50 MiB limit"
        else:
            raise AssertionError("oversized image data URL was accepted")

    decode.assert_not_called()


def test_kimi_transport_splits_current_visual_input_for_genai():
    data_url = _png_data_url(56, 28)
    request = {
        "model": "kimi-k3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "What color is this?"},
                ],
            }
        ],
        "stream": True,
        "max_tokens": 64,
    }
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "Red"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]
    )

    with patch(
        "genai_proxy.upstream.transport.requests.post",
        return_value=upstream,
    ) as post:
        chunks = list(
            _app().extensions["genai_service"].stream_openai_completion(request)
        )

    payload = post.call_args.kwargs["json"]
    assert payload["chatInfo"] == "What color is this?"
    assert payload["messages"] == []
    assert payload["imageUrl"] == data_url
    assert payload["imageUrls"] == [data_url]
    assert payload["width"] == 56
    assert payload["height"] == 28
    assert "chatGroupId" not in payload
    assert any('"content": "Red"' in chunk for chunk in chunks)
    assert request["messages"][0]["content"][0]["type"] == "image_url"


def test_kimi_transport_preserves_multiple_current_images_in_order():
    first = _png_data_url(56, 28)
    second = _png_data_url(29, 28)
    request = {
        "model": "kimi-k3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images."},
                    {"type": "image_url", "image_url": {"url": first}},
                    {"type": "image_url", "image_url": {"url": second}},
                ],
            }
        ],
        "stream": True,
    }
    upstream = fake_completion("Compared.")
    app = _app()
    prepared = app.extensions["genai_service"]._prepare_chat_request(request)

    assert prepared.image_sizes == ((56, 28), (29, 28))

    with patch(
        "genai_proxy.upstream.transport.requests.post",
        return_value=upstream,
    ) as post:
        chunks = list(app.extensions["genai_service"].stream_openai_completion(request))

    payload = post.call_args.kwargs["json"]
    assert payload["chatInfo"] == "Compare these images."
    assert payload["imageUrl"] == first
    assert payload["imageUrls"] == [first, second]
    assert (payload["width"], payload["height"]) == (56, 28)
    assert any('"content": "Compared."' in chunk for chunk in chunks)


def test_kimi_empty_current_user_gets_nonempty_transport_trigger():
    prepared = (
        _app()
        .extensions["genai_service"]
        ._prepare_chat_request(
            {
                "model": "kimi-k3",
                "messages": [{"role": "user", "content": ""}],
            }
        )
    )

    assert prepared.messages == [{"role": "user", "content": "\u200b"}]


def test_kimi_image_only_current_user_gets_nonempty_transport_trigger():
    data_url = _png_data_url(56, 28)
    prepared = (
        _app()
        .extensions["genai_service"]
        ._prepare_chat_request(
            {
                "model": "kimi-k3",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            }
        )
    )

    assert prepared.messages[0]["content"][0] == {
        "type": "text",
        "text": "\u200b",
    }


def test_kimi_responses_input_tokens_counts_visual_content():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/responses/input_tokens",
            json={
                "model": "kimi-k3",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": _png_data_url(56, 28),
                            }
                        ],
                    }
                ],
            },
        )
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "object": "response.input_tokens",
        "input_tokens": 137,
    }


def test_kimi_input_tokens_rejects_invalid_image_data():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/responses/input_tokens",
            json={
                "model": "kimi-k3",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,not-valid",
                            }
                        ],
                    }
                ],
            },
        )
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_image"


def test_kimi_anthropic_count_tokens_counts_base64_image():
    payload = _png_data_url(56, 28).partition(",")[2]
    response = (
        _app()
        .test_client()
        .post(
            "/v1/messages/count_tokens",
            json={
                "model": "kimi-k3",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": payload,
                                },
                            }
                        ],
                    }
                ],
            },
        )
    )

    assert response.status_code == 200
    assert response.get_json() == {"input_tokens": 137}


def test_kimi_claude_url_image_is_preserved_for_vision_transport():
    request = convert_claude_to_openai(
        {
            "model": "kimi-k3",
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.test/image.png",
                            },
                        },
                    ],
                }
            ],
        },
        FakeModelManager(),
    )

    assert request["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.test/image.png"},
    }


def test_kimi_active_tools_use_validated_bridge_without_chat_group_id():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    app = _app()
    client = app.test_client()
    raw_content = (
        '<k3_action>{"name":"get_weather","arguments":{"city":"Shanghai"}}</k3_action>'
    )
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": raw_content},
                            "finish_reason": None,
                        }
                    ]
                }
            ).encode(),
            json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode(),
        ]
    )
    request = {
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "Weather in Shanghai?"}],
        "tools": tools,
    }

    with patch(
        "genai_proxy.upstream.transport.requests.post", return_value=upstream
    ) as post:
        response = client.post(
            "/v1/chat/completions",
            json=request,
        )

    assert response.status_code == 200
    payload = response.get_json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Shanghai"}

    upstream_payload = post.call_args.kwargs["json"]
    assert upstream_payload["chatInfo"] == "Weather in Shanghai?"
    assert "chatGroupId" not in upstream_payload
    assert "tools" not in upstream_payload
    assert "tool_choice" not in upstream_payload
    assert all(
        not message.get("tools") and not message.get("tool_calls")
        for message in upstream_payload["messages"]
    )
    bridge_prompt = upstream_payload["messages"][-1]["content"]
    assert bridge_prompt.startswith("# Client response protocol\n")
    assert "<k3_actions>" in bridge_prompt
    assert "<k3_final>" in bridge_prompt
    assert '"name":"get_weather"' in bridge_prompt
    assert "client data channel" in bridge_prompt
    assert "not native model tool use" in bridge_prompt
    assert "Weather in Shanghai?" not in bridge_prompt

    prepared = app.extensions["genai_service"]._prepare_chat_request(request)
    expected = count_openai_completion_tokens(
        {"role": "assistant", "content": raw_content},
        "kimi-k3",
        model_record=FakeModelManager.records["kimi-k3"],
        tool_adapter="kimi_k3",
        prompt_messages=prepared.messages,
    )
    assert payload["usage"]["completion_tokens"] == expected


def test_kimi_auto_never_executes_a_plain_json_answer():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object"},
            },
        }
    ]
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "content": ('{"name":"get_weather","arguments":{}}')
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]
    )

    with patch(
        "genai_proxy.upstream.transport.requests.post", return_value=upstream
    ) as post:
        response = (
            _app()
            .test_client()
            .post(
                "/v1/chat/completions",
                json={
                    "model": "kimi-k3",
                    "messages": [{"role": "user", "content": "Answer directly."}],
                    "tools": tools,
                },
            )
        )

    assert response.status_code == 502
    assert (
        "neither a valid client action nor a final response"
        in (response.get_json()["error"]["message"])
    )
    assert post.call_count == 3


def test_kimi_auto_retries_unwrapped_response_as_required_action():
    first = fake_completion(
        "I cannot invoke a native tool in this response.",
        reasoning="The requested check still needs external evidence. ",
    )
    second = fake_completion(
        '<k3_action>{"name":"get_weather","arguments":{"city":"Shanghai"}}</k3_action>',
        reasoning="I will encode the client action as response data.",
    )
    request = {
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "Check Shanghai weather."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }

    with patch(
        "genai_proxy.upstream.transport.requests.post",
        side_effect=[first, second],
    ) as post:
        response = (
            _app()
            .test_client()
            .post(
                "/v1/chat/completions",
                json=request,
            )
        )

    assert response.status_code == 200
    choice = response.get_json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert post.call_count == 2
    retry_prompt = post.call_args_list[1].kwargs["json"]["messages"][-1]["content"]
    assert "Return at least one complete action block and no other text." in (
        retry_prompt
    )
    assert "<k3_final>" not in retry_prompt


def test_kimi_bridge_stream_waits_for_complete_action_then_emits_tool_chunks():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    raw_parts = [
        '<k3_action>{"name":"get_weather",',
        '"arguments":{"city":"Shanghai"}}</k3_action>',
    ]
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": raw_parts[0]},
                            "finish_reason": None,
                        }
                    ]
                }
            ).encode(),
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": raw_parts[1]},
                            "finish_reason": None,
                        }
                    ]
                }
            ).encode(),
            json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode(),
        ]
    )
    request = {
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "Weather in Shanghai?"}],
        "tools": tools,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    with patch("genai_proxy.upstream.transport.requests.post", return_value=upstream):
        body = "".join(
            _app().extensions["genai_service"].stream_openai_completion(request)
        )

    events = [
        json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")
    ]
    choices = [choice for event in events for choice in event.get("choices", [])]
    tool_chunks = [
        tool_call
        for choice in choices
        for tool_call in choice.get("delta", {}).get("tool_calls", []) or []
    ]
    assert len(tool_chunks) == 1
    assert tool_chunks[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_chunks[0]["function"]["arguments"]) == {"city": "Shanghai"}
    assert any(choice.get("finish_reason") == "tool_calls" for choice in choices)
    assert "<k3_action>" not in body
    usage_events = [event["usage"] for event in events if event.get("usage")]
    assert len(usage_events) == 1
    assert usage_events[0]["completion_tokens"] > 0


def test_kimi_bridge_rejects_malformed_tagged_action():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object"},
            },
        }
    ]
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "content": (
                                    '<k3_action>{"name":"get_weather","arguments":'
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]
    )

    with patch("genai_proxy.upstream.transport.requests.post", return_value=upstream):
        response = (
            _app()
            .test_client()
            .post(
                "/v1/chat/completions",
                json={
                    "model": "kimi-k3",
                    "messages": [{"role": "user", "content": "Check weather."}],
                    "tools": tools,
                },
            )
        )

    assert response.status_code == 502
    assert response.get_json()["error"]["message"] == (
        "Upstream returned an invalid tool call"
    )


def test_kimi_bridge_enforces_required_tool_choice():
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "I will answer directly."},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]
    )

    with patch("genai_proxy.upstream.transport.requests.post", return_value=upstream):
        response = (
            _app()
            .test_client()
            .post(
                "/v1/chat/completions",
                json={
                    "model": "kimi-k3",
                    "messages": [{"role": "user", "content": "Check weather."}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    "tool_choice": "required",
                },
            )
        )

    assert response.status_code == 502
    assert response.get_json()["error"]["message"] == (
        "Upstream did not return the required tool call"
    )


def test_kimi_bridge_retries_required_tool_choice_twice():
    first_failure = fake_completion(
        "I cannot invoke a native tool in this response.",
        reasoning="First attempt. ",
    )
    second_failure = fake_completion(
        '{"name":"get_weather","arguments":',
        reasoning="Second attempt. ",
    )
    success = fake_completion(
        (
            '<k3_action>{"name":"get_weather",'
            '"arguments":{"city":"Shanghai"}}</k3_action>'
        ),
        reasoning="Third attempt.",
    )
    request = {
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "Check Shanghai."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "required",
    }
    app = _app()

    with patch(
        "genai_proxy.upstream.transport.requests.post",
        side_effect=[first_failure, second_failure, success],
    ) as post:
        response = app.test_client().post(
            "/v1/chat/completions",
            json=request,
        )

    assert response.status_code == 200
    payload = response.get_json()
    message = payload["choices"][0]["message"]
    tool_call = message["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_weather"
    assert message["reasoning_content"] == (
        "First attempt. Second attempt. Third attempt."
    )
    prepared = app.extensions["genai_service"]._prepare_chat_request(request)
    expected_tokens = count_openai_completion_tokens(
        {
            "role": "assistant",
            "reasoning_content": message["reasoning_content"],
            "content": (
                '<k3_action>{"name":"get_weather",'
                '"arguments":{"city":"Shanghai"}}</k3_action>'
            ),
        },
        "kimi-k3",
        model_record=FakeModelManager.records["kimi-k3"],
        tool_adapter="kimi_k3",
        prompt_messages=prepared.messages,
    )
    assert payload["usage"]["completion_tokens"] == expected_tokens
    assert post.call_count == 3


def test_kimi_auto_accepts_explicit_final_after_tool_result():
    upstream = fake_completion(
        "<k3_final>Inspection returned beta; the task is complete.</k3_final>",
        reasoning="The available evidence is sufficient for this response.",
    )
    request = {
        "model": "kimi-k3",
        "messages": [
            {"role": "user", "content": "Inspect the project."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_inspect",
                        "type": "function",
                        "function": {
                            "name": "inspect",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_inspect",
                "content": "Inspection returned beta.",
            },
            {"role": "user", "content": "\u200b"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "inspect",
                    "parameters": {"type": "object"},
                },
            },
        ],
    }

    with patch(
        "genai_proxy.upstream.transport.requests.post",
        return_value=upstream,
    ) as post:
        response = (
            _app()
            .test_client()
            .post(
                "/v1/chat/completions",
                json=request,
            )
        )

    assert response.status_code == 200
    message = response.get_json()["choices"][0]["message"]
    assert message["content"] == "Inspection returned beta; the task is complete."
    assert "tool_calls" not in message
    assert post.call_count == 1
    assert "chatGroupId" not in post.call_args.kwargs["json"]


def test_kimi_auto_accepts_normal_text_that_mentions_tool_and_argument_names():
    upstream = fake_completion(
        (
            "<k3_final>The inspect_source path: changed.py was already "
            "checked.</k3_final>"
        ),
    )
    request = {
        "model": "kimi-k3",
        "messages": [
            {"role": "user", "content": "Summarize the completed check."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "inspect_source",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
        ],
    }

    with patch(
        "genai_proxy.upstream.transport.requests.post",
        return_value=upstream,
    ) as post:
        response = (
            _app()
            .test_client()
            .post(
                "/v1/chat/completions",
                json=request,
            )
        )

    assert response.status_code == 200
    message = response.get_json()["choices"][0]["message"]
    assert message["content"] == (
        "The inspect_source path: changed.py was already checked."
    )
    assert "tool_calls" not in message
    assert post.call_count == 1


def test_kimi_malformed_action_retries_with_structural_prompt():
    first = fake_completion(
        '<k3_action>{"name":"inspect_source","arguments":',
    )
    second = fake_completion(
        '<k3_action>{"name":"inspect_source",'
        '"arguments":{"path":"changed.py"}}</k3_action>'
    )
    request = {
        "model": "kimi-k3",
        "messages": [
            {"role": "user", "content": "Inspect changed.py."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "inspect_source",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
        ],
    }

    with patch(
        "genai_proxy.upstream.transport.requests.post",
        side_effect=[first, second],
    ) as post:
        response = (
            _app()
            .test_client()
            .post(
                "/v1/chat/completions",
                json=request,
            )
        )

    assert response.status_code == 200
    message = response.get_json()["choices"][0]["message"]
    assert message["tool_calls"][0]["function"]["name"] == "inspect_source"
    assert post.call_count == 2
    retry_payload = post.call_args_list[1].kwargs["json"]
    assert retry_payload["chatInfo"] == "Inspect changed.py."
    assert retry_payload["messages"][-1]["content"].startswith(
        "The previous response did not use a complete valid client response envelope"
    )
    assert (
        "plain response data for the client"
        in (retry_payload["messages"][-1]["content"])
    )
    assert "chatGroupId" not in retry_payload


def test_kimi_tool_bridge_counting_is_consistent_across_compatibility_routes():
    client = _app().test_client()
    responses_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather.",
        "parameters": {"type": "object"},
    }
    claude_tool = {
        "name": "get_weather",
        "description": "Get the weather.",
        "input_schema": {"type": "object"},
    }
    cases = [
        (
            "/v1/responses/input_tokens",
            {
                "model": "kimi-k3",
                "input": "Weather in Shanghai?",
                "tools": [responses_tool],
            },
        ),
        (
            "/v1/messages/count_tokens",
            {
                "model": "kimi-k3",
                "messages": [{"role": "user", "content": "Weather in Shanghai?"}],
                "tools": [claude_tool],
            },
        ),
    ]

    with patch("genai_proxy.upstream.transport.requests.post") as post:
        for path, payload in cases:
            response = client.post(path, json=payload)
            assert response.status_code == 200
            if path.startswith("/v1/responses"):
                assert response.get_json()["input_tokens"] > 0
            else:
                assert response.get_json()["input_tokens"] > 0

    post.assert_not_called()


def test_kimi_tool_bridge_generation_is_consistent_across_compatibility_routes():
    app = _app()
    raw_content = (
        '<k3_action>{"name":"get_weather","arguments":{"city":"Shanghai"}}</k3_action>'
    )

    def upstream():
        return FakeResponse(
            [
                json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {"content": raw_content},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ).encode()
            ]
        )

    responses_payload = {
        "model": "kimi-k3",
        "input": "Weather in Shanghai?",
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    }
    claude_payload = {
        "model": "kimi-k3",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "Weather in Shanghai?"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    }

    with patch(
        "genai_proxy.upstream.transport.requests.post",
        side_effect=[upstream(), upstream()],
    ) as post:
        responses_response = app.test_client().post(
            "/v1/responses", json=responses_payload
        )
        claude_response = app.test_client().post("/v1/messages", json=claude_payload)

    assert responses_response.status_code == 200
    function_call = next(
        item
        for item in responses_response.get_json()["output"]
        if item["type"] == "function_call"
    )
    assert function_call["name"] == "get_weather"
    assert json.loads(function_call["arguments"]) == {"city": "Shanghai"}

    assert claude_response.status_code == 200
    tool_use = next(
        block
        for block in claude_response.get_json()["content"]
        if block["type"] == "tool_use"
    )
    assert tool_use["name"] == "get_weather"
    assert tool_use["input"] == {"city": "Shanghai"}

    assert post.call_count == 2
    for call in post.call_args_list:
        upstream_payload = call.kwargs["json"]
        assert "chatGroupId" not in upstream_payload
        assert upstream_payload["chatInfo"] == "Weather in Shanghai?"
        assert any(
            str(message.get("content", "")).startswith("# Client response protocol\n")
            for message in upstream_payload["messages"]
        )


def test_kimi_tool_history_result_becomes_nonempty_current_input():
    app = _app()
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "Shanghai is sunny."},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]
    )
    with patch(
        "genai_proxy.upstream.transport.requests.post", return_value=upstream
    ) as post:
        response = app.test_client().post(
            "/v1/chat/completions",
            json={
                "model": "kimi-k3",
                "messages": [
                    {"role": "user", "content": "Weather in Shanghai?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Shanghai"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_weather",
                        "content": "Sunny.",
                    },
                ],
            },
        )

    assert response.status_code == 200
    assert (
        response.get_json()["choices"][0]["message"]["content"] == "Shanghai is sunny."
    )
    upstream_payload = post.call_args.kwargs["json"]
    assert upstream_payload["chatInfo"].startswith("Completed client action result: ")
    assert '"name":"get_weather"' in upstream_payload["chatInfo"]
    assert '"arguments":{"city":"Shanghai"}' in upstream_payload["chatInfo"]
    assert '"content":"Sunny."' in upstream_payload["chatInfo"]
    assert "chatGroupId" not in upstream_payload
    assert all(
        message.get("role") != "tool" for message in upstream_payload["messages"]
    )
    assert all(
        not message.get("tool_calls") for message in upstream_payload["messages"]
    )
    assert not any(
        message.get("role") == "system"
        and "tool result" in str(message.get("content", "")).lower()
        for message in upstream_payload["messages"]
    )


def test_kimi_tool_history_sends_only_latest_reasoning_as_continuation_state():
    request = {
        "model": "kimi-k3",
        "messages": [
            {"role": "user", "content": "Inspect the project in several steps."},
            {
                "role": "assistant",
                "reasoning_content": "STALE_PLAN: inspect the project root.",
                "tool_calls": [
                    {
                        "id": "call_root",
                        "type": "function",
                        "function": {
                            "name": "inspect",
                            "arguments": '{"path":"."}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_root",
                "content": "Found src and tests.",
            },
            {
                "role": "assistant",
                "name": "project-agent",
                "reasoning_content": (
                    "CURRENT_PLAN: inspect src and tests in parallel, then compare "
                    "their evidence and summarize without replanning."
                ),
                "tool_calls": [
                    {
                        "id": "call_src",
                        "type": "function",
                        "function": {
                            "name": "inspect",
                            "arguments": '{"path":"src"}',
                        },
                    },
                    {
                        "id": "call_tests",
                        "type": "function",
                        "function": {
                            "name": "inspect",
                            "arguments": '{"path":"tests"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_src",
                "content": "Source evidence.",
            },
            {
                "role": "tool",
                "tool_call_id": "call_tests",
                "content": "Test evidence.",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "inspect",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
    }

    with patch(
        "genai_proxy.upstream.transport.requests.post",
        return_value=fake_completion("<k3_final>Inspection complete.</k3_final>"),
    ) as post:
        response = (
            _app()
            .test_client()
            .post(
                "/v1/chat/completions",
                json=request,
            )
        )

    assert response.status_code == 200
    upstream_payload = post.call_args.kwargs["json"]
    reasoning_messages = [
        message
        for message in upstream_payload["messages"]
        if message.get("reasoning_content")
    ]
    assert reasoning_messages == []
    assert not any(
        message.get("role") == "assistant" for message in upstream_payload["messages"]
    )
    state_messages = [
        message
        for message in upstream_payload["messages"]
        if str(message.get("content", "")).startswith("# Continuation checkpoint\n")
    ]
    assert len(state_messages) == 1
    assert (
        "CURRENT_PLAN: inspect src and tests in parallel, then compare "
        "their evidence and summarize without replanning."
        in state_messages[0]["content"]
    )
    assert "<k3_state>" in state_messages[0]["content"]
    assert '<k3_completed>[{"id":"call_src","name":"inspect"},' in (
        state_messages[0]["content"]
    )
    assert '{"id":"call_tests","name":"inspect"}]</k3_completed>' in (
        state_messages[0]["content"]
    )
    assert "STALE_PLAN" not in json.dumps(
        upstream_payload["messages"],
        ensure_ascii=False,
    )
    assert upstream_payload["chatInfo"].startswith("Completed client action result: ")
    assert '"id":"call_tests"' in upstream_payload["chatInfo"]
    assert '"arguments":{"path":"tests"}' in upstream_payload["chatInfo"]
    assert "chatGroupId" not in upstream_payload

    encoded_prompt = render_chat_prompt(
        [
            *upstream_payload["messages"],
            {"role": "user", "content": upstream_payload["chatInfo"]},
        ],
        "kimi_k3",
        add_generation_prompt=True,
    )
    assert "CURRENT_PLAN" in encoded_prompt
    assert "STALE_PLAN" not in encoded_prompt
    assert encoded_prompt.count("<k3_state>") == 1
    assert encoded_prompt.count("<k3_completed>") == 1
    assert "<|open|>think<|sep|>" in encoded_prompt


def test_kimi_message_level_tools_are_rejected_before_counting_or_transport():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/chat/completions",
            json={
                "model": "kimi-k3",
                "messages": [
                    {
                        "role": "system",
                        "content": "",
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "parameters": {"type": "object"},
                                },
                            }
                        ],
                    },
                    {"role": "user", "content": "Search."},
                ],
            },
        )
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "unsupported_tool_transport"


def test_kimi_history_preserves_official_name_and_reasoning_fields():
    messages = [
        {"role": "user", "name": "requester", "content": "Continue."},
        {
            "role": "assistant",
            "name": "agent",
            "reasoning_content": "Prior reasoning.",
            "content": "Prior answer.",
        },
        {"role": "user", "content": "Summarize."},
    ]

    prepared = (
        _app()
        .extensions["genai_service"]
        ._prepare_chat_request({"model": "kimi-k3", "messages": messages})
    )

    assert prepared.messages[:2] == messages[:2]
    prompt = render_chat_prompt(
        prepared.messages,
        "kimi_k3",
        add_generation_prompt=True,
    )
    assert 'role="user" name="requester"' in prompt
    assert 'role="assistant" name="agent"' in prompt
    assert "Prior reasoning." in prompt


def test_kimi_current_user_name_is_removed_with_chat_info_transport():
    prepared = (
        _app()
        .extensions["genai_service"]
        ._prepare_chat_request(
            {
                "model": "kimi-k3",
                "messages": [
                    {"role": "user", "name": "requester", "content": "Hello"},
                ],
            }
        )
    )

    assert prepared.messages == [{"role": "user", "content": "Hello"}]
    assert 'name="requester"' not in render_chat_prompt(
        prepared.messages,
        "kimi_k3",
        add_generation_prompt=True,
    )


def test_kimi_rejects_invalid_history_content_shape():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/chat/completions",
            json={
                "model": "kimi-k3",
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": "invalid"}],
                    },
                    {"role": "user", "content": "Hello"},
                ],
            },
        )
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "unsupported_content_type"


def test_openai_responses_input_tokens_route_uses_official_qwen_template():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/responses/input_tokens",
            json={"model": "qwen3.5", "input": "Hello, 世界"},
        )
    )
    assert response.status_code == 200
    assert response.get_json() == {
        "object": "response.input_tokens",
        "input_tokens": 14,
    }


def test_openai_responses_input_tokens_counts_qwen_visual_patches():
    data_url = _png_data_url(32, 32)
    response = (
        _app()
        .test_client()
        .post(
            "/v1/responses/input_tokens",
            json={
                "model": "qwen3.5",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": data_url,
                            }
                        ],
                    }
                ],
            },
        )
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            ],
        }
    ]
    prompt = render_chat_prompt(
        messages,
        "qwen_3_5",
        add_generation_prompt=True,
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "object": "response.input_tokens",
        "input_tokens": (
            _count_encoded("qwen_3_5", prompt) + _qwen_image_token_count(32, 32) - 1
        ),
    }


def test_qwen_merges_openai_system_and_developer_messages_before_counting():
    service = _app().extensions["genai_service"]
    prepared = service._prepare_chat_request(
        {
            "model": "qwen3.5",
            "messages": [
                {"role": "user", "content": "Earlier"},
                {"role": "developer", "content": "Developer instruction"},
                {"role": "system", "content": "System instruction"},
                {"role": "user", "content": "Hello"},
            ],
        }
    )

    assert prepared.messages == [
        {
            "role": "system",
            "content": "Developer instruction\n\nSystem instruction",
        },
        {"role": "user", "content": "Earlier"},
        {"role": "user", "content": "Hello"},
    ]
    assert prepared.prompt_tokens == count_openai_request_tokens(
        prepared.messages,
        "qwen3.5",
    )


def test_responses_count_tokens_handles_instructions_plus_developer_input():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/responses/input_tokens",
            json={
                "model": "qwen3.5",
                "instructions": "System instruction",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": "Developer instruction",
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": "Hello",
                    },
                ],
            },
        )
    )
    assert response.status_code == 200
    assert response.get_json()["input_tokens"] > 0


def test_glm_maps_developer_role_to_supported_system_role():
    service = _app().extensions["genai_service"]
    prepared = service._prepare_chat_request(
        {
            "model": "chatglm",
            "messages": [
                {"role": "developer", "content": "Developer instruction"},
                {"role": "user", "content": "Hello"},
            ],
        }
    )
    assert prepared.messages[0] == {
        "role": "system",
        "content": "Developer instruction",
    }


def test_glm51_maps_developer_role_to_supported_system_role():
    service = _app().extensions["genai_service"]
    prepared = service._prepare_chat_request(
        {
            "model": "chatglm51",
            "messages": [
                {"role": "developer", "content": "Developer instruction"},
                {"role": "user", "content": "Hello"},
            ],
        }
    )
    assert prepared.tool_adapter == "glm_5_1"
    assert prepared.messages[0] == {
        "role": "system",
        "content": "Developer instruction",
    }
    assert prepared.prompt_tokens == count_openai_request_tokens(
        prepared.messages,
        "chatglm51",
        model_record=FakeModelManager.records["chatglm51"],
        tool_adapter=prepared.tool_adapter,
    )


def test_minimax_merges_developer_and_system_for_official_template():
    service = _app().extensions["genai_service"]
    prepared = service._prepare_chat_request(
        {
            "model": "MiniMax-M1",
            "messages": [
                {"role": "user", "content": "Earlier"},
                {"role": "developer", "content": "Developer instruction"},
                {"role": "system", "content": "System instruction"},
                {"role": "user", "content": "Hello"},
            ],
        }
    )

    assert prepared.messages == [
        {
            "role": "system",
            "content": "Developer instruction\n\nSystem instruction",
        },
        {"role": "user", "content": "Earlier"},
        {"role": "user", "content": "Hello"},
    ]
    assert prepared.prompt_tokens == count_openai_request_tokens(
        prepared.messages,
        "MiniMax-M1",
        model_record=FakeModelManager.records["MiniMax-M1"],
        tool_adapter=prepared.tool_adapter,
    )


def test_kimi_maps_developer_role_to_supported_system_role():
    prepared = (
        _app()
        .extensions["genai_service"]
        ._prepare_chat_request(
            {
                "model": "kimi-k3",
                "messages": [
                    {"role": "developer", "content": "Developer instruction"},
                    {"role": "user", "content": "Hello"},
                ],
            }
        )
    )

    assert prepared.messages[0] == {
        "role": "system",
        "content": "Developer instruction",
    }


def test_kimi_reasoning_effort_count_matches_upstream_default_max():
    prepared = (
        _app()
        .extensions["genai_service"]
        ._prepare_chat_request(
            {
                "model": "kimi-k3",
                "reasoning_effort": "low",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )
    )

    assert prepared.token_reasoning_config == {"effort": "max"}
    assert prepared.prompt_tokens == 127


def test_openai_input_token_route_covers_all_supported_model_families():
    client = _app().test_client()
    expected = {
        "chatglm": 13,
        "deepseek-pro": 5,
        "deepseek-chat": 5,
        "qwen3.5": 11,
        "MiniMax-M1": 39,
        "kimi-k3": 127,
    }
    for model, input_tokens in expected.items():
        response = client.post(
            "/v1/responses/input_tokens",
            json={"model": model, "input": "Hello"},
        )
        assert response.status_code == 200
        assert response.get_json()["input_tokens"] == input_tokens


def test_completion_serialization_matches_official_templates():
    user = {"role": "user", "content": "Hello"}
    assistant = {
        "role": "assistant",
        "reasoning_content": "think",
        "content": " answer",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": (
                        '{"city":"上海","active":true,"value":null,'
                        '"items":[1],"metadata":{"unit":"c"}}'
                    ),
                },
            }
        ],
    }
    for family in (
        "glm_5_1",
        "glm_5_2",
        "deepseek_v4_pro",
        "qwen_3_5",
        "minimax_m2_7",
        "kimi_k3",
    ):
        generation_prompt = render_chat_prompt(
            [user], family, add_generation_prompt=True
        )
        completed_prompt = render_chat_prompt(
            [user, assistant], family, add_generation_prompt=False
        )
        if family == "kimi_k3":
            assert _serialized_completion(
                assistant, family
            ) == _official_kimi_completion(assistant)
            continue
        assert completed_prompt == generation_prompt + _serialized_completion(
            assistant, family
        )


def test_completion_serialization_matches_official_templates_for_sparse_outputs():
    user = {"role": "user", "content": "Hello"}
    assistants = (
        {"role": "assistant", "content": "answer"},
        {"role": "assistant", "reasoning_content": "think", "content": ""},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "ping", "arguments": "{}"},
                }
            ],
        },
    )
    for family in (
        "glm_5_1",
        "glm_5_2",
        "deepseek_v4_pro",
        "qwen_3_5",
        "kimi_k3",
    ):
        generation_prompt = render_chat_prompt(
            [user], family, add_generation_prompt=True
        )
        for assistant in assistants:
            completed_prompt = render_chat_prompt(
                [user, assistant], family, add_generation_prompt=False
            )
            if family == "kimi_k3":
                assert _serialized_completion(
                    assistant, family
                ) == _official_kimi_completion(assistant)
                continue
            assert completed_prompt == generation_prompt + _serialized_completion(
                assistant, family
            )


def test_minimax_empty_reasoning_uses_official_generation_boundary():
    user = {"role": "user", "content": "Hello"}
    sentinel = "__GENAI2OPENAI_REASONING_SENTINEL__"
    for assistant in (
        {"role": "assistant", "content": "answer"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "ping", "arguments": "{}"},
                }
            ],
        },
    ):
        with_reasoning = {
            **assistant,
            "content": assistant.get("content") or "",
            "reasoning_content": sentinel,
        }
        generation_prompt = render_chat_prompt(
            [user],
            "minimax_m2_7",
            add_generation_prompt=True,
        )
        completed_prompt = render_chat_prompt(
            [user, with_reasoning],
            "minimax_m2_7",
            add_generation_prompt=False,
        )
        official_suffix = completed_prompt.removeprefix(generation_prompt)
        assert official_suffix.startswith(f"{sentinel}\n")
        assert _serialized_completion(
            assistant,
            "minimax_m2_7",
        ) == official_suffix.removeprefix(f"{sentinel}\n")


def test_qwen_completion_count_uses_full_sequence_boundary():
    messages = [{"role": "user", "content": "Hello"}]
    assistant = {"role": "assistant", "content": "answer"}
    prompt = render_chat_prompt(messages, "qwen_3_5", add_generation_prompt=True)
    completed_prompt = render_chat_prompt(
        [*messages, assistant],
        "qwen_3_5",
        add_generation_prompt=False,
    )
    expected = _count_encoded("qwen_3_5", completed_prompt) - _count_encoded(
        "qwen_3_5", prompt
    )

    assert expected == 5
    assert (
        count_openai_completion_tokens(
            assistant,
            "qwen3.5",
            prompt_messages=messages,
        )
        == expected
    )


def test_qwen_reasoning_count_uses_prompt_boundary_and_template_trimming():
    messages = [{"role": "user", "content": "Hello"}]
    assert (
        count_openai_reasoning_tokens(
            "\nhello",
            "qwen3.5",
            prompt_messages=messages,
        )
        == 1
    )


def test_length_completion_does_not_count_unproduced_end_tokens():
    messages = [{"role": "user", "content": "Hello"}]
    assistant = {
        "role": "assistant",
        "reasoning_content": "think",
        "content": "answer",
    }
    cases = [
        ("deepseek-pro", "deepseek_v4_pro", "deepseek_v4_pro", 3),
        ("qwen3.5", "qwen_3_5", "generic", 5),
        ("kimi-k3", "kimi_k3", "kimi_k3", 8),
    ]
    for model, family, adapter, expected in cases:
        assert (
            count_openai_completion_tokens(
                assistant,
                model,
                tool_adapter=adapter,
                prompt_messages=messages,
                finish_reason="length",
            )
            == expected
        )
        assert not _serialized_completion(
            assistant,
            family,
            finish_reason="length",
        ).endswith(("<｜end▁of▁sentence｜>", "<|im_end|>\n"))


def test_length_during_reasoning_does_not_add_thinking_close_marker():
    messages = [{"role": "user", "content": "Hello"}]
    assistant = {"role": "assistant", "reasoning_content": "unfinished"}
    for model, family, adapter in (
        ("deepseek-pro", "deepseek_v4_pro", "deepseek_v4_pro"),
        ("qwen3.5", "qwen_3_5", "generic"),
        ("kimi-k3", "kimi_k3", "kimi_k3"),
    ):
        if family == "kimi_k3":
            expected = _count_encoded(family, "unfinished")
        else:
            prompt = render_chat_prompt(messages, family, add_generation_prompt=True)
            expected = _count_encoded(family, prompt + "unfinished") - _count_encoded(
                family, prompt
            )
        actual = count_openai_completion_tokens(
            assistant,
            model,
            tool_adapter=adapter,
            prompt_messages=messages,
            finish_reason="length",
        )
        assert actual == expected
        assert "</think>" not in _serialized_completion(
            assistant,
            family,
            finish_reason="length",
        )


def test_anthropic_count_tokens_route_uses_mapped_official_glm_template():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/messages/count_tokens",
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello, 世界"}],
            },
        )
    )
    assert response.status_code == 200
    assert response.get_json() == {"input_tokens": 15}


def test_anthropic_count_token_route_covers_all_supported_model_families():
    client = _app().test_client()
    expected = {
        "chatglm": 13,
        "deepseek-pro": 5,
        "deepseek-chat": 5,
        "qwen3.5": 11,
        "MiniMax-M1": 39,
        "kimi-k3": 127,
    }
    for model, input_tokens in expected.items():
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 200
        assert response.get_json() == {"input_tokens": input_tokens}


def test_nonstream_usage_covers_all_supported_model_families():
    expected = {
        "chatglm": (13, 2),
        "deepseek-pro": (5, 2),
        "deepseek-chat": (5, 2),
        "qwen3.5": (11, 5),
        "MiniMax-M1": (39, 5),
        "kimi-k3": (127, 14),
    }
    for model, (prompt_tokens, completion_tokens) in expected.items():
        service = _app().extensions["genai_service"]
        upstream = FakeResponse(
            [
                json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {"content": "answer"},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ).encode()
            ]
        )
        with patch(
            "genai_proxy.upstream.transport.requests.post",
            return_value=upstream,
        ):
            response = service.build_openai_completion(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )

        assert response["usage"]["prompt_tokens"] == prompt_tokens
        assert response["usage"]["completion_tokens"] == completion_tokens
        assert response["usage"]["total_tokens"] == prompt_tokens + completion_tokens


def test_responses_usage_uses_exact_qwen_counts():
    service = _app().extensions["genai_service"]
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "answer"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]
    )
    with patch("genai_proxy.upstream.transport.requests.post", return_value=upstream):
        response = service.build_response({"model": "qwen3.5", "input": "Hello"})

    assert response["usage"] == {
        "input_tokens": 11,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 5,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 16,
    }


def test_anthropic_nonstream_usage_uses_exact_qwen_counts():
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "answer"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]
    )
    with patch("genai_proxy.upstream.transport.requests.post", return_value=upstream):
        response = (
            _app()
            .test_client()
            .post(
                "/v1/messages",
                json={
                    "model": "qwen3.5",
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        )

    assert response.status_code == 200
    assert response.get_json()["usage"] == {
        "input_tokens": 11,
        "output_tokens": 5,
    }


def test_anthropic_stream_usage_uses_exact_qwen_counts():
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "answer"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]
    )
    with patch("genai_proxy.upstream.transport.requests.post", return_value=upstream):
        response = (
            _app()
            .test_client()
            .post(
                "/v1/messages",
                json={
                    "model": "qwen3.5",
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )
        )
        body = response.get_data(as_text=True)

    events = []
    for block in body.split("\n\n"):
        data_line = next(
            (line for line in block.splitlines() if line.startswith("data: ")),
            None,
        )
        if data_line:
            events.append(json.loads(data_line[6:]))

    start = next(event for event in events if event["type"] == "message_start")
    delta = next(event for event in events if event["type"] == "message_delta")
    assert start["message"]["usage"] == {
        "input_tokens": 11,
        "output_tokens": 0,
    }
    assert delta["usage"] == {"input_tokens": 11, "output_tokens": 5}


def test_anthropic_count_tokens_rejects_non_object_json():
    response = _app().test_client().post("/v1/messages/count_tokens", json=[])
    assert response.status_code == 400
    assert (
        response.get_json()["error"]["message"] == "Request body must be a JSON object"
    )


def test_anthropic_messages_rejects_non_object_or_invalid_json():
    client = _app().test_client()
    responses = (
        client.post("/v1/messages", json=[]),
        client.post("/v1/messages", data="{", content_type="application/json"),
    )
    for response in responses:
        assert response.status_code == 400
        assert response.get_json()["error"]["message"] == (
            "Request body must be a JSON object"
        )


def test_token_count_routes_reject_invalid_json_as_client_error():
    client = _app().test_client()
    for path in ("/v1/messages/count_tokens", "/v1/responses/input_tokens"):
        response = client.post(path, data="{", content_type="application/json")
        assert response.status_code == 400


def test_anthropic_count_tokens_rejects_invalid_messages_shape():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/messages/count_tokens",
            json={"model": "claude-sonnet-4-5", "messages": "hello"},
        )
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["message"] == (
        "'messages' must be a list of objects"
    )


def test_anthropic_messages_rejects_invalid_messages_shape():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 10,
                "messages": 1,
            },
        )
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["message"] == (
        "'messages' must be a list of objects"
    )


def test_openai_chat_rejects_non_object_message_items():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/chat/completions",
            json={"model": "qwen3.5", "messages": [1]},
        )
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["message"] == (
        "'messages' must be a list of objects"
    )


def test_openai_chat_rejects_non_object_or_invalid_json():
    client = _app().test_client()
    responses = (
        client.post("/v1/chat/completions", json=[]),
        client.post("/v1/chat/completions", data="{", content_type="application/json"),
    )
    for response in responses:
        assert response.status_code == 400
        assert response.get_json()["error"]["message"] == (
            "Request body must be a JSON object"
        )


def test_token_interfaces_reject_invalid_tool_shapes():
    client = _app().test_client()
    openai_response = client.post(
        "/v1/responses/input_tokens",
        json={"model": "qwen3.5", "input": "Hello", "tools": "weather"},
    )
    claude_response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "qwen3.5",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [1],
        },
    )

    assert openai_response.status_code == 400
    assert claude_response.status_code == 400
    assert claude_response.get_json()["error"]["message"] == (
        "'tools' must be a list of objects"
    )


def test_openai_token_interface_rejects_non_string_model():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/responses/input_tokens",
            json={"model": 35, "input": "Hello"},
        )
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["message"] == "'model' must be a string"


def test_anthropic_interfaces_reject_zero_model_as_non_string():
    client = _app().test_client()
    payload = {
        "model": 0,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    responses = (
        client.post("/v1/messages/count_tokens", json=payload),
        client.post("/v1/messages", json={**payload, "max_tokens": 10}),
    )
    for response in responses:
        assert response.status_code == 400
        assert response.get_json()["error"]["message"] == "'model' must be a string"


def test_token_interfaces_reject_malformed_nested_content_and_tool_calls():
    client = _app().test_client()
    cases = (
        (
            "/v1/messages/count_tokens",
            {
                "model": "qwen3.5",
                "messages": [{"role": "user", "content": [1]}],
            },
        ),
        (
            "/v1/messages/count_tokens",
            {
                "model": "qwen3.5",
                "system": [None],
                "messages": [{"role": "user", "content": "Hello"}],
            },
        ),
        (
            "/v1/responses/input_tokens",
            {"model": "qwen3.5", "input": [1]},
        ),
        (
            "/v1/chat/completions",
            {
                "model": "qwen3.5",
                "messages": [{"role": "user", "content": "Hello", "tool_calls": [1]}],
            },
        ),
    )
    for path, payload in cases:
        response = client.post(path, json=payload)
        assert response.status_code == 400


def test_openai_chat_rejects_malformed_function_tool_definitions():
    client = _app().test_client()
    invalid_functions = (
        {},
        {"name": 1, "parameters": {}},
        {"name": "weather", "parameters": 1},
    )
    for function in invalid_functions:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.5",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [{"type": "function", "function": function}],
            },
        )
        assert response.status_code == 400


def test_anthropic_rejects_non_object_tool_choice():
    response = (
        _app()
        .test_client()
        .post(
            "/v1/messages/count_tokens",
            json={
                "model": "qwen3.5",
                "messages": [{"role": "user", "content": "Hello"}],
                "tool_choice": "auto",
            },
        )
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["message"] == (
        "'tool_choice' must be an object"
    )


def test_chat_stream_include_usage_counts_reasoning_and_content():
    service = _app().extensions["genai_service"]
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"reasoning_content": "think", "content": "Hi"},
                            "finish_reason": None,
                        }
                    ]
                }
            ).encode(),
            json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode(),
        ]
    )
    with patch("genai_proxy.upstream.transport.requests.post", return_value=upstream):
        body = "".join(
            service.stream_openai_completion(
                {
                    "model": "chatglm",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            )
        )

    chunks = [
        json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")
    ]
    usage = next(chunk["usage"] for chunk in chunks if not chunk["choices"])
    assert usage == {
        "prompt_tokens": 13,
        "completion_tokens": 3,
        "total_tokens": 16,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 1},
    }


def test_chat_stream_without_usage_skips_tokenizer_counting():
    service = _app().extensions["genai_service"]
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "answer"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]
    )
    with (
        patch("genai_proxy.upstream.transport.requests.post", return_value=upstream),
        patch(
            "genai_proxy.chat.preparation.count_openai_request_tokens"
        ) as prepare_count_request,
        patch(
            "genai_proxy.chat.usage.count_openai_request_tokens"
        ) as usage_count_request,
    ):
        body = "".join(
            service.stream_openai_completion(
                {
                    "model": "qwen3.5",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                }
            )
        )

    prepare_count_request.assert_not_called()
    usage_count_request.assert_not_called()
    chunks = [
        json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")
    ]
    assert not any(not chunk["choices"] for chunk in chunks)


def test_chat_stream_defers_requested_usage_until_after_first_delta():
    service = _app().extensions["genai_service"]
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "first"},
                            "finish_reason": None,
                        }
                    ]
                }
            ).encode(),
            json.dumps(
                {
                    "choices": [
                        {"delta": {"content": " second"}, "finish_reason": "stop"}
                    ]
                }
            ).encode(),
        ]
    )
    with (
        patch("genai_proxy.upstream.transport.requests.post", return_value=upstream),
        patch(
            "genai_proxy.chat.preparation.count_openai_request_tokens",
        ) as prepare_count_request,
        patch(
            "genai_proxy.chat.usage.count_openai_request_tokens",
            return_value=13,
        ) as count_request,
    ):
        stream = service.stream_openai_completion(
            {
                "model": "chatglm",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        prepare_count_request.assert_not_called()
        count_request.assert_not_called()
        first = next(stream)
        prepare_count_request.assert_not_called()
        count_request.assert_not_called()
        remaining = "".join(stream)

    assert '"content": "first"' in first
    prepare_count_request.assert_not_called()
    count_request.assert_called_once()
    assert '"usage"' in remaining


def test_chat_usage_propagates_length_finish_reason():
    service = _app().extensions["genai_service"]
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": "think",
                                "content": "answer",
                            },
                            "finish_reason": "length",
                        }
                    ]
                }
            ).encode()
        ]
    )
    request = {
        "model": "qwen3.5",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    with patch("genai_proxy.upstream.transport.requests.post", return_value=upstream):
        response = service.build_openai_completion(request)

    assert response["choices"][0]["finish_reason"] == "length"
    assert response["usage"]["completion_tokens"] == 5


def test_parsed_qwen_tool_call_usage_counts_raw_generated_syntax():
    service = _app().extensions["genai_service"]
    raw_content = (
        '<tool_call>{"name":"weather","arguments":{"city":"上海"}}</tool_call>'
    )
    upstream = FakeResponse(
        [
            json.dumps(
                {
                    "choices": [
                        {"delta": {"content": raw_content}, "finish_reason": None}
                    ]
                }
            ).encode(),
            json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode(),
        ]
    )
    request = {
        "model": "qwen3.5",
        "messages": [{"role": "user", "content": "Check weather"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }
    with patch("genai_proxy.upstream.transport.requests.post", return_value=upstream):
        response = service.build_openai_completion(request)

    assert response["choices"][0]["finish_reason"] == "tool_calls"
    assert (
        response["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        == "weather"
    )
    expected = count_openai_completion_tokens(
        {"role": "assistant", "content": raw_content},
        "qwen3.5",
        model_record=FakeModelManager.records["qwen3.5"],
        tool_adapter="generic",
        prompt_messages=service._prepare_chat_request(request).messages,
    )
    assert response["usage"]["completion_tokens"] == expected
