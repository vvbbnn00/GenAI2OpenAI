import base64
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
from flask import Flask

from genai_proxy.api.openai.routes import bp as openai_bp
from genai_proxy.api.openai.service import GenAIService
from genai_proxy.chat.tool_protocol import extract_tool_calls
from genai_proxy.config import AppConfig
from genai_proxy.errors import ProxyError
from genai_proxy.logging_utils import safe_log_code
from genai_proxy.models.deepseek_v4.tooling import extract_deepseek_tool_calls
from genai_proxy.models.qwen35.tooling import extract_qwen35_tool_calls
from genai_proxy.runtime import log_startup
from genai_proxy.upstream import transport
from genai_proxy.upstream.auth import TokenManager
from genai_proxy.upstream.catalog import ModelManager
from genai_proxy.upstream.kimi_history import _KimiHistoryCleanup

SENSITIVE_MARKER = "LOG_PRIVACY_SENTINEL_DO_NOT_EMIT"


class _TokenManager:
    token = "test-token"
    billing_user_id = None

    def refresh_after_auth_failure(self, *_args, **_kwargs):
        return False

    def update_billing_user_id(self, _user_id):
        return True

    def update_token_from_upstream(self, _token, _reason):
        return True


class _ModelManager:
    record = {
        "aiType": "chatglm",
        "aiName": "GLM-5.2",
        "rootAiType": "xinference",
        "rootModelName": "Xinference",
    }

    def resolve_model(self, model):
        return model or "chatglm"

    def get_model_record(self, model):
        return self.record if model == "chatglm" else None

    def root_ai_type_for(self, _model):
        return "xinference"


class _JsonResponse:
    def __init__(self, payload=None, *, status_code=200, text=""):
        self.payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = text
        self.closed = False

    def json(self):
        return self.payload

    def close(self):
        self.closed = True


class _StreamResponse(_JsonResponse):
    def __init__(self, lines, *, status_code=200, text=""):
        super().__init__(status_code=status_code, text=text)
        self.lines = lines

    def iter_lines(self, **_kwargs):
        yield from self.lines


def _jwt(*, username=SENSITIVE_MARKER):
    payload = {
        "username": username,
        "exp": int(time.time()) + 3600,
    }

    def encode(value):
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return ".".join((encode({"alg": "none"}), encode(payload), "signature"))


def _service(logger):
    return GenAIService(
        logger,
        _TokenManager(),
        _ModelManager(),
        max_retries=0,
    )


def test_log_codes_allow_only_http_style_numbers():
    assert safe_log_code(400) == "400"
    assert safe_log_code("503") == "503"
    assert safe_log_code(123456789) == "int"
    assert safe_log_code(SENSITIVE_MARKER) == "str"
    assert safe_log_code(f"400\n{SENSITIVE_MARKER}") == "str"


def test_startup_and_passkey_logs_redact_identity_and_paths(caplog, tmp_path):
    logger = logging.getLogger("test_logging_privacy.startup")
    keystore_path = tmp_path / f"{SENSITIVE_MARKER}.keystore"
    model_cache_path = tmp_path / f"{SENSITIVE_MARKER}.models.json"
    config = AppConfig(
        token=_jwt(),
        keystore=str(keystore_path),
        port=5000,
        debug=True,
        api_key=None,
        token_check_interval=0,
        claude_haiku_model="deepseek-chat",
        claude_sonnet_model="chatglm",
        claude_opus_model="chatglm",
        genai_model_cache=str(model_cache_path),
    )

    class FakeKeystore:
        username = SENSITIVE_MARKER

        def dump(self, _path):
            return None

    class FakeSession:
        def get(self, _url, **_kwargs):
            return _JsonResponse({"result": {"token": _jwt()}})

        def close(self):
            return None

    class FakeClient:
        session = FakeSession()

        def login(self):
            return None

        def logout(self):
            return None

    class LoginSession:
        def get(self, _url, **_kwargs):
            return SimpleNamespace(
                url=f"https://genai.example.invalid/{SENSITIVE_MARKER}?token=safe"
            )

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        log_startup(config, logger)
        manager = TokenManager(
            logger,
            token=_jwt(),
            keystore_path=str(keystore_path),
            token_check_interval=0,
        )
        manager._keystore = FakeKeystore()
        manager._ids_client = FakeClient()
        with patch.object(
            manager,
            "_get_genai_login_response",
            return_value=SimpleNamespace(
                url=f"https://genai.example.invalid/login?token={SENSITIVE_MARKER}"
            ),
        ):
            manager._refresh_token()
        manager._get_genai_login_response(SimpleNamespace(session=LoginSession()))
        manager.shutdown()

    assert SENSITIVE_MARKER not in caplog.text
    assert str(tmp_path) not in caplog.text
    assert "Keystore: configured" in caplog.text
    assert "GenAI model cache: persistent" in caplog.text
    assert "IDS passkey login successful" in caplog.text


def test_token_cache_failures_log_only_exception_types(caplog, tmp_path):
    logger = logging.getLogger("test_logging_privacy.token_cache")
    cache_owner = tmp_path / SENSITIVE_MARKER
    manager = TokenManager(
        logger,
        token=_jwt(username="test-user"),
        keystore_path=str(cache_owner),
        token_check_interval=0,
    )

    with caplog.at_level(logging.WARNING, logger=logger.name):
        with patch("builtins.open", side_effect=OSError(SENSITIVE_MARKER)):
            assert manager._load_cached_token() is False
            manager._write_cached_token()
        with patch(
            "genai_proxy.upstream.auth.os.remove",
            side_effect=OSError(SENSITIVE_MARKER),
        ):
            manager._delete_cached_token()
    manager.shutdown()

    assert SENSITIVE_MARKER not in caplog.text
    assert str(tmp_path) not in caplog.text
    assert caplog.text.count("OSError") == 3


def test_catalog_and_account_failures_do_not_log_urls_bodies_or_ids(
    caplog,
    tmp_path,
):
    logger = logging.getLogger("test_logging_privacy.account")
    model_manager = ModelManager(logger, _TokenManager(), max_retries=0)
    service = _service(logger)
    cache_path = tmp_path / f"{SENSITIVE_MARKER}.json"
    cache_path.write_text(SENSITIVE_MARKER, encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        cached_manager = ModelManager(
            logger,
            _TokenManager(),
            cache_path=str(cache_path),
        )
        with patch(
            "genai_proxy.upstream.catalog.tempfile.NamedTemporaryFile",
            side_effect=OSError(SENSITIVE_MARKER),
        ):
            cached_manager._write_persistent_cache([], time.time())

        with (
            patch(
                "genai_proxy.upstream.catalog.requests.get",
                side_effect=requests.ConnectionError(
                    f"https://genai.example.invalid/{SENSITIVE_MARKER}"
                ),
            ),
            pytest.raises(ProxyError),
        ):
            model_manager._fetch_models_once("test-token")

        with (
            patch(
                "genai_proxy.upstream.catalog.requests.get",
                return_value=_JsonResponse(
                    status_code=500,
                    text=SENSITIVE_MARKER,
                ),
            ),
            pytest.raises(ProxyError),
        ):
            model_manager._fetch_models_once("test-token")

        with (
            patch(
                "genai_proxy.upstream.catalog.requests.get",
                return_value=_JsonResponse(
                    {"success": False, "code": 400, "message": SENSITIVE_MARKER}
                ),
            ),
            pytest.raises(ProxyError),
        ):
            model_manager._fetch_models_once("test-token")

        with (
            patch(
                "genai_proxy.chat.service.upstream_transport.fetch_user_info",
                side_effect=requests.ConnectionError(
                    f"https://genai.example.invalid/?userId={SENSITIVE_MARKER}"
                ),
            ),
            pytest.raises(ProxyError),
        ):
            service._fetch_user_info_record("test-token", SENSITIVE_MARKER)

        with (
            patch(
                "genai_proxy.chat.service.upstream_transport.fetch_user_info",
                return_value=_JsonResponse(
                    {
                        "success": True,
                        "code": 200,
                        "result": {"records": [{"id": SENSITIVE_MARKER}]},
                    }
                ),
            ),
            pytest.raises(ProxyError),
        ):
            service._fetch_user_info_record("test-token", "different-id")

        with (
            patch(
                "genai_proxy.chat.service.upstream_transport.fetch_current_user",
                return_value=_JsonResponse(
                    {
                        "success": False,
                        "code": 400,
                        "message": SENSITIVE_MARKER,
                    }
                ),
            ),
            pytest.raises(ProxyError),
        ):
            service._fetch_current_user_id(SENSITIVE_MARKER)

    assert SENSITIVE_MARKER not in caplog.text
    assert str(tmp_path) not in caplog.text
    assert "ConnectionError" in caplog.text
    assert "HTTP error 500" in caplog.text
    assert "business error (code=400)" in caplog.text
    assert "did not match the current account" in caplog.text


def test_chat_debug_logs_keep_metadata_without_request_or_response_content(caplog):
    logger = logging.getLogger("test_logging_privacy.chat")
    service = _service(logger)
    request = {
        "model": "chatglm",
        "messages": [{"role": "user", "content": SENSITIVE_MARKER}],
        "stream": True,
    }
    response = _StreamResponse(
        [
            "data: "
            + json.dumps(
                {
                    "error": {
                        "message": SENSITIVE_MARKER,
                        "type": "upstream_error",
                        "code": 400,
                    }
                }
            )
        ]
    )

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        with patch.object(transport, "post_chat", return_value=response) as post:
            with pytest.raises(ProxyError) as raised:
                list(service.stream_openai_completion(request))

        invalid_frame = _StreamResponse([f"data: {SENSITIVE_MARKER}"])
        assert list(transport.iter_sse_json(invalid_frame, logger)) == []

    assert response.closed
    assert raised.value.message == SENSITIVE_MARKER
    assert "chatGroupId" not in post.call_args.args[1]
    assert SENSITIVE_MARKER not in caplog.text
    assert "content_kind=text" in caplog.text
    assert "Request metrics: prompt_tokens=" in caplog.text
    assert "prompt_tokens=None" not in caplog.text
    assert "messages=1, tools=0" in caplog.text
    assert "max_output_tokens=30000" in caplog.text
    assert "payload_bytes=" in caplog.text
    assert "SSE line" in caplog.text
    assert "structured error (status=400, code=missing)" in caplog.text
    assert "JSON decode error (JSONDecodeError" in caplog.text


def test_tool_parsers_do_not_log_generated_names_or_arguments(caplog):
    logger = logging.getLogger("test_logging_privacy.tools")
    invalid_call = f"<tool_call>{SENSITIVE_MARKER}</tool_call>"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "allowed",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        extract_tool_calls(invalid_call, logger=logger, tools=tools)
        extract_deepseek_tool_calls(invalid_call, logger=logger, tools=tools)
        extract_qwen35_tool_calls(
            f"<tool_call><function={SENSITIVE_MARKER}></function></tool_call>",
            logger=logger,
            tools=tools,
        )

    assert SENSITIVE_MARKER not in caplog.text
    assert "Failed to parse tool_call[0]" in caplog.text
    assert "DeepSeek repair failed for tool_call[0]" in caplog.text
    assert "Qwen 3.5 returned an unknown tool name" in caplog.text


def test_kimi_cleanup_errors_do_not_log_question_user_or_group_ids(caplog):
    logger = logging.getLogger("test_logging_privacy.kimi")
    service = _service(logger)
    cleanup = _KimiHistoryCleanup(
        question=SENSITIVE_MARKER,
        user_id=SENSITIVE_MARKER,
        existing_group_ids=frozenset({SENSITIVE_MARKER}),
    )

    with (
        caplog.at_level(logging.WARNING, logger=logger.name),
        patch.object(
            service,
            "_with_token_auth_retry",
            side_effect=RuntimeError(SENSITIVE_MARKER),
        ),
    ):
        assert service._prepare_kimi_history_cleanup(SENSITIVE_MARKER) is None
        service._delete_completed_kimi_history(cleanup)

    assert SENSITIVE_MARKER not in caplog.text
    assert caplog.text.count("RuntimeError") == 2


def test_unhandled_request_errors_log_the_type_without_the_message(caplog):
    logger = logging.getLogger("test_logging_privacy.route")

    class FailingService:
        def build_openai_completion(self, _request):
            raise RuntimeError(SENSITIVE_MARKER)

    app = Flask(__name__)
    app.register_blueprint(openai_bp)
    app.extensions["logger"] = logger
    app.extensions["genai_service"] = FailingService()

    with caplog.at_level(logging.ERROR, logger=logger.name):
        response = app.test_client().post(
            "/v1/chat/completions",
            json={"model": "chatglm", "messages": []},
        )

    assert response.status_code == 500
    assert SENSITIVE_MARKER in response.get_data(as_text=True)
    assert SENSITIVE_MARKER not in caplog.text
    assert "RuntimeError" in caplog.text


def test_source_logging_contract_rejects_direct_content_and_tracebacks():
    source_root = Path(__file__).resolve().parents[2] / "src" / "genai_proxy"
    package_source = "\n".join(
        path.read_text(encoding="utf-8") for path in source_root.rglob("*.py")
    )
    forbidden = (
        "Token username:",
        "successful for user:",
        "Raw line",
        "content=%s",
        "raw: %s",
        "response.text[:",
        "Keystore: %s",
    )

    for marker in forbidden:
        assert marker not in package_source

    sensitive_boundaries = (
        "api/anthropic/compat.py",
        "api/anthropic/routes.py",
        "api/openai/routes.py",
        "chat/service.py",
        "chat/streaming.py",
        "chat/tool_protocol.py",
        "models/deepseek_v4/tooling.py",
        "models/kimi_k3/tooling.py",
        "models/qwen35/tooling.py",
        "upstream/auth.py",
        "upstream/catalog.py",
        "upstream/kimi_history.py",
        "upstream/transport.py",
    )
    for relative_path in sensitive_boundaries:
        source = (source_root / relative_path).read_text(encoding="utf-8")
        assert ".exception(" not in source
