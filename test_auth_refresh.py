import base64
import json
import logging
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from flask import Flask

from genai_proxy.compat.claude import stream_openai_to_claude
from genai_proxy.compat.openai import make_error_chunk
from genai_proxy.errors import ProxyError
from genai_proxy.routes.openai import bp as openai_bp
from genai_proxy.services.genai import GenAIService
from genai_proxy.services.models import ModelManager
from genai_proxy.services.token_manager import (
    GENAI_LEGACY_CAS_SERVICE_URL,
    GENAI_LOGIN_URL,
    TokenManager,
    is_genai_auth_failure,
)


def make_jwt(exp: int | None = None) -> str:
    payload = {"username": "2025233184", "exp": exp or int(time.time()) + 3600}
    return ".".join(
        [
            _b64({"typ": "JWT", "alg": "HS256"}),
            _b64(payload),
            "signature",
        ]
    )


def _b64(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


class FakeResponse:
    def __init__(self, payload=None, status_code=200, url="https://genai.shanghaitech.edu.cn/"):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.url = url
        self.text = json.dumps(self._payload, ensure_ascii=False)

    def json(self):
        return self._payload


class FakeStreamingResponse(FakeResponse):
    def __init__(self, lines, status_code=200, payload=None):
        super().__init__(payload=payload, status_code=status_code)
        self._lines = lines
        self.closed = False

    def iter_lines(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


class FakeTokenManager:
    def __init__(self):
        self._token = "stale-token"
        self.refresh_count = 0
        self.rejected_token = None
        self.upstream_token_updates = []
        self._billing_user_id = None

    @property
    def token(self):
        return self._token

    def refresh_after_auth_failure(self, reason, rejected_token=None):
        self.refresh_count += 1
        self.rejected_token = rejected_token
        self._token = "fresh-token"
        return True

    def update_token_from_upstream(self, token, reason):
        self.upstream_token_updates.append((token, reason))
        self._token = token
        return True

    def update_billing_user_id(self, user_id):
        self._billing_user_id = str(user_id)
        return True

    @property
    def billing_user_id(self):
        return self._billing_user_id


class FailedRefreshTokenManager(FakeTokenManager):
    def refresh_after_auth_failure(self, reason, rejected_token=None):
        self.refresh_count += 1
        self.rejected_token = rejected_token
        return False


class FakeModelManager:
    def resolve_model(self, model):
        return model or "chatglm"

    def get_model_record(self, model):
        return {"aiType": model, "rootAiType": "xinference"}

    def root_ai_type_for(self, model):
        return "xinference"


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return self.responses.pop(0)


class AuthRefreshTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test_auth_refresh")

    def test_genai_auth_failure_detection_matches_current_business_errors(self):
        self.assertTrue(
            is_genai_auth_failure(
                {"success": False, "message": "Token失效，请重新登录", "code": 500, "result": None}
            )
        )
        self.assertTrue(
            is_genai_auth_failure(
                {"success": False, "message": "鉴权失败，请重新登录", "code": 500, "result": None}
            )
        )
        self.assertFalse(
            is_genai_auth_failure(
                {"success": False, "message": "模型不存在", "code": 500, "result": None}
            )
        )

    def test_model_resolution_matches_upstream_ids_case_insensitively(self):
        manager = ModelManager(self.logger, FakeTokenManager())
        record = {
            "aiType": "Kimi-k3",
            "aiName": "Kimi-K3",
            "rootAiType": "xinference",
        }
        manager._models_cache = [record]
        manager._models_cache_at = time.time()

        self.assertEqual(manager.resolve_model("kimi-k3"), "Kimi-k3")
        self.assertIs(manager.get_model_record("KIMI-K3"), record)

    def test_model_list_refreshes_once_when_cached_token_is_rejected(self):
        token_manager = FakeTokenManager()
        manager = ModelManager(self.logger, token_manager)
        responses = [
            FakeResponse({"success": False, "message": "Token失效，请重新登录", "code": 500, "result": None}),
            FakeResponse(
                {
                    "success": True,
                    "code": 200,
                    "result": {
                        "records": [
                            {
                                "aiType": "deepseek-chat",
                                "aiName": "DeepSeek-V4-Flash",
                                "createTime": "2026-04-25 00:00:00",
                            }
                        ]
                    },
                }
            ),
        ]

        with patch("genai_proxy.services.models.requests.get", side_effect=responses) as mocked_get:
            models = manager.list_genai_models(force_refresh=True)

        self.assertEqual(token_manager.refresh_count, 1)
        self.assertEqual(token_manager.rejected_token, "stale-token")
        self.assertEqual(models[0]["aiType"], "deepseek-chat")
        self.assertEqual(mocked_get.call_args_list[0].kwargs["headers"]["X-Access-Token"], "stale-token")
        self.assertEqual(mocked_get.call_args_list[1].kwargs["headers"]["X-Access-Token"], "fresh-token")

    def test_chat_preserves_structured_upstream_error_details(self):
        service = GenAIService(
            self.logger,
            FakeTokenManager(),
            FakeModelManager(),
            max_retries=0,
        )
        response = FakeStreamingResponse(
            [
                json.dumps(
                    {
                        "error": {
                            "message": "message content cannot be empty",
                            "type": "invalid_request_error",
                            "code": 400,
                        }
                    }
                ).encode()
            ]
        )

        with patch(
            "genai_proxy.services.genai.requests.post",
            return_value=response,
        ):
            with self.assertRaises(ProxyError) as raised:
                list(
                    service.stream_openai_completion(
                        {
                            "model": "chatglm",
                            "messages": [{"role": "user", "content": "Hello"}],
                        }
                    )
                )

        self.assertEqual(raised.exception.message, "message content cannot be empty")
        self.assertEqual(raised.exception.error_type, "invalid_request_error")
        self.assertEqual(raised.exception.status, 400)

    def test_service_auth_retry_passes_rejected_token_to_refresh(self):
        token_manager = FakeTokenManager()
        service = GenAIService(self.logger, token_manager, None)
        seen_tokens = []

        def fetch(token):
            seen_tokens.append(token)
            if len(seen_tokens) == 1:
                raise ProxyError(
                    "rejected",
                    error_type="authentication_error",
                    code="upstream_auth_failed",
                    status=502,
                )
            return "ok"

        self.assertEqual(service._with_token_auth_retry("unit test", fetch), "ok")
        self.assertEqual(seen_tokens, ["stale-token", "fresh-token"])
        self.assertEqual(token_manager.rejected_token, "stale-token")

    def test_non_stream_completion_raises_when_stream_auth_refresh_fails(self):
        token_manager = FailedRefreshTokenManager()
        service = GenAIService(self.logger, token_manager, FakeModelManager())
        auth_failure = {"success": False, "message": "Token失效，请重新登录", "code": 500, "result": None}

        with patch(
            "genai_proxy.services.genai.requests.post",
            return_value=FakeStreamingResponse([json.dumps(auth_failure, ensure_ascii=False).encode()]),
        ):
            with self.assertRaises(ProxyError) as raised:
                service.build_openai_completion(
                    {
                        "model": "chatglm",
                        "messages": [{"role": "user", "content": "hello"}],
                    }
                )

        self.assertEqual(raised.exception.code, "upstream_auth_failed")
        self.assertEqual(raised.exception.status, 502)
        self.assertEqual(token_manager.refresh_count, 1)
        self.assertEqual(token_manager.rejected_token, "stale-token")

    def test_tool_stream_raises_when_stream_auth_refresh_fails(self):
        token_manager = FailedRefreshTokenManager()
        service = GenAIService(self.logger, token_manager, FakeModelManager())
        auth_failure = {"success": False, "message": "Token失效，请重新登录", "code": 500, "result": None}

        with patch(
            "genai_proxy.services.genai.requests.post",
            return_value=FakeStreamingResponse([json.dumps(auth_failure, ensure_ascii=False).encode()]),
        ):
            stream = service.stream_openai_completion(
                {
                    "model": "chatglm",
                    "messages": [{"role": "user", "content": "what is the weather"}],
                    "stream": True,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"location": {"type": "string"}},
                                    "required": ["location"],
                                },
                            },
                        }
                    ],
                }
            )
            with self.assertRaises(ProxyError) as raised:
                list(stream)

        self.assertEqual(raised.exception.code, "upstream_auth_failed")
        self.assertEqual(token_manager.refresh_count, 1)

    def test_non_stream_completion_preserves_upstream_finish_reason(self):
        service = GenAIService(self.logger, FakeTokenManager(), FakeModelManager())
        completion_line = json.dumps(
            {
                "choices": [
                    {
                        "delta": {"content": "partial answer"},
                        "finish_reason": "length",
                    }
                ]
            }
        )

        with patch(
            "genai_proxy.services.genai.requests.post",
            return_value=FakeStreamingResponse([completion_line.encode()]),
        ):
            response = service.build_openai_completion(
                {
                    "model": "chatglm",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            )

        choice = response["choices"][0]
        self.assertEqual(choice["message"]["content"], "partial answer")
        self.assertEqual(choice["finish_reason"], "length")

    def test_non_stream_completion_raises_on_internal_stream_error_chunk(self):
        service = GenAIService(
            self.logger,
            FakeTokenManager(),
            FakeModelManager(),
            max_retries=0,
        )
        business_error = {
            "success": False,
            "message": "temporary upstream failure",
            "code": 500,
            "result": None,
        }

        with patch(
            "genai_proxy.services.genai.requests.post",
            return_value=FakeStreamingResponse([json.dumps(business_error).encode()]),
        ):
            with self.assertRaises(ProxyError) as raised:
                service.build_openai_completion(
                    {
                        "model": "chatglm",
                        "messages": [{"role": "user", "content": "hello"}],
                    }
                )

        self.assertEqual(raised.exception.error_type, "upstream_error")
        self.assertEqual(raised.exception.status, 502)
        self.assertEqual(raised.exception.message, "Upstream error: temporary upstream failure")

    def test_stream_completion_raises_on_initial_business_error(self):
        service = GenAIService(
            self.logger,
            FakeTokenManager(),
            FakeModelManager(),
            max_retries=0,
        )
        business_error = {
            "success": False,
            "message": "temporary upstream failure",
            "code": 500,
            "result": None,
        }

        with patch(
            "genai_proxy.services.genai.requests.post",
            return_value=FakeStreamingResponse([json.dumps(business_error).encode()]),
        ):
            stream = service.stream_openai_completion(
                {
                    "model": "chatglm",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                }
            )
            with self.assertRaises(ProxyError) as raised:
                next(stream)

        self.assertEqual(raised.exception.error_type, "upstream_error")
        self.assertEqual(raised.exception.status, 502)
        self.assertEqual(raised.exception.message, "Upstream error: temporary upstream failure")

    def test_tool_stream_raises_when_buffered_attempt_errors_after_content(self):
        service = GenAIService(
            self.logger,
            FakeTokenManager(),
            FakeModelManager(),
            max_retries=0,
        )
        lines = [
            json.dumps({"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]}).encode(),
            json.dumps(
                {
                    "success": False,
                    "message": "temporary upstream failure",
                    "code": 500,
                    "result": None,
                }
            ).encode(),
        ]

        with patch(
            "genai_proxy.services.genai.requests.post",
            return_value=FakeStreamingResponse(lines),
        ):
            stream = service.stream_openai_completion(
                {
                    "model": "chatglm",
                    "messages": [{"role": "user", "content": "what is the weather"}],
                    "stream": True,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"location": {"type": "string"}},
                                    "required": ["location"],
                                },
                            },
                        }
                    ],
                }
            )
            with self.assertRaises(ProxyError) as raised:
                list(stream)

        self.assertEqual(raised.exception.error_type, "upstream_error")
        self.assertEqual(raised.exception.status, 502)
        self.assertEqual(raised.exception.message, "Upstream error: temporary upstream failure")

    def test_non_stream_empty_error_finish_uses_generic_error_message(self):
        service = GenAIService(
            self.logger,
            FakeTokenManager(),
            FakeModelManager(),
            max_retries=0,
        )
        lines = [
            json.dumps({"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]}).encode(),
            json.dumps({"choices": [{"delta": {}, "finish_reason": "error"}]}).encode(),
        ]

        with patch(
            "genai_proxy.services.genai.requests.post",
            return_value=FakeStreamingResponse(lines),
        ):
            with self.assertRaises(ProxyError) as raised:
                service.build_openai_completion(
                    {
                        "model": "chatglm",
                        "messages": [{"role": "user", "content": "hello"}],
                    }
                )

        self.assertEqual(raised.exception.message, "Upstream error")

    def test_openai_stream_route_returns_502_when_first_chunk_raises_proxy_error(self):
        class FailingStreamService:
            def stream_openai_completion(self, req_data):
                def gen():
                    raise ProxyError(
                        "Upstream GenAI token is invalid or expired",
                        error_type="authentication_error",
                        code="upstream_auth_failed",
                        status=502,
                    )
                    yield ""

                return gen()

        app = Flask(__name__)
        app.extensions["genai_service"] = FailingStreamService()
        app.extensions["logger"] = self.logger
        app.register_blueprint(openai_bp)

        response = app.test_client().post(
            "/v1/chat/completions",
            json={
                "model": "chatglm",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json["error"]["code"], "upstream_auth_failed")

    def test_openai_stream_route_returns_500_when_first_chunk_raises_unhandled_error(self):
        class FailingStreamService:
            def stream_openai_completion(self, req_data):
                def gen():
                    raise RuntimeError("unexpected stream failure")
                    yield ""

                return gen()

        app = Flask(__name__)
        app.extensions["genai_service"] = FailingStreamService()
        app.extensions["logger"] = self.logger
        app.register_blueprint(openai_bp)

        with patch.object(self.logger, "exception"):
            response = app.test_client().post(
                "/v1/chat/completions",
                json={
                    "model": "chatglm",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json["error"]["code"], "internal_error")

    def test_claude_stream_conversion_raises_before_message_start_when_openai_auth_fails(self):
        def failing_openai_stream():
            raise ProxyError(
                "Upstream GenAI token is invalid or expired",
                error_type="authentication_error",
                code="upstream_auth_failed",
                status=502,
            )
            yield ""

        stream = stream_openai_to_claude(
            failing_openai_stream(),
            {
                "model": "claude-sonnet",
                "_estimator_model": "chatglm",
                "messages": [{"role": "user", "content": "hello"}],
            },
            self.logger,
        )

        with self.assertRaises(ProxyError):
            next(stream)

    def test_claude_stream_conversion_raises_before_message_start_for_openai_error_chunk(self):
        stream = stream_openai_to_claude(
            iter([make_error_chunk("upstream failed", "chatglm")]),
            {
                "model": "claude-sonnet",
                "_estimator_model": "chatglm",
                "messages": [{"role": "user", "content": "hello"}],
            },
            self.logger,
        )

        with self.assertRaises(ProxyError) as raised:
            next(stream)

        self.assertEqual(raised.exception.status, 502)
        self.assertEqual(raised.exception.message, "upstream failed")

    def test_claude_stream_conversion_emits_error_event_for_late_openai_error_chunk(self):
        normal_chunk = (
            "data: "
            + json.dumps({"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]})
            + "\n\n"
        )
        stream = stream_openai_to_claude(
            iter([normal_chunk, make_error_chunk("late failure", "chatglm")]),
            {
                "model": "claude-sonnet",
                "_estimator_model": "chatglm",
                "messages": [{"role": "user", "content": "hello"}],
            },
            self.logger,
        )

        events = list(stream)

        self.assertTrue(any("event: content_block_delta" in event for event in events))
        self.assertTrue(any("event: error" in event and "late failure" in event for event in events))

    def test_billing_stores_token_returned_by_current_user_response(self):
        old_token = make_jwt(exp=int(time.time()) + 3600)
        new_exp = int(time.time()) + 7200
        new_token = make_jwt(exp=new_exp)
        token_manager = FakeTokenManager()
        token_manager._token = old_token
        service = GenAIService(self.logger, token_manager, None)

        responses = [
            FakeResponse(
                {
                    "success": True,
                    "code": 200,
                    "result": {
                        "token": new_token,
                        "userInfo": {"id": "42"},
                    },
                }
            ),
            FakeResponse(
                {
                    "success": True,
                    "code": 200,
                    "result": {
                        "records": [
                            {
                                "id": "42",
                                "quota": "12.5",
                            }
                        ]
                    },
                }
            ),
        ]

        with patch("genai_proxy.services.genai.requests.get", side_effect=responses) as mocked_get:
            result = service.fetch_openai_billing_subscription()

        self.assertEqual(result["access_until"], new_exp)
        self.assertEqual(token_manager.token, new_token)
        self.assertEqual(token_manager.billing_user_id, "42")
        self.assertEqual(token_manager.upstream_token_updates, [(new_token, "current user response")])
        self.assertEqual(mocked_get.call_args_list[1].kwargs["headers"]["X-Access-Token"], old_token)

    def test_billing_reuses_cached_user_id_after_first_lookup(self):
        old_token = make_jwt(exp=int(time.time()) + 3600)
        new_token = make_jwt(exp=int(time.time()) + 7200)
        token_manager = FakeTokenManager()
        token_manager._token = old_token
        service = GenAIService(self.logger, token_manager, None)

        responses = [
            FakeResponse(
                {
                    "success": True,
                    "code": 200,
                    "result": {
                        "token": new_token,
                        "userInfo": {"id": "42"},
                    },
                }
            ),
            FakeResponse(
                {
                    "success": True,
                    "code": 200,
                    "result": {"records": [{"id": "42", "quota": "12.5"}]},
                }
            ),
            FakeResponse(
                {
                    "success": True,
                    "code": 200,
                    "result": {"records": [{"id": "42", "monthSurplus": "0.5"}]},
                }
            ),
        ]

        with patch("genai_proxy.services.genai.requests.get", side_effect=responses) as mocked_get:
            service.fetch_openai_billing_subscription()
            usage = service.fetch_openai_billing_usage()

        self.assertEqual(usage["total_usage"], 50.0)
        self.assertEqual(mocked_get.call_count, 3)
        self.assertIn("/htk/user/info/", mocked_get.call_args_list[0].args[0])
        self.assertIn("/htk/ai-user-info/list", mocked_get.call_args_list[1].args[0])
        self.assertIn("/htk/ai-user-info/list", mocked_get.call_args_list[2].args[0])
        self.assertEqual(mocked_get.call_args_list[2].kwargs["headers"]["X-Access-Token"], new_token)

    def test_billing_uses_cached_user_id_without_current_user_lookup(self):
        token = make_jwt()
        token_manager = FakeTokenManager()
        token_manager._token = token
        token_manager._billing_user_id = "42"
        service = GenAIService(self.logger, token_manager, None)

        with patch(
            "genai_proxy.services.genai.requests.get",
            return_value=FakeResponse(
                {
                    "success": True,
                    "code": 200,
                    "result": {"records": [{"id": "42", "monthSurplus": "0.5"}]},
                }
            ),
        ) as mocked_get:
            usage = service.fetch_openai_billing_usage()

        self.assertEqual(usage["total_usage"], 50.0)
        self.assertEqual(mocked_get.call_count, 1)
        self.assertIn("/htk/ai-user-info/list", mocked_get.call_args.args[0])

    def test_model_list_result_null_raises_proxy_error_instead_of_attribute_error(self):
        token_manager = FakeTokenManager()
        manager = ModelManager(self.logger, token_manager, max_retries=0)

        with patch(
            "genai_proxy.services.models.requests.get",
            return_value=FakeResponse({"success": True, "code": 200, "result": None}),
        ):
            with self.assertRaises(ProxyError) as raised:
                manager.list_genai_models(force_refresh=True)

        self.assertEqual(raised.exception.error_type, "upstream_error")

    def test_refresh_after_auth_failure_deletes_cache_and_forces_refresh(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            keystore_path = os.path.join(temp_dir, "docker-deploy.keystore")
            cache_path = f"{keystore_path}.token.json"
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                json.dump({"token": "stale-token", "exp": int(time.time()) + 3600}, cache_file)

            manager = TokenManager(
                self.logger,
                token=make_jwt(),
                keystore_path=keystore_path,
                token_check_interval=0,
            )

            def fake_refresh(force=False, rejected_token=None):
                self.assertTrue(force)
                self.assertIsNone(rejected_token)
                manager._delete_cached_token()
                manager._token = make_jwt()
                manager._token_exp = int(time.time()) + 3600

            with patch.object(manager, "_refresh_token", side_effect=fake_refresh) as refresh:
                self.assertTrue(manager.refresh_after_auth_failure("unit test"))

            self.assertEqual(refresh.call_count, 1)
            self.assertFalse(os.path.exists(cache_path))

    def test_forced_refresh_deletes_cache_before_login_attempt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            keystore_path = os.path.join(temp_dir, "missing.keystore")
            cache_path = f"{keystore_path}.token.json"
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                json.dump({"token": "rejected-token", "exp": int(time.time()) + 3600}, cache_file)

            manager = TokenManager(
                self.logger,
                token=make_jwt(),
                keystore_path=keystore_path,
                token_check_interval=0,
            )

            with (
                patch.object(manager._logger, "exception"),
                self.assertRaises(Exception),
            ):
                manager._refresh_token(force=True)

            self.assertFalse(os.path.exists(cache_path))

    def test_refresh_after_auth_failure_reuses_newer_cached_token_for_rejected_token(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            keystore_path = os.path.join(temp_dir, "docker-deploy.keystore")
            cache_path = f"{keystore_path}.token.json"
            rejected_token = make_jwt(exp=int(time.time()) + 3600)
            cached_token = make_jwt(exp=int(time.time()) + 7200)
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                json.dump({"token": cached_token, "exp": int(time.time()) + 7200}, cache_file)

            manager = TokenManager(
                self.logger,
                token=rejected_token,
                keystore_path=keystore_path,
                token_check_interval=0,
            )

            self.assertTrue(
                manager.refresh_after_auth_failure(
                    "unit test",
                    rejected_token=rejected_token,
                )
            )

            self.assertEqual(manager.token, cached_token)
            self.assertTrue(os.path.exists(cache_path))

    def test_token_access_keeps_existing_token_when_proactive_refresh_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            keystore_path = os.path.join(temp_dir, "docker-deploy.keystore")
            current_token = make_jwt(exp=int(time.time()) + 60)
            manager = TokenManager(
                self.logger,
                token=current_token,
                keystore_path=keystore_path,
                token_check_interval=0,
            )

            with patch.object(manager, "_refresh_token", side_effect=RuntimeError("ids failed")) as refresh:
                self.assertEqual(manager.token, current_token)
                self.assertEqual(manager.token, current_token)

            self.assertEqual(refresh.call_count, 1)
            self.assertGreater(manager._last_refresh_failure_at, 0)

    def test_rejected_token_refresh_failure_keeps_existing_token(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            keystore_path = os.path.join(temp_dir, "docker-deploy.keystore")
            current_token = make_jwt()
            manager = TokenManager(
                self.logger,
                token=current_token,
                keystore_path=keystore_path,
                token_check_interval=0,
            )

            with patch.object(manager, "_refresh_token", side_effect=RuntimeError("ids failed")):
                self.assertFalse(
                    manager.refresh_after_auth_failure(
                        "unit test",
                        rejected_token=current_token,
                    )
                )

            self.assertEqual(manager.token, current_token)
            self.assertGreater(manager._last_refresh_failure_at, 0)

    def test_background_confirmation_refreshes_before_expiry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            keystore_path = os.path.join(temp_dir, "docker-deploy.keystore")
            manager = TokenManager(
                self.logger,
                token=make_jwt(exp=int(time.time()) + 60),
                keystore_path=keystore_path,
                token_check_interval=0,
            )

            def fake_refresh(force=False):
                self.assertFalse(force)
                manager._token = make_jwt()
                manager._token_exp = int(time.time()) + 3600

            with patch.object(manager, "_refresh_token", side_effect=fake_refresh) as refresh:
                manager._confirm_token_for_background()

            self.assertEqual(refresh.call_count, 1)

    def test_background_confirmation_loads_newer_cached_token(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            keystore_path = os.path.join(temp_dir, "docker-deploy.keystore")
            cache_path = f"{keystore_path}.token.json"
            old_token = make_jwt(exp=int(time.time()) + 3600)
            cached_token = make_jwt(exp=int(time.time()) + 7200)
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                json.dump({"token": cached_token, "exp": int(time.time()) + 7200}, cache_file)

            manager = TokenManager(
                self.logger,
                token=old_token,
                keystore_path=keystore_path,
                token_check_interval=0,
            )

            with patch.object(manager, "_refresh_token") as refresh:
                manager._confirm_token_for_background()

            self.assertEqual(refresh.call_count, 0)
            self.assertEqual(manager.token, cached_token)

    def test_background_confirmation_loads_cached_billing_user_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            keystore_path = os.path.join(temp_dir, "docker-deploy.keystore")
            cache_path = f"{keystore_path}.token.json"
            cached_token = make_jwt(exp=int(time.time()) + 7200)
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                json.dump(
                    {
                        "token": cached_token,
                        "exp": int(time.time()) + 7200,
                        "user_id": "42",
                    },
                    cache_file,
                )

            manager = TokenManager(
                self.logger,
                token=cached_token,
                keystore_path=keystore_path,
                token_check_interval=0,
            )

            self.assertEqual(manager.billing_user_id, "42")

    def test_background_confirmation_does_not_refresh_during_shutdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            keystore_path = os.path.join(temp_dir, "docker-deploy.keystore")
            manager = TokenManager(
                self.logger,
                token=make_jwt(),
                keystore_path=keystore_path,
                token_check_interval=0,
            )
            manager._shutdown_done = True

            with patch.object(manager, "_refresh_token") as refresh:
                manager._confirm_token_for_background()

            self.assertEqual(refresh.call_count, 0)

    def test_background_confirmation_thread_starts_and_shutdown_stops_it(self):
        called = threading.Event()

        def confirm(manager):
            called.set()

        with tempfile.TemporaryDirectory() as temp_dir:
            keystore_path = os.path.join(temp_dir, "docker-deploy.keystore")
            with patch.object(TokenManager, "_confirm_token_for_background", confirm):
                manager = TokenManager(
                    self.logger,
                    token=make_jwt(),
                    keystore_path=keystore_path,
                    token_check_interval=30,
                )
                self.assertTrue(called.wait(1))
                self.assertTrue(manager._token_check_thread.is_alive())
                manager.shutdown()
                self.assertFalse(manager._token_check_thread.is_alive())

    def test_genai_login_flow_prefers_current_oauth_entry_and_falls_back_to_legacy_cas(self):
        manager = TokenManager(
            self.logger,
            token=make_jwt(),
            keystore_path="unused.keystore",
            token_check_interval=0,
        )
        current_response = FakeResponse(url="https://genai.shanghaitech.edu.cn/?token=current")
        client = type("FakeClient", (), {"session": FakeSession([current_response])})()

        self.assertIs(manager._get_genai_login_response(client), current_response)
        self.assertEqual(client.session.urls, [GENAI_LOGIN_URL])

        fallback_response = FakeResponse(url="https://genai.shanghaitech.edu.cn/?token=legacy")
        client = type(
            "FakeClient",
            (),
            {
                "session": FakeSession(
                    [
                        FakeResponse(url="https://genai.shanghaitech.edu.cn/no-token"),
                        fallback_response,
                    ]
                )
            },
        )()

        self.assertIs(manager._get_genai_login_response(client), fallback_response)
        self.assertEqual(client.session.urls, [GENAI_LOGIN_URL, GENAI_LEGACY_CAS_SERVICE_URL])


if __name__ == "__main__":
    unittest.main()
