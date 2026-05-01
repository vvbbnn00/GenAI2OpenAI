import base64
import json
import logging
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import requests

from genai_proxy.errors import ProxyError
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


class InvalidJsonResponse(FakeResponse):
    def json(self):
        raise ValueError("invalid json")


class FakeTokenManager:
    def __init__(self):
        self._token = "stale-token"
        self.refresh_count = 0
        self.rejected_token = None

    @property
    def token(self):
        return self._token

    def refresh_after_auth_failure(self, reason, rejected_token=None):
        self.refresh_count += 1
        self.rejected_token = rejected_token
        self._token = "fresh-token"
        return True


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

    def test_model_list_result_null_raises_proxy_error_instead_of_attribute_error(self):
        token_manager = FakeTokenManager()
        manager = ModelManager(self.logger, token_manager)

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

            with (
                patch.object(manager, "_refresh_token", side_effect=fake_refresh) as refresh,
                patch.object(manager, "_token_was_rejected_by_upstream", return_value=False) as confirm,
            ):
                manager._confirm_token_for_background()

            self.assertEqual(refresh.call_count, 1)
            self.assertEqual(confirm.call_count, 1)

    def test_background_confirmation_refreshes_when_upstream_rejects_token(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            keystore_path = os.path.join(temp_dir, "docker-deploy.keystore")
            manager = TokenManager(
                self.logger,
                token=make_jwt(),
                keystore_path=keystore_path,
                token_check_interval=0,
            )

            old_token = manager._token

            def fake_refresh(force=False, rejected_token=None):
                self.assertTrue(force)
                self.assertEqual(rejected_token, old_token)
                manager._token = make_jwt()
                manager._token_exp = int(time.time()) + 3600

            with (
                patch.object(manager, "_token_was_rejected_by_upstream", return_value=True) as confirm,
                patch.object(manager, "_refresh_token", side_effect=fake_refresh) as refresh,
            ):
                manager._confirm_token_for_background()

            self.assertEqual(confirm.call_count, 1)
            self.assertEqual(refresh.call_count, 1)

    def test_background_confirmation_does_not_refresh_during_shutdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            keystore_path = os.path.join(temp_dir, "docker-deploy.keystore")
            manager = TokenManager(
                self.logger,
                token=make_jwt(),
                keystore_path=keystore_path,
                token_check_interval=0,
            )

            def reject_and_stop(token):
                manager._stop_event.set()
                return True

            with (
                patch.object(manager, "_token_was_rejected_by_upstream", side_effect=reject_and_stop) as confirm,
                patch.object(manager, "_refresh_token") as refresh,
            ):
                manager._confirm_token_for_background()

            self.assertEqual(confirm.call_count, 1)
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

    def test_periodic_confirmation_classifies_auth_failures_only(self):
        manager = TokenManager(self.logger, token=make_jwt(), token_check_interval=0)

        with patch(
            "genai_proxy.services.token_manager.requests.get",
            return_value=FakeResponse(status_code=401),
        ):
            self.assertTrue(manager._token_was_rejected_by_upstream("token"))

        with patch(
            "genai_proxy.services.token_manager.requests.get",
            return_value=FakeResponse({"success": False, "message": "Token失效，请重新登录", "code": 500}),
        ):
            self.assertTrue(manager._token_was_rejected_by_upstream("token"))

        with patch(
            "genai_proxy.services.token_manager.requests.get",
            return_value=FakeResponse({"success": True, "code": 200}),
        ):
            self.assertFalse(manager._token_was_rejected_by_upstream("token"))

        with patch(
            "genai_proxy.services.token_manager.requests.get",
            return_value=FakeResponse(status_code=500),
        ):
            self.assertFalse(manager._token_was_rejected_by_upstream("token"))

        with patch(
            "genai_proxy.services.token_manager.requests.get",
            return_value=InvalidJsonResponse(status_code=200),
        ):
            self.assertFalse(manager._token_was_rejected_by_upstream("token"))

        with patch(
            "genai_proxy.services.token_manager.requests.get",
            side_effect=requests.Timeout(),
        ):
            self.assertFalse(manager._token_was_rejected_by_upstream("token"))

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
