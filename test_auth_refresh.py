import base64
import json
import logging
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from genai_proxy.errors import ProxyError
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


class FakeTokenManager:
    def __init__(self):
        self._token = "stale-token"
        self.refresh_count = 0

    @property
    def token(self):
        return self._token

    def refresh_after_auth_failure(self, reason):
        self.refresh_count += 1
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
        self.assertEqual(models[0]["aiType"], "deepseek-chat")
        self.assertEqual(mocked_get.call_args_list[0].kwargs["headers"]["X-Access-Token"], "stale-token")
        self.assertEqual(mocked_get.call_args_list[1].kwargs["headers"]["X-Access-Token"], "fresh-token")

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

            manager = TokenManager(self.logger, token=make_jwt(), keystore_path=keystore_path)

            def fake_refresh(force=False):
                self.assertTrue(force)
                manager._token = make_jwt()
                manager._token_exp = int(time.time()) + 3600

            with patch.object(manager, "_refresh_token", side_effect=fake_refresh) as refresh:
                self.assertTrue(manager.refresh_after_auth_failure("unit test"))

            self.assertEqual(refresh.call_count, 1)
            self.assertFalse(os.path.exists(cache_path))

    def test_genai_login_flow_prefers_current_oauth_entry_and_falls_back_to_legacy_cas(self):
        manager = TokenManager(self.logger, token=make_jwt(), keystore_path="unused.keystore")
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
