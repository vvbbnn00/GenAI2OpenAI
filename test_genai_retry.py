import json
import logging
import os
import unittest
from unittest.mock import patch

import requests

from genai_proxy.config import parse_args
from genai_proxy.errors import ProxyError
from genai_proxy.retry import (
    TRANSIENT_UPSTREAM_ERROR_CODE,
    is_retryable_business_error,
    retry_delay,
)
from genai_proxy.services.genai import GenAIService
from genai_proxy.services.models import ModelManager


class FakeTokenManager:
    token = "token"
    billing_user_id = None

    def refresh_after_auth_failure(self, *_args, **_kwargs):
        return False


class FakeModelManager:
    def resolve_model(self, model):
        return model or "chatglm"

    def get_model_record(self, model):
        return {"aiType": model, "rootAiType": "xinference"}

    def root_ai_type_for(self, _model):
        return "xinference"


class FakeResponse:
    def __init__(self, lines=(), status_code=200, text="", payload=None):
        self._lines = lines
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.closed = False

    def iter_lines(self):
        yield from self._lines

    def close(self):
        self.closed = True

    def json(self):
        if self._payload is None:
            raise ValueError("missing JSON payload")
        return self._payload


def make_service(max_retries=3):
    return GenAIService(
        logging.getLogger("test_genai_retry"),
        FakeTokenManager(),
        FakeModelManager(),
        max_retries=max_retries,
        retry_backoff=0,
    )


def make_request():
    return {
        "model": "chatglm",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }


def completion_lines(content="ok"):
    return [
        ("data: " + json.dumps({"choices": [{"delta": {"content": content}}]})).encode(),
        ("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})).encode(),
    ]


class GenAIRetryTests(unittest.TestCase):
    def test_retry_count_defaults_to_ten(self):
        with patch.dict(os.environ, {"GENAI_MAX_RETRIES": ""}):
            config = parse_args(["--token", "token"])

        self.assertEqual(config.genai_max_retries, 10)
        self.assertEqual(retry_delay(0.5, 10), 5.0)

    def test_retryable_business_error_excludes_permanent_code_500_errors(self):
        self.assertTrue(is_retryable_business_error(500, "temporary upstream failure"))
        self.assertFalse(is_retryable_business_error(500, "模型不存在"))
        self.assertFalse(is_retryable_business_error(500, "Invalid request parameters"))
        self.assertTrue(is_retryable_business_error(502, "bad gateway"))

    def test_retries_connection_failure_before_stream_starts(self):
        service = make_service()
        success = FakeResponse(completion_lines())

        with patch(
            "genai_proxy.services.genai.requests.post",
            side_effect=[requests.ConnectionError("[Errno 101] Network is unreachable"), success],
        ) as post:
            chunks = list(service.stream_openai_completion(make_request()))

        self.assertEqual(post.call_count, 2)
        self.assertTrue(success.closed)
        self.assertTrue(any('"content": "ok"' in chunk for chunk in chunks))

    def test_retries_connect_timeout_before_stream_starts(self):
        service = make_service()
        success = FakeResponse(completion_lines())

        with patch(
            "genai_proxy.services.genai.requests.post",
            side_effect=[requests.ConnectTimeout("connect timed out"), success],
        ) as post:
            chunks = list(service.stream_openai_completion(make_request()))

        self.assertEqual(post.call_count, 2)
        self.assertTrue(success.closed)
        self.assertTrue(any('"content": "ok"' in chunk for chunk in chunks))

    def test_retries_retryable_http_status_before_stream_starts(self):
        service = make_service()
        unavailable = FakeResponse(status_code=502, text="bad gateway")
        success = FakeResponse(completion_lines())

        with patch(
            "genai_proxy.services.genai.requests.post",
            side_effect=[unavailable, success],
        ) as post:
            chunks = list(service.stream_openai_completion(make_request()))

        self.assertEqual(post.call_count, 2)
        self.assertTrue(unavailable.closed)
        self.assertTrue(success.closed)
        self.assertTrue(any('"content": "ok"' in chunk for chunk in chunks))

    def test_raises_proxy_error_after_retries_are_exhausted(self):
        service = make_service(max_retries=2)

        with patch(
            "genai_proxy.services.genai.requests.post",
            side_effect=requests.ConnectionError("connection refused"),
        ) as post:
            with self.assertRaises(ProxyError) as raised:
                list(service.stream_openai_completion(make_request()))

        self.assertEqual(raised.exception.status, 502)
        self.assertEqual(raised.exception.message, "Failed to connect to upstream GenAI")
        self.assertEqual(post.call_count, 3)

    def test_does_not_retry_after_stream_content_was_emitted(self):
        service = make_service()

        class InterruptedResponse(FakeResponse):
            def iter_lines(self):
                yield completion_lines("partial")[0]
                raise requests.ConnectionError("connection reset")

        response = InterruptedResponse()
        with patch("genai_proxy.services.genai.requests.post", return_value=response) as post:
            chunks = list(service.stream_openai_completion(make_request()))

        self.assertEqual(post.call_count, 1)
        self.assertTrue(response.closed)
        self.assertTrue(any('"content": "partial"' in chunk for chunk in chunks))
        self.assertTrue(any("Failed to connect to upstream GenAI" in chunk for chunk in chunks))

    def test_non_stream_retries_after_partial_upstream_disconnect(self):
        service = make_service()

        class InterruptedResponse(FakeResponse):
            def iter_lines(self):
                yield completion_lines("discarded partial")[0]
                raise requests.exceptions.ChunkedEncodingError("response ended prematurely")

        interrupted = InterruptedResponse()
        success = FakeResponse(completion_lines("complete"))
        request = {**make_request(), "stream": False}

        with patch(
            "genai_proxy.services.genai.requests.post",
            side_effect=[interrupted, success],
        ) as post:
            response = service.build_openai_completion(request)

        self.assertEqual(post.call_count, 2)
        self.assertTrue(interrupted.closed)
        self.assertEqual(response["choices"][0]["message"]["content"], "complete")

    def test_non_stream_retries_when_upstream_eof_has_no_finish_reason(self):
        service = make_service()
        incomplete = FakeResponse([completion_lines("discarded partial")[0]])
        success = FakeResponse(completion_lines("complete"))
        request = {**make_request(), "stream": False}

        with patch(
            "genai_proxy.services.genai.requests.post",
            side_effect=[incomplete, success],
        ) as post:
            response = service.build_openai_completion(request)

        self.assertEqual(post.call_count, 2)
        self.assertEqual(response["choices"][0]["message"]["content"], "complete")

    def test_retries_transient_business_error_from_stream(self):
        service = make_service()
        business_error = FakeResponse(
            [
                (
                    "data: "
                    + json.dumps(
                        {
                            "success": False,
                            "code": 500,
                            "message": "temporary upstream failure",
                        }
                    )
                ).encode()
            ]
        )
        success = FakeResponse(completion_lines())

        with patch(
            "genai_proxy.services.genai.requests.post",
            side_effect=[business_error, success],
        ) as post:
            chunks = list(service.stream_openai_completion(make_request()))

        self.assertEqual(post.call_count, 2)
        self.assertTrue(any('"content": "ok"' in chunk for chunk in chunks))

    def test_does_not_retry_permanent_business_error_from_stream(self):
        service = make_service(max_retries=3)
        business_error = FakeResponse(
            [
                (
                    "data: "
                    + json.dumps(
                        {
                            "success": False,
                            "code": 500,
                            "message": "模型不存在",
                        }
                    )
                ).encode()
            ]
        )

        with patch(
            "genai_proxy.services.genai.requests.post",
            return_value=business_error,
        ) as post:
            with self.assertRaises(ProxyError) as raised:
                list(service.stream_openai_completion(make_request()))

        self.assertEqual(post.call_count, 1)
        self.assertEqual(raised.exception.message, "Upstream error: 模型不存在")

    def test_tool_stream_retries_partial_upstream_disconnect_before_client_output(self):
        service = make_service()

        class InterruptedResponse(FakeResponse):
            def iter_lines(self):
                yield completion_lines("discarded partial")[0]
                raise requests.ConnectionError("stream closed")

        request = {
            **make_request(),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    },
                }
            ],
        }

        with patch(
            "genai_proxy.services.genai.requests.post",
            side_effect=[InterruptedResponse(), FakeResponse(completion_lines("complete"))],
        ) as post:
            chunks = list(service.stream_openai_completion(request))

        rendered = "".join(chunks)
        self.assertEqual(post.call_count, 2)
        self.assertIn("complete", rendered)
        self.assertNotIn("discarded partial", rendered)

    def test_tool_stream_does_not_retry_after_reasoning_reaches_client(self):
        service = make_service()

        class InterruptedResponse(FakeResponse):
            def iter_lines(self):
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "choices": [
                                {"delta": {"reasoning_content": "visible reasoning"}}
                            ]
                        }
                    )
                ).encode()
                raise requests.ConnectionError("stream closed")

        request = {
            **make_request(),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    },
                }
            ],
        }

        with patch(
            "genai_proxy.services.genai.requests.post",
            return_value=InterruptedResponse(),
        ) as post:
            chunks = list(service.stream_openai_completion(request))

        rendered = "".join(chunks)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(rendered.count("visible reasoning"), 1)
        self.assertIn("Failed to connect to upstream GenAI", rendered)

    def test_non_chat_upstream_operation_uses_same_retry_budget(self):
        service = make_service(max_retries=2)
        calls = []

        def fetch(_token):
            calls.append(True)
            if len(calls) < 3:
                raise ProxyError(
                    "temporary failure",
                    error_type="upstream_error",
                    code=TRANSIENT_UPSTREAM_ERROR_CODE,
                    status=502,
                )
            return "ok"

        self.assertEqual(service._with_token_auth_retry("billing request", fetch), "ok")
        self.assertEqual(len(calls), 3)

    def test_model_list_retries_connection_and_502_failures(self):
        manager = ModelManager(
            logging.getLogger("test_genai_retry.models"),
            FakeTokenManager(),
            max_retries=2,
            retry_backoff=0,
        )
        success = FakeResponse(
            payload={
                "success": True,
                "code": 200,
                "result": {"records": [{"aiType": "chatglm"}]},
            }
        )

        with patch(
            "genai_proxy.services.models.requests.get",
            side_effect=[
                requests.ConnectionError("connection reset"),
                FakeResponse(status_code=502, text="bad gateway"),
                success,
            ],
        ) as get:
            models = manager.list_genai_models()

        self.assertEqual(get.call_count, 3)
        self.assertEqual(models, [{"aiType": "chatglm"}])

    def test_model_list_uses_stale_cache_after_retries_are_exhausted(self):
        manager = ModelManager(
            logging.getLogger("test_genai_retry.models"),
            FakeTokenManager(),
            max_retries=1,
            retry_backoff=0,
        )
        manager._models_cache = [{"aiType": "cached-model"}]
        manager._models_cache_at = 0

        with patch(
            "genai_proxy.services.models.requests.get",
            side_effect=requests.ConnectionError("connection reset"),
        ) as get:
            models = manager.list_genai_models()

        self.assertEqual(get.call_count, 2)
        self.assertEqual(models, [{"aiType": "cached-model"}])

    def test_model_list_does_not_hide_auth_failure_with_stale_cache(self):
        manager = ModelManager(
            logging.getLogger("test_genai_retry.models"),
            FakeTokenManager(),
            max_retries=0,
            retry_backoff=0,
        )
        manager._models_cache = [{"aiType": "cached-model"}]
        manager._models_cache_at = 0

        with patch(
            "genai_proxy.services.models.requests.get",
            return_value=FakeResponse(status_code=401),
        ):
            with self.assertRaises(ProxyError) as raised:
                manager.list_genai_models()

        self.assertEqual(raised.exception.code, "upstream_auth_failed")

    def test_retry_settings_are_available_as_cli_options(self):
        config = parse_args(
            [
                "--token",
                "token",
                "--genai-max-retries",
                "10",
                "--genai-retry-backoff",
                "0.25",
            ]
        )

        self.assertEqual(config.genai_max_retries, 10)
        self.assertEqual(config.genai_retry_backoff, 0.25)
