import json
import logging
import os
import unittest
from unittest.mock import patch

import requests

from genai_proxy.config import parse_args
from genai_proxy.errors import ProxyError
from genai_proxy.services.genai import GenAIService


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
    def __init__(self, lines=(), status_code=200, text=""):
        self._lines = lines
        self.status_code = status_code
        self.text = text
        self.closed = False

    def iter_lines(self):
        yield from self._lines

    def close(self):
        self.closed = True


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
    def test_retry_count_defaults_to_five(self):
        with patch.dict(os.environ, {"GENAI_MAX_RETRIES": ""}):
            config = parse_args(["--token", "token"])

        self.assertEqual(config.genai_max_retries, 5)

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
        unavailable = FakeResponse(status_code=503, text="unavailable")
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

    def test_retry_settings_are_available_as_cli_options(self):
        config = parse_args(
            [
                "--token",
                "token",
                "--genai-max-retries",
                "5",
                "--genai-retry-backoff",
                "0.25",
            ]
        )

        self.assertEqual(config.genai_max_retries, 5)
        self.assertEqual(config.genai_retry_backoff, 0.25)
