import json
import logging
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import requests
from flask import Flask

from genai_proxy.api.openai.routes import bp as openai_bp
from genai_proxy.api.openai.service import GenAIService
from genai_proxy.chat.streaming import (
    GENAI_DEEPSEEK_TIMEOUT_MAX_RETRIES,
    GENAI_TIMEOUT_MAX_RETRIES,
)
from genai_proxy.config import parse_args
from genai_proxy.errors import ProxyError
from genai_proxy.retry import (
    TRANSIENT_UPSTREAM_ERROR_CODE,
    is_retryable_business_error,
    retry_delay,
)
from genai_proxy.upstream.catalog import ModelManager
from genai_proxy.upstream.transport import (
    GENAI_DEEPSEEK_STREAM_TIMEOUT,
    GENAI_STREAM_TIMEOUT,
)


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

    def iter_lines(self, *args, **kwargs):
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


def make_request(model="chatglm"):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }


def completion_lines(content="ok"):
    return [
        (
            "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})
        ).encode(),
        (
            "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        ).encode(),
    ]


def completion_events(chunks):
    events = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            events.append(json.loads(line[6:]))
    return events


class GenAIRetryTests(unittest.TestCase):
    def test_retry_count_defaults_to_ten(self):
        with patch.dict(os.environ, {"GENAI_MAX_RETRIES": ""}):
            config = parse_args(["--token", "token"])

        self.assertEqual(config.genai_max_retries, 10)
        self.assertEqual(retry_delay(0.5, 10), 5.0)

    def test_retryable_business_error_excludes_permanent_code_500_errors(self):
        self.assertTrue(is_retryable_business_error(500, "temporary upstream failure"))
        self.assertFalse(is_retryable_business_error(500, "模型不存在"))
        self.assertFalse(
            is_retryable_business_error(
                500,
                "未找到对应节点信息，请重新设置",
            )
        )
        self.assertFalse(is_retryable_business_error(500, "Invalid request parameters"))
        self.assertTrue(is_retryable_business_error(502, "bad gateway"))

    def test_retries_connection_failure_before_stream_starts(self):
        service = make_service()
        success = FakeResponse(completion_lines())

        with patch(
            "genai_proxy.upstream.transport.requests.post",
            side_effect=[
                requests.ConnectionError("[Errno 101] Network is unreachable"),
                success,
            ],
        ) as post:
            chunks = list(service.stream_openai_completion(make_request()))

        self.assertEqual(post.call_count, 2)
        self.assertTrue(success.closed)
        self.assertTrue(any('"content": "ok"' in chunk for chunk in chunks))

    def test_retries_connect_timeout_before_stream_starts(self):
        service = make_service()
        success = FakeResponse(completion_lines())

        with patch(
            "genai_proxy.upstream.transport.requests.post",
            side_effect=[requests.ConnectTimeout("connect timed out"), success],
        ) as post:
            chunks = list(service.stream_openai_completion(make_request()))

        self.assertEqual(post.call_count, 2)
        self.assertTrue(success.closed)
        self.assertTrue(any('"content": "ok"' in chunk for chunk in chunks))

    def test_timeout_retries_are_bounded_independently_of_general_retries(self):
        service = make_service(max_retries=10)

        with patch(
            "genai_proxy.upstream.transport.requests.post",
            side_effect=requests.ReadTimeout("upstream stalled"),
        ) as post:
            with self.assertRaises(ProxyError) as raised:
                list(service.stream_openai_completion(make_request()))

        self.assertEqual(GENAI_STREAM_TIMEOUT, (10, 90))
        self.assertEqual(GENAI_TIMEOUT_MAX_RETRIES, 1)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(
            raised.exception.message,
            "Upstream stream timed out or stalled",
        )

    def test_deepseek_stall_retries_recover_on_the_third_attempt(self):
        for model in ("deepseek-chat", "deepseek-pro"):
            with self.subTest(model=model):
                service = make_service(max_retries=10)
                success = FakeResponse(completion_lines())

                with patch(
                    "genai_proxy.upstream.transport.requests.post",
                    side_effect=[
                        requests.ReadTimeout("first stall"),
                        requests.ReadTimeout("second stall"),
                        success,
                    ],
                ) as post:
                    chunks = list(
                        service.stream_openai_completion(make_request(model))
                    )

                self.assertEqual(GENAI_DEEPSEEK_STREAM_TIMEOUT, (10, 60))
                self.assertEqual(GENAI_DEEPSEEK_TIMEOUT_MAX_RETRIES, 2)
                self.assertEqual(post.call_count, 3)
                self.assertTrue(success.closed)
                self.assertTrue(any('"content": "ok"' in chunk for chunk in chunks))
                self.assertTrue(
                    all(
                        call.kwargs["timeout"] == GENAI_DEEPSEEK_STREAM_TIMEOUT
                        for call in post.call_args_list
                    )
                )
                self.assertTrue(
                    all(
                        "chatGroupId" not in call.kwargs["json"]
                        for call in post.call_args_list
                    )
                )

    def test_deepseek_stall_retry_budget_is_still_bounded(self):
        service = make_service(max_retries=10)

        with patch(
            "genai_proxy.upstream.transport.requests.post",
            side_effect=requests.ReadTimeout("upstream stalled"),
        ) as post:
            with self.assertRaises(ProxyError) as raised:
                list(
                    service.stream_openai_completion(
                        make_request("deepseek-chat")
                    )
                )

        self.assertEqual(post.call_count, 3)
        self.assertEqual(
            raised.exception.message,
            "Upstream stream timed out or stalled",
        )

    def test_deepseek_connect_timeout_keeps_the_default_retry_budget(self):
        service = make_service(max_retries=10)

        with patch(
            "genai_proxy.upstream.transport.requests.post",
            side_effect=requests.ConnectTimeout("connect timed out"),
        ) as post:
            with self.assertRaises(ProxyError):
                list(
                    service.stream_openai_completion(
                        make_request("deepseek-chat")
                    )
                )

        self.assertEqual(post.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["timeout"] == GENAI_DEEPSEEK_STREAM_TIMEOUT
                for call in post.call_args_list
            )
        )

    def test_kimi_keeps_the_default_stall_budget(self):
        service = make_service(max_retries=10)

        with patch(
            "genai_proxy.upstream.transport.requests.post",
            side_effect=requests.ReadTimeout("upstream stalled"),
        ) as post:
            with self.assertRaises(ProxyError):
                list(service.stream_openai_completion(make_request("kimi-k3")))

        self.assertEqual(post.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["timeout"] == GENAI_STREAM_TIMEOUT
                for call in post.call_args_list
            )
        )

    def test_deepseek_does_not_retry_a_timeout_after_client_output(self):
        service = make_service(max_retries=10)

        class InterruptedResponse(FakeResponse):
            def iter_lines(self, *args, **kwargs):
                yield completion_lines("partial")[0]
                raise requests.ReadTimeout("upstream stalled")

        response = InterruptedResponse()
        with patch(
            "genai_proxy.upstream.transport.requests.post",
            return_value=response,
        ) as post:
            chunks = list(
                service.stream_openai_completion(make_request("deepseek-pro"))
            )

        self.assertEqual(post.call_count, 1)
        self.assertEqual(
            post.call_args.kwargs["timeout"],
            GENAI_DEEPSEEK_STREAM_TIMEOUT,
        )
        self.assertTrue(response.closed)
        self.assertTrue(any('"content": "partial"' in chunk for chunk in chunks))
        self.assertTrue(
            any("Upstream stream timed out or stalled" in chunk for chunk in chunks)
        )

    def test_retries_retryable_http_status_before_stream_starts(self):
        service = make_service()
        unavailable = FakeResponse(status_code=502, text="bad gateway")
        success = FakeResponse(completion_lines())

        with patch(
            "genai_proxy.upstream.transport.requests.post",
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
            "genai_proxy.upstream.transport.requests.post",
            side_effect=requests.ConnectionError("connection refused"),
        ) as post:
            with self.assertRaises(ProxyError) as raised:
                list(service.stream_openai_completion(make_request()))

        self.assertEqual(raised.exception.status, 502)
        self.assertEqual(
            raised.exception.message, "Failed to connect to upstream GenAI"
        )
        self.assertEqual(post.call_count, 3)

    def test_does_not_retry_after_stream_content_was_emitted(self):
        service = make_service()

        class InterruptedResponse(FakeResponse):
            def iter_lines(self, *args, **kwargs):
                yield completion_lines("partial")[0]
                raise requests.ConnectionError("connection reset")

        response = InterruptedResponse()
        with patch(
            "genai_proxy.upstream.transport.requests.post", return_value=response
        ) as post:
            chunks = list(service.stream_openai_completion(make_request()))

        self.assertEqual(post.call_count, 1)
        self.assertTrue(response.closed)
        self.assertTrue(any('"content": "partial"' in chunk for chunk in chunks))
        self.assertTrue(
            any("Failed to connect to upstream GenAI" in chunk for chunk in chunks)
        )
        events = completion_events(chunks)
        self.assertEqual(len({event["id"] for event in events}), 1)
        self.assertEqual(len({event["created"] for event in events}), 1)

    def test_stream_content_finish_and_usage_share_completion_metadata(self):
        service = make_service()
        request = {
            **make_request(),
            "stream_options": {"include_usage": True},
        }

        with patch(
            "genai_proxy.upstream.transport.requests.post",
            return_value=FakeResponse(completion_lines()),
        ):
            chunks = list(service.stream_openai_completion(request))

        events = completion_events(chunks)
        self.assertGreaterEqual(len(events), 3)
        self.assertEqual(len({event["id"] for event in events}), 1)
        self.assertEqual(len({event["created"] for event in events}), 1)
        self.assertEqual(events[-1]["choices"], [])
        self.assertIn("usage", events[-1])

    def test_non_stream_retries_after_partial_upstream_disconnect(self):
        service = make_service()

        class InterruptedResponse(FakeResponse):
            def iter_lines(self, *args, **kwargs):
                yield completion_lines("discarded partial")[0]
                raise requests.exceptions.ChunkedEncodingError(
                    "response ended prematurely"
                )

        interrupted = InterruptedResponse()
        success = FakeResponse(completion_lines("complete"))
        request = {**make_request(), "stream": False}

        with patch(
            "genai_proxy.upstream.transport.requests.post",
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
            "genai_proxy.upstream.transport.requests.post",
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
            "genai_proxy.upstream.transport.requests.post",
            side_effect=[business_error, success],
        ) as post:
            chunks = list(service.stream_openai_completion(make_request()))

        self.assertEqual(post.call_count, 2)
        self.assertTrue(any('"content": "ok"' in chunk for chunk in chunks))

    def test_does_not_retry_permanent_business_error_from_stream(self):
        for message in ("模型不存在", "未找到对应节点信息，请重新设置"):
            with self.subTest(message=message):
                service = make_service(max_retries=3)
                business_error = FakeResponse(
                    [
                        (
                            "data: "
                            + json.dumps(
                                {
                                    "success": False,
                                    "code": 500,
                                    "message": message,
                                }
                            )
                        ).encode()
                    ]
                )

                with patch(
                    "genai_proxy.upstream.transport.requests.post",
                    return_value=business_error,
                ) as post:
                    with self.assertRaises(ProxyError) as raised:
                        list(service.stream_openai_completion(make_request()))

                self.assertEqual(post.call_count, 1)
                self.assertEqual(
                    raised.exception.message,
                    f"Upstream error: {message}",
                )

    def test_tool_stream_retries_partial_upstream_disconnect_before_client_output(self):
        service = make_service()

        class InterruptedResponse(FakeResponse):
            def iter_lines(self, *args, **kwargs):
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
            "genai_proxy.upstream.transport.requests.post",
            side_effect=[
                InterruptedResponse(),
                FakeResponse(completion_lines("complete")),
            ],
        ) as post:
            chunks = list(service.stream_openai_completion(request))

        rendered = "".join(chunks)
        self.assertEqual(post.call_count, 2)
        self.assertIn("complete", rendered)
        self.assertNotIn("discarded partial", rendered)

    def test_tool_stream_does_not_retry_after_reasoning_reaches_client(self):
        service = make_service()

        class InterruptedResponse(FakeResponse):
            def iter_lines(self, *args, **kwargs):
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
            "genai_proxy.upstream.transport.requests.post",
            return_value=InterruptedResponse(),
        ) as post:
            chunks = list(service.stream_openai_completion(request))

        rendered = "".join(chunks)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(rendered.count("visible reasoning"), 1)
        self.assertIn("Failed to connect to upstream GenAI", rendered)
        events = completion_events(chunks)
        self.assertEqual(len({event["id"] for event in events}), 1)
        self.assertEqual(len({event["created"] for event in events}), 1)

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
            "genai_proxy.upstream.catalog.requests.get",
            side_effect=[
                requests.ConnectionError("connection reset"),
                FakeResponse(status_code=502, text="bad gateway"),
                success,
            ],
        ) as get:
            models = manager.list_genai_models(force_refresh=True)

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
            "genai_proxy.upstream.catalog.requests.get",
            side_effect=requests.ConnectionError("connection reset"),
        ) as get:
            models = manager.list_genai_models(force_refresh=True)

        self.assertEqual(get.call_count, 2)
        self.assertEqual(models, [{"aiType": "cached-model"}])

    def test_model_list_uses_stale_cache_after_auth_refresh_fails(self):
        manager = ModelManager(
            logging.getLogger("test_genai_retry.models"),
            FakeTokenManager(),
            max_retries=0,
            retry_backoff=0,
        )
        manager._models_cache = [{"aiType": "cached-model"}]
        manager._models_cache_at = 0

        with patch(
            "genai_proxy.upstream.catalog.requests.get",
            return_value=FakeResponse(status_code=401),
        ):
            models = manager.list_genai_models(force_refresh=True)

        self.assertEqual(models, [{"aiType": "cached-model"}])

    def test_stale_model_cache_returns_before_background_refresh_finishes(self):
        manager = ModelManager(
            logging.getLogger("test_genai_retry.models"),
            FakeTokenManager(),
            max_retries=0,
            retry_backoff=0,
        )
        manager._models_cache = [{"aiType": "cached-model"}]
        manager._models_cache_at = 0
        refresh_started = threading.Event()
        allow_refresh = threading.Event()

        def fetch():
            refresh_started.set()
            allow_refresh.wait(timeout=2)
            return [{"aiType": "fresh-model"}]

        with patch.object(manager, "_fetch_models", side_effect=fetch) as mocked:
            models = manager.list_genai_models()
            self.assertEqual(models, [{"aiType": "cached-model"}])
            self.assertTrue(refresh_started.wait(timeout=1))

            for _ in range(10):
                self.assertEqual(
                    manager.list_genai_models(),
                    [{"aiType": "cached-model"}],
                )
            self.assertEqual(mocked.call_count, 1)

            allow_refresh.set()
            for _ in range(100):
                if not manager._models_refreshing:
                    break
                time.sleep(0.01)

        self.assertFalse(manager._models_refreshing)
        self.assertEqual(manager.list_genai_models(), [{"aiType": "fresh-model"}])

    def test_persistent_model_cache_survives_restart_and_upstream_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, "models.json")
            first = ModelManager(
                logging.getLogger("test_genai_retry.models"),
                FakeTokenManager(),
                max_retries=0,
                retry_backoff=0,
                cache_path=cache_path,
            )
            with patch.object(
                first,
                "_fetch_models",
                return_value=[{"aiType": "persisted-model"}],
            ):
                self.assertEqual(
                    first.list_genai_models(force_refresh=True),
                    [{"aiType": "persisted-model"}],
                )

            restarted = ModelManager(
                logging.getLogger("test_genai_retry.models"),
                FakeTokenManager(),
                max_retries=0,
                retry_backoff=0,
                cache_path=cache_path,
            )
            with patch.object(
                restarted,
                "_fetch_models",
                side_effect=ProxyError(
                    "offline",
                    error_type="upstream_error",
                    status=502,
                ),
            ) as fetch:
                self.assertEqual(
                    restarted.list_genai_models(force_refresh=True),
                    [{"aiType": "persisted-model"}],
                )
                self.assertEqual(fetch.call_count, 1)

    def test_corrupt_persistent_model_cache_falls_back_without_502(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, "models.json")
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                cache_file.write("{broken")

            manager = ModelManager(
                logging.getLogger("test_genai_retry.models"),
                FakeTokenManager(),
                max_retries=0,
                retry_backoff=0,
                cache_path=cache_path,
            )
            with patch.object(
                manager,
                "_fetch_models",
                side_effect=ProxyError(
                    "offline",
                    error_type="upstream_error",
                    status=502,
                ),
            ):
                models = manager.list_genai_models(force_refresh=True)

        self.assertIn("GPT-4.1", {model["aiType"] for model in models})

    def test_failed_background_refresh_obeys_cooldown(self):
        manager = ModelManager(
            logging.getLogger("test_genai_retry.models"),
            FakeTokenManager(),
            max_retries=0,
            retry_backoff=0,
        )
        with patch.object(
            manager,
            "_fetch_models",
            side_effect=ProxyError(
                "offline",
                error_type="upstream_error",
                status=502,
            ),
        ) as fetch:
            manager.list_genai_models()
            for _ in range(100):
                if not manager._models_refreshing:
                    break
                time.sleep(0.01)
            manager.list_genai_models()

        self.assertEqual(fetch.call_count, 1)
        self.assertGreater(manager._models_refresh_after, time.time())

    def test_models_endpoint_serves_fallback_with_http_cache_policy(self):
        manager = ModelManager(
            logging.getLogger("test_genai_retry.models"),
            FakeTokenManager(),
            max_retries=0,
            retry_backoff=0,
        )
        manager._models_cache_at = time.time()
        app = Flask(__name__)
        app.extensions["model_manager"] = manager
        app.register_blueprint(openai_bp)

        with patch.object(
            manager,
            "_fetch_models",
            side_effect=ProxyError(
                "offline",
                error_type="upstream_error",
                status=502,
            ),
        ):
            manager._models_cache_at = 0
            response = app.test_client().get("/v1/models")
            for _ in range(100):
                if not manager._models_refreshing:
                    break
                time.sleep(0.01)

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "GPT-4.1",
            {model["id"] for model in response.get_json()["data"]},
        )
        self.assertIn(
            "qwen-instruct",
            {model["id"] for model in response.get_json()["data"]},
        )
        self.assertNotIn(
            "MiniMax-M1",
            {model["id"] for model in response.get_json()["data"]},
        )
        self.assertEqual(
            response.headers["Cache-Control"],
            ("private, max-age=60, stale-while-revalidate=300, stale-if-error=86400"),
        )

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
        self.assertTrue(config.genai_model_cache.endswith("models.json"))
