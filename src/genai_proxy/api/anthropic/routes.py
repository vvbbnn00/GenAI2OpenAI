"""Anthropic-compatible Flask routes."""

import time
import uuid

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    request,
    stream_with_context,
)

from genai_proxy.api.anthropic.compat import (
    claude_error,
    convert_claude_to_openai,
    convert_openai_to_claude_response,
    stream_openai_to_claude,
)
from genai_proxy.api.common import prime_stream
from genai_proxy.errors import ProxyError

bp = Blueprint("claude", __name__)


def map_claude_model_alias(model: str | None, config) -> str | None:
    if model is not None and not isinstance(model, str):
        raise ProxyError("'model' must be a string")
    if not model:
        return model

    lowered = model.lower()
    if "haiku" in lowered:
        return config.claude_haiku_model
    if "sonnet" in lowered:
        return config.claude_sonnet_model
    if "opus" in lowered:
        return config.claude_opus_model
    return model


def _stream_with_completion_log(gen, logger, request_id: str, start_time: float):
    try:
        yield from gen
    finally:
        elapsed = time.monotonic() - start_time
        logger.info("[%s] completed in %.2fs", request_id, elapsed)


@bp.route("/v1/messages", methods=["POST"])
def create_message():
    request_id = f"claude_{uuid.uuid4().hex[:16]}"
    start_time = time.monotonic()
    service = current_app.extensions["genai_service"]
    model_manager = current_app.extensions["model_manager"]
    logger = current_app.extensions["logger"]
    config = current_app.extensions["config"]
    stream = False
    streaming_response_started = False

    try:
        original_req_data = request.get_json(silent=True)
        if not isinstance(original_req_data, dict):
            raise ProxyError("Request body must be a JSON object")
        req_data = original_req_data
        original_model = original_req_data.get("model")
        messages = original_req_data.get("messages")
        message_count = len(messages) if isinstance(messages, list) else 0
        mapped_model = map_claude_model_alias(original_model, config)
        original_req_with_estimator = {
            **original_req_data,
            "_estimator_model": mapped_model or original_model,
        }
        if mapped_model != original_model:
            req_data = {**req_data, "model": mapped_model}
        model_context = (
            service.resolve_model_context(mapped_model) if mapped_model else None
        )
        openai_request = convert_claude_to_openai(
            req_data,
            model_manager,
            resolved_model=(model_context.model if model_context else None),
        )

        logger.info(
            "[%s] claude-model=%s mapped-model=%s stream=%s messages=%d",
            request_id,
            original_model,
            openai_request.get("model"),
            openai_request.get("stream", False),
            message_count,
        )

        stream = bool(openai_request.get("stream"))
        if stream:
            original_req_with_estimator["_input_tokens"] = (
                service.count_openai_input_tokens(
                    openai_request,
                    model_context=model_context,
                )
            )
            gen = prime_stream(
                stream_openai_to_claude(
                    service.stream_openai_completion(
                        openai_request,
                        model_context=model_context,
                    ),
                    original_req_with_estimator,
                    logger,
                )
            )
            streaming_response_started = True
            return Response(
                stream_with_context(
                    _stream_with_completion_log(gen, logger, request_id, start_time)
                ),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                    "X-Accel-Buffering": "no",
                },
            )

        response = service.build_openai_completion(
            openai_request,
            model_context=model_context,
        )
        return jsonify(
            convert_openai_to_claude_response(response, original_req_with_estimator)
        )
    except ProxyError as exc:
        return claude_error(exc.message, exc.error_type, exc.status)
    except Exception as exc:
        logger.error(
            "[%s] Unhandled Claude error (%s)",
            request_id,
            type(exc).__name__,
        )
        return claude_error(str(exc), "api_error", 500)
    finally:
        if not streaming_response_started:
            elapsed = time.monotonic() - start_time
            logger.info("[%s] completed in %.2fs", request_id, elapsed)


@bp.route("/v1/messages/count_tokens", methods=["POST"])
def count_tokens():
    try:
        req_data = request.get_json(silent=True)
        if not isinstance(req_data, dict):
            raise ProxyError("Request body must be a JSON object")
        config = current_app.extensions["config"]
        model_manager = current_app.extensions["model_manager"]
        service = current_app.extensions["genai_service"]
        mapped_model = map_claude_model_alias(req_data.get("model"), config)
        converted_request = {
            **req_data,
            "model": mapped_model,
            # Anthropic's count endpoint does not require a generation limit,
            # while the shared message converter intentionally does.
            "max_tokens": req_data.get("max_tokens", 1),
            "stream": False,
        }
        model_context = (
            service.resolve_model_context(mapped_model) if mapped_model else None
        )
        openai_request = convert_claude_to_openai(
            converted_request,
            model_manager,
            resolved_model=(model_context.model if model_context else None),
        )
        return jsonify(
            {
                "input_tokens": service.count_openai_input_tokens(
                    openai_request,
                    model_context=model_context,
                )
            }
        )
    except ProxyError as exc:
        return claude_error(exc.message, exc.error_type, exc.status)
    except Exception as exc:
        return claude_error(str(exc), "api_error", 500)
