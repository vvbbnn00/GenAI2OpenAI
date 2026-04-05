import time
import uuid

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from genai_proxy.compat.claude import (
    claude_error,
    convert_claude_to_openai,
    convert_openai_to_claude_response,
    estimate_claude_tokens,
    stream_openai_to_claude,
)
from genai_proxy.errors import ProxyError


bp = Blueprint("claude", __name__)


def map_claude_model_alias(model: str | None, config) -> str | None:
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

    try:
        original_req_data = request.get_json() or {}
        req_data = original_req_data
        original_model = original_req_data.get("model")
        message_count = len(original_req_data.get("messages", []))
        mapped_model = map_claude_model_alias(original_model, config)
        original_req_with_estimator = {
            **original_req_data,
            "_estimator_model": mapped_model or original_model,
        }
        if mapped_model != original_model:
            req_data = {**req_data, "model": mapped_model}
        openai_request = convert_claude_to_openai(req_data, model_manager)

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
            gen = stream_openai_to_claude(
                service.stream_openai_completion(openai_request),
                original_req_with_estimator,
                logger,
            )
            return Response(
                stream_with_context(_stream_with_completion_log(gen, logger, request_id, start_time)),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                },
            )

        response = service.build_openai_completion(openai_request)
        return jsonify(convert_openai_to_claude_response(response, original_req_with_estimator))
    except ProxyError as exc:
        return claude_error(exc.message, exc.error_type, exc.status)
    except Exception as exc:
        logger.exception("[%s] Unhandled Claude error", request_id)
        return claude_error(str(exc), "api_error", 500)
    finally:
        if not stream:
            elapsed = time.monotonic() - start_time
            logger.info("[%s] completed in %.2fs", request_id, elapsed)


@bp.route("/v1/messages/count_tokens", methods=["POST"])
def count_tokens():
    try:
        req_data = request.get_json() or {}
        config = current_app.extensions["config"]
        req_data = {
            **req_data,
            "_estimator_model": map_claude_model_alias(req_data.get("model"), config),
        }
        return jsonify(estimate_claude_tokens(req_data))
    except Exception as exc:
        return claude_error(str(exc), "api_error", 500)
