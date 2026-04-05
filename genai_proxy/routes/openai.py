import time
import uuid

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from genai_proxy.compat.openai import openai_error
from genai_proxy.errors import ProxyError


bp = Blueprint("openai", __name__)


def _billing_error_response(exc: ProxyError, fallback_message: str, fallback_error_type: str):
    message = exc.message
    error_type = exc.error_type

    if exc.error_type == "upstream_error":
        message = fallback_message
        error_type = fallback_error_type

    return openai_error(
        message,
        error_type=error_type,
        code=exc.code,
        status=exc.status,
    )


def _stream_with_completion_log(gen, logger, request_id: str, start_time: float):
    try:
        yield from gen
    finally:
        elapsed = time.monotonic() - start_time
        logger.info("[%s] completed in %.2fs", request_id, elapsed)


@bp.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    start_time = time.monotonic()
    service = current_app.extensions["genai_service"]
    logger = current_app.extensions["logger"]
    stream = False

    try:
        req_data = request.get_json()
        stream = bool((req_data or {}).get("stream", False))
        logger.info(
            "[%s] model=%s stream=%s tools=%s messages=%d",
            request_id,
            (req_data or {}).get("model", "GPT-4.1"),
            stream,
            bool((req_data or {}).get("tools")),
            len((req_data or {}).get("messages", [])),
        )

        if stream:
            gen = service.stream_openai_completion(req_data)
            return Response(
                stream_with_context(_stream_with_completion_log(gen, logger, request_id, start_time)),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                },
            )

        return jsonify(service.build_openai_completion(req_data))
    except ProxyError as exc:
        return openai_error(
            exc.message,
            error_type=exc.error_type,
            code=exc.code,
            status=exc.status,
        )
    except Exception as exc:
        logger.exception("[%s] Unhandled error", request_id)
        return openai_error(
            str(exc),
            error_type="server_error",
            code="internal_error",
            status=500,
        )
    finally:
        if not stream:
            elapsed = time.monotonic() - start_time
            logger.info("[%s] completed in %.2fs", request_id, elapsed)


@bp.route("/v1/models", methods=["GET"])
def list_models():
    model_manager = current_app.extensions["model_manager"]
    return jsonify({"object": "list", "data": model_manager.list_openai_models()})


@bp.route("/v1/dashboard/billing/subscription", methods=["GET"])
def billing_subscription():
    service = current_app.extensions["genai_service"]

    try:
        return jsonify(service.fetch_openai_billing_subscription())
    except ProxyError as exc:
        return _billing_error_response(
            exc,
            fallback_message="Failed to fetch subscription quota",
            fallback_error_type="upstream_error",
        )


@bp.route("/v1/dashboard/billing/usage", methods=["GET"])
def billing_usage():
    service = current_app.extensions["genai_service"]

    try:
        return jsonify(service.fetch_openai_billing_usage())
    except ProxyError as exc:
        return _billing_error_response(
            exc,
            fallback_message="Failed to fetch usage",
            fallback_error_type="new_api_error",
        )


@bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200
