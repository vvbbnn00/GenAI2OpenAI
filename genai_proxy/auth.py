from flask import request

from genai_proxy.compat.claude import claude_error
from genai_proxy.compat.openai import openai_error


def register_auth(app, config, logger):
    @app.before_request
    def check_api_key():
        if not config.api_key:
            return

        if request.path == "/health":
            return

        if not request.path.startswith("/v1/"):
            return

        client_key = None
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            client_key = authorization[7:]
        elif request.path.startswith("/v1/messages"):
            client_key = request.headers.get("x-api-key")

        if not client_key:
            logger.warning("Missing API key for path %s", request.path)
            if request.path.startswith("/v1/messages"):
                return claude_error("Missing API key", "authentication_error", 401)
            return openai_error(
                "Missing Authorization header with Bearer token",
                error_type="invalid_request_error",
                code="invalid_api_key",
                status=401,
            )

        if client_key != config.api_key:
            logger.warning("Invalid API key for path %s", request.path)
            if request.path.startswith("/v1/messages"):
                return claude_error("Invalid API key", "authentication_error", 401)
            return openai_error(
                "Incorrect API key provided",
                error_type="invalid_request_error",
                code="invalid_api_key",
                status=401,
            )

