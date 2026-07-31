from flask import Flask
from flask_cors import CORS

from genai_proxy.auth import register_auth
from genai_proxy.retry import DEFAULT_MAX_RETRIES, DEFAULT_RETRY_BACKOFF
from genai_proxy.routes.claude import bp as claude_bp
from genai_proxy.routes.openai import bp as openai_bp
from genai_proxy.services.genai import GenAIService
from genai_proxy.services.models import ModelManager
from genai_proxy.services.token_manager import TokenManager


def create_app(config, logger):
    app = Flask(__name__)
    CORS(app)

    token_manager = TokenManager(
        logger,
        token=config.token,
        keystore_path=config.keystore,
        token_check_interval=config.token_check_interval,
    )
    max_retries = getattr(config, "genai_max_retries", DEFAULT_MAX_RETRIES)
    retry_backoff = getattr(config, "genai_retry_backoff", DEFAULT_RETRY_BACKOFF)
    model_manager = ModelManager(
        logger,
        token_manager,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        cache_path=getattr(config, "genai_model_cache", None),
        fallback_model_ids=(
            getattr(config, "claude_haiku_model", None),
            getattr(config, "claude_sonnet_model", None),
            getattr(config, "claude_opus_model", None),
        ),
    )
    if getattr(config, "genai_model_cache", None):
        model_manager.refresh_in_background()
    genai_service = GenAIService(
        logger,
        token_manager,
        model_manager,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        cleanup_kimi_history=True,
    )

    app.extensions["logger"] = logger
    app.extensions["config"] = config
    app.extensions["model_manager"] = model_manager
    app.extensions["token_manager"] = token_manager
    app.extensions["genai_service"] = genai_service

    register_auth(app, config, logger)
    app.register_blueprint(openai_bp)
    app.register_blueprint(claude_bp)
    return app
