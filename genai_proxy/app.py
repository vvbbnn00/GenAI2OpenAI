from flask import Flask
from flask_cors import CORS

from genai_proxy.auth import register_auth
from genai_proxy.routes.claude import bp as claude_bp
from genai_proxy.routes.openai import bp as openai_bp
from genai_proxy.services.genai import GenAIService
from genai_proxy.services.models import ModelManager
from genai_proxy.services.token_manager import TokenManager


def create_app(config, logger):
    app = Flask(__name__)
    CORS(app)

    token_manager = TokenManager(logger, token=config.token, keystore_path=config.keystore)
    model_manager = ModelManager(logger, token_manager)
    genai_service = GenAIService(logger, token_manager, model_manager)

    app.extensions["logger"] = logger
    app.extensions["config"] = config
    app.extensions["model_manager"] = model_manager
    app.extensions["token_manager"] = token_manager
    app.extensions["genai_service"] = genai_service

    register_auth(app, config, logger)
    app.register_blueprint(openai_bp)
    app.register_blueprint(claude_bp)
    return app
