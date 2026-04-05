import os

from genai_proxy.app import create_app
from genai_proxy.config import AppConfig, parse_args
from genai_proxy.logging_utils import setup_logging


def _config_from_env() -> AppConfig:
    return AppConfig(
        token=os.environ.get("GENAI_TOKEN") or None,
        keystore=os.environ.get("KEYSTORE_PATH") or None,
        port=int(os.environ.get("APP_PORT", 5000)),
        debug=os.environ.get("APP_DEBUG", "0") == "1",
        api_key=os.environ.get("API_KEY") or None,
        claude_haiku_model=os.environ.get("CLAUDE_HAIKU_MODEL", "qwen-instruct"),
        claude_sonnet_model=os.environ.get("CLAUDE_SONNET_MODEL", "qwen-instruct"),
        claude_opus_model=os.environ.get("CLAUDE_OPUS_MODEL", "deepseek-v3:671b"),
    )


def _log_startup(config: AppConfig, logger) -> None:
    if config.api_key:
        logger.info("API key authentication enabled")
    else:
        logger.info("No API key set — running in open mode (no auth)")

    logger.info("Starting GenAI2OpenAI proxy on port %d", config.port)
    logger.info("Debug: %s, Auth: %s", config.debug, "enabled" if config.api_key else "disabled")
    logger.info(
        "Token mode: %s",
        "passkey auto-refresh" if config.keystore else "static token (no auto-refresh)",
    )
    if config.keystore:
        logger.info("Keystore: %s", config.keystore)
    logger.info(
        "Claude alias mapping: haiku=%s sonnet=%s opus=%s",
        config.claude_haiku_model,
        config.claude_sonnet_model,
        config.claude_opus_model,
    )


def create_app_from_env():
    config = _config_from_env()
    logger = setup_logging(config.debug)
    _log_startup(config, logger)
    return create_app(config, logger)


def main():
    config = parse_args()
    logger = setup_logging(config.debug)
    _log_startup(config, logger)

    app = create_app(config, logger)
    try:
        app.run(host="0.0.0.0", port=config.port, debug=False, threaded=True)
    finally:
        app.extensions["token_manager"].shutdown()


if __name__ == "__main__":
    main()
else:
    app = create_app_from_env()
