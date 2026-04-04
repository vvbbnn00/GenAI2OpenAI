from genai_proxy.app import create_app
from genai_proxy.config import parse_args
from genai_proxy.logging_utils import setup_logging


def main():
    config = parse_args()
    logger = setup_logging(config.debug)

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

    app = create_app(config, logger)
    try:
        app.run(host="0.0.0.0", port=config.port, debug=False)
    finally:
        app.extensions["token_manager"].shutdown()


if __name__ == "__main__":
    main()
