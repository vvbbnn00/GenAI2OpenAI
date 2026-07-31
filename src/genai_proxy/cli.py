"""Development server command-line entrypoint."""

from genai_proxy.app import create_app
from genai_proxy.config import parse_args
from genai_proxy.logging_utils import setup_logging
from genai_proxy.runtime import log_startup


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    logger = setup_logging(config.debug)
    log_startup(config, logger)

    app = create_app(config, logger)
    try:
        app.run(host="0.0.0.0", port=config.port, debug=False, threaded=True)
    finally:
        app.extensions["token_manager"].shutdown()


__all__ = ["main"]
