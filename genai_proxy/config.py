import argparse
import os
from dataclasses import dataclass


@dataclass(slots=True)
class AppConfig:
    token: str | None
    keystore: str | None
    port: int
    debug: bool
    api_key: str | None
    token_check_interval: int
    claude_haiku_model: str
    claude_sonnet_model: str
    claude_opus_model: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GenAI Flask API Server")
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="GenAI API Access Token (JWT)",
    )
    parser.add_argument(
        "--keystore",
        type=str,
        default=None,
        help="Path to shanghaitech-ids-passkey keystore file for auto-login/refresh",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Flask server port (default: 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for client authentication (or set API_KEY env var)",
    )
    parser.add_argument(
        "--token-check-interval",
        type=int,
        default=os.environ.get("TOKEN_CHECK_INTERVAL") or "60",
        help="Seconds between background GenAI token expiry/cache checks; set 0 to disable",
    )
    parser.add_argument(
        "--claude-haiku-model",
        type=str,
        default=os.environ.get("CLAUDE_HAIKU_MODEL", "deepseek-chat"),
        help="Mapped GenAI model for Claude model names containing 'haiku'",
    )
    parser.add_argument(
        "--claude-sonnet-model",
        type=str,
        default=os.environ.get("CLAUDE_SONNET_MODEL", "chatglm"),
        help="Mapped GenAI model for Claude model names containing 'sonnet'",
    )
    parser.add_argument(
        "--claude-opus-model",
        type=str,
        default=os.environ.get("CLAUDE_OPUS_MODEL", "chatglm"),
        help="Mapped GenAI model for Claude model names containing 'opus'",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> AppConfig:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.token and not args.keystore:
        parser.error("At least one of --token or --keystore must be provided")
    if args.token_check_interval < 0:
        parser.error("--token-check-interval must be 0 or greater")

    return AppConfig(
        token=args.token,
        keystore=args.keystore,
        port=args.port,
        debug=args.debug,
        api_key=args.api_key or os.environ.get("API_KEY"),
        token_check_interval=args.token_check_interval,
        claude_haiku_model=args.claude_haiku_model,
        claude_sonnet_model=args.claude_sonnet_model,
        claude_opus_model=args.claude_opus_model,
    )
