"""Shared process bootstrap for CLI and WSGI entrypoints."""

import os

from genai_proxy.app import create_app
from genai_proxy.config import DEFAULT_MODEL_CACHE_PATH, AppConfig
from genai_proxy.logging_utils import setup_logging
from genai_proxy.retry import DEFAULT_MAX_RETRIES, DEFAULT_RETRY_BACKOFF
from genai_proxy.version import format_program_version, get_program_version


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return float(value)


def config_from_env() -> AppConfig:
    return AppConfig(
        token=os.environ.get("GENAI_TOKEN") or None,
        keystore=os.environ.get("KEYSTORE_PATH") or None,
        port=int(os.environ.get("APP_PORT", "5000")),
        debug=os.environ.get("APP_DEBUG", "0") == "1",
        api_key=os.environ.get("API_KEY") or None,
        token_check_interval=max(0, _int_env("TOKEN_CHECK_INTERVAL", 60)),
        claude_haiku_model=os.environ.get("CLAUDE_HAIKU_MODEL", "deepseek-chat"),
        claude_sonnet_model=os.environ.get("CLAUDE_SONNET_MODEL", "chatglm"),
        claude_opus_model=os.environ.get("CLAUDE_OPUS_MODEL", "chatglm"),
        genai_max_retries=max(0, _int_env("GENAI_MAX_RETRIES", DEFAULT_MAX_RETRIES)),
        genai_retry_backoff=max(
            0.0,
            _float_env("GENAI_RETRY_BACKOFF", DEFAULT_RETRY_BACKOFF),
        ),
        genai_model_cache=(
            os.environ.get("GENAI_MODEL_CACHE") or DEFAULT_MODEL_CACHE_PATH
        ),
    )


def log_startup(config: AppConfig, logger) -> None:
    logger.info("Program version: %s", format_program_version(get_program_version()))
    if config.api_key:
        logger.info("API key authentication enabled")
    else:
        logger.info("No API key set — running in open mode (no auth)")

    logger.info("Starting GenAI2OpenAI proxy on port %d", config.port)
    logger.info(
        "Debug: %s, Auth: %s",
        config.debug,
        "enabled" if config.api_key else "disabled",
    )
    logger.info(
        "Token mode: %s",
        "passkey auto-refresh" if config.keystore else "static token (no auto-refresh)",
    )
    if config.keystore:
        logger.info("Keystore: %s", config.keystore)
        if config.token_check_interval:
            logger.info(
                "Token maintenance interval: %d seconds",
                config.token_check_interval,
            )
        else:
            logger.info("Token maintenance interval: disabled")
    logger.info(
        "Claude alias mapping: haiku=%s sonnet=%s opus=%s",
        config.claude_haiku_model,
        config.claude_sonnet_model,
        config.claude_opus_model,
    )
    logger.info(
        "GenAI upstream retries: %d, initial backoff: %.2f seconds",
        config.genai_max_retries,
        config.genai_retry_backoff,
    )
    logger.info(
        "GenAI model cache: %s",
        config.genai_model_cache or "memory only",
    )


def create_app_from_env():
    config = config_from_env()
    logger = setup_logging(config.debug)
    log_startup(config, logger)
    return create_app(config, logger)


__all__ = ["config_from_env", "create_app_from_env", "log_startup"]
