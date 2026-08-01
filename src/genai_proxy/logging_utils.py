import logging


def safe_log_code(value: object) -> str:
    """Return a bounded, single-line representation for an upstream error code."""
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return str(value) if 100 <= value <= 599 else "int"
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        numeric = int(value)
        return value if len(value) == 3 and 100 <= numeric <= 599 else "str"
    return type(value).__name__


def setup_logging(debug: bool) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger("genai_proxy")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger
