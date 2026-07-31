"""Production WSGI entrypoint."""

from genai_proxy.runtime import create_app_from_env

app = create_app_from_env()

__all__ = ["app"]
