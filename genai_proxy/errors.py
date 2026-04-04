from dataclasses import dataclass


@dataclass
class ProxyError(Exception):
    message: str
    error_type: str = "invalid_request_error"
    code: str | None = None
    status: int = 400

    def __str__(self) -> str:
        return self.message

