import atexit
import base64
import json
import threading
import time
from datetime import datetime
from urllib.parse import parse_qs, quote, urlparse


GENAI_LOGIN_URL = "https://genai.shanghaitech.edu.cn/htk/user/login"
GENAI_CAS_SERVICE_URL = (
    "https://ids.shanghaitech.edu.cn/authserver/login"
    f"?service={quote(GENAI_LOGIN_URL, safe='')}"
)
GENAI_GET_TOKEN_URL = (
    "https://genai.shanghaitech.edu.cn/htk/user/info/{token}?_t={timestamp}"
)


def parse_jwt_payload(token: str) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT token format")

    payload_b64 = parts[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    payload_bytes = base64.urlsafe_b64decode(payload_b64)
    return json.loads(payload_bytes)


class TokenManager:
    REFRESH_MARGIN = 300

    def __init__(self, logger, token: str | None = None, keystore_path: str | None = None):
        self._logger = logger
        self._token = token
        self._keystore_path = keystore_path
        self._token_exp = None
        self._lock = threading.Lock()
        self._ids_client = None
        self._keystore = None
        self._used_ids = False
        self._shutdown_done = False

        if token:
            self._update_expiry()

        if not token and keystore_path:
            self._logger.info("No initial token provided, logging in via passkey...")
            self._refresh_token()

        atexit.register(self.shutdown)

    def _update_expiry(self) -> None:
        if not self._token:
            self._token_exp = None
            return

        try:
            payload = parse_jwt_payload(self._token)
            self._token_exp = payload.get("exp")
            if self._token_exp:
                exp_dt = datetime.fromtimestamp(self._token_exp)
                remaining = self._token_exp - time.time()
                self._logger.info(
                    "Token expires at %s (%.0f minutes remaining)",
                    exp_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    remaining / 60,
                )
            username = payload.get("username")
            if username:
                self._logger.info("Token username: %s", username)
        except Exception as exc:
            self._logger.warning("Failed to parse JWT token: %s", exc)
            self._token_exp = None

    def _needs_refresh(self) -> bool:
        if not self._keystore_path:
            return False
        if not self._token:
            return True
        if not self._token_exp:
            return False
        return time.time() >= (self._token_exp - self.REFRESH_MARGIN)

    def _refresh_token(self) -> None:
        if not self._keystore_path:
            self._logger.warning(
                "Token expired or missing, but no keystore configured for refresh"
            )
            return

        try:
            from shanghaitech_ids_passkey import IDSClient, PasskeyKeystore

            self._logger.info("Refreshing GenAI token via passkey login...")
            if self._keystore is None:
                self._keystore = PasskeyKeystore.load(self._keystore_path)
            if self._ids_client is None:
                self._ids_client = IDSClient(self._keystore)

            client = self._ids_client
            keystore = self._keystore
            client.login()
            self._used_ids = True
            self._logger.info("IDS passkey login successful for user: %s", keystore.username)

            response = client.session.get(
                GENAI_CAS_SERVICE_URL,
                allow_redirects=True,
                timeout=30,
            )

            final_url = response.url
            self._logger.debug("Final redirect URL: %s", final_url)

            parsed = urlparse(final_url)
            params = parse_qs(parsed.query)

            if "token" not in params:
                raise RuntimeError(
                    f"Could not extract GenAI token from login flow. Final URL: {final_url}"
                )

            real_token = client.session.get(
                GENAI_GET_TOKEN_URL.format(
                    token=params["token"][0],
                    timestamp=int(time.time() * 1000),
                ),
                timeout=30,
            ).json().get("result", {}).get("token")

            if not real_token:
                raise RuntimeError(
                    "Failed to retrieve real token from GenAI after CAS login"
                )

            self._token = real_token
            self._update_expiry()
            keystore.dump(self._keystore_path)
            self._logger.info("GenAI token refreshed successfully")
        except ImportError:
            self._logger.error(
                "shanghaitech-ids-passkey not installed. Install with: "
                "pip install shanghaitech-ids-passkey"
            )
            raise
        except Exception:
            self._logger.exception("Failed to refresh token via passkey")
            raise

    @property
    def token(self) -> str | None:
        with self._lock:
            if self._needs_refresh():
                self._refresh_token()
            elif self._token and self._token_exp and not self._keystore_path:
                remaining = self._token_exp - time.time()
                if remaining < self.REFRESH_MARGIN:
                    self._logger.warning(
                        "Token expires in %.0f seconds but no keystore for auto-refresh!",
                        remaining,
                    )
            return self._token

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True

            if not self._ids_client:
                return

            try:
                if self._used_ids:
                    self._logger.info("Logging out from IDS before shutdown...")
                    self._ids_client.logout()
                    self._logger.info("IDS logout successful")
            except Exception:
                self._logger.exception("Failed to logout from IDS during shutdown")
            finally:
                self._ids_client.session.close()

