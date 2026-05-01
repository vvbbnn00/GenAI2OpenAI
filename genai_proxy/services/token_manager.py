import atexit
import base64
import json
import os
import threading
import time
from datetime import datetime
from urllib.parse import parse_qs, quote, urlparse

import requests

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


GENAI_LOGIN_URL = "https://genai.shanghaitech.edu.cn/htk/user/login"
GENAI_LEGACY_CAS_SERVICE_URL = (
    "https://ids.shanghaitech.edu.cn/authserver/login"
    f"?service={quote(GENAI_LOGIN_URL, safe='')}"
)
GENAI_GET_TOKEN_URL = (
    "https://genai.shanghaitech.edu.cn/htk/user/info/{token}?_t={timestamp}"
)
GENAI_CONFIRM_TOKEN_URL = "https://genai.shanghaitech.edu.cn/htk/user/info/{token}"

GENAI_AUTH_FAILURE_MESSAGES = (
    "token失效",
    "鉴权失败",
    "重新登录",
    "invalid token",
    "token expired",
    "authentication failed",
)


def parse_jwt_payload(token: str) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT token format")

    payload_b64 = parts[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    payload_bytes = base64.urlsafe_b64decode(payload_b64)
    return json.loads(payload_bytes)


def is_genai_auth_failure(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False

    code = payload.get("code")
    message = str(payload.get("message") or "").lower()
    if code in (401, 403):
        return True
    if payload.get("success") is False:
        return any(marker in message for marker in GENAI_AUTH_FAILURE_MESSAGES)
    return False


class TokenManager:
    REFRESH_MARGIN = 300
    DEFAULT_TOKEN_CHECK_INTERVAL = 60
    REFRESH_FAILURE_RETRY_INTERVAL = 60
    TOKEN_CONFIRM_TIMEOUT = 10

    def __init__(
        self,
        logger,
        token: str | None = None,
        keystore_path: str | None = None,
        token_check_interval: int | None = None,
    ):
        self._logger = logger
        self._token = token
        self._keystore_path = keystore_path
        self._token_check_interval = (
            self.DEFAULT_TOKEN_CHECK_INTERVAL
            if token_check_interval is None
            else max(0, token_check_interval)
        )
        self._token_cache_path = f"{keystore_path}.token.json" if keystore_path else None
        self._token_exp = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._token_check_thread = None
        self._last_refresh_failure_at = 0.0
        self._ids_client = None
        self._keystore = None
        self._used_ids = False
        self._shutdown_done = False

        if token:
            self._update_expiry()

        if not token and keystore_path:
            self._refresh_token()

        if keystore_path and self._token_check_interval:
            self._start_token_check_thread()

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

    def _with_process_refresh_lock(self):
        if not self._token_cache_path or fcntl is None:
            return None

        lock_path = f"{self._token_cache_path}.lock"
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return lock_fd

    def _release_process_refresh_lock(self, lock_fd) -> None:
        if lock_fd is None or fcntl is None:
            return
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    def _load_cached_token(self, rejected_token: str | None = None) -> bool:
        if not self._token_cache_path:
            return False

        try:
            with open(self._token_cache_path, encoding="utf-8") as cache_file:
                payload = json.load(cache_file)
        except FileNotFoundError:
            return False
        except Exception as exc:
            self._logger.warning("Failed to read token cache %s: %s", self._token_cache_path, exc)
            return False

        cached_token = payload.get("token")
        cached_exp = payload.get("exp")
        if not cached_token or not cached_exp:
            return False
        if rejected_token is not None and cached_token == rejected_token:
            return False

        if time.time() >= (float(cached_exp) - self.REFRESH_MARGIN):
            return False

        self._token = cached_token
        self._update_expiry()
        self._logger.info("Loaded GenAI token from cache: %s", self._token_cache_path)
        return True

    def _write_cached_token(self) -> None:
        if not self._token_cache_path or not self._token:
            return

        payload = {
            "token": self._token,
            "exp": self._token_exp,
            "updated_at": int(time.time()),
        }
        temp_path = f"{self._token_cache_path}.tmp"

        try:
            with open(temp_path, "w", encoding="utf-8") as cache_file:
                json.dump(payload, cache_file)
            os.replace(temp_path, self._token_cache_path)
        except Exception as exc:
            self._logger.warning("Failed to update token cache %s: %s", self._token_cache_path, exc)

    def _delete_cached_token(self) -> None:
        if not self._token_cache_path:
            return
        try:
            os.remove(self._token_cache_path)
        except FileNotFoundError:
            return
        except Exception as exc:
            self._logger.warning("Failed to delete token cache %s: %s", self._token_cache_path, exc)

    def _refresh_token(self, force: bool = False, rejected_token: str | None = None) -> None:
        if not self._keystore_path:
            self._logger.warning(
                "Token expired or missing, but no keystore configured for refresh"
            )
            return

        lock_fd = self._with_process_refresh_lock()
        try:
            if force:
                if rejected_token is not None and self._load_cached_token(rejected_token=rejected_token):
                    self._last_refresh_failure_at = 0.0
                    return
                self._delete_cached_token()
            elif self._load_cached_token():
                self._last_refresh_failure_at = 0.0
                return

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

            login_response = self._get_genai_login_response(client)
            params = parse_qs(urlparse(login_response.url).query)

            token_response = client.session.get(
                GENAI_GET_TOKEN_URL.format(
                    token=params["token"][0],
                    timestamp=int(time.time() * 1000),
                ),
                timeout=30,
            )
            token_payload = token_response.json()
            token_result = token_payload.get("result")
            real_token = token_result.get("token") if isinstance(token_result, dict) else None

            if not real_token:
                raise RuntimeError(
                    "Failed to retrieve real token from GenAI login flow: "
                    f"{token_payload.get('message', 'missing result.token')}"
                )

            self._token = real_token
            self._update_expiry()
            keystore.dump(self._keystore_path)
            self._write_cached_token()
            self._last_refresh_failure_at = 0.0
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
        finally:
            self._release_process_refresh_lock(lock_fd)

    def _get_genai_login_response(self, client):
        for login_url in (GENAI_LOGIN_URL, GENAI_LEGACY_CAS_SERVICE_URL):
            response = client.session.get(
                login_url,
                allow_redirects=True,
                timeout=30,
            )
            parsed = urlparse(response.url)
            params = parse_qs(parsed.query)
            self._logger.debug(
                "GenAI login flow ended at %s://%s%s with query keys: %s",
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                sorted(params),
            )
            if params.get("token"):
                return response

        parsed = urlparse(response.url)
        raise RuntimeError(
            "Could not extract GenAI token from login flow. "
            f"Final URL path: {parsed.scheme}://{parsed.netloc}{parsed.path}, "
            f"query keys: {sorted(parse_qs(parsed.query))}"
        )

    def _start_token_check_thread(self) -> None:
        self._token_check_thread = threading.Thread(
            target=self._run_token_check_loop,
            name="genai-token-check",
            daemon=True,
        )
        self._token_check_thread.start()

    def _run_token_check_loop(self) -> None:
        self._logger.info(
            "Periodic GenAI token confirmation enabled: every %d seconds",
            self._token_check_interval,
        )
        while not self._stop_event.is_set():
            try:
                self._confirm_token_for_background()
            except Exception:
                self._logger.exception("Periodic GenAI token confirmation failed")

            if self._stop_event.wait(self._token_check_interval):
                break

    def _confirm_token_for_background(self) -> None:
        with self._lock:
            if self._shutdown_done:
                return
            if self._needs_refresh():
                self._refresh_token_for_access("periodic token confirmation")
            token = self._token

        if not token:
            return

        if self._stop_event.is_set():
            return

        if self._token_was_rejected_by_upstream(token):
            if self._stop_event.is_set():
                return
            self.refresh_after_auth_failure(
                "periodic token confirmation",
                rejected_token=token,
            )

    def _token_was_rejected_by_upstream(self, token: str) -> bool:
        try:
            response = requests.get(
                GENAI_CONFIRM_TOKEN_URL.format(token=token),
                params={"_t": int(time.time() * 1000)},
                headers={
                    "Accept": "application/json",
                    "X-Access-Token": token,
                },
                timeout=self.TOKEN_CONFIRM_TIMEOUT,
            )
        except requests.RequestException as exc:
            self._logger.warning(
                "Periodic token confirmation request failed: %s",
                exc.__class__.__name__,
            )
            return False

        if response.status_code in (401, 403):
            return True
        if response.status_code != 200:
            self._logger.warning(
                "Periodic token confirmation HTTP error %d",
                response.status_code,
            )
            return False

        try:
            payload = response.json()
        except ValueError:
            self._logger.warning("Periodic token confirmation returned invalid JSON")
            return False

        if is_genai_auth_failure(payload):
            return True
        code = payload.get("code", 200)
        try:
            failed_code = int(code) >= 400
        except (TypeError, ValueError):
            failed_code = False
        if payload.get("success") is False or failed_code:
            self._logger.warning(
                "Periodic token confirmation business error: code=%s message=%s",
                code,
                payload.get("message"),
            )
        return False

    def refresh_after_auth_failure(
        self,
        reason: str = "upstream authentication failure",
        rejected_token: str | None = None,
    ) -> bool:
        if not self._keystore_path:
            self._logger.warning(
                "GenAI token was rejected by upstream (%s), but no keystore is configured",
                reason,
            )
            return False

        with self._lock:
            if self._shutdown_done or self._stop_event.is_set():
                return False

            if rejected_token is not None and self._token and self._token != rejected_token:
                self._logger.info(
                    "Skipping token refresh for %s because another refresh already replaced it",
                    reason,
                )
                return True

            self._logger.warning("GenAI token was rejected by upstream (%s); refreshing", reason)
            self._token = None
            self._token_exp = None
            try:
                self._refresh_token(force=True, rejected_token=rejected_token)
            except Exception:
                self._last_refresh_failure_at = time.time()
                self._logger.exception("Failed to refresh GenAI token after upstream rejection")
                return False
            return bool(self._token)

    def _refresh_token_for_access(self, reason: str) -> None:
        if self._token and self._recent_refresh_failure():
            return

        try:
            self._refresh_token()
        except Exception:
            self._last_refresh_failure_at = time.time()
            if self._token:
                self._logger.warning(
                    "Continuing with existing GenAI token after refresh failure during %s",
                    reason,
                )
                return
            raise

    def _recent_refresh_failure(self) -> bool:
        return (
            self._last_refresh_failure_at > 0
            and time.time() - self._last_refresh_failure_at < self.REFRESH_FAILURE_RETRY_INTERVAL
        )

    @property
    def token(self) -> str | None:
        with self._lock:
            if self._needs_refresh():
                self._refresh_token_for_access("token access")
            elif self._token and self._token_exp and not self._keystore_path:
                remaining = self._token_exp - time.time()
                if remaining < self.REFRESH_MARGIN:
                    self._logger.warning(
                        "Token expires in %.0f seconds but no keystore for auto-refresh!",
                        remaining,
                    )
            return self._token

    def shutdown(self) -> None:
        self._stop_event.set()
        token_check_thread = self._token_check_thread
        if (
            token_check_thread
            and token_check_thread.is_alive()
            and token_check_thread is not threading.current_thread()
        ):
            token_check_thread.join(timeout=5)

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
