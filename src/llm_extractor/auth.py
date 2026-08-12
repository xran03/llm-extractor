"""OAuth2 client-credentials token provider.

Some gateways issue a long-lived API key; others require a short-lived bearer
token minted from a client id + secret. This provider implements the second
path with in-memory caching and pre-emptive refresh so callers always get a
currently-valid token. The HTTP call is isolated in ``_http_fetch`` and can be
replaced through the ``fetcher`` argument for network-free tests.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request


class OAuthError(RuntimeError):
    """Raised when a token cannot be minted."""


class OAuthTokenProvider:
    def __init__(self, client_id: str, client_secret: str, token_url: str,
                 timeout: float = 30.0, refresh_skew: float = 60.0, fetcher=None):
        if not (client_id and client_secret and token_url):
            raise OAuthError("client_id, client_secret and token_url are all required")
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.timeout = timeout
        self.refresh_skew = refresh_skew
        self._fetcher = fetcher or self._http_fetch
        self._token = ""
        self._expiry = 0.0
        self.last_expires_in = 0.0

    def get_token(self, force: bool = False) -> str:
        if not force and self._token and time.time() < self._expiry - self.refresh_skew:
            return self._token
        data = self._fetcher()
        token = data.get("access_token") or data.get("token")
        if not token:
            raise OAuthError(f"token endpoint returned no access_token: {sorted(data)}")
        self._token = token
        self.last_expires_in = float(data.get("expires_in", 900))
        self._expiry = time.time() + self.last_expires_in
        return self._token

    def _http_fetch(self) -> dict:
        basic = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode("utf-8")
        ).decode("ascii")
        body = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
        req = urllib.request.Request(
            self.token_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {basic}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise OAuthError(f"token endpoint HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise OAuthError(f"token endpoint unreachable: {exc}") from exc
