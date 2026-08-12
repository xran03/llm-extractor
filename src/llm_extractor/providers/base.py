"""Provider interface shared by every API backend.

A provider hides *how* a gateway is called (chat-completions vs responses) so
the extraction pipeline only ever sees one small surface:

* :meth:`Provider.complete` — messages in, assistant text out;
* :meth:`Provider.list_models` — connectivity check / model discovery.

Messages use the familiar OpenAI chat shape. Image parts use the portable form
``{"type": "image", "mime": "image/png", "data_b64": "..."}`` and each provider
translates them into its own wire format, so callers never branch on backend.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class ProviderError(RuntimeError):
    """Non-retryable API or transport failure."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached: bool = False

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached": self.cached,
        }


@dataclass
class Completion:
    """One model response: the text plus token accounting."""

    text: str
    usage: Usage = field(default_factory=Usage)
    raw: dict = field(default_factory=dict)


def image_part(image_bytes: bytes, mime: str = "image/png") -> dict:
    """Build a backend-neutral image content part."""
    return {
        "type": "image",
        "mime": mime,
        "data_b64": base64.b64encode(image_bytes).decode("ascii"),
    }


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def user_message(*parts) -> dict:
    """Build a user message from strings and/or content parts."""
    content = [text_part(p) if isinstance(p, str) else p for p in parts]
    return {"role": "user", "content": content}


@dataclass
class HTTPProvider:
    """Shared HTTP transport: retries, backoff, and auth-refresh on 401/403."""

    name: str = "provider"
    base_url: str = ""
    api_key: str = ""
    timeout: float = 90.0
    max_retries: int = 4
    backoff_base: float = 1.5
    token_provider: object = None

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or "").rstrip("/")
        if not self.base_url:
            raise ProviderError(
                f"{self.name}: no base URL configured. Set it in .env "
                f"(see .env.example) or pass --base-url."
            )
        if not self.api_key and self.token_provider is None:
            raise ProviderError(
                f"{self.name}: no credentials. Set an API key in .env, pass "
                f"--api-key, or use --api-key - to paste one."
            )

    # ------------------------------ transport ------------------------------
    def _auth_header(self, force_refresh: bool = False) -> str:
        if self.token_provider is not None:
            return f"Bearer {self.token_provider.get_token(force=force_refresh)}"
        return f"Bearer {self.api_key}"

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        data = None
        base_headers: dict = {"Accept": "application/json"}
        if payload is not None:
            base_headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")

        last_err: Exception | None = None
        forced_refresh = False
        for attempt in range(self.max_retries):
            try:
                headers = dict(base_headers)
                headers["Authorization"] = self._auth_header()
                req = urllib.request.Request(url, data=data, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                if exc.code in (401, 403) and self.token_provider is not None and not forced_refresh:
                    forced_refresh = True
                    self._auth_header(force_refresh=True)
                    continue
                if exc.code != 429 and 400 <= exc.code < 500:
                    raise ProviderError(f"{self.name} HTTP {exc.code} from {path}: {body[:500]}") from exc
                last_err = ProviderError(f"{self.name} HTTP {exc.code} from {path}: {body[:500]}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_err = ProviderError(f"{self.name} network error to {path}: {exc}")
            time.sleep(self.backoff_base ** attempt)
        raise last_err or ProviderError(f"{self.name}: request to {path} failed")

    # -------------------------------- API ---------------------------------
    def list_models(self) -> list:
        return [m.get("id") for m in self.request("GET", "/v1/models").get("data", [])]

    def complete(self, messages, model, temperature=0.0, max_tokens=None,
                 json_schema=None, **kwargs) -> Completion:  # pragma: no cover - abstract
        raise NotImplementedError

    def complete_text(self, messages, model, **kwargs) -> str:
        return self.complete(messages, model, **kwargs).text


def usage_from(payload: dict) -> Usage:
    """Read token usage from either API's usage object."""
    usage = payload.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
