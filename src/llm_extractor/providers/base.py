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
import uuid
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

    #: Endpoint this backend sends inference to, and the batch endpoint it
    #: names for queued requests. Subclasses override.
    INFERENCE_PATH = "/v1/chat/completions"
    #: OpenAI batch protocol paths. Uniform across gateways that implement it.
    BATCHES_PATH = "/v1/batches"
    FILES_PATH = "/v1/files"

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

    def request_raw(self, method: str, path: str, data: bytes | None = None,
                    content_type: str = "", accept: str = "application/json") -> bytes:
        """Send one request, retrying transient failures, and return raw bytes.

        Every call in this package goes through here, so retries, backoff and
        OAuth refresh are defined once regardless of body type — JSON for
        inference, multipart for a batch upload, JSONL for a result download.
        """
        url = f"{self.base_url}{path}"
        base_headers: dict = {"Accept": accept}
        if content_type:
            base_headers["Content-Type"] = content_type

        last_err: Exception | None = None
        forced_refresh = False
        for attempt in range(self.max_retries):
            try:
                headers = dict(base_headers)
                headers["Authorization"] = self._auth_header()
                req = urllib.request.Request(url, data=data, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read()
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

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        raw = self.request_raw(method, path, data,
                               content_type="application/json" if data else "")
        return json.loads(raw.decode("utf-8"))

    # -------------------------------- API ---------------------------------
    def list_models(self) -> list:
        return [m.get("id") for m in self.request("GET", "/v1/models").get("data", [])]

    def build_payload(self, messages, model, temperature=0.0, max_tokens=None,
                      json_schema=None, **kwargs) -> dict:  # pragma: no cover - abstract
        """Render a call into this backend's wire format."""
        raise NotImplementedError

    def parse_completion(self, raw: dict) -> Completion:  # pragma: no cover - abstract
        """Read this backend's response shape back into a :class:`Completion`."""
        raise NotImplementedError

    #: Parameters a gateway may reject for a particular model. Some models
    #: deprecate sampling controls entirely and answer 400 rather than ignoring
    #: them, so the request is retried once without the offending key.
    OPTIONAL_PARAMS = ("temperature", "top_p")

    def complete(self, messages, model, temperature=0.0, max_tokens=None,
                 json_schema=None, **kwargs) -> Completion:
        payload = self.build_payload(messages, model, temperature=temperature,
                                     max_tokens=max_tokens, json_schema=json_schema,
                                     **kwargs)
        try:
            raw = self.request("POST", self.INFERENCE_PATH, payload)
        except ProviderError as exc:
            rejected = self._rejected_param(exc)
            if rejected is None:
                raise
            # Which models accept which sampling controls changes without
            # notice, so this reacts to what the gateway said rather than
            # carrying a list of model names that will quietly rot.
            payload.pop(rejected, None)
            raw = self.request("POST", self.INFERENCE_PATH, payload)
        return self.parse_completion(raw)

    def _rejected_param(self, error: Exception):
        """The optional parameter this error blames, if it blames exactly one."""
        message = str(error)
        if " 400" not in message:
            return None
        lowered = message.lower()
        if not any(word in lowered for word in
                   ("deprecated", "unsupported", "not supported", "does not support")):
            return None
        named = [p for p in self.OPTIONAL_PARAMS if p in lowered]
        return named[0] if len(named) == 1 else None

    def complete_text(self, messages, model, **kwargs) -> str:
        return self.complete(messages, model, **kwargs).text

    # ------------------------------- batch --------------------------------
    # The OpenAI batch protocol: upload a JSONL of requests, create a batch over
    # it, poll, then download a JSONL of responses. The wire format of each line
    # is whatever ``build_payload`` produces, so a batched call and a live call
    # can never drift apart.

    #: Where queued requests are sent. Batch is an OpenAI-protocol feature and
    #: gateways implement it over chat-completions; a provider whose live path
    #: is something else overrides this and the payload builder together.
    BATCH_INFERENCE_PATH = ""

    def batch_endpoint(self) -> str:
        return self.BATCH_INFERENCE_PATH or self.INFERENCE_PATH

    def build_batch_payload(self, messages, model, **kwargs) -> dict:
        """Render one queued request. Defaults to the live wire format."""
        return self.build_payload(messages, model, **kwargs)

    def parse_batch_completion(self, raw: dict) -> Completion:
        """Read one queued answer. Defaults to the live response shape."""
        return self.parse_completion(raw)

    def upload_file(self, content: bytes, filename: str = "batch.jsonl",
                    purpose: str = "batch") -> dict:
        body, content_type = encode_multipart(
            {"purpose": purpose}, "file", filename, content, "application/jsonl")
        raw = self.request_raw("POST", f"{self.FILES_PATH}", body, content_type=content_type)
        return json.loads(raw.decode("utf-8"))

    def create_batch(self, input_file_id: str, endpoint: str = "",
                     completion_window: str = "24h", metadata: dict | None = None) -> dict:
        payload = {
            "input_file_id": input_file_id,
            "endpoint": endpoint or self.INFERENCE_PATH,
            "completion_window": completion_window,
        }
        if metadata:
            payload["metadata"] = metadata
        return self.request("POST", self.BATCHES_PATH, payload)

    def get_batch(self, batch_id: str) -> dict:
        return self.request("GET", f"{self.BATCHES_PATH}/{batch_id}")

    def list_batches(self, limit: int = 20, after: str = "") -> dict:
        query = f"?limit={int(limit)}" + (f"&after={after}" if after else "")
        return self.request("GET", f"{self.BATCHES_PATH}{query}")

    def cancel_batch(self, batch_id: str) -> dict:
        return self.request("POST", f"{self.BATCHES_PATH}/{batch_id}/cancel")

    def download_file(self, file_id: str) -> bytes:
        return self.request_raw("GET", f"{self.FILES_PATH}/{file_id}/content",
                                accept="application/json")


def encode_multipart(fields: dict, file_field: str, filename: str,
                     content: bytes, file_type: str = "application/octet-stream"):
    """Build a multipart/form-data body; returns ``(bytes, content_type)``."""
    boundary = f"----llmextractor{uuid.uuid4().hex}"
    buf = bytearray()
    for name, value in fields.items():
        buf += f"--{boundary}\r\n".encode()
        buf += f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    buf += f"--{boundary}\r\n".encode()
    buf += (f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\n').encode()
    buf += f"Content-Type: {file_type}\r\n\r\n".encode()
    buf += content
    buf += f"\r\n--{boundary}--\r\n".encode()
    return bytes(buf), f"multipart/form-data; boundary={boundary}"


def usage_from(payload: dict) -> Usage:
    """Read token usage from either API's usage object."""
    usage = payload.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
