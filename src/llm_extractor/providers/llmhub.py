"""LLMHub backend — OpenAI-compatible ``/v1/chat/completions``.

This is the proven path carried over from the original pipeline. It is kept
because many gateways still only expose chat-completions; new deployments
should prefer the ``aimodelhub`` (Responses API) backend.
"""
from __future__ import annotations

from .base import Completion, HTTPProvider, usage_from


class LLMHubProvider(HTTPProvider):
    """Chat-completions provider."""

    API_STYLE = "chat.completions"
    INFERENCE_PATH = "/v1/chat/completions"

    def build_payload(self, messages, model, temperature=0.0, max_tokens=None,
                      json_schema=None, reasoning_effort=None, **kwargs) -> dict:
        return chat_payload(messages, model, temperature=temperature,
                            max_tokens=max_tokens, json_schema=json_schema,
                            reasoning_effort=reasoning_effort, **kwargs)

    def parse_completion(self, raw: dict) -> Completion:
        return parse_chat_completion(raw)


def chat_payload(messages, model, temperature=0.0, max_tokens=None,
                 json_schema=None, reasoning_effort=None, **kwargs) -> dict:
    """Render a call in chat-completions wire format.

    Shared rather than owned by one provider: batch jobs are queued against
    ``/v1/chat/completions`` even on gateways whose live path is the Responses
    API, so the Responses backend needs this shape too.
    """
    payload: dict = {
        "model": model,
        "messages": [_to_chat_message(m) for m in messages],
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": json_schema.get("name", "output"),
                "schema": json_schema["schema"],
                "strict": bool(json_schema.get("strict", False)),
            },
        }
    payload.update(kwargs.get("extra") or {})
    return payload


def parse_chat_completion(raw: dict) -> Completion:
    """Read a chat-completions response back into a :class:`Completion`."""
    choices = raw.get("choices") or [{}]
    text = (choices[0].get("message", {}) or {}).get("content") or ""
    if isinstance(text, list):  # some gateways return content parts
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    return Completion(text=text, usage=usage_from(raw), raw=raw)


def _to_chat_message(message: dict) -> dict:
    """Translate a neutral message into chat-completions wire format."""
    content = message.get("content")
    if isinstance(content, str):
        return {"role": message["role"], "content": content}

    parts = []
    for part in content or []:
        if part.get("type") == "image":
            url = f"data:{part.get('mime', 'image/png')};base64,{part['data_b64']}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            parts.append({"type": "text", "text": part.get("text", "")})
    return {"role": message["role"], "content": parts}
