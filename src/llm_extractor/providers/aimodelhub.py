"""AI Model Hub backend — OpenAI-compatible ``/v1/responses``.

The Responses API supersedes chat-completions: system prompts become
``instructions``, messages become typed ``input`` blocks, structured output is
declared under ``text.format`` instead of ``response_format``, and the answer
is returned as ``output_text`` (or reassembled from ``output[].content[]``).
"""
from __future__ import annotations

from .base import Completion, HTTPProvider, usage_from


class AIModelHubProvider(HTTPProvider):
    """Responses-API provider."""

    API_STYLE = "responses"

    def complete(self, messages, model, temperature=0.0, max_tokens=None,
                 json_schema=None, reasoning_effort=None, **kwargs) -> Completion:
        instructions, input_blocks = _split_messages(messages)
        payload: dict = {
            "model": model,
            "input": input_blocks,
            "temperature": temperature,
            "stream": False,
        }
        if instructions:
            payload["instructions"] = instructions
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        if json_schema:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": json_schema.get("name", "output"),
                    "schema": json_schema["schema"],
                    "strict": bool(json_schema.get("strict", False)),
                }
            }
        payload.update(kwargs.get("extra") or {})

        raw = self.request("POST", "/v1/responses", payload)
        return Completion(text=response_text(raw), usage=usage_from(raw), raw=raw)


def response_text(raw: dict) -> str:
    """Extract assistant text from a Responses-API payload.

    Prefers the ``output_text`` convenience field, then walks ``output`` blocks
    and concatenates every text part, ignoring reasoning/tool blocks.
    """
    direct = raw.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    if isinstance(direct, list):
        joined = "".join(t for t in direct if isinstance(t, str))
        if joined.strip():
            return joined

    chunks = []
    for block in raw.get("output") or []:
        if not isinstance(block, dict) or block.get("type") not in (None, "message"):
            continue
        for part in block.get("content") or []:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                chunks.append(part.get("text", ""))
    return "".join(chunks)


def _split_messages(messages):
    """Split neutral messages into ``instructions`` + typed input blocks."""
    instructions_parts = []
    blocks = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        if role == "system":
            instructions_parts.append(
                content if isinstance(content, str) else _plain_text(content)
            )
            continue
        blocks.append({"role": role, "content": _to_input_parts(role, content)})
    return "\n\n".join(p for p in instructions_parts if p), blocks


def _plain_text(content) -> str:
    return "".join(
        part.get("text", "") for part in content or []
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _to_input_parts(role: str, content):
    text_type = "output_text" if role == "assistant" else "input_text"
    if isinstance(content, str):
        return [{"type": text_type, "text": content}]

    parts = []
    for part in content or []:
        if part.get("type") == "image":
            parts.append({
                "type": "input_image",
                "image_url": f"data:{part.get('mime', 'image/png')};base64,{part['data_b64']}",
            })
        else:
            parts.append({"type": text_type, "text": part.get("text", "")})
    return parts
