"""Both API backends must send their own wire format and read their own replies."""
from __future__ import annotations

import unittest

from llm_extractor.providers import BACKENDS, build_provider
from llm_extractor.providers.aimodelhub import AIModelHubProvider, response_text
from llm_extractor.providers.base import ProviderError, image_part, user_message
from llm_extractor.providers.llmhub import LLMHubProvider
from llm_extractor.settings import Settings

from ._fakes import PNG_BYTES, RecordingTransport

SCHEMA = {"name": "out", "strict": True,
          "schema": {"type": "object", "properties": {}, "required": [],
                     "additionalProperties": False}}


def _messages():
    return [
        {"role": "system", "content": "be precise"},
        user_message("read this figure", image_part(PNG_BYTES, mime="image/png")),
    ]


class LLMHubProviderTest(unittest.TestCase):
    def setUp(self):
        self.provider = LLMHubProvider(name="llmhub", base_url="https://gw.example",
                                       api_key="k")
        self.transport = RecordingTransport(
            {"choices": [{"message": {"content": '{"records": []}'}}],
             "usage": {"prompt_tokens": 11, "completion_tokens": 3}}
        )
        self.provider.request = self.transport

    def test_posts_to_chat_completions(self):
        self.provider.complete(_messages(), model="m", max_tokens=99, json_schema=SCHEMA)
        call = self.transport.calls[0]
        self.assertEqual(call["path"], "/v1/chat/completions")
        self.assertEqual(call["payload"]["max_tokens"], 99)
        self.assertEqual(call["payload"]["response_format"]["type"], "json_schema")

    def test_system_message_stays_in_messages(self):
        self.provider.complete(_messages(), model="m")
        messages = self.transport.calls[0]["payload"]["messages"]
        self.assertEqual(messages[0]["role"], "system")

    def test_image_becomes_image_url_data_uri(self):
        self.provider.complete(_messages(), model="m")
        parts = self.transport.calls[0]["payload"]["messages"][1]["content"]
        image = [p for p in parts if p["type"] == "image_url"][0]
        self.assertTrue(image["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_reads_usage_and_text(self):
        completion = self.provider.complete(_messages(), model="m")
        self.assertEqual(completion.text, '{"records": []}')
        self.assertEqual(completion.usage.prompt_tokens, 11)
        self.assertEqual(completion.usage.completion_tokens, 3)

    def test_content_parts_are_joined(self):
        self.provider.request = RecordingTransport(
            {"choices": [{"message": {"content": [{"type": "text", "text": "ab"},
                                                  {"type": "text", "text": "cd"}]}}]}
        )
        self.assertEqual(self.provider.complete(_messages(), model="m").text, "abcd")


class AIModelHubProviderTest(unittest.TestCase):
    def setUp(self):
        self.provider = AIModelHubProvider(name="aimodelhub",
                                           base_url="https://gw.example", api_key="k")
        self.transport = RecordingTransport(
            {"output_text": '{"records": []}',
             "usage": {"input_tokens": 7, "output_tokens": 2}}
        )
        self.provider.request = self.transport

    def test_posts_to_responses_endpoint(self):
        self.provider.complete(_messages(), model="m", max_tokens=99, json_schema=SCHEMA)
        call = self.transport.calls[0]
        self.assertEqual(call["path"], "/v1/responses")
        self.assertEqual(call["payload"]["max_output_tokens"], 99)
        self.assertEqual(call["payload"]["text"]["format"]["type"], "json_schema")
        self.assertNotIn("messages", call["payload"])

    def test_system_message_becomes_instructions(self):
        self.provider.complete(_messages(), model="m")
        payload = self.transport.calls[0]["payload"]
        self.assertEqual(payload["instructions"], "be precise")
        self.assertEqual([b["role"] for b in payload["input"]], ["user"])

    def test_image_becomes_input_image(self):
        self.provider.complete(_messages(), model="m")
        parts = self.transport.calls[0]["payload"]["input"][0]["content"]
        kinds = {p["type"] for p in parts}
        self.assertEqual(kinds, {"input_text", "input_image"})

    def test_reads_input_output_token_usage(self):
        completion = self.provider.complete(_messages(), model="m")
        self.assertEqual(completion.usage.prompt_tokens, 7)
        self.assertEqual(completion.usage.completion_tokens, 2)

    def test_response_text_falls_back_to_output_blocks(self):
        raw = {"output": [
            {"type": "reasoning", "content": [{"type": "text", "text": "ignore"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "hello"},
                                            {"type": "output_text", "text": " world"}]},
        ]}
        self.assertEqual(response_text(raw), "hello world")

    def test_response_text_prefers_output_text(self):
        self.assertEqual(response_text({"output_text": "direct"}), "direct")


class BuildProviderTest(unittest.TestCase):
    def test_both_backends_registered(self):
        self.assertEqual(sorted(BACKENDS), ["aimodelhub", "llmhub"])

    def test_unknown_backend_is_rejected(self):
        settings = Settings(api="nope", base_url="https://x", api_key="k",
                            cache_enabled=False)
        with self.assertRaises(ProviderError):
            build_provider(settings)

    def test_missing_base_url_is_rejected(self):
        settings = Settings(api="llmhub", base_url="", api_key="k", cache_enabled=False)
        with self.assertRaises(ProviderError) as ctx:
            build_provider(settings)
        self.assertIn("base URL", str(ctx.exception))

    def test_missing_credentials_are_rejected(self):
        settings = Settings(api="llmhub", base_url="https://x", cache_enabled=False)
        with self.assertRaises(ProviderError) as ctx:
            build_provider(settings)
        self.assertIn("credentials", str(ctx.exception))

    def test_oauth_credentials_take_precedence_over_api_key(self):
        settings = Settings(api="llmhub", base_url="https://x", api_key="k",
                            client_id="id", client_secret="secret",
                            token_url="https://token", cache_enabled=False)
        provider = build_provider(settings)
        self.assertIsNotNone(provider.token_provider)


if __name__ == "__main__":
    unittest.main()
