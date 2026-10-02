"""Tests for request normalization — no model or network needed (except one
local-data-URL / base64 case)."""

import base64
import unittest

from vllm_omni_mlx.schemas import ApiError, normalize_anthropic, normalize_openai

PNG = base64.b64encode(b"\x89PNG fake bytes").decode()


def openai_payload(**overrides):
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
    }
    payload.update(overrides)
    return payload


def anthropic_payload(**overrides):
    payload = {
        "model": "test-model",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return payload


class OpenAINormalization(unittest.TestCase):
    def test_plain_text(self):
        req = normalize_openai(openai_payload())
        self.assertEqual(len(req.messages), 1)
        self.assertEqual(req.messages[0].parts[0].kind, "text")
        self.assertEqual(req.messages[0].parts[0].text, "hello")
        self.assertFalse(req.stream)

    def test_system_message_extracted(self):
        req = normalize_openai(
            openai_payload(messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}])
        )
        self.assertEqual(req.system, "be brief")
        self.assertEqual([m.role for m in req.messages], ["user"])

    def test_multimodal_parts(self):
        req = normalize_openai(
            openai_payload(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
                            {"type": "input_audio", "input_audio": {"data": PNG, "format": "wav"}},
                        ],
                    }
                ]
            )
        )
        parts = req.messages[0].parts
        self.assertEqual([p.kind for p in parts], ["text", "image", "audio"])
        self.assertEqual(parts[1].data, b"\x89PNG fake bytes")
        self.assertEqual(parts[1].media_type, "image/png")
        self.assertEqual(parts[2].media_type, "audio/wav")

    def test_stop_normalization(self):
        req = normalize_openai(openai_payload(stop="END"))
        self.assertEqual(req.stop, ("END",))
        req = normalize_openai(openai_payload(stop=["A", "B"]))
        self.assertEqual(req.stop, ("A", "B"))

    def test_max_completion_tokens_fallback(self):
        req = normalize_openai({"model": "m", "messages": [{"role": "user", "content": "x"}], "max_completion_tokens": 7})
        self.assertEqual(req.max_tokens, 7)

    def test_errors(self):
        with self.assertRaises(ApiError):
            normalize_openai({"model": "m"})
        with self.assertRaises(ApiError):
            normalize_openai(openai_payload(max_tokens=0))
        with self.assertRaises(ApiError):
            normalize_openai(openai_payload(messages=[{"role": "user", "content": [{"type": "video", "video": {}}]}]))
        with self.assertRaises(ApiError):
            normalize_openai(openai_payload(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "ftp://nope"}}]}]))
        with self.assertRaises(ApiError):
            normalize_openai(openai_payload(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%"}}]}]))

    def test_user_role_required(self):
        with self.assertRaises(ApiError):
            normalize_openai(openai_payload(messages=[{"role": "assistant", "content": "hi"}]))


class AnthropicNormalization(unittest.TestCase):
    def test_plain_text(self):
        req = normalize_anthropic(anthropic_payload())
        self.assertEqual(req.messages[0].parts[0].text, "hello")
        self.assertEqual(req.max_tokens, 64)

    def test_system_string_and_blocks(self):
        req = normalize_anthropic(anthropic_payload(system="be brief"))
        self.assertEqual(req.system, "be brief")
        req = normalize_anthropic(
            anthropic_payload(system=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
        )
        self.assertEqual(req.system, "ab")

    def test_image_base64_and_url_source(self):
        req = normalize_anthropic(
            anthropic_payload(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "look"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
                        ],
                    }
                ]
            )
        )
        part = req.messages[0].parts[1]
        self.assertEqual(part.kind, "image")
        self.assertEqual(part.data, b"\x89PNG fake bytes")

        req = normalize_anthropic(
            anthropic_payload(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": {"type": "url", "url": f"data:image/png;base64,{PNG}"}}],
                    }
                ]
            )
        )
        self.assertEqual(req.messages[0].parts[0].data, b"\x89PNG fake bytes")

    def test_stop_sequences(self):
        req = normalize_anthropic(anthropic_payload(stop_sequences=["END"]))
        self.assertEqual(req.stop, ("END",))

    def test_errors(self):
        with self.assertRaises(ApiError):
            normalize_anthropic({"model": "m", "messages": [{"role": "user", "content": "x"}]})  # no max_tokens
        with self.assertRaises(ApiError):
            normalize_anthropic(anthropic_payload(messages=[{"role": "user", "content": [{"type": "tool_use", "id": "t", "name": "n", "input": {}}]}]))
        with self.assertRaises(ApiError):
            normalize_anthropic(anthropic_payload(max_tokens="lots"))
        with self.assertRaises(ApiError):
            normalize_anthropic(anthropic_payload(system=[{"type": "image", "source": {}}]))


if __name__ == "__main__":
    unittest.main()
