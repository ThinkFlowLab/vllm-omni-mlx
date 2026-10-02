"""Streaming fixes: model-name fallback in SSE payloads and prompt_tokens on
the first chunk (Anthropic message_start input usage)."""

import json
import unittest

from starlette.testclient import TestClient

from vllm_omni_mlx.backends import Backend, Chunk, _usage_first_emitter
from vllm_omni_mlx.schemas import UnifiedRequest
from vllm_omni_mlx.server import create_app


class FakeBackend(Backend):
    def __init__(self, chunks):
        self._chunks = chunks
        self.name = "fake-model"

    def chat(self, req: UnifiedRequest):
        yield from self._chunks


def _sse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    event = None
    for line in body.splitlines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: ") and line[6:] != "[DONE]":
            events.append((event, json.loads(line[6:])))
    return events


class UsageFirstEmitterTest(unittest.TestCase):
    def test_first_chunk_carries_prompt_tokens_once(self):
        make = _usage_first_emitter()
        first = make("a", 12)
        second = make("b", 12)
        self.assertEqual((first.text, first.prompt_tokens), ("a", 12))
        self.assertEqual((second.text, second.prompt_tokens), ("b", None))

    def test_none_prompt_tokens_stays_none(self):
        make = _usage_first_emitter()
        chunk = make("a", None)
        self.assertIsNone(chunk.prompt_tokens)
        later = make("b", 12)
        self.assertEqual(later.prompt_tokens, 12)


class StreamingResponsesTest(unittest.TestCase):
    def test_openai_stream_uses_backend_model_name(self):
        backend = FakeBackend([Chunk(text="hi", prompt_tokens=3), Chunk(text="", finish_reason="stop", completion_tokens=1)])
        client = TestClient(create_app(backend))
        body = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        ).text
        chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertEqual(chunk["model"], "fake-model")
        self.assertIn("data: [DONE]", body)

    def test_anthropic_stream_message_start_has_model_and_input_tokens(self):
        backend = FakeBackend([Chunk(text="hi", prompt_tokens=3), Chunk(text="", finish_reason="stop", completion_tokens=1)])
        client = TestClient(create_app(backend))
        body = client.post(
            "/v1/messages",
            json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 10, "stream": True},
        ).text
        events = _sse_events(body)
        kinds = [e for e, _ in events]
        self.assertEqual(kinds[0], "message_start")
        message = events[0][1]["message"]
        self.assertEqual(message["model"], "fake-model")
        self.assertEqual(message["usage"]["input_tokens"], 3)
        self.assertEqual(kinds[-1], "message_stop")


if __name__ == "__main__":
    unittest.main()
