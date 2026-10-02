"""TextBackend single-entry prefix cache: suffix-only prefill on conversation
continuation, full re-prefill on divergence, and full prompt-token usage."""

import threading
import unittest
from types import SimpleNamespace

from vllm_omni_mlx.backends import TextBackend
from vllm_omni_mlx.schemas import Message, Part, UnifiedRequest


def user(text):
    return Message(role="user", parts=[Part(kind="text", text=text)])


def assistant(text):
    return Message(role="assistant", parts=[Part(kind="text", text=text)])


class FakeTokenizer:
    """Chat template shaped like a real one: the generation prompt ends with
    the assistant header 'a>', and turn 2 re-renders the reply directly after
    those same bytes, so turn-1 prompt + reply tokens is a strict prefix of
    turn-2's rendering."""

    bos_token = None

    HEADERS = {"user": "u:", "assistant": "a>", "system": "s:"}

    def apply_chat_template(self, messages, add_generation_prompt=True):
        out = "".join(self.HEADERS[m["role"]] + m["content"] + "<" for m in messages)
        return out + ("a>" if add_generation_prompt else "")

    def encode(self, prompt, add_special_tokens=False):
        return [ord(ch) for ch in prompt]


class FakeMlxLm:
    """Emits reply 'abc' one token per response, then the turn terminator '<'
    (empty text, like an eos token the next template rendering re-emits)."""

    def __init__(self):
        self.calls = []

    def stream_generate(self, model, tokenizer, prompt, **kwargs):
        self.calls.append({"prompt": list(prompt), **kwargs})
        for ch in "abc":
            yield SimpleNamespace(text=ch, token=ord(ch), prompt_tokens=len(prompt), finish_reason=None)
        yield SimpleNamespace(text="", token=ord("<"), prompt_tokens=len(prompt), finish_reason="stop")


def make_backend():
    backend = TextBackend.__new__(TextBackend)
    backend._mlx_lm = FakeMlxLm()
    backend._make_sampler = lambda **kw: None
    backend._make_cache = lambda model, max_kv_size=None: [f"cache-{id(object())}"]
    backend._draft_model = None
    backend._kv_kwargs = {}
    backend.model = None
    backend.tokenizer = FakeTokenizer()
    backend.name = "fake"
    backend._lock = threading.Lock()
    backend._cached = (None, None)
    return backend


def drain(gen):
    return [c for c in gen]


class PromptCacheTest(unittest.TestCase):
    def test_first_turn_full_prefill_populates_cache(self):
        backend = make_backend()
        req = UnifiedRequest(model="", messages=[user("hi")])
        chunks = drain(backend.chat(req))
        call = backend._mlx_lm.calls[0]
        cached_tokens, cache = backend._cached
        self.assertEqual(call["prompt"], cached_tokens[: len(call["prompt"])])
        self.assertIs(call["prompt_cache"], cache)
        # usage reports the full prompt count; generated ids stored for the next turn
        self.assertEqual(chunks[-1].prompt_tokens, len(cached_tokens) - 4)
        self.assertEqual(cached_tokens[-4:], [ord(c) for c in "abc<"])

    def test_continuation_feeds_suffix_only(self):
        backend = make_backend()
        drain(backend.chat(UnifiedRequest(model="", messages=[user("hi")])))
        first_cache = backend._cached[1]
        req2 = UnifiedRequest(model="", messages=[user("hi"), assistant("abc"), user("q")])
        chunks = drain(backend.chat(req2))
        second = backend._mlx_lm.calls[1]
        # everything after the cached prompt+reply: 'u:q<a>'
        self.assertEqual(second["prompt"], [ord(c) for c in "u:q<a>"])
        self.assertIs(second["prompt_cache"], first_cache)
        # usage still reports the full second-turn prompt length
        full_render = "u:hi<a>abc<u:q<a>"
        self.assertEqual(chunks[-1].prompt_tokens, len(full_render))

    def test_divergent_prompt_re_prefills_from_scratch(self):
        backend = make_backend()
        drain(backend.chat(UnifiedRequest(model="", messages=[user("hi")])))
        first_cache = backend._cached[1]
        drain(backend.chat(UnifiedRequest(model="", messages=[user("other")])))
        second = backend._mlx_lm.calls[1]
        self.assertEqual(second["prompt"], [ord(c) for c in "u:other<a>"])
        self.assertIsNot(second["prompt_cache"], first_cache)

    def test_identical_prompt_re_prefills(self):
        backend = make_backend()
        drain(backend.chat(UnifiedRequest(model="", messages=[user("hi")])))
        first_cache = backend._cached[1]
        drain(backend.chat(UnifiedRequest(model="", messages=[user("hi")])))
        # the suffix would be empty (prompt equals the cached prefix) — re-prefill instead
        self.assertEqual(backend._mlx_lm.calls[1]["prompt"], backend._mlx_lm.calls[0]["prompt"])
        self.assertIsNot(backend._mlx_lm.calls[1]["prompt_cache"], first_cache)


if __name__ == "__main__":
    unittest.main()
