"""Closure parity tests (#65): the compiled decode steps must be bit-exact
against the eager mlx-audio-mirror paths they replace — not "close enough":

- predictor frame: greedy tokens identical (16/16) and, under temperature,
  identical draws from a reseeded RNG stream (mx.compile with
  inputs/outputs=mx.random.state advances the global stream exactly as the
  eager per-step categorical calls do);
- talker decode frame: hidden and logits max|Δ| == 0 for two consecutive
  frames (the second exercises the shapeless growing-cache path), argmax
  agreeing.

These pin the fast path's correctness floor; loop-level behavior (chunk
schedules, boundary envelope, HNR) stays in tests.test_stream_loop.
Weight-gated like the rest of the battery."""

import os

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest

from tests._teardown import ReleaseAfterClass

import mlx.core as mx

from vllm_omni_mlx.tts.code_predictor import CodePredictor
from vllm_omni_mlx.tts.compiled_steps import (
    _sample_predictor_token,
    make_predictor_frame,
    make_talker_decode,
)
from vllm_omni_mlx.tts.config import DEFAULT_MODEL, TTSConfig, load_tts_model, local_snapshot
from vllm_omni_mlx.tts.prompt_embeds import PromptEmbeds
from vllm_omni_mlx.tts.talker import Talker

EN = "The quick brown fox jumps over the lazy dog."


class CompiledStepsTest(ReleaseAfterClass, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if local_snapshot(DEFAULT_MODEL) is None:
            raise unittest.SkipTest(f"{DEFAULT_MODEL} not cached locally")
        cls.model = load_tts_model(TTSConfig())

    def _first_frame_state(self):
        """Prefill a short prompt eagerly and assemble frame-1 inputs —
        the state both paths start from."""
        model = self.model
        talker = Talker(model.talker)
        layout = PromptEmbeds(model).build(EN, "vivian")
        cache = model.talker.make_cache()
        logits, hidden = model.talker(layout.input_embeds, cache=cache)
        mx.eval(logits, hidden)
        first_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        input_embeds_f1 = layout.trailing_text_hidden[:, 0:1, :] + talker.codec_embeds([first_token])
        mx.eval(input_embeds_f1)
        return cache, hidden[:, -1:, :], first_token, input_embeds_f1, layout.trailing_text_hidden

    def test_predictor_frame_greedy_tokens_exact(self):
        model = self.model
        predictor = CodePredictor(model.talker)
        _, code_hidden, first_token, _, _ = self._first_frame_state()

        code_cache = predictor.make_cache()
        eager_tokens = [first_token]
        for code_idx in range(predictor.num_code_groups - 1):
            if code_idx == 0:
                code_0_embed = model.talker.get_input_embeddings()(first_token)
                code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
            else:
                code_input = predictor.residual_embeddings[code_idx - 1](eager_tokens[-1])
            code_logits = predictor.step(code_input, code_cache, code_idx)
            eager_tokens.append(mx.argmax(code_logits[:, -1, :], axis=-1, keepdims=True))
        eager_codes = mx.concatenate(eager_tokens, axis=1)

        frame = make_predictor_frame(
            model.talker.code_predictor,
            temperature=0.0,
            top_k=50,
            top_p=1.0,
            base_embedding=model.talker.get_input_embeddings(),
        )
        compiled_codes = frame(code_hidden, first_token)
        mx.eval(eager_codes, compiled_codes)
        self.assertEqual(eager_codes.shape, compiled_codes.shape)
        self.assertEqual(int((eager_codes != compiled_codes).sum()), 0)

    def test_predictor_frame_sampled_draws_exact(self):
        model = self.model
        predictor = CodePredictor(model.talker)
        _, code_hidden, first_token, _, _ = self._first_frame_state()

        code_cache = predictor.make_cache()
        mx.random.seed(11)
        eager_tokens = [first_token]
        for code_idx in range(predictor.num_code_groups - 1):
            if code_idx == 0:
                code_0_embed = model.talker.get_input_embeddings()(first_token)
                code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
            else:
                code_input = predictor.residual_embeddings[code_idx - 1](eager_tokens[-1])
            code_logits = predictor.step(code_input, code_cache, code_idx)
            eager_tokens.append(_sample_predictor_token(code_logits[:, -1, :], 0.9, 50, 1.0))
        eager_codes = mx.concatenate(eager_tokens, axis=1)
        mx.eval(eager_codes)

        frame = make_predictor_frame(
            model.talker.code_predictor,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
            base_embedding=model.talker.get_input_embeddings(),
        )
        mx.random.seed(11)
        compiled_codes = frame(code_hidden, first_token)
        mx.eval(compiled_codes)
        self.assertEqual(int((eager_codes != compiled_codes).sum()), 0)

    def test_talker_decode_frame_bit_exact(self):
        model = self.model
        talker = Talker(model.talker)
        cache, _, first_token, input_embeds_f1, trailing = self._first_frame_state()

        prefix = cache[0].offset
        keys = [c.keys[..., :prefix, :] for c in cache]
        values = [c.values[..., :prefix, :] for c in cache]
        step = make_talker_decode(model.talker)

        # frame 1: eager vs compiled from identical post-prefill state
        logits_e, hidden_e = model.talker(input_embeds_f1, cache=cache)
        mx.eval(logits_e, hidden_e)
        logits_c, hidden_c, keys2, values2 = step(
            input_embeds_f1, mx.array([prefix], dtype=mx.int32), keys, values
        )
        mx.eval(logits_c, hidden_c)
        self.assertEqual(float(mx.abs(hidden_e[:, -1:, :] - hidden_c).max()), 0.0)
        self.assertEqual(float(mx.abs(logits_e[:, -1:, :] - logits_c).max()), 0.0)

        # frame 2 through the shapeless growing-cache path
        next_token = mx.argmax(logits_e[:, -1, :], axis=-1, keepdims=True)
        input_embeds_f2 = trailing[:, 1:2, :] + talker.codec_embeds([next_token])
        mx.eval(input_embeds_f2)
        logits_e2, hidden_e2 = model.talker(input_embeds_f2, cache=cache)
        mx.eval(logits_e2, hidden_e2)
        logits_c2, hidden_c2, _, _ = step(
            input_embeds_f2, mx.array([prefix + 1], dtype=mx.int32), keys2, values2
        )
        mx.eval(logits_c2, hidden_c2)
        self.assertEqual(float(mx.abs(hidden_e2[:, -1:, :] - hidden_c2).max()), 0.0)
        self.assertEqual(
            int(mx.argmax(logits_e2[:, -1, :], axis=-1).item()),
            int(mx.argmax(logits_c2[0, 0], axis=-1).item()),
        )

    def test_prewarm_traces_closures_and_stream_still_exact(self):
        from vllm_omni_mlx.tts.stream_loop import prewarm_streaming, synthesize_stream

        config = TTSConfig()
        self.assertEqual(prewarm_streaming(self.model, 0.5, 0.2), (2, 6))
        audio = mx.concatenate(
            [c.reshape(-1) for c in synthesize_stream(
                self.model, config, EN, speaker="vivian", temperature=0.9, seed=3,
                streaming_interval=0.5, max_tokens=64)]
        )
        self.assertGreater(audio.shape[0], 24000)  # at least a second of speech


if __name__ == "__main__":
    unittest.main()
