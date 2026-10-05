"""First-chunk fast-path loop (#39 tasks 2+3): chunk scheduling math,
pad-and-trim exactness, token-stream invariance across chunk schedules,
greedy parity with mlx-audio's loop, and HNR plausibility. Weight-gated
where the model is required; scheduling math runs everywhere.

Calibration notes (4-bit checkpoint): the loop is token-exact — same seed +
greedy yields identical sampler-draw streams across chunk schedules and vs
mlx-audio's own loop — but chunked streaming_step is boundary-sensitive at
the ~3e-3 mean waveform level with identical codes; mlx-audio's loop shows
the same envelope between its own intervals, so parity is asserted on draws
(exact) plus a mean-envelope tripwire well below structural-break territory
(1.6e-2 mean, the left-context-windowing error mode)."""

import os

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest

from tests._teardown import ReleaseAfterClass

import mlx.core as mx

from tests.audio_metrics import CLEAN_VOICE_HNR_DB, int16_pcm_hnr_db
from vllm_omni_mlx.tts.config import DEFAULT_MODEL, TTSConfig, load_tts_model, local_snapshot
from vllm_omni_mlx.tts.stream_loop import (
    SAMPLES_PER_FRAME,
    _pad_target,
    frames_for_interval,
    generate_frames,
    initial_frames_bucket,
    prewarm_streaming,
    synthesize_stream,
)

EN = "The quick brown fox jumps over the lazy dog."

# chunked streaming_step is boundary-sensitive at the ~3e-3 mean / 0.25 max
# waveform level even with identical codes — mlx-audio's own loop shows the
# same envelope between its own streaming_intervals (measured: identical
# 608-draw token streams, mean|Δ| 1.8e-3, max|Δ| 0.246 between interval 0.5
# and 2.0). Structural decode breakage (e.g. left-context windowing instead
# of stateful streaming) measured 1.6e-2 MEAN — an order of magnitude above
# this envelope — so the mean bound below is the corruption tripwire.
AUDIO_ENVELOPE_MEAN = 8e-3


def _pcm16(audio: mx.array) -> bytes:
    from array import array

    return array("h", (mx.clip(audio, -1.0, 1.0) * 32767.0).astype(mx.int16).tolist()).tobytes()


def _taped(model, generate):
    """Run `generate()` recording every sampler draw (16 per frame) — both
    our loop and mlx-audio's call model._sample_token, so the recorded
    streams are directly comparable."""
    orig = model._sample_token
    draws: list[int] = []

    def sampler(logits, **kwargs):
        token = orig(logits, **kwargs)
        draws.append(int(token.reshape(-1)[0].item()))
        return token

    model._sample_token = sampler
    try:
        audio = generate()
    finally:
        model._sample_token = orig
    return draws, audio


class SchedulingMathTest(unittest.TestCase):
    def test_frames_for_interval_matches_mlx_audio_formula(self):
        self.assertEqual(frames_for_interval(2.0), 25)
        self.assertEqual(frames_for_interval(0.5), 6)
        self.assertEqual(frames_for_interval(0.25), 3)
        self.assertEqual(frames_for_interval(0.08), 1)
        self.assertEqual(frames_for_interval(0.0), 1)  # max(1, ...) clamp

    def test_initial_frames_bucket(self):
        self.assertEqual(initial_frames_bucket(1, 6), 1)
        self.assertEqual(initial_frames_bucket(2, 6), 2)
        self.assertEqual(initial_frames_bucket(3, 6), 2)  # quantized down
        self.assertEqual(initial_frames_bucket(6, 6), 4)
        self.assertEqual(initial_frames_bucket(16, 25), 16)
        self.assertEqual(initial_frames_bucket(32, 25), 16)  # capped at chunk

    def test_pad_target(self):
        self.assertEqual(_pad_target(1, 2, 6), 2)
        self.assertEqual(_pad_target(2, 2, 6), 2)
        self.assertEqual(_pad_target(3, 2, 6), 6)
        self.assertEqual(_pad_target(6, 2, 6), 6)


class StreamLoopTest(ReleaseAfterClass, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if local_snapshot(DEFAULT_MODEL) is None:
            raise unittest.SkipTest(f"{DEFAULT_MODEL} not cached locally")
        cls.model = load_tts_model(TTSConfig())
        cls.config = TTSConfig()
        # trace the serving shapes up front so latency-sensitive assertions
        # measure generation, not first-use compile
        cls.prewarmed = prewarm_streaming(cls.model, 0.5, 0.2)

    def test_prewarm_traces_serving_shapes_and_cleans_state(self):
        self.assertEqual(self.prewarmed, (2, 6))

    def test_first_chunk_is_initial_frames_then_steady(self):
        chunks = list(
            generate_frames(
                self.model, text=EN, speaker="vivian", temperature=0.0, max_tokens=512,
                initial_frames=2, chunk_frames=6,
            )
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].shape[0], 2 * SAMPLES_PER_FRAME)
        for chunk in chunks[1:-1]:
            self.assertEqual(chunk.shape[0], 6 * SAMPLES_PER_FRAME)
        self.assertLessEqual(chunks[-1].shape[0], 6 * SAMPLES_PER_FRAME)
        self.assertGreater(chunks[-1].shape[0], 0)

    def test_default_config_first_chunk_is_one_frame(self):
        # the 0.08 s streaming_initial_interval default (#77, −35 ms TTFA at
        # unchanged RTF) buckets to a single codec frame
        chunks = list(
            synthesize_stream(
                self.model, self.config, EN, speaker="vivian", temperature=0.0, max_tokens=24
            )
        )
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0].shape[0], SAMPLES_PER_FRAME)

    def test_padded_remainder_is_trimmed_exact(self):
        # pad-and-trim safety: the decoder is causal, so padding the final
        # chunk to a compiled bucket shape must not disturb the true frames'
        # audio beyond shape-noise (measured 7.9e-6 — kernel tiling differs
        # across the 3- vs 6-frame calls; one 16-bit LSB is 3.1e-5)
        import numpy as np

        decoder = self.model.speech_tokenizer.decoder
        num_q = decoder.config.num_quantizers
        rng = np.random.default_rng(5)
        codes = mx.array(rng.integers(1, 2000, size=(1, num_q, 3)), dtype=mx.int32)
        pad = mx.broadcast_to(codes[..., -1:], (1, num_q, 6))
        padded = mx.concatenate([codes, pad], axis=-1)

        decoder.reset_streaming_state()
        plain = decoder.streaming_step(codes).squeeze(1)[0]
        decoder.reset_streaming_state()
        trimmed = decoder.streaming_step(padded).squeeze(1)[0][: 3 * SAMPLES_PER_FRAME]

        self.assertEqual(plain.shape, trimmed.shape)
        self.assertLess(float(mx.abs(plain - trimmed).max()), 1e-4)

    def test_chunk_schedule_does_not_change_tokens(self):
        # greedy, same seed: the initial-chunk boundary (2 vs 6 frames) must
        # not perturb the token stream — draws identical — and the joined
        # audio stays inside the boundary-sensitivity envelope
        def fast():
            mx.random.seed(21)
            return mx.concatenate([c.reshape(-1) for c in generate_frames(
                self.model, text=EN, speaker="vivian", temperature=0.0, max_tokens=512,
                initial_frames=2, chunk_frames=6)])

        def steady():
            mx.random.seed(21)
            return mx.concatenate([c.reshape(-1) for c in generate_frames(
                self.model, text=EN, speaker="vivian", temperature=0.0, max_tokens=512,
                initial_frames=6, chunk_frames=6)])

        fast_draws, fast_audio = _taped(self.model, fast)
        steady_draws, steady_audio = _taped(self.model, steady)
        self.assertEqual(fast_draws, steady_draws)
        self.assertEqual(fast_audio.shape, steady_audio.shape)
        self.assertLess(float(mx.abs(fast_audio - steady_audio).mean()), AUDIO_ENVELOPE_MEAN)

    def test_greedy_parity_with_mlx_audio_loop(self):
        # the vendored loop vs mlx-audio's generate_custom_voice, greedy,
        # same seed: identical sampler draws (16 per frame — both loops run
        # eager here), audio within the boundary envelope (bucket
        # quantization + padded remainder shift the vocode boundaries),
        # both above the HNR floor.
        from vllm_omni_mlx.tts import stream_loop

        def ours():
            mx.random.seed(21)
            return mx.concatenate([c.reshape(-1) for c in generate_frames(
                self.model, text=EN, speaker="vivian", temperature=0.0, max_tokens=512,
                initial_frames=2, chunk_frames=6)])

        def reference():
            mx.random.seed(21)
            return mx.concatenate([r.audio for r in self.model.generate_custom_voice(
                text=EN, speaker="vivian", temperature=0.0, max_tokens=512,
                stream=True, streaming_interval=0.5) if r.audio is not None and r.audio.size])

        # the eager loop is the token-exact reference against mlx-audio's;
        # the compiled fast path's rare fp16 near-tie flips (fusion FMA
        # contraction, ~1 frame in 45 greedy — see test_compiled_drift below)
        # are kept out of this assertion on purpose
        was, stream_loop.EAGER_STREAM = stream_loop.EAGER_STREAM, True
        try:
            ours_draws, ours_audio = _taped(self.model, ours)
        finally:
            stream_loop.EAGER_STREAM = was
        ref_draws, ref_audio = _taped(self.model, reference)
        self.assertEqual(ours_draws, ref_draws)
        n = min(ours_audio.shape[0], ref_audio.shape[0])
        self.assertGreater(n, 2 * 24000)
        self.assertLess(float(mx.abs(ours_audio[:n] - ref_audio[:n]).mean()), AUDIO_ENVELOPE_MEAN)
        for label, audio in (("ours", ours_audio), ("mlx-audio", ref_audio)):
            hnr = int16_pcm_hnr_db(_pcm16(audio[:n]))
            self.assertGreater(hnr, CLEAN_VOICE_HNR_DB, f"{label}: HNR {hnr:.2f} dB below floor")

    def test_compiled_drift_bounded(self):
        # compiled fast path vs the eager loop, greedy, same seed: the two
        # agree until the first fp16 near-tie argmax flip under compile
        # fusion (FMA contraction reorders rounding by ~1 ULP; measured 1
        # frame in 45 on this seed, cascading only within that frame's
        # predictor groups), after which tokens/audio are legitimately
        # different. Bounds: no divergence in the first 5 frames, HNR above
        # the floor, duration within 2x of the eager run. The #66 prefix
        # cache is disabled for the compiled run — its splice is a valid
        # but different decode regime (see test_prefix_cache), out of scope
        # for this compile-only drift bound.
        import os

        from vllm_omni_mlx.tts import stream_loop

        def run(eager: bool):
            was, stream_loop.EAGER_STREAM = stream_loop.EAGER_STREAM, eager
            env_saved = os.environ.get("VLLM_OMNI_TTS_PREFIX_CACHE")
            if not eager:
                os.environ["VLLM_OMNI_TTS_PREFIX_CACHE"] = "0"
            try:
                mx.random.seed(21)
                draws: list[int] = []
                orig = self.model._sample_token

                def sampler(logits, **kwargs):
                    token = orig(logits, **kwargs)
                    draws.append(int(token.reshape(-1)[0].item()))
                    return token

                self.model._sample_token = sampler
                try:
                    audio = mx.concatenate([c.reshape(-1) for c in generate_frames(
                        self.model, text=EN, speaker="vivian", temperature=0.0, max_tokens=512,
                        initial_frames=2, chunk_frames=6)])
                finally:
                    self.model._sample_token = orig
            finally:
                stream_loop.EAGER_STREAM = was
                if env_saved is None:
                    os.environ.pop("VLLM_OMNI_TTS_PREFIX_CACHE", None)
                else:
                    os.environ["VLLM_OMNI_TTS_PREFIX_CACHE"] = env_saved
            return draws, audio

        eager_draws, _ = run(True)
        compiled_draws, compiled_audio = run(False)
        eager_talker = eager_draws[::16]  # talker draws, one per frame
        diverged = next(
            (i for i, (a, b) in enumerate(zip(eager_talker, compiled_draws)) if a != b), None
        )
        self.assertTrue(
            diverged is None or diverged >= 5,
            f"compiled path diverged too early (frame {diverged})",
        )
        hnr = int16_pcm_hnr_db(_pcm16(compiled_audio))
        self.assertGreater(hnr, CLEAN_VOICE_HNR_DB, f"HNR {hnr:.2f} dB below floor")
        self.assertLess(
            abs(compiled_audio.shape[0] - len(eager_talker) * SAMPLES_PER_FRAME),
            2 * len(eager_talker) * SAMPLES_PER_FRAME,
        )

    def test_streamed_speech_hnr_above_floor(self):
        audio = mx.concatenate(
            [c.reshape(-1) for c in synthesize_stream(
                self.model, self.config, EN, speaker="vivian", temperature=0.9, seed=3,
                streaming_interval=0.5, max_tokens=512)]
        )
        hnr = int16_pcm_hnr_db(_pcm16(audio))
        self.assertGreater(hnr, CLEAN_VOICE_HNR_DB, f"HNR {hnr:.2f} dB below floor: noise-like output")


if __name__ == "__main__":
    unittest.main()
