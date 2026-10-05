"""Per-voice prefix caches (#66): Level-1 prompt pieces, Level-2 voice-prefix
KV splice, and their serving semantics.

The load-bearing assertions:

- **Level 1 is bit-exact by construction**: the assembled prompt (all three
  ``_prepare_generation_inputs`` outputs) equals mlx-audio's direct call to
  the last bit — the cached pieces are the very arrays a fresh call would
  recompute, so caching them cannot perturb anything.
- **Level 2 is reproducible**: a cache miss computes exactly what a hit
  replays (one batched forward over the static rows + the first-text row
  through the compiled decode), so the same (voice, text, seed) yields
  bitwise-identical audio on the first and every later request. Streams are
  NOT draw-identical to the uncached path — the first-text row moves between
  mlx-audio's multi-row prefill kernels and the single-row decode kernels,
  and fp16 chaos amplifies that within a few frames (measured: divergence
  from frame ~2, 32–35 of 44 talker draws) — which is why the uncached
  reference is kept (eager loop never caches; the env kill switch) and the
  correctness bar here is HNR, the catastrophic-decode detector.
- the eager loop (the #43 mlx-audio-mirror oracle) never touches either
  cache level.
"""

import gc
import os
import unittest
from array import array

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx

from tests._teardown import ReleaseAfterClass
from tests.audio_metrics import CLEAN_VOICE_HNR_DB, int16_pcm_hnr_db

from vllm_omni_mlx.tts import prefix_cache
from vllm_omni_mlx.tts.config import DEFAULT_MODEL, TTSConfig, load_tts_model, local_snapshot
from vllm_omni_mlx.tts.prompt_embeds import PromptEmbeds
from vllm_omni_mlx.tts.stream_loop import generate_frames, prewarm_streaming

EN = "The quick brown fox jumps over the lazy dog."
ZH = "今天天气真不错，我们一起去公园散步吧。"


def _pcm16(audio: mx.array) -> bytes:
    return array("h", (mx.clip(audio, -1.0, 1.0) * 32767.0).astype(mx.int16).tolist()).tobytes()


def _greedy(model, text, speaker, language="auto", instruct=None, max_tokens=200):
    mx.random.seed(21)
    return mx.concatenate(
        [c.reshape(-1) for c in generate_frames(
            model, text=text, speaker=speaker, language=language, instruct=instruct,
            temperature=0.0, max_tokens=max_tokens)]
    )


class VoiceKeyTest(unittest.TestCase):
    def test_normalizes_case_and_empties(self):
        self.assertEqual(
            prefix_cache.voice_key("Vivian", "Auto", None),
            prefix_cache.voice_key("vivian", "auto", ""),
        )
        self.assertEqual(prefix_cache.voice_key(None, "EN", "  "), ("", "en", ""))
        self.assertEqual(prefix_cache.voice_key("vivian", "auto", " Calm. "), ("vivian", "auto", "Calm."))

    def test_distinct_keys(self):
        self.assertNotEqual(
            prefix_cache.voice_key("vivian", "auto", None),
            prefix_cache.voice_key("vivian", "en", None),
        )
        self.assertNotEqual(
            prefix_cache.voice_key("vivian", "auto", None),
            prefix_cache.voice_key("vivian", "auto", "cheerful"),
        )


class _DummyModel:
    """Weak-referenceable stand-in for a loaded model (object() is not)."""


class PrefixStoreTest(unittest.TestCase):
    """Unit level: LRU order, bounds, accounting, kill switch, lifetime."""

    def setUp(self):
        prefix_cache.clear_prefix_caches()
        self._env = os.environ.pop("VLLM_OMNI_TTS_PREFIX_CACHE", None)

    def tearDown(self):
        if self._env is not None:
            os.environ["VLLM_OMNI_TTS_PREFIX_CACHE"] = self._env
        prefix_cache.clear_prefix_caches()

    def test_enabled_by_default_and_switchable(self):
        self.assertTrue(prefix_cache.prefix_cache_enabled())
        for off in ("0", "false"):
            os.environ["VLLM_OMNI_TTS_PREFIX_CACHE"] = off
            self.assertFalse(prefix_cache.prefix_cache_enabled())
        os.environ["VLLM_OMNI_TTS_PREFIX_CACHE"] = "1"
        self.assertTrue(prefix_cache.prefix_cache_enabled())

    def test_store_lookup_roundtrip_and_lru_eviction(self):
        model = _DummyModel()
        def put(key, tag):
            arr = [mx.full((1, 2, 3, 4), float(tag))]
            prefix_cache.store_prefix_kv(model, *key, keys=arr, values=arr, length=3)

        k1, k2, k3 = ("vivian", "auto", ""), ("ryan", "auto", ""), ("vivian", "en", "")
        put(k1, 1)
        put(k2, 2)
        put(k3, 3)
        stats = prefix_cache.prefix_cache_stats(model)
        self.assertEqual(stats["entries"], 3)

        # touch k1, then fill to the cap (default 8): the two least
        # recently used (k2, k3) are evicted, k1 (recently touched) survives
        self.assertIsNotNone(prefix_cache.lookup_prefix_kv(model, *k1))
        for i in range(7):
            put((f"spk{i}", "auto", ""), i)
        self.assertIsNotNone(prefix_cache.lookup_prefix_kv(model, *k1))
        self.assertIsNone(prefix_cache.lookup_prefix_kv(model, *k2))
        self.assertIsNone(prefix_cache.lookup_prefix_kv(model, *k3))
        self.assertEqual(prefix_cache.prefix_cache_stats(model)["entries"], 8)

    def test_lookup_and_store_respect_the_kill_switch(self):
        os.environ["VLLM_OMNI_TTS_PREFIX_CACHE"] = "0"
        model = _DummyModel()
        arr = [mx.zeros((1, 1, 1, 1))]
        prefix_cache.store_prefix_kv(model, "vivian", "auto", None, arr, arr, 1)
        self.assertIsNone(prefix_cache.lookup_prefix_kv(model, "vivian", "auto", None))
        self.assertEqual(prefix_cache.prefix_cache_stats(model)["entries"], 0)

    def test_bytes_accounting_counts_stored_arrays(self):
        model = _DummyModel()
        keys = [mx.zeros((1, 2, 8, 4), dtype=mx.float16)]
        values = [mx.zeros((1, 2, 8, 4), dtype=mx.float16)]
        prefix_cache.store_prefix_kv(model, "vivian", "auto", None, keys, values, 8)
        stats = prefix_cache.prefix_cache_stats(model)
        self.assertEqual(stats["bytes"], 2 * 2 * 8 * 4 * 2)  # K+V, fp16

    def test_store_released_with_the_model(self):
        model = _DummyModel()
        arr = [mx.zeros((1, 1, 1, 1))]
        prefix_cache.store_prefix_kv(model, "vivian", "auto", None, arr, arr, 1)
        self.assertEqual(prefix_cache.prefix_cache_stats(model)["entries"], 1)
        del model
        gc.collect()
        self.assertEqual(prefix_cache.prefix_cache_stats(_DummyModel())["entries"], 0)


class PrefixCacheWeightsTest(ReleaseAfterClass, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if local_snapshot(DEFAULT_MODEL) is None:
            raise unittest.SkipTest(f"{DEFAULT_MODEL} not cached locally")
        cls.model = load_tts_model(TTSConfig())
        prewarm_streaming(cls.model, 0.5, 0.2)

    def setUp(self):
        prefix_cache.clear_prefix_caches()
        self._env = os.environ.pop("VLLM_OMNI_TTS_PREFIX_CACHE", None)

    def tearDown(self):
        if self._env is not None:
            os.environ["VLLM_OMNI_TTS_PREFIX_CACHE"] = self._env
        prefix_cache.clear_prefix_caches()

    def test_level1_assembly_bit_identical_to_mlx_audio(self):
        builder = PromptEmbeds(self.model)
        for speaker, language, instruct in (
            ("vivian", "auto", None),
            ("ryan", "en", None),
            ("vivian", "en", None),
            ("vivian", "auto", "Speak with cheerful energy"),
        ):
            layout = builder.build(EN, speaker=speaker, language=language, instruct=instruct)
            embeds, trailing, pad = self.model._prepare_generation_inputs(
                text=EN, language=language, speaker=speaker, instruct=instruct
            )
            with self.subTest(speaker=speaker, language=language, instruct=instruct):
                self.assertEqual(float(mx.abs(layout.input_embeds - embeds).max()), 0.0)
                self.assertEqual(float(mx.abs(layout.trailing_text_hidden - trailing).max()), 0.0)
                self.assertEqual(float(mx.abs(layout.decode_text_embed - pad).max()), 0.0)

    def test_level1_second_build_reuses_identical_piece_arrays(self):
        builder = PromptEmbeds(self.model)
        first = builder.build(EN, speaker="vivian")
        pieces1 = prefix_cache.prompt_pieces(self.model, "vivian", "auto", None)
        second = builder.build(ZH, speaker="vivian")  # different text, same voice
        pieces2 = prefix_cache.prompt_pieces(self.model, "vivian", "auto", None)
        self.assertIs(pieces1, pieces2)  # cached, not recomputed
        self.assertEqual(first.codec_prefix_len, second.codec_prefix_len)
        # the codec prefix rows are shared, the text rows are not
        n = first.prefix_rows
        self.assertEqual(float(mx.abs(first.input_embeds[:, :n] - second.input_embeds[:, :n]).max()), 0.0)
        self.assertGreater(
            float(mx.abs(first.input_embeds[:, -1] - second.input_embeds[:, -1]).max()), 0.0
        )

    def test_level2_miss_and_hit_streams_bitwise_identical(self):
        # the #66 bar, satisfied within the cached system: the first request
        # (miss) computes exactly what later requests (hits) replay, so the
        # same voice + text + seed is reproducible bitwise — across voices,
        # languages, and instructs
        for speaker, language, instruct, text in (
            ("vivian", "auto", None, EN),
            ("vivian", "auto", None, ZH),
            ("ryan", "en", None, EN),
            ("vivian", "auto", "Speak with cheerful energy", EN),
        ):
            prefix_cache.clear_prefix_caches()  # cases 1+2 share a voice key
            with self.subTest(speaker=speaker, language=language, instruct=instruct):
                entry = prefix_cache.lookup_prefix_kv(self.model, speaker, language, instruct)
                self.assertIsNone(entry)  # cold
                first = _greedy(self.model, text, speaker, language, instruct)
                entry = prefix_cache.lookup_prefix_kv(self.model, speaker, language, instruct)
                self.assertIsNotNone(entry, "miss did not populate the store")
                second = _greedy(self.model, text, speaker, language, instruct)
                third = _greedy(self.model, text, speaker, language, instruct)
                self.assertEqual(first.shape, second.shape)
                self.assertEqual(float(mx.abs(first - second).max()), 0.0)
                self.assertEqual(float(mx.abs(second - third).max()), 0.0)

    def test_level2_entry_geometry_and_footprint(self):
        _greedy(self.model, EN, "vivian")
        stats = prefix_cache.prefix_cache_stats(self.model)
        self.assertEqual(stats["entries"], 1)
        # role(3) + codec prefix(6−1) = 8 static rows on Auto with a speaker;
        # ~0.9 MiB measured — bounded, per-voice, far under any paging scheme
        entry = prefix_cache.lookup_prefix_kv(self.model, "vivian", "auto", None)
        self.assertEqual(entry.length, 8)
        self.assertLess(stats["bytes"], 4 * 2**20)  # < 4 MiB

    def test_kill_switch_restores_the_uncached_prefill(self):
        os.environ["VLLM_OMNI_TTS_PREFIX_CACHE"] = "0"
        a = _greedy(self.model, EN, "vivian")
        b = _greedy(self.model, EN, "vivian")
        self.assertEqual(prefix_cache.prefix_cache_stats(self.model)["entries"], 0)
        self.assertEqual(float(mx.abs(a - b).max()), 0.0)  # deterministic uncached path

    def test_cached_stream_hnr_above_floor(self):
        mx.random.seed(3)
        audio = mx.concatenate(
            [c for c in generate_frames(
                self.model, text=EN, speaker="vivian", temperature=0.9, max_tokens=300)]
        )
        hnr = int16_pcm_hnr_db(_pcm16(audio))
        self.assertGreater(hnr, CLEAN_VOICE_HNR_DB, f"HNR {hnr:.2f} dB below floor: noise-like output")

    def test_eager_loop_never_touches_the_caches(self):
        from vllm_omni_mlx.tts import stream_loop

        was, stream_loop.EAGER_STREAM = stream_loop.EAGER_STREAM, True
        try:
            a = _greedy(self.model, EN, "vivian")
            b = _greedy(self.model, EN, "vivian")
        finally:
            stream_loop.EAGER_STREAM = was
        self.assertEqual(prefix_cache.prefix_cache_stats(self.model)["entries"], 0)
        self.assertEqual(float(mx.abs(a - b).max()), 0.0)

    def test_warm_voice_prefix_builds_the_entry_a_miss_would(self):
        from vllm_omni_mlx.tts.stream_loop import warm_voice_prefix

        prefix_cache.clear_prefix_caches()
        self.assertTrue(warm_voice_prefix(self.model, "vivian"))
        entry = prefix_cache.lookup_prefix_kv(self.model, "vivian", "auto", None)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.length, 8)
        # every entry originates from the canonical build (see
        # stream_loop._PREFIX_TEXT), so a stream served from the warmed
        # entry matches one served from a request-built entry bitwise
        a = _greedy(self.model, EN, "vivian")
        prefix_cache.clear_prefix_caches()
        b = _greedy(self.model, EN, "vivian")
        self.assertEqual(float(mx.abs(a - b).max()), 0.0)

    def test_service_boot_prefills_the_default_voice(self):
        from vllm_omni_mlx.tts.service import TTSService

        prefix_cache.clear_prefix_caches()
        TTSService(self.model)
        self.assertIsNotNone(
            prefix_cache.lookup_prefix_kv(self.model, "vivian", "auto", None)
        )


if __name__ == "__main__":
    unittest.main()
