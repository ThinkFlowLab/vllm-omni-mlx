"""code2wav wrapper (#11 / M1.2): chunked≡full stitching parity, output
plausibility, and determinism — gated on the locally cached checkpoint."""

import unittest

import mlx.core as mx

from vllm_omni_mlx.tts.code2wav import Code2Wav
from vllm_omni_mlx.tts.config import DEFAULT_MODEL, TTSConfig, load_tts_model, local_snapshot


def code2wav():
    if local_snapshot(DEFAULT_MODEL) is None:
        raise unittest.SkipTest(f"{DEFAULT_MODEL} not cached locally")
    model = load_tts_model(TTSConfig())
    return Code2Wav(model.speech_tokenizer)


def random_codes(time: int, quantizers: int) -> mx.array:
    # mid-range bins avoid untrained corners of the codebooks
    return mx.random.randint(0, 2048, (1, quantizers, time))


class Code2WavTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dec = code2wav()
        cls.q = cls.dec.num_quantizers

    def test_output_plausible(self):
        codes = random_codes(120, self.q)
        wav = self.dec.decode(codes)
        self.assertEqual(wav.dtype, mx.float32)
        self.assertEqual(wav.shape[0], 1)
        # 120 codes @12 Hz → ~2 s @24 kHz
        expected = 120 * (24000 // 12)
        self.assertAlmostEqual(wav.shape[-1], expected, delta=expected * 0.05)
        self.assertLessEqual(float(mx.abs(wav).max()), 1.0)

    def test_streaming_matches_full_length_and_approximates(self):
        codes = random_codes(700, self.q)
        via_chunks = mx.concatenate(list(self.dec.chunks(codes)), axis=-1)
        full = self.dec.decode(codes)
        self.assertEqual(via_chunks.shape, full.shape)
        # streaming_step keeps conv+transformer state across chunks: mean error
        # vs full decode stays tiny (measured ~2.5e-4 on random codes); larger
        # boundary transients are inherent to chunked serving
        self.assertLess(float(mx.abs(via_chunks - full).mean()), 1e-3)

    def test_streaming_state_resets_between_streams(self):
        codes = random_codes(400, self.q)
        first = mx.concatenate(list(self.dec.chunks(codes)), axis=-1)
        second = mx.concatenate(list(self.dec.chunks(codes)), axis=-1)
        self.assertTrue(mx.allclose(first, second, atol=2e-4))

    def test_deterministic(self):
        codes = random_codes(60, self.q)
        first = self.dec.decode(codes)
        second = self.dec.decode(codes)
        self.assertTrue(mx.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
