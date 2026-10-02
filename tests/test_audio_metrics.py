"""HNR metric sanity (#37): periodic speech scores high, noise fails the
floor — the property that makes the assertion a catastrophic-decode detector."""

import unittest

import numpy as np

from tests.audio_metrics import CATASTROPHIC_HNR_DB, CLEAN_VOICE_HNR_DB, int16_pcm_hnr_db, pcm_hnr_db


def _tone(seconds: float, freq: float = 180.0, sr: int = 24000) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    wave = 0.6 * np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * freq * 2 * t)
    return wave.astype(np.float32)


def _noise(seconds: float, sr: int = 24000) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (0.3 * rng.standard_normal(int(seconds * sr))).astype(np.float32)


class HnrMetricTest(unittest.TestCase):
    def test_periodic_signal_passes_floor(self):
        self.assertGreater(pcm_hnr_db(_tone(2.0)), CLEAN_VOICE_HNR_DB)

    def test_white_noise_fails_catastrophic_floor(self):
        # white noise measures ~-10 dB; the catastrophic floor is the detector
        # the TTS tests gate on (issue #37 calibration)
        self.assertLess(pcm_hnr_db(_noise(2.0)), CATASTROPHIC_HNR_DB)

    def test_int16_roundtrip_matches_float(self):
        pcm = (_tone(1.0) * 32767).astype(np.int16).tobytes()
        hnr_int16 = int16_pcm_hnr_db(pcm)
        hnr_float = pcm_hnr_db(np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0)
        self.assertAlmostEqual(hnr_int16, hnr_float, places=3)

    def test_silence_scores_zero(self):
        self.assertEqual(pcm_hnr_db(np.zeros(24000, dtype=np.float32)), 0.0)


if __name__ == "__main__":
    unittest.main()
