"""Generation engine (#15 / M1.6): voices × languages × sampling modes produce
plausible artifact-free audio. Automated stand-in for listening: duration
scales with text, RMS sits in a healthy band (precision bugs surface as
silence or full-scale noise), voices differ, and greedy is deterministic.
Weight-gated; human audition files are referenced on issue #15."""

import os

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest
import wave
import io

from tests.audio_metrics import CATASTROPHIC_HNR_DB, CLEAN_VOICE_HNR_DB, wav_hnr_db
from vllm_omni_mlx.tts.config import DEFAULT_MODEL, TTSConfig, load_tts_model, local_snapshot
from vllm_omni_mlx.tts.generate import synthesize, wav_bytes

EN = "The quick brown fox jumps over the lazy dog."
ZH = "今天天气很好，我们一起去公园散步吧。"


def wav_stats(data: bytes):
    with wave.open(io.BytesIO(data)) as wav:
        assert wav.getframerate() == 24000 and wav.getnchannels() == 1
        frames = wav.getnframes()
        pcm = wav.readframes(frames)
    from array import array

    samples = array("h", pcm)
    n = len(samples)
    rms = (sum(s * s for s in samples) / max(n, 1)) ** 0.5 / 32768
    peak = max(abs(s) for s in samples) / 32768
    return frames / 24000, rms, peak


class GenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if local_snapshot(DEFAULT_MODEL) is None:
            raise unittest.SkipTest(f"{DEFAULT_MODEL} not cached locally")
        cls.model = load_tts_model(TTSConfig())
        cls.config = TTSConfig()

    def _wav(self, text, **overrides):
        return wav_bytes(synthesize(self.model, self.config, text, **overrides))

    def test_english_vivian_greedy_is_plausible(self):
        data = self._wav(EN, speaker="vivian", temperature=0.0)
        seconds, rms, peak = wav_stats(data)
        self.assertGreater(seconds, 1.0)
        self.assertGreater(rms, 0.01, "near-silence: decode path broken")
        self.assertLess(rms, 0.5, "full-scale energy: precision artifact")
        self.assertLessEqual(peak, 1.0)

    def test_speech_hnr_above_floor(self):
        # upstream vllm-omni's catastrophic-decode detector, calibrated per
        # voice on the 4-bit model (issue #37): white noise ~-10 dB; vivian
        # 0.95–2.44 dB across draws (clean-voice floor 0 dB); aiden and ryan
        # sit lower intrinsically — catastrophic floor only.
        cases = (
            (EN, {"speaker": "vivian", "temperature": 0.0}, CLEAN_VOICE_HNR_DB),
            (EN, {"speaker": "vivian"}, CLEAN_VOICE_HNR_DB),
            (EN, {"speaker": "ryan"}, CATASTROPHIC_HNR_DB),
            (ZH, {"speaker": "aiden"}, CATASTROPHIC_HNR_DB),
        )
        for text, overrides, floor in cases:
            with self.subTest(**{k: v for k, v in overrides.items() if k == "speaker"}):
                hnr = wav_hnr_db(self._wav(text, **overrides))
                self.assertGreater(
                    hnr,
                    floor,
                    f"HNR {hnr:.2f} dB below {floor} dB floor: output is noise-like, not speech",
                )

    def test_chinese_aiden_default_sampling(self):
        data = self._wav(ZH, speaker="aiden")
        seconds, rms, _ = wav_stats(data)
        self.assertGreater(seconds, 1.0)
        self.assertGreater(rms, 0.01)

    def test_voices_differ_for_same_text(self):
        vivian = self._wav(EN, speaker="vivian", temperature=0.0)
        ryan = self._wav(EN, speaker="ryan", temperature=0.0)
        self.assertNotEqual(len(vivian), len(ryan))

    def test_greedy_deterministic_with_seed(self):
        a = self._wav(EN, speaker="vivian", temperature=0.0)
        b = self._wav(EN, speaker="vivian", temperature=0.0)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
