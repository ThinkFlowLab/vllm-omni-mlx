"""0.6B-CustomVoice validation (#48) — weight-gated battery: buffered +
streamed preset synthesis, dialect speaker, and the per-voice HNR floors
calibrated for the small model (not reused from 1.7B).

Calibration (fixed sentence, seeds 1–3 + greedy, this M4): vivian 2.41–3.10,
sohee 1.57–4.63, aiden 1.07–1.94 → CLEAN floor 0 dB; ryan −2.69–1.21 and
eric −1.74–1.47 → CATASTROPHIC bucket −5 dB — same two-bucket method as the
#37 calibration, small-model numbers."""

import os

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest

from tests._teardown import ReleaseAfterClass

from vllm_omni_mlx.tts.config import TTSConfig, load_tts_model, local_snapshot
from vllm_omni_mlx.tts.service import TTSService

SMALL = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
TEXT = "This sentence measures the small custom voice model end to end."


class SmallCustomVoiceE2ETest(ReleaseAfterClass, unittest.TestCase):
    """Runs where the 0.6B snapshot is cached; skips in CI — green CI does
    not mean this battery ran."""

    @classmethod
    def setUpClass(cls):
        if local_snapshot(SMALL) is None:
            raise unittest.SkipTest(f"{SMALL} not cached locally")
        cls.model = load_tts_model(TTSConfig(model_ref=SMALL))
        cls.service = TTSService(cls.model, TTSConfig(model_ref=SMALL))

    def test_serves_custom_voice_presets(self):
        self.assertEqual(self.service.model_type, "custom_voice")
        self.assertEqual(
            sorted(self.service.voices),
            ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"],
        )

    def _hnr(self, pcm: bytes) -> float:
        from tests.audio_metrics import int16_pcm_hnr_db

        return int16_pcm_hnr_db(pcm)

    def test_buffered_wav_is_speech(self):
        payload, content_type = self.service.speech_bytes(TEXT, voice="vivian")
        self.assertEqual(content_type, "audio/wav")
        self.assertGreater(len(payload), 2 * 24000)  # >1s of 24 kHz 16-bit mono
        self.assertGreater(self._hnr(payload[44:]), 0.0, "vivian below calibrated 0 dB floor")

    def test_streamed_pcm_is_speech(self):
        chunks = list(self.service.speech_stream(TEXT, voice="sohee", streaming_interval=0.5))
        self.assertGreater(len(chunks), 1, "expected several chunks, not one buffered blob")
        pcm = b"".join(chunks)
        self.assertGreater(len(pcm), 2 * 24000)
        self.assertGreater(self._hnr(pcm), 0.0, "sohee below calibrated 0 dB floor")

    def test_low_hnr_voices_clear_catastrophic_floor(self):
        # ryan and eric calibrate below the clean floor on the small model
        # (#37 method: catastrophic bucket, not silence)
        for voice in ("ryan", "eric"):
            payload, _ = self.service.speech_bytes(TEXT, voice=voice)
            self.assertGreater(
                self._hnr(payload[44:]), -5.0, f"{voice} below catastrophic floor: noise-like output"
            )

    def test_dialect_speaker_serves(self):
        # eric = sichuan dialect per spk_is_dialect; synthesis must still
        # resolve the dialect language id and produce speech
        payload, _ = self.service.speech_bytes("今天天气真不错。", voice="eric")
        self.assertGreater(len(payload), 24000)
        self.assertGreater(self._hnr(payload[44:]), -5.0)

    def test_instructions_rejected_on_small_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "1.7B CustomVoice"):
            self.service.speech_bytes(TEXT, voice="vivian", instructions="very happy")


if __name__ == "__main__":
    unittest.main()
