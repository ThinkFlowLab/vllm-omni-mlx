"""VoxCPM2 end-to-end battery (#71) — weight-gated: runs where the
mlx-community snapshot is cached, skips in CI (green CI does not mean this
battery ran). Zero-shot (default voice), Chinese, voice design via
``instructions``, cloning from the checkpoint's bundled reference clip, and
the stream path — 48 kHz throughout, HNR floors calibrated for the 4-bit
model on this M4 (see the constants below; the #37 two-bucket method: a
floor at the bottom of the clean range detects catastrophic decode, not
quality)."""

import base64
import io
import os

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest
import wave
from pathlib import Path

from tests._teardown import ReleaseAfterClass
from vllm_omni_mlx.tts.voxcpm2 import (
    VoxCPM2Config,
    VoxCPM2Service,
    is_voxcpm2_model,
    load_voxcpm2_model,
    local_snapshot,
)

MODEL = "mlx-community/VoxCPM2-4bit"
SR = 48000
TEXT_EN = "This sentence measures the VoxCPM2 model end to end."
TEXT_ZH = "这句话用来测试中文语音合成的效果。"
INSTRUCT = "A young woman with a warm and gentle voice"

#: calibrated floor (dB) — the #37 two-bucket method's catastrophic-decode
#: bucket: clean 4-bit output measured −3.8…3.1 dB across modes (zero-shot
#: en −0.8…2.0, zh −3.8…3.1 with one low draw, design 1.9…2.4, clone
#: −0.1…0.2; the noise reference is −10.1), so −5.0 separates garbage from
#: clean while tolerating the stochastic CFM draws
HNR_FLOOR_DB = -5.0


class VoxCPM2E2ETest(ReleaseAfterClass, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snapshot = local_snapshot(MODEL)
        if snapshot is None:
            raise unittest.SkipTest(f"{MODEL} not cached locally")
        cls.snapshot = Path(snapshot)
        config = VoxCPM2Config(model_ref=MODEL)
        cls.model = load_voxcpm2_model(config)
        cls.service = VoxCPM2Service(cls.model, config)

    def _hnr(self, pcm: bytes) -> float:
        from tests.audio_metrics import int16_pcm_hnr_db

        return int16_pcm_hnr_db(pcm, sr=SR)

    def _wav_rate(self, payload: bytes) -> int:
        with wave.open(io.BytesIO(payload)) as wav:
            return wav.getframerate()

    def test_model_is_detected_and_serves_default_voice(self):
        self.assertTrue(is_voxcpm2_model(self.model))
        self.assertEqual(self.service.model_type, "voxcpm2")
        self.assertEqual(self.service.voices, ["default"])
        self.assertEqual(self.service.sample_rate, SR)

    def test_zero_shot_english_wav_is_speech(self):
        payload, content_type = self.service.speech_bytes(TEXT_EN, voice="default")
        self.assertEqual(content_type, "audio/wav")
        self.assertEqual(self._wav_rate(payload), SR)
        self.assertGreater(len(payload), 2 * SR)  # >1s of 48 kHz 16-bit mono
        hnr = self._hnr(payload[44:])
        self.assertGreater(hnr, HNR_FLOOR_DB, f"zero-shot HNR {hnr:.2f} dB below floor: noise-like output")

    def test_zero_shot_chinese_wav_is_speech(self):
        payload, _ = self.service.speech_bytes(TEXT_ZH)
        self.assertEqual(self._wav_rate(payload), SR)
        self.assertGreater(len(payload), 2 * SR)
        hnr = self._hnr(payload[44:])
        self.assertGreater(hnr, HNR_FLOOR_DB, f"zh HNR {hnr:.2f} dB below floor: noise-like output")

    def test_voice_design_from_instructions(self):
        payload, _ = self.service.speech_bytes(TEXT_EN, instructions=INSTRUCT)
        self.assertGreater(len(payload), 2 * SR)
        hnr = self._hnr(payload[44:])
        self.assertGreater(hnr, HNR_FLOOR_DB, f"design HNR {hnr:.2f} dB below floor: noise-like output")

    def test_cloning_from_bundled_reference(self):
        # the checkpoint ships reference clips (test_en.wav); cloning is
        # ref-audio-driven — no transcript needed on VoxCPM2
        ref = (self.snapshot / "test_en.wav").read_bytes()
        payload, _ = self.service.speech_bytes(TEXT_EN, voice={"ref_audio": base64.b64encode(ref).decode()})
        self.assertGreater(len(payload), 2 * SR)
        hnr = self._hnr(payload[44:])
        self.assertGreater(hnr, HNR_FLOOR_DB, f"clone HNR {hnr:.2f} dB below floor: noise-like output")

    def test_stream_slices_pcm_chunks(self):
        chunks = list(self.service.speech_stream(TEXT_EN, streaming_interval=0.5))
        self.assertGreater(len(chunks), 1, "expected interval-sized chunks, not one blob")
        pcm = b"".join(chunks)
        self.assertEqual(len(pcm) % 2, 0)
        self.assertGreater(len(pcm), 2 * SR)
        hnr = self._hnr(pcm)
        self.assertGreater(hnr, HNR_FLOOR_DB, f"stream HNR {hnr:.2f} dB below floor: noise-like output")

    def test_pcm_format_is_speech(self):
        # a fresh generate() is stochastic (CFM noise), so pcm and wav calls
        # can differ in length — assert the pcm path on its own terms
        pcm_payload, content_type = self.service.speech_bytes(TEXT_EN, response_format="pcm")
        self.assertEqual(content_type, "audio/pcm")
        self.assertEqual(len(pcm_payload) % 2, 0)
        self.assertGreater(len(pcm_payload), 2 * SR)
        hnr = self._hnr(pcm_payload)
        self.assertGreater(hnr, HNR_FLOOR_DB, f"pcm HNR {hnr:.2f} dB below floor: noise-like output")

    def test_generation_in_a_worker_thread(self):
        # the server synthesizes in worker threads (asyncio.to_thread /
        # starlette iterate_in_threadpool); a CPU-pinned load-time buffer
        # made the first cross-thread decode raise "There is no Stream(cpu,
        # 1) in current thread" — regression guard for the _unpin fix
        import threading

        outcome: dict = {}

        def work():
            try:
                payload, _ = self.service.speech_bytes("Thread synthesis works.")
                outcome["bytes"] = len(payload)
            except Exception as exc:  # noqa: BLE001 — the failure mode under test
                outcome["error"] = str(exc)

        worker = threading.Thread(target=work)
        worker.start()
        worker.join()
        if "error" in outcome:
            self.fail(f"worker-thread synthesis failed: {outcome['error']}")
        self.assertGreater(outcome["bytes"], 2 * SR)


if __name__ == "__main__":
    unittest.main()
