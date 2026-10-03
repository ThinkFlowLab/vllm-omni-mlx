"""VoiceDesign serving (#51): the `design` generation path — instructions is
the voice description, `voice` rejected, streaming names #52. Stub tests
need no weights; the e2e class is weight-gated on the 4-bit checkpoint and
carries an HNR floor calibrated per the #37 method (fixed description +
seed, not a preset speaker — designed voices are different speakers, preset
floors don't transfer).
"""

import io
import unittest
import wave
from types import SimpleNamespace

import mlx.core as mx

from tests.audio_metrics import wav_hnr_db
from vllm_omni_mlx.tts.config import TTSConfig, load_tts_model, local_snapshot
from vllm_omni_mlx.tts.generate import synthesize, synthesize_design, wav_bytes
from vllm_omni_mlx.tts.service import TTSService

VD_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"
EN = "The quick brown fox jumps over the lazy dog."
DESC = "A cheerful young female voice with high pitch and energetic tone."
PRESETS = {"vivian": 3065, "ryan": 3061}

# Calibrated 10-03 on M4 (greedy, seed 7, this description+text): HNR 1.96 dB
# across 3 runs, 0.00 dB drift (greedy is deterministic here; #38 measured
# ~1 dB kernel-nondeterminism bound on CustomVoice). White noise sits near
# -10 dB (audio_metrics). Same floor as vivian's preset with ~2 dB headroom.
DESIGN_HNR_FLOOR_DB = 0.0


def stub_model(tts_model_type, spk_id=None):
    """Minimal model surface for the seams touched before generation
    (mirrors tests/test_variants.py)."""
    return SimpleNamespace(
        config=SimpleNamespace(
            tts_model_type=tts_model_type,
            talker_config=SimpleNamespace(
                spk_id=spk_id,
                codec_language_id={"chinese": 2055, "english": 2050},
            ),
        ),
    )


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


class DesignEntryGuardTest(unittest.TestCase):
    """Guards fire before model internals are touched — the stubs carry a
    generate_voice_design that would fail loudly if misrouted."""

    def _design_stub(self):
        model = stub_model("voice_design")
        seen = {}

        def fake_generate_voice_design(**kw):
            seen.update(kw)
            return iter([SimpleNamespace(audio=mx.zeros((240,)))])

        model.generate_voice_design = fake_generate_voice_design
        return model, seen

    def test_routes_to_generate_voice_design_with_instruct(self):
        model, seen = self._design_stub()
        chunks = list(
            synthesize_design(model, TTSConfig(instruct=DESC), EN, temperature=0.0)
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(seen["text"], EN)
        self.assertEqual(seen["instruct"], DESC)
        self.assertNotIn("speaker", seen, "design prompt must carry no speaker")

    def test_missing_instruct_is_a_request_error(self):
        model, _ = self._design_stub()
        with self.assertRaisesRegex(ValueError, "voice description"):
            list(synthesize_design(model, TTSConfig(), EN))

    def test_other_variants_not_served_on_design_path(self):
        with self.assertRaisesRegex(ValueError, "design path"):
            list(
                synthesize_design(
                    stub_model("custom_voice", PRESETS), TTSConfig(instruct=DESC), EN
                )
            )

    def test_preset_entry_still_rejects_voice_design(self):
        with self.assertRaisesRegex(ValueError, "instructions"):
            list(synthesize(stub_model("voice_design"), TTSConfig(), EN))


class DesignServiceTest(unittest.TestCase):
    def _service(self):
        model = stub_model("voice_design")
        model.generate_voice_design = lambda **kw: iter(
            [SimpleNamespace(audio=mx.zeros((240,)))]
        )
        return TTSService(model)

    def test_preset_voice_rejected_with_guidance(self):
        with self.assertRaisesRegex(ValueError, "instructions"):
            self._service().speech_bytes(EN, voice="vivian", instructions=DESC)

    def test_instructions_required(self):
        with self.assertRaisesRegex(ValueError, "required"):
            self._service().speech_bytes(EN)

    def test_synthesizes_wav_through_service(self):
        payload, ctype = self._service().speech_bytes(EN, instructions=DESC)
        self.assertEqual(ctype, "audio/wav")
        self.assertEqual(payload[:4], b"RIFF")

    def test_language_override_flows(self):
        service = self._service()
        seen = {}

        def fake(**kw):
            seen.update(kw)
            return iter([SimpleNamespace(audio=mx.zeros((240,)))])

        service._model.generate_voice_design = fake
        service.speech_bytes(EN, instructions=DESC, language="english")
        self.assertEqual(seen["language"], "english")

    def test_streaming_names_its_issue(self):
        with self.assertRaisesRegex(ValueError, "#52"):
            self._service().speech_stream(EN, instructions=DESC)


class VoiceDesignWeightGatedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if local_snapshot(VD_MODEL) is None:
            raise unittest.SkipTest(f"{VD_MODEL} not cached locally")
        cls.model = load_tts_model(TTSConfig(model_ref=VD_MODEL))
        cls.config = TTSConfig(model_ref=VD_MODEL)
        cls.service = TTSService(cls.model, cls.config)

    def _wav(self, text=EN, **overrides):
        overrides.setdefault("instruct", DESC)
        return wav_bytes(synthesize_design(self.model, self.config, text, **overrides))

    def test_checkpoint_is_voice_design_with_no_presets(self):
        self.assertEqual(self.service.model_type, "voice_design")
        self.assertEqual(self.service.voices, [])

    def test_greedy_described_voice_passes_calibrated_hnr_floor(self):
        data = self._wav(temperature=0.0, seed=7)
        seconds, rms, peak = wav_stats(data)
        self.assertGreater(seconds, 1.0)
        self.assertGreater(rms, 0.01, "near-silence: decode path broken")
        self.assertLess(rms, 0.5, "full-scale energy: precision artifact")
        self.assertLessEqual(peak, 1.0)
        hnr = wav_hnr_db(data)
        self.assertGreaterEqual(
            hnr,
            DESIGN_HNR_FLOOR_DB,
            f"HNR {hnr:.2f}dB below the design-voice floor — decode degraded",
        )

    def test_greedy_hnr_stable_across_reruns(self):
        # #38's kernel-nondeterminism bound; the 10-03 calibration observed
        # 0.00 dB drift on this exact pair
        hnr_a = wav_hnr_db(self._wav(temperature=0.0, seed=7))
        hnr_b = wav_hnr_db(self._wav(temperature=0.0, seed=7))
        self.assertLess(abs(hnr_a - hnr_b), 1.5)

    def test_service_end_to_end_wav(self):
        # service path (default sampling, unseeded): shape + plausibility
        payload, ctype = self.service.speech_bytes(EN, instructions=DESC)
        self.assertEqual(ctype, "audio/wav")
        seconds, rms, _ = wav_stats(payload)
        self.assertGreater(seconds, 1.0)
        self.assertLess(seconds, 30.0, "no-EOS runaway (spike 10-03 hit max_tokens)")
        self.assertGreater(rms, 0.01)
        self.assertLess(rms, 0.5)

    def test_audition_file_for_human_listening(self):
        # writes under /tmp for the pending human-audition pass; not an
        # assertion about the audio itself
        data = self._wav(temperature=0.0, seed=7)
        with open("/tmp/vd_e2e_vivian_design.wav", "wb") as fh:
            fh.write(data)
        self.assertGreater(len(data), 24000 * 2)


if __name__ == "__main__":
    unittest.main()
