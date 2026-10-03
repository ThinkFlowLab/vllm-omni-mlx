"""Variant dispatch scaffold (#47, extended per-path in #49): the
tts_model_type probe, the served-guard per generation path (preset vs
clone), and per-type service validation — all against stub models, no
weights needed."""

import unittest
from types import SimpleNamespace

from vllm_omni_mlx.tts import variants
from vllm_omni_mlx.tts.config import TTSConfig
from vllm_omni_mlx.tts.generate import synthesize
from vllm_omni_mlx.tts.service import TTSService
from vllm_omni_mlx.tts.stream_loop import synthesize_stream

PRESETS = {"vivian": 3065, "ryan": 3061}


def stub_model(tts_model_type, spk_id=None):
    """Minimal model surface for the seams touched before generation:
    variants probe reads config.tts_model_type, PromptEmbeds reads
    talker_config.spk_id / codec_language_id."""
    return SimpleNamespace(
        config=SimpleNamespace(
            tts_model_type=tts_model_type,
            talker_config=SimpleNamespace(
                spk_id=spk_id,
                codec_language_id={"chinese": 2055, "english": 2050},
            ),
        ),
    )


class ModelVariantTest(unittest.TestCase):
    def test_known_types_round_trip(self):
        for variant in variants.KNOWN:
            self.assertEqual(variants.model_variant(stub_model(variant)), variant)

    def test_normalizes_case_and_whitespace(self):
        self.assertEqual(variants.model_variant(stub_model(" Custom_Voice ")), "custom_voice")

    def test_unknown_type_raises_with_raw_value(self):
        with self.assertRaisesRegex(ValueError, "'turbo'"):
            variants.model_variant(stub_model("turbo"))

    def test_missing_type_raises(self):
        model = SimpleNamespace(config=SimpleNamespace())
        with self.assertRaisesRegex(ValueError, "unknown TTS model type None"):
            variants.model_variant(model)


class EnsureServedTest(unittest.TestCase):
    def test_custom_voice_passes(self):
        model = stub_model("custom_voice", PRESETS)
        self.assertEqual(variants.ensure_served(model), "custom_voice")

    def test_base_preset_path_points_at_cloning_shape(self):
        with self.assertRaisesRegex(ValueError, "ref_audio"):
            variants.ensure_served(stub_model("base"))

    def test_base_clone_path_passes(self):
        model = stub_model("base")
        self.assertEqual(variants.ensure_served(model, path="clone"), "base")

    def test_custom_voice_clone_path_needs_base(self):
        with self.assertRaisesRegex(ValueError, "needs a Base checkpoint"):
            variants.ensure_served(stub_model("custom_voice", PRESETS), path="clone")

    def test_unknown_path_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown generation path"):
            variants.require_served("base", path="bogus")

    def test_voice_design_preset_path_points_at_instructions(self):
        with self.assertRaisesRegex(ValueError, "instructions"):
            variants.ensure_served(stub_model("voice_design"))


class GenerationEntryGuardTest(unittest.TestCase):
    """The guards must fire before any model internals are touched — the
    stubs have no generation methods, so reaching past the guard would fail
    with AttributeError, not ValueError."""

    def test_synthesize_rejects_base(self):
        with self.assertRaisesRegex(ValueError, "no preset voices"):
            list(synthesize(stub_model("base"), TTSConfig(), "hello"))

    def test_synthesize_stream_rejects_voice_design(self):
        with self.assertRaisesRegex(ValueError, "#52"):
            list(synthesize_stream(stub_model("voice_design"), TTSConfig(), "hello"))

    def test_synthesize_clone_rejects_custom_voice(self):
        import mlx.core as mx

        from vllm_omni_mlx.tts.generate import synthesize_clone

        with self.assertRaisesRegex(ValueError, "needs a Base checkpoint"):
            list(synthesize_clone(stub_model("custom_voice", PRESETS), TTSConfig(), "hi", mx.zeros(24000), "ref"))


class ServicePerTypeTest(unittest.TestCase):
    def test_unknown_type_fails_at_construction(self):
        with self.assertRaisesRegex(ValueError, "unknown TTS model type"):
            TTSService(stub_model("turbo", PRESETS))

    def test_base_serves_empty_voices_and_rejects_preset_speech(self):
        service = TTSService(stub_model("base"))
        self.assertEqual(service.voices, [])
        with self.assertRaisesRegex(ValueError, "ref_audio"):
            service.speech_bytes("hello")

    def test_voice_design_rejects_speech_even_with_instructions(self):
        service = TTSService(stub_model("voice_design"))
        with self.assertRaisesRegex(ValueError, "#52"):
            service.speech_stream("hello", instructions="a cheerful young voice")

    def test_custom_voice_validation_unchanged(self):
        service = TTSService(stub_model("custom_voice", PRESETS))
        self.assertEqual(sorted(service.voices), ["ryan", "vivian"])
        self.assertEqual(service.model_type, "custom_voice")
        with self.assertRaisesRegex(ValueError, "not one of the preset voices"):
            service.speech_bytes("hello", voice="chloe")


if __name__ == "__main__":
    unittest.main()
