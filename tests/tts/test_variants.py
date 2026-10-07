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


def stub_model(tts_model_type, spk_id=None, size="1b7"):
    """Minimal model surface for the seams touched before generation:
    variants probe reads config.tts_model_type (+ tts_model_size for the
    instruct policy), PromptEmbeds reads talker_config.spk_id /
    codec_language_id."""
    return SimpleNamespace(
        config=SimpleNamespace(
            tts_model_type=tts_model_type,
            tts_model_size=size,
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

    def test_base_clone_stream_path_passes(self):
        model = stub_model("base")
        self.assertEqual(variants.ensure_served(model, path="clone_stream"), "base")

    def test_custom_voice_clone_paths_need_base(self):
        for path in ("clone", "clone_stream"):
            with self.assertRaisesRegex(ValueError, "needs a Base checkpoint"):
                variants.ensure_served(stub_model("custom_voice", PRESETS), path=path)

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

    def test_synthesize_stream_design_needs_instruct(self):
        with self.assertRaisesRegex(ValueError, "voice description"):
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

    def test_voice_design_streaming_validates_then_generates(self):
        # #52: streaming design is served — validation passes and generation
        # starts; the stub has no engine, so the only failure left must not
        # be a service-level rejection
        service = TTSService(stub_model("voice_design"))
        stream = service.speech_stream("hello", instructions="a cheerful young voice")
        with self.assertRaises((AttributeError, TypeError)) as ctx:
            next(stream)
        self.assertNotIn("#52", str(ctx.exception))

    def test_custom_voice_validation_unchanged(self):
        service = TTSService(stub_model("custom_voice", PRESETS))
        self.assertEqual(sorted(service.voices), ["ryan", "vivian"])
        self.assertEqual(service.model_type, "custom_voice")
        with self.assertRaisesRegex(ValueError, "not one of the preset voices"):
            service.speech_bytes("hello", voice="chloe")

    def test_instructions_rejected_on_small_model(self):
        service = TTSService(stub_model("custom_voice", PRESETS, size="0b6"))
        with self.assertRaisesRegex(ValueError, "1.7B CustomVoice"):
            service.speech_bytes("hello", voice="vivian", instructions="very happy")

    def test_instructions_pass_validation_on_1_7b(self):
        service = TTSService(stub_model("custom_voice", PRESETS))
        overrides = service._validated_overrides("hello", "vivian", 1.0, "very happy", None)
        self.assertEqual(overrides, {"speaker": "vivian", "instruct": "very happy"})


if __name__ == "__main__":
    unittest.main()
