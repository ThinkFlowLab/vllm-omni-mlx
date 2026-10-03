"""TTS scaffold (#10): config defaults, [tts]-extra guard, and — when the
checkpoint is cached locally — the full-checkpoint load that validates
mlx-audio's tensor mapping end to end."""

import sys
import os

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest
from unittest import mock

from vllm_omni_mlx.tts.config import (
    DEFAULT_MODEL,
    DEFAULT_SPEAKER,
    TTSConfig,
    load_tts_model,
    local_snapshot,
)


class TTSConfigTest(unittest.TestCase):
    def test_defaults_match_qwen3_tts(self):
        config = TTSConfig()
        self.assertEqual(config.temperature, 0.9)
        self.assertEqual(config.top_k, 50)
        self.assertEqual(config.top_p, 1.0)
        self.assertEqual(config.repetition_penalty, 1.05)
        self.assertEqual(config.streaming_interval, 2.0)

    def test_with_overrides_ignores_none_and_unknown_and_model_ref(self):
        config = TTSConfig()
        overridden = config.with_overrides(speaker="Ryan", temperature=None, bogus=1, model_ref="other")
        self.assertEqual(overridden.speaker, "Ryan")
        self.assertEqual(overridden.model_ref, config.model_ref)
        self.assertNotIn("bogus", overridden.__dict__)

    def test_frozen(self):
        with self.assertRaises(Exception):
            TTSConfig().speaker = "Ryan"


class LoaderGuardTest(unittest.TestCase):
    def test_missing_mlx_audio_raises_install_hint(self):
        with mock.patch.dict(sys.modules, {"mlx_audio": None, "mlx_audio.tts": None, "mlx_audio.tts.utils": None}):
            with self.assertRaises(RuntimeError) as ctx:
                load_tts_model(TTSConfig())
        self.assertIn("vllm-omni-mlx[tts]", str(ctx.exception))


class CheckpointMappingTest(unittest.TestCase):
    """mlx-audio owns the checkpoint→MLX mapping; a full load of the model with
    strict weight application IS the no-orphan check. Runs where the snapshot
    is cached (~2.2 GiB), skips in CI."""

    @classmethod
    def setUpClass(cls):
        snapshot = local_snapshot(DEFAULT_MODEL)
        if snapshot is None:
            raise unittest.SkipTest(f"{DEFAULT_MODEL} not cached locally")
        cls.model = load_tts_model(TTSConfig())

    def test_every_stage_loaded_with_parameters(self):
        for stage in ("talker", "speech_tokenizer"):
            module = getattr(self.model, stage, None)
            self.assertIsNotNone(module, f"missing stage: {stage}")
            self.assertGreater(len(list(module.parameters())), 0, f"no parameters mapped for {stage}")
        # speaker_encoder is optional: the 4-bit CustomVoice conversion carries
        # predefined speaker embeds without it (voices still resolve below)

    def test_custom_voice_surface(self):
        self.assertEqual(self.model.config.tts_model_type, "custom_voice")
        speakers = [s.lower() for s in self.model.supported_speakers]
        self.assertIn(DEFAULT_SPEAKER.lower(), speakers)
        self.assertEqual(self.model.sample_rate, 24000)


if __name__ == "__main__":
    unittest.main()
