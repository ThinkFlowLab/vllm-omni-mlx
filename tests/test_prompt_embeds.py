"""CustomVoice prompt layout (#14 / M1.5): spec invariants on the real
checkpoint config — two preset voices, Auto vs explicit language, speaker
slot exactness — gated on the cached checkpoint."""

import os

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest

import mlx.core as mx

from vllm_omni_mlx.tts.config import DEFAULT_MODEL, TTSConfig, load_tts_model, local_snapshot
from vllm_omni_mlx.tts.prompt_embeds import PromptEmbeds

TEXT = "The quick brown fox jumps over the lazy dog."


class PromptEmbedsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if local_snapshot(DEFAULT_MODEL) is None:
            raise unittest.SkipTest(f"{DEFAULT_MODEL} not cached locally")
        cls.model = load_tts_model(TTSConfig())
        cls.builder = PromptEmbeds(cls.model)
        cls.hidden = cls.model.talker.config.hidden_size

    def test_layout_length_auto_and_explicit_language(self):
        vivian = self.builder.build(TEXT, speaker="Vivian", language="auto")
        english = self.builder.build(TEXT, speaker="Vivian", language="English")
        # auto: 3 think rows + spk + [pad, bos] = 6; explicit adds a language row = 7
        self.assertEqual(vivian.codec_prefix_len, 6)
        self.assertEqual(english.codec_prefix_len, 7)
        self.assertEqual(vivian.input_embeds.shape, (1, vivian.expected_len, self.hidden))
        self.assertEqual(english.input_embeds.shape, (1, english.expected_len, self.hidden))
        # decode-step text row and trailing text carry the talker hidden width
        self.assertEqual(vivian.decode_text_embed.shape[-1], self.hidden)
        self.assertGreater(vivian.trailing_text_hidden.shape[1], 0)

    def test_speaker_slot_differs_exactly_between_voices(self):
        vivian = self.builder.build(TEXT, speaker="Vivian", language="auto")
        ryan = self.builder.build(TEXT, speaker="Ryan", language="auto")
        self.assertEqual(vivian.input_embeds.shape, ryan.input_embeds.shape)
        diff = mx.abs(vivian.input_embeds - ryan.input_embeds[0]).sum(axis=-1)[0]
        changed = [i for i in range(diff.shape[0]) if float(diff[i]) > 1e-6]
        self.assertEqual(changed, [vivian.speaker_position])

    def test_speaker_row_is_spk_embedding_channel_summed(self):
        vivian = self.builder.build(TEXT, speaker="Vivian", language="auto")
        codec_embed = self.model.talker.get_input_embeddings()
        spk = codec_embed(mx.array([[self.builder.speaker_id("vivian")]]))[0, 0]
        # channel-summed with the tts_pad row occupying that slot
        pad = vivian.decode_text_embed[0, 0]
        row = vivian.input_embeds[0, vivian.speaker_position]
        self.assertTrue(mx.allclose(row, (spk + pad).astype(row.dtype), atol=1e-5))

    def test_speakers_and_languages_from_config(self):
        self.assertIn("vivian", self.builder.speakers)
        self.assertIn("ryan", self.builder.speakers)
        self.assertIn("english", self.builder.languages)
        with self.assertRaises(ValueError):
            self.builder.speaker_id("nonexistent")


if __name__ == "__main__":
    unittest.main()
