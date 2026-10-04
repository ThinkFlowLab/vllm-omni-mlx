"""Talker backbone seam (#13 / M1.4): dual-track structure from checkpoint
config, channel-summed codec embeds, and the config-derived codec-id
suppression list — gated on the cached checkpoint."""

import os

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest

import mlx.core as mx

from vllm_omni_mlx.tts.config import DEFAULT_MODEL, TTSConfig, load_tts_model, local_snapshot
from vllm_omni_mlx.tts.talker import Talker


class TalkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if local_snapshot(DEFAULT_MODEL) is None:
            raise unittest.SkipTest(f"{DEFAULT_MODEL} not cached locally")
        cls.model = load_tts_model(TTSConfig())
        cls.talker = Talker(cls.model.talker)

    def test_dual_track_config(self):
        # values are read from the checkpoint config, not hardcoded; the two
        # tracks share the talker hidden width and differ in vocabulary
        self.assertEqual(self.talker.text_vocab_size, self.model.talker.config.text_vocab_size)
        self.assertEqual(self.talker.codec_vocab_size, self.model.talker.config.vocab_size)
        self.assertEqual(self.talker.text_hidden_size, self.talker.codec_hidden_size)
        self.assertLess(self.talker.codec_vocab_size, self.talker.text_vocab_size)

    def test_codec_embeds_channel_sum_all_groups(self):
        groups = self.model.talker.config.code_predictor_config.num_code_groups
        codes = [mx.array([[10 + i]]) for i in range(groups)]
        embeds = self.talker.codec_embeds(codes)
        self.assertEqual(embeds.shape, (1, 1, self.talker.codec_hidden_size))

    def test_suppression_list_covers_specials_except_eos(self):
        suppressed = self.talker.suppressed_codec_ids()
        eos = self.talker.codec_eos_token_id
        base = self.talker.codec_vocab_size - 1024
        self.assertEqual(suppressed, [i for i in range(base, base + 1024) if i != eos])
        self.assertNotIn(eos, suppressed)
        self.assertNotIn(base - 1, suppressed)  # real codes are untouched


if __name__ == "__main__":
    unittest.main()
