"""MTP code predictor seam (#12 / M1.3): structure facts, per-step logits with
KV cache advance, and determinism — gated on the cached checkpoint."""

import os

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest

import mlx.core as mx

from vllm_omni_mlx.tts.code_predictor import CodePredictor
from vllm_omni_mlx.tts.config import DEFAULT_MODEL, TTSConfig, load_tts_model, local_snapshot


def predictor():
    if local_snapshot(DEFAULT_MODEL) is None:
        raise unittest.SkipTest(f"{DEFAULT_MODEL} not cached locally")
    model = load_tts_model(TTSConfig())
    return CodePredictor(model.talker)


class CodePredictorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred = predictor()

    def test_mtp_structure(self):
        # Q-1 residual groups, each with its own embedding and LM head
        residuals = self.pred.num_code_groups - 1
        self.assertEqual(residuals, 15)
        self.assertEqual(len(self.pred.residual_embeddings), residuals)
        self.assertEqual(len(self.pred.residual_heads), residuals)
        self.assertEqual(self.pred.num_layers, 5)

    def test_step_advances_logits_per_group(self):
        pred = self.pred
        cache = pred.make_cache()
        self.assertEqual(len(cache), pred.num_layers)
        embeds = pred.residual_embeddings[0](mx.array([[1234]]))  # (1, 1, hidden)
        logits0 = pred.step(embeds, cache, generation_step=0)
        logits1 = pred.step(embeds, cache, generation_step=1)
        self.assertEqual(logits0.shape, (1, 1, 2048))
        self.assertEqual(logits1.shape, (1, 1, 2048))
        # different head per step → different logits for identical input
        self.assertFalse(mx.allclose(logits0, logits1, atol=1e-4))

    def test_deterministic_given_cache(self):
        pred = self.pred
        embeds = pred.residual_embeddings[0](mx.array([[1234]]))
        first = pred.step(embeds, pred.make_cache(), 0)
        second = pred.step(embeds, pred.make_cache(), 0)
        self.assertTrue(mx.allclose(first, second, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
