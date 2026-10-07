"""Vendored compiled-loop parity (#79): the loop must reproduce the library.

Weight-gated — runs where the snapshot is cached, skips in CI. Three gates:

1. the vendored EAGER loop is bitwise-identical to mlx-audio's generate
   (same RNG stream, same op order — the control-flow copy is exact);
2. the compiled closures drift only by fp-fusion rounding (max|Δ| ~2e-3,
   mean ~2e-5 on the final waveform, same stop point);
3. instruct and reference-clone modes run through the compiled loop and
   still produce speech (size floor + the #71 HNR floor).
"""

import base64
import os
import unittest
import unittest.mock

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pathlib import Path

import mlx.core as mx

from tests._teardown import ReleaseAfterClass

from vllm_omni_mlx.tts.voxcpm2 import (
    VoxCPM2Config,
    decode_ref_audio,
    local_snapshot,
)

#: the #71 calibrated catastrophic-decode floor (see test_voxcpm2_e2e.py —
#: clean 4-bit output measured −3.8…3.1 dB, noise reference −10.1)
HNR_FLOOR_DB = -5.0

MODEL = "mlx-community/VoxCPM2-4bit"
SR = 48000
TEXT = "Parity between the vendored loop and the library."
INSTRUCT = "A calm, low male voice"


class VendoredLoopParityTest(ReleaseAfterClass, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snapshot = local_snapshot(MODEL)
        if snapshot is None:
            raise unittest.SkipTest(f"{MODEL} not cached locally")
        cls.snapshot = Path(snapshot)
        from vllm_omni_mlx.tts.voxcpm2 import load_voxcpm2_model

        cls.config = VoxCPM2Config(model_ref=MODEL, max_tokens=60)
        cls.model = load_voxcpm2_model(cls.config)

    def _hnr(self, samples: mx.array) -> float:
        import numpy as np

        from tests.audio_metrics import pcm_hnr_db

        return pcm_hnr_db(np.asarray(samples, dtype=np.float32), sr=SR)

    def _library(self, seed: int, **kwargs) -> mx.array:
        mx.random.seed(seed)
        for result in self.model.generate(text=TEXT, max_tokens=self.config.max_tokens, **kwargs):
            return result.audio
        raise AssertionError("library produced no audio")

    def _vendored(self, seed: int, compiled: bool, **kwargs) -> mx.array:
        from vllm_omni_mlx.tts import voxcpm2_loop

        mx.random.seed(seed)
        for audio in voxcpm2_loop.generate_frames(self.model, self.config, TEXT, compiled=compiled, **kwargs):
            return audio
        raise AssertionError("vendored loop produced no audio")

    def test_vendored_eager_is_bitwise_library(self):
        for seed in (1, 2):
            with self.subTest(seed=seed):
                a = self._library(seed)
                b = self._vendored(seed, compiled=False)
                self.assertEqual(a.shape, b.shape)
                self.assertTrue(bool(mx.all(a == b).item()), f"seed {seed}: vendored-eager differs from library")

    def test_compiled_drift_is_fusion_rounding(self):
        a = self._vendored(7, compiled=False)
        b = self._vendored(7, compiled=True)
        # same stop point expected: the drift does not flip the stop argmax
        self.assertEqual(a.shape, b.shape, "compiled loop stopped at a different patch")
        delta = mx.abs(a - b)
        # fp-fusion rounding (~2e-6 per op) amplified through the AR chain:
        # the mean stays tiny (measured ~2e-5) but rare samples reach ~2e-2
        # on a [-1, 1] waveform — envelope calibrated over repeated runs
        self.assertLess(float(delta.mean().item()), 8e-3, "mean drift beyond the fusion-rounding envelope")
        self.assertLess(float(delta.max().item()), 0.05, "max drift beyond the amplified-rounding envelope")
        self.assertGreater(self._hnr(b), HNR_FLOOR_DB)

    def test_compiled_instruct_mode(self):
        from vllm_omni_mlx.tts import voxcpm2_loop

        mx.random.seed(3)
        out = None
        for audio in voxcpm2_loop.generate_frames(self.model, self.config, TEXT, instruct=INSTRUCT):
            out = audio
            break
        self.assertIsNotNone(out)
        self.assertGreater(out.size, SR)  # >1s
        self.assertGreater(self._hnr(out), HNR_FLOOR_DB)

    def test_compiled_clone_mode(self):
        from vllm_omni_mlx.tts import voxcpm2_loop

        ref = base64.b64encode((self.snapshot / "test_en.wav").read_bytes()).decode()
        ref_audio = decode_ref_audio(ref, SR)
        mx.random.seed(4)
        out = None
        for audio in voxcpm2_loop.generate_frames(self.model, self.config, TEXT, ref_audio=ref_audio):
            out = audio
            break
        self.assertIsNotNone(out)
        self.assertGreater(out.size, SR)
        self.assertGreater(self._hnr(out), HNR_FLOOR_DB)

    def test_escape_env_selects_library_path(self):
        from vllm_omni_mlx.tts import voxcpm2, voxcpm2_loop

        with unittest.mock.patch.dict(os.environ, {"VLLM_OMNI_VOXCPM2_EAGER": "1"}):
            self.assertTrue(voxcpm2_loop.eager_escape())
            mx.random.seed(5)
            chunks = list(voxcpm2.synthesize(self.model, self.config, TEXT))
        a = chunks[0]
        b = self._library(5)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(bool(mx.all(a == b).item()), "escape env must reproduce the library exactly")


if __name__ == "__main__":
    unittest.main()
