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

from tests.asr_oracle import requires_oracle
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
        # mirror the config's solver knobs so parity holds at whatever the
        # serving default is (the library's own generate default is 10 steps)
        for result in self.model.generate(
            text=TEXT,
            max_tokens=self.config.max_tokens,
            inference_timesteps=self.config.inference_timesteps,
            cfg_value=self.config.cfg_value,
            warmup_patches=self.config.warmup_patches,
            **kwargs,
        ):
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
        # the stop argmax sits on a near-tie under quantized weights, so the
        # compiled run may stop a couple of patches early/late, and the AR
        # chain amplifies low-bit differences once trajectories part — the
        # honest comparison is the shared pre-divergence prefix (the vocoder
        # calibration rule: envelope tripwires, never bitwise across paths)
        patch_samples = 4 * 960  # patch_size × decode chunk, at 48 kHz
        self.assertLessEqual(
            abs(a.size - b.size), 2 * patch_samples, "compiled loop drifted more than two stop patches"
        )
        n = min(a.size, b.size, SR)  # first second: before near-ties compound
        delta = mx.abs(a[:n] - b[:n])
        self.assertLess(float(delta.mean().item()), 8e-3, "mean drift beyond the fusion-rounding envelope")
        self.assertLess(float(delta.max().item()), 0.05, "max drift beyond the amplified-rounding envelope")
        self.assertGreater(self._hnr(b), HNR_FLOOR_DB)
        self.assertGreater(self._hnr(a), HNR_FLOOR_DB)

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

    @requires_oracle
    def test_default_timesteps_quality_equivalent(self):
        """The t=6 default's #88 evidence, kept as a gate: paired-seed ASR
        round-trip against the checkpoint's t=10 (same noise draws, only the
        solver's step count differs) must not lose similarity, per text —
        the paired median absorbs the oracle's per-draw decode wobble.
        Texts mirror the #88 study (long + short English, zh). What this
        gate CAN resolve: catastrophic degradation (per-draw absolute
        similarity — the rope-sign bug scored 0.2–0.3 where clean speech
        scores 0.6–1.0) and the big compound effect (t=6 under quantized
        weights measured −0.18 mean, ~2.5σ). What it CANNOT resolve at a
        test budget: sub-0.1 deltas — per-pair noise is ±0.2–0.4 (whisper
        decode wobble; en2 is bimodal ±0.4), so n=6 paired means carry a
        ~0.07 standard error. Those stats are printed as evidence, not
        asserted; zh is report-only for the same reason as the fleet
        battery (scripts/acc_all_checkpoints.py)."""
        from vllm_omni_mlx.tts import voxcpm2_loop

        from tests.asr_oracle import load_oracle, round_trip_similarity

        oracle = load_oracle()
        texts = {
            "en": "The paired equivalence gate speaks a clear sentence for the transcriber to check against the reference text.",
            "en2": "The quick synthesis equivalence check speaks a shorter sentence.",
        }
        # zh: POOLED over three sentences — one-sentence zh evidence is
        # sentence-level oracle noise (a reviewer's single-sentence −0.173
        # vs our pooled n=18 mean +0.003 ± 0.066 on the same comparison);
        # pooling across sentences is the honest statistic
        zh_sentences = [
            ["这句话验证减少求解步数之后中文输出的可懂度没有下降。", "這句話驗證減少求解步數之後中文輸出的可懂度沒有下降。"],
            ["今天的天气很好，适合出去散步和购物。", "今天的天氣很好，適合出去散步和購物。"],
            ["科技的发展改变了人们的日常生活和工作方式。", "科技的發展改變了人們的日常生活和工作方式。"],
        ]
        configs = {
            10: VoxCPM2Config(model_ref=MODEL, max_tokens=120, inference_timesteps=10),
            self.config.inference_timesteps: VoxCPM2Config(
                model_ref=MODEL, max_tokens=120, inference_timesteps=self.config.inference_timesteps
            ),
        }
        for name, text in texts.items():
            with self.subTest(text=name):
                sims = {}
                for t, config in configs.items():
                    scores = []
                    for seed in range(6):
                        mx.random.seed(seed)
                        out = None
                        for audio in voxcpm2_loop.generate_frames(self.model, config, text):
                            out = audio
                        mx.eval(out)
                        scores.append(round_trip_similarity(oracle, out, SR, text))
                    sims[t] = scores
                deltas = [b - a for a, b in zip(sims[10], sims[self.config.inference_timesteps])]
                mean = sum(deltas) / len(deltas)
                print(f"{name}: paired deltas {['%+.3f' % d for d in deltas]} mean {mean:+.3f} (evidence; oracle resolution ~±0.1 at n=6)")
                for draw in sims[self.config.inference_timesteps]:
                    self.assertGreater(
                        draw, 0.5,
                        f"{name}: round-trip similarity {draw:.3f} below the catastrophic floor (audio broken, not noisy)",
                    )
        with self.subTest(text="zh-pooled"):
            zh_deltas = []
            for refs in zh_sentences:
                sims = {}
                for t, config in configs.items():
                    scores = []
                    for seed in range(4):
                        mx.random.seed(seed)
                        out = None
                        for audio in voxcpm2_loop.generate_frames(self.model, config, refs[0]):
                            out = audio
                        mx.eval(out)
                        scores.append(round_trip_similarity(oracle, out, SR, refs))
                    sims[t] = scores
                zh_deltas.extend(b - a for a, b in zip(sims[10], sims[self.config.inference_timesteps]))
            mean = sum(zh_deltas) / len(zh_deltas)
            print(f"zh-pooled: n={len(zh_deltas)} paired mean {mean:+.3f} (evidence, report-only — fleet-battery zh rule)")

    @requires_oracle
    def test_default_quantization_quality_equivalent(self):
        """The 8-bit-DiT/4-bit-encoder default's evidence (review finding
        on #98): paired-seed ASR round-trip against the bf16 blocks (same
        seeds, same solver knobs, only load-time quantization differs).
        Same statistics contract as the timestep gate: catastrophic
        per-draw floor is asserted; sub-0.1 paired deltas are printed as
        evidence only — absolute similarity is machine- and
        oracle-dependent (the reviewer's M1 Max measured different
        absolutes and the direction agrees within noise), and per-pair
        wobble ±0.2–0.4 puts a hard sub-0.1 gate below the oracle's
        resolution at test budgets."""
        from vllm_omni_mlx.tts import voxcpm2, voxcpm2_loop

        from tests.asr_oracle import load_oracle, round_trip_similarity

        oracle = load_oracle()
        texts = {
            "en": "Welcome to the VoxCPM2 benchmark. This paragraph is long enough that the speech path runs for many seconds, which makes the real time factor meaningful over a sustained generation.",
            "en2": "The paired quantization gate speaks a clear sentence for the transcriber.",
            "zh": [
                "这句话用来测试中文语音合成在量化求解器之后的音质是否保持一致。",
                "這句話用來測試中文語音合成在量化求解器之後的音質是否保持一致。",
            ],
        }
        config = VoxCPM2Config(model_ref=MODEL, max_tokens=160)

        def scores_for(model) -> dict:
            out_by_text = {}
            for name, text in texts.items():
                spoken = text if isinstance(text, str) else text[0]
                sims = []
                for seed in range(6):
                    mx.random.seed(seed)
                    out = None
                    for audio in voxcpm2_loop.generate_frames(model, config, spoken):
                        out = audio
                    mx.eval(out)
                    sims.append(round_trip_similarity(oracle, out, SR, text))
                out_by_text[name] = sims
            return out_by_text

        with unittest.mock.patch.dict(os.environ, {"VLLM_OMNI_VOXCPM2_QUANT": "off"}):
            bf16_model = voxcpm2.load_voxcpm2_model(config)
            mx.eval(bf16_model.parameters())
            bf16 = scores_for(bf16_model)
            voxcpm2_loop._CLOSURES.clear()
            del bf16_model
            mx.clear_cache()
        # self.model was loaded with the default (8-bit DiT / 4-bit encoder)
        quant = scores_for(self.model)
        for name in texts:
            with self.subTest(text=name):
                deltas = [b - a for a, b in zip(bf16[name], quant[name])]
                mean = sum(deltas) / len(deltas)
                print(f"{name}: paired deltas {['%+.3f' % d for d in deltas]} mean {mean:+.3f} (evidence; oracle resolution ~±0.1 at n=6)")
                if name == "zh":
                    continue  # report-only: whisper-base zh spans 0.3-1.0 on clean speech (fleet battery rule)
                for draw in quant[name]:
                    self.assertGreater(
                        draw, 0.5,
                        f"{name}: round-trip similarity {draw:.3f} below the catastrophic floor (audio broken, not noisy)",
                    )

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
