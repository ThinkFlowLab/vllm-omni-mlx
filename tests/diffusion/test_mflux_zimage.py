"""Real Z-Image-Turbo round-trip through the mflux seam (#99).

Weight-gated: runs against the local HF snapshot when cached and skips
elsewhere (CI has no weights, so a green CI run does not exercise this
file). One file = one process (CONTRIBUTING) — the 4-bit model peaks
around 6–7 GiB resident and holds it for the class.
"""

import os

# offline keeps the load fast and hang-free when the network is flaky
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest

from vllm_omni_mlx.diffusion.config import DEFAULT_MODEL, load_image_service


def _snapshot_cached() -> bool:
    # direct check rather than snapshot_download(local_files_only=True):
    # a patterns-filtered download counts as "incomplete" to the hub API
    # even when every weight file is present
    import glob
    import os

    pattern = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{DEFAULT_MODEL.replace('/', '--')}/snapshots/*/transformer/*.safetensors"
    )
    return bool(glob.glob(pattern))


@unittest.skipUnless(_snapshot_cached(), f"{DEFAULT_MODEL} not in the local HF cache")
class ZImageTurboE2ETest(unittest.TestCase):
    """512² × 9 steps keeps the run quick while exercising the full path:
    pre-quant load → text encode → 9 denoise steps → VAE decode → PNG."""

    @classmethod
    def setUpClass(cls):
        import mlx.core as mx

        cls.mx = mx
        cls.service = load_image_service(DEFAULT_MODEL)
        cls.prompt = "a crisp red apple on a wooden table, studio light"

    @classmethod
    def tearDownClass(cls):
        # donor release: one heavy thing at a time on a 16 GB machine
        cls.service = None
        cls.mx.clear_cache()

    def test_generate_png_shape_and_gate(self):
        (result,) = self.service.generate(self.prompt, width=512, height=512, steps=9, seed=20261007)
        self.assertTrue(result.png.startswith(b"\x89PNG"), "output must be a PNG")
        self.assertGreater(len(result.png), 50_000, "a 512² PNG should not be tiny (non-degenerate gate)")
        self.assertEqual((result.width, result.height, result.steps), (512, 512, 9))
        self.assertEqual(result.seed, 20261007)
        self.assertGreater(result.generation_time, 0.1)
        print(
            f"\nz-image-turbo 512x512/9 steps: {result.generation_time:.2f}s, "
            f"peak {result.peak_memory_gib:.2f} GiB, png {len(result.png) / 1e6:.2f} MB",
            flush=True,
        )

    def test_same_seed_reproduces_bytes(self):
        first = self.service.generate(self.prompt, width=512, height=512, steps=9, seed=7)[0].png
        second = self.service.generate(self.prompt, width=512, height=512, steps=9, seed=7)[0].png
        self.assertEqual(
            first, second, "same-seed calls must reproduce byte-identical PNGs (fp16 fusion-flip precedent: characterize if this trips)"
        )

    def test_family_capability_validation(self):
        with self.assertRaises(ValueError):
            self.service.generate(self.prompt, width=512, height=512, guidance=3.5)


if __name__ == "__main__":
    unittest.main()
