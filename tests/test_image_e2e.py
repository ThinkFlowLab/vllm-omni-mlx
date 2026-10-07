"""Weight-gated Qwen-Image-2.1 e2e (#101): the seam produces real images.

Gates per the #91 metrics doctrine: deterministic-seed reproducibility
(byte-identical PNGs), non-degenerate output stats, and the issue's
acceptance run — a valid 1024² image from the default 40-step guidance-free
path. Skips where the checkpoint is not locally cached (CI); run on a
machine with the weights, one file per process (checkpoints stack).
"""

import os

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import io
import unittest

from tests._teardown import ReleaseAfterClass

from vllm_omni_mlx.diffusion.config import DEFAULT_MODEL, ImageConfig, load_image_model, local_snapshot
from vllm_omni_mlx.diffusion.service import ImageService

PROMPT = "A puffin standing on a cliff, natural lighting"


def png_stats(png: bytes):
    from PIL import Image, ImageStat

    image = Image.open(io.BytesIO(png)).convert("RGB")
    stat = ImageStat.Stat(image)
    return image.size, stat.mean, (sum(s ** 2 for s in stat.stddev) / 3) ** 0.5


class ImageEndToEndTest(ReleaseAfterClass, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if local_snapshot(DEFAULT_MODEL) is None:
            raise unittest.SkipTest(f"{DEFAULT_MODEL} not cached locally")
        cls.config = ImageConfig()
        cls.model = load_image_model(cls.config)
        cls.service = ImageService(cls.model, cls.config)

    def test_same_seed_is_byte_identical(self):
        first = self.service.generate(PROMPT, size="512x512", steps=8, seed=42)[0]
        second = self.service.generate(PROMPT, size="512x512", steps=8, seed=42)[0]
        self.assertEqual(first.png, second.png, "same seed must reproduce byte-identical output")
        self.assertEqual(first.png[1:4], b"PNG")

    def test_output_gates_small(self):
        result = self.service.generate(PROMPT, size="512x512", steps=8, seed=7)[0]
        (width, height), means, stddev = png_stats(result.png)
        self.assertEqual((width, height), (512, 512))
        self.assertGreater(stddev, 8.0, "near-uniform output: sampling path broken")
        self.assertLess(stddev, 120.0, "full-scale noise: precision artifact")
        for mean in means:
            self.assertLess(abs(mean - 128), 80, "channel mean pinned: decode path broken")

    def test_acceptance_1024_default_path(self):
        # #101 acceptance: a valid image from the default 40-step guidance-free
        # path. Heavy (~14 min on an M4) — gated so the default suite stays
        # light; run explicitly for the acceptance gate: IMAGE_E2E_FULL=1
        # python -m unittest tests.test_image_e2e
        if os.environ.get("IMAGE_E2E_FULL") != "1":
            raise unittest.SkipTest("set IMAGE_E2E_FULL=1 for the 1024²/40-step acceptance run")
        result = self.service.generate(PROMPT, size="1024x1024", seed=123)[0]
        self.assertEqual(result.steps, 40)
        (width, height), _, stddev = png_stats(result.png)
        self.assertEqual((width, height), (1024, 1024))
        self.assertGreater(stddev, 8.0, "near-uniform output: sampling path broken")
        self.assertGreater(len(result.png), 100_000, "PNG suspiciously small for a 1024² photo")


if __name__ == "__main__":
    unittest.main()
