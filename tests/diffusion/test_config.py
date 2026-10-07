"""Family registry + request parsing for /v1/images/generations (#91).

CI-safe: no mflux import (config.py keeps backend imports lazy) and no
weights — resolution and defaults only.
"""

import unittest

from vllm_omni_mlx.diffusion.config import DEFAULT_MODEL, FAMILIES, resolve_model
from vllm_omni_mlx.schemas import ApiError, parse_image_generation


class ResolveModelTest(unittest.TestCase):
    def test_alias_routes_to_prequant_default(self):
        for alias in ("z-image-turbo", "ZImage-Turbo"):
            resolved = resolve_model(alias)
            self.assertEqual(resolved.family.name, "z-image-turbo")
            self.assertEqual(resolved.weights_ref, DEFAULT_MODEL)
            self.assertIsNone(resolved.quantize)  # stored 4-bit level honored

    def test_canonical_repo_quantizes_on_load(self):
        resolved = resolve_model("Tongyi-MAI/Z-Image-Turbo")
        self.assertEqual(resolved.weights_ref, "Tongyi-MAI/Z-Image-Turbo")
        self.assertEqual(resolved.quantize, 4)

    def test_prequant_mirror(self):
        resolved = resolve_model(DEFAULT_MODEL)
        self.assertEqual(resolved.weights_ref, DEFAULT_MODEL)
        self.assertIsNone(resolved.quantize)

    def test_unknown_ref_lists_supported(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_model("stabilityai/stable-diffusion-xl-base-1.0")
        self.assertIn("z-image-turbo", str(ctx.exception))

    def test_every_family_has_serving_defaults(self):
        for family in FAMILIES:
            self.assertGreaterEqual(family.steps, 1, family.name)


class ParseImageGenerationTest(unittest.TestCase):
    def test_minimal_request(self):
        req = parse_image_generation({"prompt": "a cat"})
        self.assertEqual((req.width, req.height, req.n, req.steps), (1024, 1024, 1, None))

    def test_size_and_diffusion_params(self):
        req = parse_image_generation(
            {"prompt": "a cat", "n": 2, "size": "768X1344", "steps": 9, "guidance": 0.0, "seed": 7}
        )
        self.assertEqual((req.width, req.height, req.n, req.steps, req.guidance, req.seed), (768, 1344, 2, 9, 0.0, 7))

    def test_prompt_must_be_nonempty_string(self):
        for bad in (None, 5, "", "   "):
            with self.assertRaises(ApiError):
                parse_image_generation({"prompt": bad})

    def test_size_format(self):
        for bad in ("1024", "ax1024", "1024x", 1024, "1024.5x1024"):
            with self.assertRaises(ApiError):
                parse_image_generation({"prompt": "a cat", "size": bad})

    def test_dimension_bounds_and_grid(self):
        for bad in ("240x1024", "2112x1024", "1024x1000"):  # below/above bounds, not %16
            with self.assertRaises(ApiError):
                parse_image_generation({"prompt": "a cat", "size": bad})

    def test_n_range(self):
        for bad in (0, 5, "2", True):
            with self.assertRaises(ApiError):
                parse_image_generation({"prompt": "a cat", "n": bad})

    def test_scalar_types(self):
        for payload in ({"prompt": "a cat", "steps": "9"}, {"prompt": "a cat", "guidance": "0"}, {"prompt": "a cat", "seed": 1.5}):
            with self.assertRaises(ApiError):
                parse_image_generation(payload)


if __name__ == "__main__":
    unittest.main()
