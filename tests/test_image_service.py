"""Image service + /v1/images/generations (#101, seam per #91): size/steps/
guidance/n validation, seed determinism, the viggle_turbo steps gate, the
single-user lock, and the HTTP surface with a fake service — no weights, no
mflux import (CI-safe)."""

import json
import unittest

from starlette.testclient import TestClient

from vllm_omni_mlx.diffusion.service import ImageError, ImageResult, ImageService, parse_size
from vllm_omni_mlx.server import create_app


class _FakePILImage:
    """Stands in for generated.image so the unit path needs no PIL."""

    def __init__(self, tag):
        self._tag = tag

    def save(self, buf, format=None):
        buf.write(b"FAKE-PNG-" + str(self._tag).encode())


class FakeMfluxModel:
    """Records generate_image kwargs; returns distinguishable images."""

    def __init__(self):
        self.calls = []

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        return type("Generated", (), {"image": _FakePILImage(len(self.calls))})()


class ImageServiceValidationTest(unittest.TestCase):
    def test_parse_size(self):
        self.assertEqual(parse_size("1024x1024"), (1024, 1024))
        self.assertEqual(parse_size("1280X720"), (1280, 720))
        self.assertEqual(parse_size("1328x896"), (1328, 896))

    def test_parse_size_rejects(self):
        for bad in ("1024", "axb", "1024x", "", None, "1000x1000", "128x128", "4096x4096", "256x0"):
            with self.assertRaises(ImageError, msg=repr(bad)):
                parse_size(bad)

    def test_defaults_come_from_config(self):
        model = FakeMfluxModel()
        service = ImageService(model)
        service.generate("a puffin", seed=42)
        call = model.calls[0]
        self.assertEqual(call["width"], 1024)
        self.assertEqual(call["height"], 1024)
        self.assertEqual(call["num_inference_steps"], 40)
        self.assertEqual(call["guidance"], 1.0)
        self.assertIsNone(call["negative_prompt"])

    def test_seed_deterministic_and_incrementing_for_n(self):
        model = FakeMfluxModel()
        service = ImageService(model)
        results = service.generate("a puffin", seed=7, n=3)
        self.assertEqual([r.seed for r in results], [7, 8, 9])
        self.assertEqual([r.png for r in results], [b"FAKE-PNG-1", b"FAKE-PNG-2", b"FAKE-PNG-3"])

    def test_seed_random_when_absent(self):
        model = FakeMfluxModel()
        service = ImageService(model)
        results = service.generate("a puffin", n=2)
        self.assertNotEqual(results[0].seed, results[1].seed)

    def test_request_validation_errors(self):
        service = ImageService(FakeMfluxModel())
        for kwargs in (
            {"prompt": "   "},
            {"prompt": "ok", "size": "1000x1000"},
            {"prompt": "ok", "steps": 0},
            {"prompt": "ok", "steps": 1000},
            {"prompt": "ok", "guidance": 0.5},
            {"prompt": "ok", "n": 0},
            {"prompt": "ok", "n": 5},
            {"prompt": "ok", "seed": "42"},
        ):
            with self.assertRaises(ImageError, msg=kwargs):
                service.generate(kwargs.get("prompt", ""), **{k: v for k, v in kwargs.items() if k != "prompt"})

    def test_viggle_turbo_steps_gate(self):
        from vllm_omni_mlx.diffusion.config import ImageConfig

        model = FakeMfluxModel()
        config = ImageConfig(scheduler="viggle_turbo")
        service = ImageService(model, config)
        with self.assertRaisesRegex(ImageError, "6"):
            service.generate("a puffin", steps=40)
        results = service.generate("a puffin", steps=6, seed=1)
        self.assertEqual(model.calls[0]["num_inference_steps"], 6)
        self.assertEqual(results[0].steps, 6)


class ImagesApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(image_service=FakeImageService()), raise_server_exceptions=False)

    def test_generations_returns_openai_shape(self):
        response = self.client.post("/v1/images/generations", json={"prompt": "a puffin", "seed": 42})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("created", body)
        self.assertEqual(len(body["data"]), 1)
        self.assertTrue(body["data"][0]["b64_json"].startswith("RkFLRS1QTkct"))  # base64("FAKE-PNG-")

    def test_n_images(self):
        response = self.client.post("/v1/images/generations", json={"prompt": "a puffin", "n": 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["data"]), 2)

    def test_validation_maps_to_400(self):
        for payload in ({}, {"prompt": ""}, {"prompt": "ok", "size": "bad"}, {"prompt": "ok", "steps": 0}, {"prompt": "ok", "n": 9}):
            response = self.client.post("/v1/images/generations", json=payload)
            self.assertEqual(response.status_code, 400, payload)
            self.assertIn("error", response.json())

    def test_url_response_format_rejected(self):
        response = self.client.post("/v1/images/generations", json={"prompt": "ok", "response_format": "url"})
        self.assertEqual(response.status_code, 400)

    def test_model_listing_includes_image_service(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertIn("fake-image", [entry["id"] for entry in response.json()["data"]])

    def test_health_has_no_image_side_effects(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(json.dumps(self.client.get("/health").json()), '{"status": "ok"}')


class FakeImageService(ImageService):
    """The real validation path behind the API tests, with a fake model."""

    name = "fake-image"
    license = "Fake License — testing only"

    def __init__(self):
        super().__init__(FakeMfluxModel())


if __name__ == "__main__":
    unittest.main()
