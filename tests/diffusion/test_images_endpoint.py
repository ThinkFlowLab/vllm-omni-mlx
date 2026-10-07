"""POST /v1/images/generations (#91): validation, auth, model listing,
and the b64/seeds response shape with a fake service. The real
Z-Image-Turbo round-trip is weight-gated in test_mflux_zimage.py."""

import base64
import unittest

from starlette.testclient import TestClient

from vllm_omni_mlx.diffusion.service import ImageResult
from vllm_omni_mlx.server import create_app


class FakeImageService:
    name = "fake-image-model"
    model_type = "image"

    def __init__(self):
        self.calls = []

    def generate(self, prompt, n=1, width=1024, height=1024, steps=None, guidance=None, seed=None):
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not 1 <= n <= 4:
            raise ValueError(f"n must be between 1 and 4, got {n}")
        if guidance is not None:
            raise ValueError("guidance is not supported on z-image-turbo (guidance-distilled)")
        self.calls.append((prompt, n, width, height, steps, guidance, seed))
        return [
            ImageResult(
                png=b"\x89PNG-fake-" + bytes([i]),
                seed=(seed if seed is not None else 1234) + i,
                width=width,
                height=height,
                steps=steps or 9,
                generation_time=0.5,
                peak_memory_gib=1.0,
            )
            for i in range(n)
        ]


class ImagesEndpointTest(unittest.TestCase):
    def setUp(self):
        self.service = FakeImageService()
        self.client = TestClient(create_app(image_service=self.service))

    def test_round_trip(self):
        response = self.client.post(
            "/v1/images/generations", json={"prompt": "a cat", "size": "512x512", "steps": 9, "seed": 42}
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model"], "fake-image-model")
        self.assertEqual(len(body["data"]), 1)
        self.assertEqual(base64.b64decode(body["data"][0]["b64_json"]), b"\x89PNG-fake-\x00")
        self.assertEqual(body["seeds"], [42])
        self.assertEqual(body["generation_time_ms"], [500])
        self.assertEqual(self.service.calls, [("a cat", 1, 512, 512, 9, None, 42)])

    def test_n_sequential_seeds(self):
        body = self.client.post("/v1/images/generations", json={"prompt": "a cat", "n": 3, "seed": 10}).json()
        self.assertEqual(body["seeds"], [10, 11, 12])
        self.assertEqual(len(body["data"]), 3)

    def test_validation_errors_are_400(self):
        for payload in ({"prompt": ""}, {"prompt": "a cat", "size": "1000x1024"}, {"prompt": "a cat", "n": 9}):
            response = self.client.post("/v1/images/generations", json=payload)
            self.assertEqual(response.status_code, 400, payload)
            self.assertEqual(response.json()["error"]["type"], "invalid_request_error")

    def test_family_capability_error_is_400(self):
        response = self.client.post("/v1/images/generations", json={"prompt": "a cat", "guidance": 3.5})
        self.assertEqual(response.status_code, 400)
        self.assertIn("guidance-distilled", response.json()["error"]["message"])

    def test_auth(self):
        client = TestClient(create_app(image_service=self.service, api_key="secret"))
        self.assertEqual(client.post("/v1/images/generations", json={"prompt": "a cat"}).status_code, 401)
        self.assertEqual(
            client.post(
                "/v1/images/generations", json={"prompt": "a cat"}, headers={"Authorization": "Bearer secret"}
            ).status_code,
            200,
        )

    def test_route_absent_without_service(self):
        client = TestClient(create_app())
        self.assertEqual(client.post("/v1/images/generations", json={"prompt": "a cat"}).status_code, 404)

    def test_model_listing(self):
        body = self.client.get("/v1/models").json()
        self.assertIn("fake-image-model", [entry["id"] for entry in body["data"]])


if __name__ == "__main__":
    unittest.main()
