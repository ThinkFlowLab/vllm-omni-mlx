"""POST /v1/audio/speech + GET /v1/audio/voices (#16 / M1.7): validation,
auth, formats, and model listing with a fake service; plus a weight-gated
real round-trip through the actual TTS model."""

import unittest

from starlette.testclient import TestClient

from vllm_omni_mlx.server import create_app
from vllm_omni_mlx.tts.service import TTSService


class FakeTTSService:
    name = "fake-tts"
    voices = ["vivian", "ryan"]

    def speech_bytes(self, input, voice=None, response_format="wav", speed=1.0, instructions=None, language=None):
        if not input or not input.strip():
            raise ValueError("input must be a non-empty string")
        if response_format not in ("wav", "pcm"):
            raise ValueError(f"response_format must be 'wav' or 'pcm', got '{response_format}'")
        if speed != 1.0:
            raise ValueError("speed != 1.0 is not supported yet")
        speaker = (voice or "vivian").lower()
        if speaker not in self.voices:
            raise ValueError(f"voice '{voice}' is not one of the preset voices")
        payload = f"{speaker}|{input}|{instructions}|{language}".encode()
        return (payload if response_format == "pcm" else b"RIFF" + payload), "audio/pcm" if response_format == "pcm" else "audio/wav"


class AudioRoutesTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(tts_service=FakeTTSService(), api_key="k1"))

    def test_speech_round_trip(self):
        response = self.client.post(
            "/v1/audio/speech",
            json={"input": "hello", "voice": "vivian"},
            headers={"Authorization": "Bearer k1"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertTrue(response.content.startswith(b"RIFF"))
        self.assertIn(b"vivian|hello", response.content)

    def test_pcm_format_and_default_voice(self):
        response = self.client.post("/v1/audio/speech", json={"input": "hi"}, headers={"Authorization": "Bearer k1"})
        self.assertEqual(response.headers["content-type"], "audio/wav")  # wav default
        pcm = self.client.post(
            "/v1/audio/speech", json={"input": "hi", "response_format": "pcm"}, headers={"Authorization": "Bearer k1"}
        )
        self.assertEqual(pcm.headers["content-type"], "audio/pcm")
        self.assertIn(b"vivian|hi", pcm.content)

    def test_validation_errors(self):
        cases = [
            ({"input": "  "}, "input"),
            ({"input": "hi", "response_format": "mp3"}, "response_format"),
            ({"input": "hi", "speed": 1.5}, "speed"),
            ({"input": "hi", "voice": "celebrity"}, "voice"),
        ]
        for body, hint in cases:
            response = self.client.post("/v1/audio/speech", json=body, headers={"Authorization": "Bearer k1"})
            self.assertEqual(response.status_code, 400, body)
            self.assertIn(hint, response.json()["error"]["message"])
        missing = self.client.post("/v1/audio/speech", json={}, headers={"Authorization": "Bearer k1"})
        self.assertEqual(missing.status_code, 400)

    def test_auth_guards_audio_routes(self):
        self.assertEqual(self.client.post("/v1/audio/speech", json={"input": "hi"}).status_code, 401)
        self.assertEqual(self.client.get("/v1/audio/voices").status_code, 401)
        ok = self.client.get("/v1/audio/voices", headers={"x-api-key": "k1"})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json(), {"object": "list", "voices": ["vivian", "ryan"]})

    def test_models_list_includes_tts_without_backend(self):
        data = self.client.get("/v1/models").json()
        self.assertEqual([m["id"] for m in data["data"]], ["fake-tts"])


class RealSpeechRoundTripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vllm_omni_mlx.tts.config import DEFAULT_MODEL, TTSConfig, load_tts_model, local_snapshot

        if local_snapshot(DEFAULT_MODEL) is None:
            raise unittest.SkipTest(f"{DEFAULT_MODEL} not cached locally")
        cls.service = TTSService(load_tts_model(TTSConfig()))
        cls.client = TestClient(create_app(tts_service=cls.service, api_key="k1"))

    def test_wav_round_trip_playable(self):
        import io
        import wave

        response = self.client.post(
            "/v1/audio/speech",
            json={"input": "The audio endpoint works.", "voice": "vivian", "response_format": "wav"},
            headers={"Authorization": "Bearer k1"},
        )
        self.assertEqual(response.status_code, 200)
        with wave.open(io.BytesIO(response.content)) as wav:
            self.assertEqual(wav.getframerate(), 24000)
            self.assertEqual(wav.getnchannels(), 1)
            self.assertGreater(wav.getnframes(), 24000)  # >1s of audio


if __name__ == "__main__":
    unittest.main()
