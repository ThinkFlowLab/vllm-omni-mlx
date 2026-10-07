"""POST /v1/audio/speech + GET /v1/audio/voices (#16 / M1.7): validation,
auth, formats, and model listing with a fake service; plus a weight-gated
real round-trip through the actual TTS model."""

import os

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest

from tests._teardown import ReleaseAfterClass
from unittest import mock

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

    def speech_stream(self, input, voice=None, speed=1.0, instructions=None, language=None, streaming_interval=None, streaming_initial_interval=None):
        # eager validation like the real service, then a lazy byte generator
        if not input or not input.strip():
            raise ValueError("input must be a non-empty string")
        if speed != 1.0:
            raise ValueError("speed != 1.0 is not supported yet")
        speaker = (voice or "vivian").lower()
        if speaker not in self.voices:
            raise ValueError(f"voice '{voice}' is not one of the preset voices")
        if streaming_interval is not None and not 0 < streaming_interval <= 10:
            raise ValueError("streaming_interval must be in (0, 10] seconds")
        if streaming_initial_interval is not None and not 0 < streaming_initial_interval <= 10:
            raise ValueError("streaming_initial_interval must be in (0, 10] seconds")

        def gen():
            yield b"CHUNK-ONE|"
            yield f"{speaker}|{input}".encode()

        return gen()


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

    def test_stream_returns_chunked_pcm(self):
        response = self.client.post(
            "/v1/audio/speech",
            json={"input": "hello stream", "voice": "vivian", "stream": True},
            headers={"Authorization": "Bearer k1"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("audio/pcm"))
        self.assertEqual(response.headers["x-audio-sample-rate"], "24000")
        self.assertEqual(response.content, b"CHUNK-ONE|vivian|hello stream")

    def test_stream_validation(self):
        wav = self.client.post(
            "/v1/audio/speech",
            json={"input": "hi", "stream": True, "response_format": "wav"},
            headers={"Authorization": "Bearer k1"},
        )
        self.assertEqual(wav.status_code, 400)
        self.assertIn("pcm", wav.json()["error"]["message"])
        bad_interval = self.client.post(
            "/v1/audio/speech",
            json={"input": "hi", "stream": True, "streaming_interval": 0},
            headers={"Authorization": "Bearer k1"},
        )
        self.assertEqual(bad_interval.status_code, 400)
        bad_initial = self.client.post(
            "/v1/audio/speech",
            json={"input": "hi", "stream": True, "streaming_initial_interval": 0},
            headers={"Authorization": "Bearer k1"},
        )
        self.assertEqual(bad_initial.status_code, 400)
        bad_voice = self.client.post(
            "/v1/audio/speech",
            json={"input": "hi", "stream": True, "voice": "celebrity"},
            headers={"Authorization": "Bearer k1"},
        )
        self.assertEqual(bad_voice.status_code, 400)


class RealSpeechRoundTripTest(ReleaseAfterClass, unittest.TestCase):
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
        # upstream-style floor: >1s of 24 kHz 16-bit mono (min_audio_bytes)
        self.assertGreater(len(response.content), 2 * 24000)
        with wave.open(io.BytesIO(response.content)) as wav:
            self.assertEqual(wav.getframerate(), 24000)
            self.assertEqual(wav.getnchannels(), 1)
            self.assertGreater(wav.getnframes(), 24000)  # >1s of audio

    def test_pcm_output_is_speech_not_noise(self):
        from tests.audio_metrics import CLEAN_VOICE_HNR_DB, int16_pcm_hnr_db

        response = self.client.post(
            "/v1/audio/speech",
            json={"input": "The audio endpoint works end to end.", "voice": "vivian", "response_format": "pcm"},
            headers={"Authorization": "Bearer k1"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.content) % 2, 0)
        self.assertGreater(len(response.content), 2 * 24000)
        hnr = int16_pcm_hnr_db(response.content)
        self.assertGreater(hnr, CLEAN_VOICE_HNR_DB, f"HNR {hnr:.2f} dB below floor: noise-like output")

    def test_streamed_pcm_is_speech(self):
        from tests.audio_metrics import CLEAN_VOICE_HNR_DB, int16_pcm_hnr_db

        response = self.client.post(
            "/v1/audio/speech",
            json={"input": "The streamed audio endpoint works end to end.", "voice": "vivian", "stream": True, "streaming_interval": 1.0},
            headers={"Authorization": "Bearer k1"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("audio/pcm"))
        self.assertEqual(len(response.content) % 2, 0)
        self.assertGreater(len(response.content), 2 * 24000)
        hnr = int16_pcm_hnr_db(response.content)
        self.assertGreater(hnr, CLEAN_VOICE_HNR_DB, f"HNR {hnr:.2f} dB below floor: noise-like output")


class StreamLockReleaseTest(unittest.TestCase):
    """#40 review P3: closing a stream mid-generation releases the service
    lock at the next chunk boundary — the batch-1 guarantee for disconnects."""

    def test_early_close_releases_service_lock(self):
        from types import SimpleNamespace

        import mlx.core as mx

        import vllm_omni_mlx.tts.service as service_module
        from vllm_omni_mlx.tts.config import TTSConfig
        from vllm_omni_mlx.tts.service import TTSService

        def fake_synthesize(model, config, text, **kwargs):
            yield mx.zeros(2400)
            yield mx.zeros(2400)

        model = SimpleNamespace(
            config=SimpleNamespace(
                tts_model_type="custom_voice",
                talker_config=SimpleNamespace(spk_id={"vivian": 1}, codec_language_id={})
            )
        )
        with mock.patch.object(service_module, "synthesize_stream", fake_synthesize):
            service = TTSService(model, TTSConfig())
            stream = service.speech_stream("hold the line", voice="vivian")
            next(stream)  # first chunk arrives -> lock held
            self.assertFalse(service._lock.acquire(blocking=False))
            stream.close()  # client disconnect
            self.assertTrue(service._lock.acquire(blocking=False), "lock leaked after stream close")
            service._lock.release()


if __name__ == "__main__":
    unittest.main()
