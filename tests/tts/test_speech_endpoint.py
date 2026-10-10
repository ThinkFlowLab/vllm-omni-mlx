"""POST /v1/audio/speech + GET /v1/audio/voices (#16 / M1.7): validation,
auth, formats, and model listing with a fake service; plus a weight-gated
real round-trip through the actual TTS model."""

import asyncio
import json
import os
import threading
from types import SimpleNamespace

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest

from tests._teardown import ReleaseAfterClass
from unittest import mock

from starlette.testclient import TestClient
from starlette.requests import ClientDisconnect

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


class ControlledAudioStream:
    """A source whose second chunk stays in synthesis until the test releases it."""

    def __init__(self):
        self.next_calls = 0
        self.next_started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.closed = threading.Event()
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        self.next_calls += 1
        if self.cancelled.is_set() or self.next_calls > 2:
            raise StopIteration
        if self.next_calls == 1:
            return b"\x01\x00"
        self.next_started.set()
        if not self.release.wait(5):
            raise RuntimeError("test did not release synthesis")
        if self.cancelled.is_set():
            raise StopIteration
        self.finished = True
        return b"\x02\x00"

    def cancel(self):
        self.cancelled.set()
        self.release.set()

    def close(self):
        self.closed.set()


class AudioStreamLifecycleTest(unittest.IsolatedAsyncioTestCase):
    """Exercise real ASGI sends; TestClient buffers the response before returning."""

    async def asyncSetUp(self):
        self.source = ControlledAudioStream()
        self.addCleanup(self.source.release.set)
        self.incoming = asyncio.Queue()
        self.messages = []
        self.first_chunk = asyncio.Event()

    async def _send(self, message):
        self.messages.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            self.first_chunk.set()

    def _start(self, *, send=None, source=None, service=None, payload=None, spec_version="2.3"):
        self.incoming.put_nowait({
            "type": "http.request",
            "body": json.dumps(payload or {"input": "hello", "stream": True}).encode(),
            "more_body": False,
        })
        if service is None:
            service = SimpleNamespace(
                name="fake-live-tts", voices=[], sample_rate=48000,
                speech_stream=lambda *args: self.source if source is None else source,
            )
        app = create_app(tts_service=service)
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": spec_version},
            "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": "/v1/audio/speech", "raw_path": b"/v1/audio/speech",
            "query_string": b"", "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234), "server": ("testserver", 80),
        }
        task = asyncio.create_task(app(scope, self.incoming.get, send or self._send))

        async def cleanup():
            self.source.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        self.addAsyncCleanup(cleanup)
        return task

    async def _wait_thread_event(self, event):
        self.assertTrue(await asyncio.to_thread(event.wait, 2), "stream worker did not reach the expected state")

    async def test_first_chunk_arrives_before_synthesis_finishes_with_backpressure(self):
        allow_send = asyncio.Event()

        async def send(message):
            await self._send(message)
            if message["type"] == "http.response.body" and message.get("body") == b"\x01\x00":
                await allow_send.wait()

        task = self._start(send=send)
        await asyncio.wait_for(self.first_chunk.wait(), 2)
        self.assertFalse(self.source.finished)
        self.assertFalse(task.done())
        self.assertEqual(self.source.next_calls, 1)
        self.assertFalse(self.source.next_started.is_set())
        allow_send.set()
        await self._wait_thread_event(self.source.next_started)
        self.source.release.set()
        await asyncio.wait_for(task, 2)
        bodies = [m["body"] for m in self.messages if m["type"] == "http.response.body"]
        self.assertEqual(bodies, [b"\x01\x00", b"\x02\x00", b""])
        self.assertEqual(dict(self.messages[0]["headers"])[b"x-audio-sample-rate"], b"48000")
        self.assertTrue(self.source.closed.is_set())

    async def test_disconnect_before_first_next_closes_source(self):
        headers_sent = asyncio.Event()

        async def send(message):
            if message["type"] == "http.response.start":
                headers_sent.set()
                await asyncio.Event().wait()

        task = self._start(send=send)
        await asyncio.wait_for(headers_sent.wait(), 2)
        self.incoming.put_nowait({"type": "http.disconnect"})
        await asyncio.wait_for(task, 2)
        self.assertEqual(self.source.next_calls, 0)
        self.assertTrue(self.source.cancelled.is_set())
        self.assertTrue(self.source.closed.is_set())

    async def test_disconnect_interrupts_pending_next_and_closes_source(self):
        task = self._start()
        await self._wait_thread_event(self.source.next_started)
        self.incoming.put_nowait({"type": "http.disconnect"})
        await asyncio.wait_for(task, 2)
        self.assertTrue(self.source.cancelled.is_set())
        await self._wait_thread_event(self.source.closed)
        self.assertFalse(self.source.finished)

    async def test_send_failure_closes_source_even_before_first_next(self):
        async def send(message):
            raise OSError("client connection closed")

        task = self._start(send=send, spec_version="2.4")
        with self.assertRaises(ClientDisconnect):
            await asyncio.wait_for(task, 2)
        self.assertEqual(self.source.next_calls, 0)
        self.assertTrue(self.source.closed.is_set())

    async def test_chunk_send_failure_closes_source(self):
        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("client connection closed")

        task = self._start(send=send, spec_version="2.4")
        with self.assertRaises(ClientDisconnect):
            await asyncio.wait_for(task, 2)
        self.assertEqual(self.source.next_calls, 1)
        self.assertTrue(self.source.cancelled.is_set())
        self.assertTrue(self.source.closed.is_set())

    async def test_task_cancellation_interrupts_pending_next_and_closes_source(self):
        task = self._start(spec_version="2.4")
        await self._wait_thread_event(self.source.next_started)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.source.cancelled.is_set())
        await self._wait_thread_event(self.source.closed)

    async def test_nano_first_chunk_and_disconnect_reach_generation_worker(self):
        import mlx.core as mx

        from vllm_omni_mlx.tts import moss_nano

        model = SimpleNamespace(
            sample_rate=48000, config=SimpleNamespace(model_type="moss_tts_nano")
        )
        service = moss_nano.MossNanoService(model)
        self.addCleanup(service._pool.shutdown)
        finished = threading.Event()

        def synthesize(*args, cancel, **kwargs):
            try:
                yield mx.array([0.5, -0.5])
                if not cancel.wait(5):
                    raise RuntimeError("HTTP disconnect did not cancel Nano generation")
            finally:
                finished.set()

        with (
            mock.patch.object(moss_nano, "decode_ref_audio", return_value=mx.zeros(24000)),
            mock.patch.object(moss_nano, "validate_streaming_codec"),
            mock.patch.object(moss_nano, "synthesize_stream", side_effect=synthesize),
        ):
            task = self._start(service=service, payload={
                "input": "hello", "stream": True, "voice": {"ref_audio": "fixture"},
            })
            await asyncio.wait_for(self.first_chunk.wait(), 2)
            self.assertFalse(finished.is_set())
            bodies = [m["body"] for m in self.messages if m["type"] == "http.response.body"]
            self.assertEqual(bodies, [b"\x00\x40\x00\xc0"])
            self.assertFalse(service._lock.acquire(blocking=False))
            self.incoming.put_nowait({"type": "http.disconnect"})
            await asyncio.wait_for(task, 2)
            await self._wait_thread_event(finished)
            await asyncio.to_thread(service._pool.shutdown)
        self.assertTrue(service._lock.acquire(blocking=False))
        service._lock.release()

    async def test_plain_generator_closes_on_its_iteration_worker_after_disconnect(self):
        started = threading.Event()
        release = threading.Event()
        closed = threading.Event()
        threads = {}
        self.addCleanup(release.set)

        def chunks():
            try:
                yield b"\x01\x00"
                threads["next"] = threading.get_ident()
                started.set()
                if not release.wait(5):
                    raise RuntimeError("test did not release synthesis")
                yield b"\x02\x00"
            finally:
                threads["close"] = threading.get_ident()
                closed.set()

        task = self._start(source=chunks())
        await self._wait_thread_event(started)
        self.incoming.put_nowait({"type": "http.disconnect"})
        await asyncio.wait_for(task, 2)
        self.assertFalse(closed.is_set())
        release.set()
        await self._wait_thread_event(closed)
        self.assertEqual(threads["close"], threads["next"])


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
