"""ASR adapter (#68 tasks 1–2): capability matrix, service validation with a
fake model, POST /v1/audio/transcriptions with a fake service, and a
weight-gated real transcription."""

import importlib.util
import io
import os

# weight-gated loads resolve from the local HF cache (see test_audio_speech.py)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest
import wave
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
from starlette.testclient import TestClient

from tests._teardown import ReleaseAfterClass
from vllm_omni_mlx.asr import capabilities
from vllm_omni_mlx.asr.config import DEFAULT_MODEL, SAMPLE_RATE, ASRConfig, local_snapshot
from vllm_omni_mlx.asr.service import ASRService, Transcription, render
from vllm_omni_mlx.server import create_app

HAS_MULTIPART = importlib.util.find_spec("multipart") is not None
HAS_MLX_AUDIO = importlib.util.find_spec("mlx_audio") is not None



def weights_cached(model_ref):
    """local_snapshot() is true for a config.json-only cache; the real-model
    test needs the safetensors too."""
    path = local_snapshot(model_ref)
    return bool(path) and any(f.endswith(".safetensors") for f in os.listdir(path))


LANGS = ["Chinese", "English", "Japanese"]


def make_wav(seconds=1.0, rate=SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


class FakeModel:
    """Stands in for mlx-audio's qwen3_asr Model: records the generate call."""

    def __init__(self, model_type="qwen3_asr"):
        self.config = SimpleNamespace(model_type=model_type, support_languages=LANGS)
        self.calls = []

    def generate(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return SimpleNamespace(
            text="  hello world ",
            language="English",
            segments=[{"text": "hello world", "start": 0.0, "end": 1.0}],
            prompt_tokens=12,
            generation_tokens=3,
        )


def fake_decode(data, sample_rate, label="audio"):
    return mx.zeros((sample_rate,), dtype=mx.float32)  # 1 s


class CapabilitiesTest(unittest.TestCase):
    def test_qwen3_asr_is_the_served_decoder_family(self):
        family = capabilities.require_served("qwen3_asr")
        self.assertEqual(family.style, capabilities.DECODER)
        metrics, cacheable = capabilities.CLASS_TRAITS[family.style]
        self.assertEqual(metrics, ("TTFT", "TPOT", "ITL"))
        self.assertTrue(cacheable)

    def test_known_but_unserved_families_get_guidance(self):
        for model_type in ("qwen2_audio", "voxtral", "whisper", "parakeet_tdt"):
            with self.assertRaises(ValueError) as ctx:
                capabilities.require_served(model_type)
            self.assertIn("not served yet", str(ctx.exception))
            self.assertIn("qwen3_asr", str(ctx.exception))

    def test_unknown_and_missing_model_type(self):
        with self.assertRaises(ValueError) as ctx:
            capabilities.require_served("totally_new")
        self.assertIn("totally_new", str(ctx.exception))
        self.assertEqual(capabilities.family_of({}), "")
        self.assertEqual(capabilities.family_of({"model_type": " Qwen3_ASR "}), "qwen3_asr")


class ServiceTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.service = ASRService(self.model)
        patcher = mock.patch("vllm_omni_mlx.asr.service.decode_audio", fake_decode)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_wrong_family_fails_at_boot(self):
        with self.assertRaises(ValueError):
            ASRService(FakeModel("whisper"))

    def test_defaults_are_greedy_and_auto_language(self):
        result = self.service.transcribe(b"x")
        _, kwargs = self.model.calls[0]
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertIsNone(kwargs["language"])
        self.assertNotIn("system_prompt", kwargs)
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.duration, 1.0)
        self.assertEqual(result.generation_tokens, 3)

    def test_detected_language_list_is_flattened(self):
        self.model.generate = lambda audio, **kw: SimpleNamespace(text="x", language=["English"], segments=None)
        self.assertEqual(self.service.transcribe(b"x").language, "English")

    def test_prompt_rides_system_prompt_and_hotwords_pass_through(self):
        self.service.transcribe(b"x", prompt=" meeting about MLX ", hotwords=["Qwen", "mlx"])
        _, kwargs = self.model.calls[0]
        self.assertEqual(kwargs["system_prompt"], "meeting about MLX")
        self.assertEqual(kwargs["hotwords"], ["Qwen", "mlx"])

    def test_language_accepts_iso_code_and_name_case_insensitively(self):
        for given, expected in (("en", "English"), ("EN", "English"), ("japanese", "Japanese"), ("zh", "Chinese")):
            self.service.transcribe(b"x", language=given)
            self.assertEqual(self.model.calls[-1][1]["language"], expected)

    def test_validation_errors(self):
        cases = [
            (dict(audio=b""), "non-empty"),
            (dict(audio=b"x", language="klingon"), "not supported"),
            (dict(audio=b"x", language="fr"), "not supported"),  # known ISO, not in this checkpoint's list
            (dict(audio=b"x", temperature=5.0), "temperature"),
        ]
        for kwargs, hint in cases:
            with self.assertRaises(ValueError, msg=kwargs) as ctx:
                self.service.transcribe(**kwargs)
            self.assertIn(hint, str(ctx.exception))
        self.assertEqual(self.model.calls, [])  # nothing reached the model

    def test_upload_cap(self):
        with mock.patch("vllm_omni_mlx.asr.service.MAX_UPLOAD_BYTES", 4):
            with self.assertRaises(ValueError) as ctx:
                self.service.transcribe(b"12345")
        self.assertIn("cap", str(ctx.exception))

    def test_render_formats(self):
        result = Transcription(text="hi", language="English", duration=1.23456, segments=[{"text": "hi"}], prompt_tokens=5, generation_tokens=2)
        self.assertEqual(render(result, "text"), ("hi", "text"))
        self.assertEqual(render(result, "json"), ({"text": "hi"}, "json"))
        verbose, kind = render(result, "verbose_json")
        self.assertEqual(kind, "json")
        self.assertEqual((verbose["task"], verbose["language"], verbose["duration"]), ("transcribe", "English", 1.235))
        self.assertEqual(verbose["usage"], {"prompt_tokens": 5, "completion_tokens": 2})
        with self.assertRaises(ValueError):
            render(result, "srt")


class FakeASRService:
    name = "fake-asr"

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None, prompt=None, hotwords=None, temperature=None):
        if not audio:
            raise ValueError("file must be non-empty audio")
        self.calls.append(dict(audio=audio, language=language, prompt=prompt, hotwords=hotwords, temperature=temperature))
        return Transcription(text="hello", language="English", duration=2.0, segments=[], prompt_tokens=4, generation_tokens=1)


@unittest.skipUnless(HAS_MULTIPART, "needs python-multipart (the [asr] extra)")
class TranscriptionRouteTest(unittest.TestCase):
    AUTH = {"Authorization": "Bearer k1"}

    def setUp(self):
        self.asr = FakeASRService()
        self.client = TestClient(create_app(asr_service=self.asr, api_key="k1"))

    def post(self, data=None, files=True, headers=None):
        kwargs = {"files": {"file": ("a.wav", b"RIFFdata", "audio/wav")}} if files else {}
        return self.client.post("/v1/audio/transcriptions", data=data or {}, headers=self.AUTH if headers is None else headers, **kwargs)

    def test_json_default_and_forwarded_fields(self):
        response = self.post({"language": "en", "prompt": "ctx", "hotwords": "Qwen, MLX\nvLLM", "temperature": "0.2"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "hello"})
        call = self.asr.calls[0]
        self.assertEqual(call["audio"], b"RIFFdata")
        self.assertEqual((call["language"], call["prompt"], call["temperature"]), ("en", "ctx", 0.2))
        self.assertEqual(call["hotwords"], ["Qwen", "MLX", "vLLM"])

    def test_text_and_verbose_json(self):
        text = self.post({"response_format": "text"})
        self.assertEqual(text.text, "hello")
        self.assertTrue(text.headers["content-type"].startswith("text/plain"))
        verbose = self.post({"response_format": "verbose_json"}).json()
        self.assertEqual((verbose["task"], verbose["duration"], verbose["language"]), ("transcribe", 2.0, "English"))

    def test_validation_errors_are_400(self):
        cases = [
            (dict(files=False), "file is required"),
            (dict(data={"response_format": "srt"}), "response_format"),
            (dict(data={"stream": "true"}), "streaming"),
            (dict(data={"temperature": "hot"}), "temperature"),
        ]
        for kwargs, hint in cases:
            response = self.post(**kwargs)
            self.assertEqual(response.status_code, 400, kwargs)
            self.assertIn(hint, response.json()["error"]["message"])
        not_multipart = self.client.post("/v1/audio/transcriptions", json={"file": "x"}, headers=self.AUTH)
        self.assertEqual(not_multipart.status_code, 400)

    def test_oversized_upload_is_413_without_reaching_the_service(self):
        with mock.patch("vllm_omni_mlx.server.asr_max_upload", 8):
            response = self.client.post(
                "/v1/audio/transcriptions", files={"file": ("a.wav", b"123456789", "audio/wav")}, headers=self.AUTH
            )
        self.assertEqual(response.status_code, 413)
        self.assertIn("cap", response.json()["error"]["message"])
        self.assertEqual(self.asr.calls, [])

    def test_oversized_content_length_is_413_before_parsing(self):
        with mock.patch("vllm_omni_mlx.server.asr_max_upload", 8):
            with mock.patch("starlette.requests.Request.form", side_effect=AssertionError("parsed")) as form:
                response = self.client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("a.wav", b"x" * 100_000, "audio/wav")},
                    headers=self.AUTH,
                )
        self.assertEqual(response.status_code, 413)
        form.assert_not_called()
        self.assertEqual(self.asr.calls, [])

    def test_file_typed_text_fields_are_400(self):
        for name in ("prompt", "language", "hotwords", "temperature", "response_format"):
            response = self.client.post(
                "/v1/audio/transcriptions",
                files={"file": ("a.wav", b"RIFFdata", "audio/wav"), name: ("x.txt", b"hi", "text/plain")},
                headers=self.AUTH,
            )
            self.assertEqual(response.status_code, 400, name)
            self.assertIn(name, response.json()["error"]["message"])
        self.assertEqual(self.asr.calls, [])

    def test_service_value_error_maps_to_400(self):
        response = self.client.post(
            "/v1/audio/transcriptions", files={"file": ("a.wav", b"", "audio/wav")}, headers=self.AUTH
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("non-empty", response.json()["error"]["message"])

    def test_auth_and_models_listing(self):
        self.assertEqual(self.post(headers={}).status_code, 401)
        self.assertEqual([m["id"] for m in self.client.get("/v1/models").json()["data"]], ["fake-asr"])

    def test_route_absent_without_asr_service(self):
        client = TestClient(create_app())
        response = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"x", "audio/wav")})
        self.assertEqual(response.status_code, 404)


@unittest.skipUnless(HAS_MLX_AUDIO, "needs mlx-audio (the [asr] extra)")
class DecodeTest(unittest.TestCase):
    def test_wav_decodes_to_16k_mono(self):
        from vllm_omni_mlx.audio_io import decode_audio

        wave_16k = decode_audio(make_wav(0.5, rate=8000), SAMPLE_RATE, "file")
        self.assertEqual(wave_16k.dtype, mx.float32)
        self.assertAlmostEqual(wave_16k.size / SAMPLE_RATE, 0.5, delta=0.02)

    def test_garbage_is_a_value_error_naming_the_field(self):
        from vllm_omni_mlx.audio_io import decode_audio

        with self.assertRaises(ValueError) as ctx:
            decode_audio(b"not audio at all", SAMPLE_RATE, "file")
        self.assertIn("file could not be decoded", str(ctx.exception))


@unittest.skipUnless(HAS_MLX_AUDIO and weights_cached(DEFAULT_MODEL), "ASR weights not cached")
class RealModelTest(ReleaseAfterClass, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vllm_omni_mlx.asr.config import load_asr_model

        cls.service = ASRService(load_asr_model(ASRConfig()))

    def test_silence_transcribes_without_error(self):
        result = self.service.transcribe(make_wav(1.0))
        self.assertIsInstance(result.text, str)
        self.assertAlmostEqual(result.duration, 1.0, delta=0.05)
        self.assertGreater(result.prompt_tokens, 0)


if __name__ == "__main__":
    unittest.main()
