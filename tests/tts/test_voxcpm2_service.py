"""VoxCPM2 serving seam (#71): config, detection, service validation, and
the stream/wav plumbing with a stub model — no checkpoint needed. The
weight-gated battery lives in test_voxcpm2_e2e.py and runs only where the
snapshot is cached."""

import base64
import io
import os
import wave

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest
import unittest.mock

import mlx.core as mx
from starlette.testclient import TestClient

from vllm_omni_mlx.server import create_app
from vllm_omni_mlx.tts.voxcpm2 import (
    ARCH,
    VoxCPM2Config,
    VoxCPM2Service,
    config_is_voxcpm2,
    is_voxcpm2_model,
)

try:  # decode_ref_audio routes through mlx-audio, which the core install omits
    import mlx_audio  # noqa: F401 — availability probe only; decode is per-test

    HAS_MLX_AUDIO = True
except ImportError:
    HAS_MLX_AUDIO = False

#: tests that decode a reference clip need the [tts] extra; the rest cover
#: validation that rejects before decoding and run everywhere
requires_mlx_audio = unittest.skipUnless(HAS_MLX_AUDIO, "needs the [tts] extra (mlx-audio)")

SR = 48000


def _sine(seconds: float = 1.0, freq: float = 220.0) -> mx.array:
    import math

    n = int(seconds * SR)
    return mx.array([0.5 * math.sin(2 * math.pi * freq * i / SR) for i in range(n)], dtype=mx.float32)


def _sine_wav_bytes(seconds: float, freq: float = 220.0, sample_rate: int = SR) -> bytes:
    """A real 16-bit mono WAV — exercises the actual ref-audio decoder."""
    import math

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        n = int(seconds * sample_rate)
        frames = b"".join(
            int(0.8 * math.sin(2 * math.pi * freq * i / sample_rate) * 32767).to_bytes(2, "little", signed=True)
            for i in range(n)
        )
        wav.writeframes(frames)
    return buffer.getvalue()


class _StubVoxCPM2Model:
    """Duck-types mlx-audio's voxcpm2.Model: module identity (the #71
    detection seam), 48 kHz sample_rate, and a generate() that records the
    call and yields one chunk — the library's generate is single-yield."""

    sample_rate = SR

    def __init__(self, seconds: float = 1.0):
        self._audio = _sine(seconds)
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        yield type("Result", (), {"audio": self._audio})


def _stub_service(seconds: float = 1.0) -> tuple[VoxCPM2Service, _StubVoxCPM2Model]:
    # detection goes through the class's module; point the stub at the real
    # package so the service accepts it exactly like a loaded checkpoint
    _StubVoxCPM2Model.__module__ = "mlx_audio.tts.models.voxcpm2.voxcpm2"
    model = _StubVoxCPM2Model(seconds)
    return VoxCPM2Service(model, VoxCPM2Config(model_ref="stub/voxcpm2")), model


class DetectionTest(unittest.TestCase):
    def test_config_json_architecture_key(self):
        self.assertTrue(config_is_voxcpm2({"architecture": "voxcpm2"}))
        self.assertFalse(config_is_voxcpm2({"architecture": "minicpm"}))
        self.assertFalse(config_is_voxcpm2({"tts_model_type": "custom_voice"}))

    def test_module_prefix_after_load(self):
        service, _ = _stub_service()
        self.assertTrue(is_voxcpm2_model(service._model))
        self.assertFalse(is_voxcpm2_model(object()))


class ConfigTest(unittest.TestCase):
    def test_defaults_match_serving_knobs(self):
        config = VoxCPM2Config()
        # t=8: the knee that survives the quantized-blocks compound
        # (t=6 was equivalent on bf16 alone; see voxcpm2.VoxCPM2Config)
        self.assertEqual(config.inference_timesteps, 8)
        self.assertEqual(config.cfg_value, 2.0)
        self.assertEqual(config.max_tokens, 2000)  # AR patches, ~20 ms each
        self.assertEqual(config.warmup_patches, 0)
        self.assertEqual(config.model_ref, "mlx-community/VoxCPM2-4bit")

    def test_with_overrides_filters_none_and_model_ref(self):
        config = VoxCPM2Config().with_overrides(instruct=None, inference_timesteps=6, model_ref="other/repo")
        self.assertEqual(config.inference_timesteps, 6)
        self.assertEqual(config.model_ref, "mlx-community/VoxCPM2-4bit")




def _fake_frames(model, config, text, instruct=None, ref_audio=None, ref_text=None, compiled=True):
    """Stub stand-in for voxcpm2_loop.generate_frames (#79): forwards to the
    stub model's generate with the config knobs, so call-recording tests keep
    working on the default (vendored-loop) route."""
    for result in model.generate(
        text=text, instruct=instruct, ref_audio=ref_audio, ref_text=ref_text,
        max_tokens=config.max_tokens, inference_timesteps=config.inference_timesteps,
        cfg_value=config.cfg_value, warmup_patches=config.warmup_patches,
    ):
        if result.audio is not None and result.audio.size:
            yield result.audio


def _patch_loop_for_stubs():
    return unittest.mock.patch("vllm_omni_mlx.tts.voxcpm2_loop.generate_frames", _fake_frames)


class ServiceSurfaceTest(unittest.TestCase):
    def setUp(self):
        _patch_loop_for_stubs().start()
        self.addCleanup(unittest.mock.patch.stopall)
        self.service, self.model = _stub_service()

    def test_duck_type_surface(self):
        self.assertEqual(self.service.model_type, ARCH)
        self.assertEqual(self.service.voices, ["default"])  # zero-shot; no speaker presets
        self.assertEqual(self.service.sample_rate, SR)
        self.assertEqual(self.service.name, "stub/voxcpm2")

    def test_rejects_non_voxcpm2_models(self):
        with self.assertRaises(ValueError):
            VoxCPM2Service(object())

    def test_zero_shot_is_the_default_voice(self):
        payload, content_type = self.service.speech_bytes("hello")
        self.assertEqual(content_type, "audio/wav")
        self.assertEqual(self.model.calls[-1]["instruct"], None)
        self.assertEqual(self.model.calls[-1]["ref_audio"], None)
        self.assertTrue(payload.startswith(b"RIFF"))
        with wave.open(io.BytesIO(payload)) as wav:
            self.assertEqual(wav.getframerate(), SR)

    def test_voice_default_is_case_insensitive(self):
        self.service.speech_bytes("hello", voice="DEFAULT")
        self.assertEqual(self.model.calls[-1]["instruct"], None)

    def test_instructions_design_a_voice(self):
        self.service.speech_bytes("hello", instructions="A warm, low voice")
        self.assertEqual(self.model.calls[-1]["instruct"], "A warm, low voice")

    def test_pcm_format(self):
        payload, content_type = self.service.speech_bytes("hello", response_format="pcm")
        self.assertEqual(content_type, "audio/pcm")
        self.assertEqual(len(payload) % 2, 0)
        self.assertEqual(len(payload), 2 * int(1.0 * SR))

    def test_stream_slices_the_finished_buffer(self):
        # mlx-audio's generate is single-yield (#71 baseline): the stream
        # delivers the finished buffer in interval-sized chunks
        chunks = list(self.service.speech_stream("hello", streaming_interval=0.25))
        self.assertGreater(len(chunks), 1)
        self.assertEqual(sum(len(c) for c in chunks), 2 * int(1.0 * SR))
        for chunk in chunks[:-1]:
            self.assertEqual(len(chunk), 2 * int(0.25 * SR))

    def test_generate_receives_config_knobs(self):
        service, model = _stub_service()
        service.config = VoxCPM2Config(inference_timesteps=6, cfg_value=3.0, warmup_patches=1)
        service.speech_bytes("hello")
        self.assertEqual(model.calls[-1]["inference_timesteps"], 6)
        self.assertEqual(model.calls[-1]["cfg_value"], 3.0)
        self.assertEqual(model.calls[-1]["warmup_patches"], 1)


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.service, self.model = _stub_service()

    def test_empty_input(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.service.speech_bytes("   ")

    def test_speed_must_be_one(self):
        with self.assertRaisesRegex(ValueError, "speed"):
            self.service.speech_bytes("hello", speed=1.5)

    def test_unknown_voice_gets_guidance(self):
        with self.assertRaisesRegex(ValueError, "ref_audio"):
            self.service.speech_bytes("hello", voice="vivian")

    def test_language_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "language"):
            self.service.speech_bytes("hello", language="zh")

    def test_stream_interval_bounds(self):
        with self.assertRaisesRegex(ValueError, "streaming_interval"):
            list(self.service.speech_stream("hello", streaming_interval=0.0))
        with self.assertRaisesRegex(ValueError, "streaming_initial_interval"):
            list(self.service.speech_stream("hello", streaming_initial_interval=11.0))

    def test_voice_object_unknown_keys(self):
        with self.assertRaisesRegex(ValueError, "ref_audio and ref_text"):
            self.service.speech_bytes("hello", voice={"spk": "x"})

    def test_voice_object_requires_ref_audio(self):
        with self.assertRaisesRegex(ValueError, "ref_audio"):
            self.service.speech_bytes("hello", voice={"ref_text": "hi"})

    def test_voice_object_empty_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.service.speech_bytes("   ", voice={"ref_audio": "x"})


@requires_mlx_audio
class CloneDecodeTest(unittest.TestCase):
    """Reference-clip decoding + duration caps — real WAVs through
    mlx-audio's decoder, no checkpoint."""

    def setUp(self):
        _patch_loop_for_stubs().start()
        self.addCleanup(unittest.mock.patch.stopall)
        self.service, self.model = _stub_service()

    def test_clone_passes_ref_waveform(self):
        ref = base64.b64encode(_sine_wav_bytes(1.0)).decode()
        self.service.speech_bytes("hello", voice={"ref_audio": ref})
        call = self.model.calls[-1]
        self.assertIsNone(call["instruct"])
        self.assertIsNotNone(call["ref_audio"])
        self.assertEqual(call["ref_audio"].size, SR)  # decoded to the output rate

    def test_clone_accepts_unused_ref_text(self):
        # mlx-audio's reference mode ignores the transcript (cloning is
        # ref-audio-driven); accepted for API parity with the Qwen3 object
        ref = base64.b64encode(_sine_wav_bytes(1.0)).decode()
        self.service.speech_bytes("hello", voice={"ref_audio": ref, "ref_text": "spoken words"})
        self.assertEqual(self.model.calls[-1]["ref_text"], "spoken words")

    def test_ref_too_short(self):
        ref = base64.b64encode(_sine_wav_bytes(0.2)).decode()
        with self.assertRaisesRegex(ValueError, "at least 0.5s"):
            self.service.speech_bytes("hello", voice={"ref_audio": ref})

    def test_ref_too_long(self):
        ref = base64.b64encode(_sine_wav_bytes(31.0, sample_rate=8000)).decode()
        with self.assertRaisesRegex(ValueError, "cap is 30"):
            self.service.speech_bytes("hello", voice={"ref_audio": ref})

    def test_ref_not_audio(self):
        with self.assertRaisesRegex(ValueError, "ref_audio"):
            self.service.speech_bytes("hello", voice={"ref_audio": base64.b64encode(b"not audio").decode()})

    def test_ref_bad_base64(self):
        with self.assertRaisesRegex(ValueError, "base64"):
            self.service.speech_bytes("hello", voice={"ref_audio": "!!!not-base64!!!"})

    def test_ref_blank_ref_text_rejected(self):
        ref = base64.b64encode(_sine_wav_bytes(1.0)).decode()
        with self.assertRaisesRegex(ValueError, "ref_text"):
            self.service.speech_bytes("hello", voice={"ref_audio": ref, "ref_text": "  "})


class VoxCPM2RoutesTest(unittest.TestCase):
    """Endpoint wiring with the stub service: 48 kHz headers, voices list,
    voice-object cloning through the JSON API."""

    def setUp(self):
        _patch_loop_for_stubs().start()
        self.addCleanup(unittest.mock.patch.stopall)
        self.service, self.model = _stub_service()
        self.client = TestClient(create_app(tts_service=self.service))

    def test_voices_endpoint(self):
        response = self.client.get("/v1/audio/voices")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"object": "list", "voices": ["default"]})

    def test_stream_headers_carry_model_rate(self):
        with self.client.stream(
            "POST", "/v1/audio/speech", json={"input": "hello", "stream": True, "response_format": "pcm"}
        ) as response:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["X-Audio-Sample-Rate"], "48000")
            body = b"".join(response.iter_bytes())
        self.assertGreater(len(body), 0)

    def test_stream_wav_rejected(self):
        response = self.client.post(
            "/v1/audio/speech", json={"input": "hello", "stream": True, "response_format": "wav"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("stream: false", response.json()["error"]["message"])

    def test_stream_omitted_format_is_pcm(self):
        # the established stream contract: an omitted response_format means
        # pcm (only an explicit wav is rejected)
        response = self.client.post("/v1/audio/speech", json={"input": "hello", "stream": True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("audio/pcm"))

    def test_unknown_voice_is_a_400_with_guidance(self):
        response = self.client.post("/v1/audio/speech", json={"input": "hello", "voice": "ryan"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("ref_audio", response.json()["error"]["message"])

    def test_buffered_wav_round_trip(self):
        response = self.client.post("/v1/audio/speech", json={"input": "hello", "voice": "default"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        with wave.open(io.BytesIO(response.content)) as wav:
            self.assertEqual(wav.getframerate(), SR)


class SynthesizeRoutingTest(unittest.TestCase):
    """#79: synthesize rides the vendored compiled loop by default and the
    VLLM_OMNI_VOXCPM2_EAGER env selects the library generate — verified with
    stubs, no checkpoint (the weight-gated parity battery covers the real
    numerics)."""

    def setUp(self):
        unittest.mock.patch("vllm_omni_mlx.tts.voxcpm2_loop.generate_frames").start()
        self.addCleanup(unittest.mock.patch.stopall)
        self.service, self.model = _stub_service()

    def test_default_routes_to_the_vendored_loop(self):
        from vllm_omni_mlx.tts import voxcpm2, voxcpm2_loop

        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_OMNI_VOXCPM2_EAGER", None)
            list(voxcpm2.synthesize(self.service._model, self.service.config, "hello", instruct="a voice"))
        voxcpm2_loop.generate_frames.assert_called_once()
        kwargs = voxcpm2_loop.generate_frames.call_args.kwargs
        self.assertEqual(kwargs["instruct"], "a voice")
        self.assertEqual(self.model.calls, [], "library generate must not run on the default path")

    def test_eager_env_routes_to_library_generate(self):
        from vllm_omni_mlx.tts import voxcpm2, voxcpm2_loop

        with unittest.mock.patch.dict(os.environ, {"VLLM_OMNI_VOXCPM2_EAGER": "1"}):
            list(voxcpm2.synthesize(self.service._model, self.service.config, "hello"))
        self.assertEqual(len(self.model.calls), 1, "escape env must use the library generate")
        self.assertFalse(voxcpm2_loop.generate_frames.called)
