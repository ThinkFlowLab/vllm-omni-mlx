"""Voice-cloning scaffold (#49): ref-audio decode + resample, the cloning
``voice`` object contract, and service routing — stub models and real tiny
WAVs, no checkpoint needed. The weight-gated e2e at the bottom runs when the
Base checkpoint is cached and skips elsewhere (CI)."""

import base64
import io
import os

# weight-gated loads resolve from the local HF cache; direct hub access
# only adds a hang when the network is flaky (offline mode keeps loads fast)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest
import wave
from types import SimpleNamespace

import mlx.core as mx

from vllm_omni_mlx.tts.config import TTSConfig
from vllm_omni_mlx.tts.generate import MAX_REF_SECONDS, decode_ref_audio, synthesize_clone
from vllm_omni_mlx.tts.service import TTSService

try:  # decode_ref_audio routes through mlx-audio, which the core install omits
    import mlx_audio  # noqa: F401 — availability probe only; decode is per-test

    HAS_MLX_AUDIO = True
except ImportError:
    HAS_MLX_AUDIO = False

#: tests that decode a reference clip need the [tts] extra; the rest cover
#: validation that rejects before decoding and run everywhere
requires_mlx_audio = unittest.skipUnless(HAS_MLX_AUDIO, "needs the [tts] extra (mlx-audio)")


def wav_bytes_tone(seconds: float, sample_rate: int = 8000, freq: float = 220.0) -> bytes:
    """A real 16-bit mono WAV sine tone — exercises the actual decoder."""
    import math
    import struct

    frames = int(seconds * sample_rate)
    samples = struct.pack(
        f"<{frames}h",
        *[int(12000 * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(frames)],
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples)
    return buffer.getvalue()


def b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def stub_model(tts_model_type):
    return SimpleNamespace(
        config=SimpleNamespace(
            tts_model_type=tts_model_type,
            talker_config=SimpleNamespace(spk_id=None, codec_language_id={}),
        )
    )


class DecodeRefAudioTest(unittest.TestCase):
    @requires_mlx_audio
    def test_decodes_and_resamples_to_24k_mono(self):
        # 1 s of 8 kHz tone → ~24000 samples at 24 kHz
        audio = decode_ref_audio(b64(wav_bytes_tone(1.0, sample_rate=8000)))
        self.assertEqual(audio.dtype, mx.float32)
        self.assertAlmostEqual(audio.size, 24000, delta=4800)

    def test_invalid_base64_is_a_request_error(self):
        with self.assertRaisesRegex(ValueError, "base64"):
            decode_ref_audio("not!!base64!!")

    def test_valid_base64_non_audio_is_a_request_error(self):
        with self.assertRaisesRegex(ValueError, "could not be decoded"):
            decode_ref_audio(b64(b"just some text"))

    @requires_mlx_audio
    def test_zero_sample_audio_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero samples"):
            decode_ref_audio(b64(wav_bytes_tone(0.0)))


class CloneVoiceObjectTest(unittest.TestCase):
    def base_service(self):
        return TTSService(stub_model("base"))

    def test_missing_ref_text_rejected(self):
        with self.assertRaisesRegex(ValueError, "ref_text"):
            self.base_service().speech_bytes("hello", voice={"ref_audio": b64(wav_bytes_tone(2.0))})

    def test_missing_ref_audio_rejected(self):
        with self.assertRaisesRegex(ValueError, "ref_audio"):
            self.base_service().speech_bytes("hello", voice={"ref_text": "what it says"})

    def test_unknown_keys_rejected(self):
        voice = {"ref_audio": b64(wav_bytes_tone(2.0)), "ref_text": "what it says", "style": "warm"}
        with self.assertRaisesRegex(ValueError, "ref_audio and ref_text, got \\['style'\\]"):
            self.base_service().speech_bytes("hello", voice=voice)

    @requires_mlx_audio
    def test_clip_shorter_than_half_second_rejected(self):
        voice = {"ref_audio": b64(wav_bytes_tone(0.2)), "ref_text": "too short"}
        with self.assertRaisesRegex(ValueError, "at least 0.5s"):
            self.base_service().speech_bytes("hello", voice=voice)

    @requires_mlx_audio
    def test_clip_over_cap_rejected(self):
        voice = {"ref_audio": b64(wav_bytes_tone(MAX_REF_SECONDS + 1.5)), "ref_text": "too long"}
        with self.assertRaisesRegex(ValueError, "cap is 30s"):
            self.base_service().speech_bytes("hello", voice=voice)

    def test_cloning_on_custom_voice_needs_base_checkpoint(self):
        service = TTSService(stub_model("custom_voice"))
        voice = {"ref_audio": b64(wav_bytes_tone(2.0)), "ref_text": "what it says"}
        with self.assertRaisesRegex(ValueError, "needs a Base checkpoint"):
            service.speech_bytes("hello", voice=voice)

    def test_streaming_clone_on_custom_voice_needs_base(self):
        # streaming clone is served on Base (#50); a CustomVoice checkpoint
        # rejects the voice object at validation, before any generation
        voice = {"ref_audio": b64(wav_bytes_tone(2.0)), "ref_text": "what it says"}
        service = TTSService(stub_model("custom_voice"))
        with self.assertRaisesRegex(ValueError, "needs a Base checkpoint"):
            service.speech_stream("hello", voice=voice)

    def test_empty_input_rejected_on_clone_path(self):
        voice = {"ref_audio": b64(wav_bytes_tone(2.0)), "ref_text": "what it says"}
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.base_service().speech_bytes("   ", voice=voice)


class SynthesizeCloneGuardTest(unittest.TestCase):
    @requires_mlx_audio
    def test_cap_enforced_at_entry_too(self):
        # decode a >cap clip and pass it straight to the generation entry
        audio = decode_ref_audio(b64(wav_bytes_tone(MAX_REF_SECONDS + 1.5)))
        with self.assertRaisesRegex(ValueError, "cap is 30s"):
            list(synthesize_clone(stub_model("base"), TTSConfig(), "hi", audio, "ref"))


class CloneEndpointTest(unittest.TestCase):
    """Dict ``voice`` through the real server handler + real service, with
    only the generation entry mocked — proves the JSON → routing → 400/200
    wiring (#49's API surface), no weights needed."""

    def setUp(self):
        from unittest import mock

        import vllm_omni_mlx.tts.service as service_module
        from vllm_omni_mlx.server import create_app
        from starlette.testclient import TestClient

        received = {}

        def fake_synthesize_clone(model, config, text, ref_audio, ref_text, **kwargs):
            received["text"], received["ref_text"], received["ref_audio_len"] = text, ref_text, ref_audio.size
            yield mx.zeros(26400)  # 1.1 s of 24 kHz float

        def fake_synthesize_clone_stream(model, config, text, ref_audio, ref_text, **kwargs):
            received["stream_kwargs"] = kwargs
            yield mx.zeros(4800)   # 0.2 s chunk
            yield mx.zeros(48000)  # 2 s tail

        for name, fake in (("synthesize_clone", fake_synthesize_clone), ("synthesize_clone_stream", fake_synthesize_clone_stream)):
            patcher = mock.patch.object(service_module, name, fake)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.received = received
        service = TTSService(stub_model("base"), TTSConfig())
        self.client = TestClient(create_app(tts_service=service, api_key="k1"))
        self.voice = {"ref_audio": b64(wav_bytes_tone(2.0)), "ref_text": "what it says"}

    @requires_mlx_audio
    def test_clone_request_round_trips_through_http(self):
        response = self.client.post(
            "/v1/audio/speech",
            json={"input": "The cloned voice says this.", "voice": self.voice},
            headers={"Authorization": "Bearer k1"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content.startswith(b"RIFF"))
        self.assertGreater(len(response.content), 2 * 24000 + 44)  # >1s of 16-bit mono
        self.assertEqual(self.received["text"], "The cloned voice says this.")
        self.assertEqual(self.received["ref_text"], "what it says")
        self.assertAlmostEqual(self.received["ref_audio_len"], 48000, delta=9600)  # 2s at 24 kHz

    def test_validation_errors_surface_as_400s(self):
        missing_ref_text = {"ref_audio": self.voice["ref_audio"]}
        response = self.client.post(
            "/v1/audio/speech",
            json={"input": "hello", "voice": missing_ref_text},
            headers={"Authorization": "Bearer k1"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("ref_text", response.json()["error"]["message"])

    @requires_mlx_audio  # decode_ref_audio routes through mlx-audio; 400s on the core install
    def test_streaming_clone_streams_pcm(self):
        # #50: stream + voice object routes to the ICL fast path, PCM out
        response = self.client.post(
            "/v1/audio/speech",
            json={"input": "hello", "voice": self.voice, "stream": True, "response_format": "pcm"},
            headers={"Authorization": "Bearer k1"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("audio/pcm"))
        self.assertEqual(len(response.content), (4800 + 48000) * 2)  # two chunks, 16-bit
        self.assertIn("streaming_interval", self.received["stream_kwargs"])


class BaseCloneE2ETest(unittest.TestCase):
    """Weight-gated: full ICL round-trip on the Base checkpoint. Runs where
    `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit` is cached, skips in CI —
    a green CI run does not mean this path was exercised."""

    MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"

    @classmethod
    def setUpClass(cls):
        from vllm_omni_mlx.tts.config import local_snapshot, load_tts_model

        if local_snapshot(cls.MODEL) is None:
            raise unittest.SkipTest(f"{cls.MODEL} not cached locally — download it to run this battery")
        cls.model = load_tts_model(TTSConfig(model_ref=cls.MODEL))
        cls.service = TTSService(cls.model, TTSConfig(model_ref=cls.MODEL))

    @classmethod
    def reference_clip(cls) -> dict:
        # synthesize a reference with the sibling CustomVoice checkpoint if
        # cached; else skip — a real speech clip, not a tone (HNR gates need
        # voiced audio). Cached per class: the donor model is released as
        # soon as the clip exists so it never coexists with Base in memory.
        if getattr(cls, "_clip", None) is None:
            from vllm_omni_mlx.tts.config import DEFAULT_MODEL, load_tts_model, local_snapshot

            if local_snapshot(DEFAULT_MODEL) is None:
                raise unittest.SkipTest("no CustomVoice snapshot cached to synthesize a reference clip")
            from vllm_omni_mlx.tts.generate import synthesize, wav_bytes

            donor = load_tts_model(TTSConfig())
            try:
                clip = wav_bytes(synthesize(donor, TTSConfig(), "This is the voice we are cloning today.", seed=7))
            finally:
                del donor
                mx.clear_cache()
            cls._clip = {"ref_audio": b64(clip), "ref_text": "This is the voice we are cloning today."}
        return cls._clip

    def test_clone_round_trip_is_speech(self):
        from tests.audio_metrics import CATASTROPHIC_HNR_DB, int16_pcm_hnr_db

        voice = self.reference_clip()
        payload, content_type = self.service.speech_bytes("The cloned voice says this.", voice=voice)
        self.assertEqual(content_type, "audio/wav")
        self.assertGreater(len(payload), 24000)  # >1s of 24 kHz 16-bit mono
        pcm = payload[44:]  # past the RIFF header
        hnr = int16_pcm_hnr_db(pcm)
        self.assertGreater(hnr, CATASTROPHIC_HNR_DB, f"HNR {hnr:.2f} dB below catastrophic floor: noise-like clone")

    def test_base_checkpoint_serves_no_preset_voices(self):
        # the correction at the heart of #45: Base ships no spk_id map
        self.assertEqual(self.service.voices, [])
        self.assertEqual(self.service.model_type, "base")


if __name__ == "__main__":
    unittest.main()
