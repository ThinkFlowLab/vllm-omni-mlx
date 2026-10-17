"""Nano cloning service: reference decoding, PCM output and generator lifetime.

Buffered input/format and empty-output validation belong to HTTP/CLI tests.
The service owns reference duration, speed and streaming interval validation.
"""

import base64
import importlib.util
import io
import struct
import threading
import unittest
import wave
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx

from vllm_omni_mlx.tts import moss_nano, moss_nano_loop
from vllm_omni_mlx.tts.moss_nano import (
    MossNanoConfig,
    MossNanoService,
    decode_ref_audio,
)

HAS_MLX_AUDIO = importlib.util.find_spec("mlx_audio") is not None


def reference_wav(sample_rate=48000, channels=1, seconds=0.5):
    """Independent PCM fixture; unequal stereo channels expose bad downmixing."""
    frame = struct.pack("<h", 8192) if channels == 1 else struct.pack("<2h", 0, 16384)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frame * int(sample_rate * seconds))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeModel:
    sample_rate = 48000
    model_type = "moss_tts_nano"

    def __init__(self):
        self.kwargs = None
        self.audio = mx.array([[1.0, -1.0], [0.25, 0.75], [-2.0, -2.0]])
        self.finished = False

    def generate(self, **kwargs):
        self.kwargs = kwargs
        try:
            yield SimpleNamespace(audio=self.audio)
        finally:
            self.finished = True


@unittest.skipUnless(HAS_MLX_AUDIO, "reference decoding needs the [tts] extra")
class NanoReferenceTest(unittest.TestCase):
    def test_resamples_stereo_24k_to_mono_48k(self):
        reference = decode_ref_audio(reference_wav(sample_rate=24000, channels=2))
        self.assertEqual(reference.shape, (24000,))
        self.assertEqual(reference.dtype, mx.float32)
        self.assertTrue(
            mx.allclose(reference[100:-100], mx.array(0.25), atol=0.001).item()
        )

    def test_encoded_bytes_and_base64_decode_identically(self):
        encoded = reference_wav()
        expected = decode_ref_audio(encoded)
        actual = decode_ref_audio(base64.b64decode(encoded))
        self.assertEqual(actual.shape, (24000,))
        self.assertTrue(mx.array_equal(actual, expected).item())

    def test_invalid_reference_payloads(self):
        for data in (None, 42, b"raw audio", b"", "", "not base64", "bm90IGF1ZGlv"):
            with (
                self.subTest(data=data),
                self.assertRaisesRegex(ValueError, "ref_audio"),
            ):
                decode_ref_audio(data)

    def test_empty_decoded_audio_is_a_request_error(self):
        with mock.patch("mlx_audio.audio_io.read", return_value=(mx.zeros(0), 48000)):
            with self.assertRaisesRegex(ValueError, "ref_audio.*zero samples"):
                decode_ref_audio("YQ==")


class NanoPCMTest(unittest.TestCase):
    def test_float32_pcm16_clips_and_truncates_toward_zero(self):
        # The model loop yields float32 audio; +/-0.5 maps to +/-16383.
        audio = mx.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=mx.float32)
        expected = struct.pack("<7h", -32767, -32767, -16383, 0, 16383, 32767, 32767)
        self.assertEqual(moss_nano._pcm16(audio), expected)

    def test_pcm16_preserves_strided_sample_order(self):
        audio = mx.array(
            [-0.75, 9.0, -0.5, 9.0, -0.25, 9.0, 0.0, 9.0, 0.25, 9.0, 0.5, 9.0]
        )[::2]
        self.assertEqual(
            moss_nano._pcm16(audio),
            struct.pack("<6h", -24575, -16383, -8191, 0, 8191, 16383),
        )

    def test_mono_and_stereo_have_same_duration_and_sample_values(self):
        mono = mx.array([0.0, 0.5, -0.5])
        stereo = mx.array([[1.0, -1.0], [0.25, 0.75], [-0.75, -0.25]])
        self.assertEqual(moss_nano._pcm16(stereo), moss_nano._pcm16(mono))

    def test_invalid_audio_shapes_fail(self):
        for shape in ((), (2, 0), (1, 2, 3)):
            with (
                self.subTest(shape=shape),
                self.assertRaisesRegex(ValueError, "audio shaped"),
            ):
                moss_nano._pcm16(mx.zeros(shape))


class NanoServiceTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.service = MossNanoService(self.model)
        self.voice = {"ref_audio": "fixture"}
        self.reference = mx.zeros(24000)
        patcher = mock.patch.object(
            moss_nano, "decode_ref_audio", return_value=self.reference
        )
        self.decode = patcher.start()
        self.addCleanup(patcher.stop)

    def assert_unlocked(self):
        self.assertTrue(self.service._lock.acquire(blocking=False))
        self.service._lock.release()

    def test_cloning_metadata(self):
        self.assertEqual(self.service.voices, [])
        self.assertEqual(self.service.name, self.service.config.model_ref)
        self.assertEqual(self.service.model_type, "moss_tts_nano")
        self.assertEqual(self.service.sample_rate, 48000)

    def test_buffered_output_is_downmixed_pcm16_and_48k_wav(self):
        pcm, media_type = self.service.speech_bytes(
            "hello", voice=self.voice, response_format="pcm"
        )
        self.assertEqual(media_type, "audio/pcm")
        self.assertEqual(pcm, struct.pack("<3h", 0, 16383, -32767))
        payload, media_type = self.service.speech_bytes("hello", voice=self.voice)
        self.assertEqual(media_type, "audio/wav")
        with wave.open(io.BytesIO(payload)) as wav:
            self.assertEqual(
                (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()),
                (48000, 1, 2),
            )
            self.assertEqual(wav.getnframes(), 3)
            self.assertEqual(wav.readframes(3), pcm)
        self.assertTrue(self.model.finished)
        self.assert_unlocked()

    def test_reference_and_all_sampling_options_reach_model(self):
        config = MossNanoConfig(
            max_tokens=42,
            voice_clone_max_text_tokens=30,
            do_sample=False,
            text_temperature=0.7,
            text_top_k=3,
            text_top_p=0.8,
            audio_temperature=0.2,
            audio_top_k=4,
            audio_top_p=0.9,
            audio_repetition_penalty=1.4,
        )
        service = MossNanoService(self.model, config)
        service.speech_bytes("hello", voice=self.voice)
        kwargs = dict(self.model.kwargs)
        self.assertIs(kwargs.pop("ref_audio"), self.reference)
        self.assertEqual(kwargs.pop("text"), "hello")
        self.assertEqual(
            kwargs,
            {
                "max_tokens": 42,
                "voice_clone_max_text_tokens": 30,
                "do_sample": False,
                "text_temperature": 0.7,
                "text_top_k": 3,
                "text_top_p": 0.8,
                "audio_temperature": 0.2,
                "audio_top_k": 4,
                "audio_top_p": 0.9,
                "audio_repetition_penalty": 1.4,
            },
        )

    def test_reference_text_instructions_and_language_are_ignored(self):
        with mock.patch.object(
            moss_nano_loop,
            "synthesize_clone_stream",
            side_effect=lambda *a: (chunk for chunk in [self.model.audio]),
        ) as run:
            for ref_text in (None, "", "transcript"):
                voice = {**self.voice, "ref_text": ref_text}
                self.service.speech_bytes(
                    "hello", voice=voice, instructions="whisper", language="English"
                )
                self.assertNotIn("ref_text", self.model.kwargs)
                self.assertNotIn("instructions", self.model.kwargs)
                self.assertNotIn("language", self.model.kwargs)
                list(
                    self.service.speech_stream(
                        "hello", voice=voice, instructions="whisper", language="English"
                    )
                )
                self.assertEqual(len(run.call_args.args), 4)
                self.assertEqual(run.call_args.kwargs, {})

    def test_clone_and_speed_validation_precedes_lock_and_iteration(self):
        cases = [
            ({"voice": None}, "no preset voices"),
            ({"voice": "Vivian"}, "no preset voices"),
            ({"voice": {}}, "ref_audio"),
            ({"speed": 2.0}, "speed"),
        ]
        with mock.patch.object(self.service, "_lock") as lock:
            lock.__enter__.side_effect = AssertionError(
                "validation entered generation lock"
            )
            for method in (self.service.speech_bytes, self.service.speech_stream):
                for overrides, hint in cases:
                    with (
                        self.subTest(method=method.__name__, overrides=overrides),
                        self.assertRaisesRegex(ValueError, hint),
                    ):
                        method(**{"input": "hello", "voice": self.voice, **overrides})
                self.decode.side_effect = ValueError("ref_audio is invalid")
                with self.assertRaisesRegex(ValueError, "ref_audio"):
                    method("hello", voice=self.voice)
                self.decode.side_effect = None
            lock.__enter__.assert_not_called()

    def test_duration_limits_are_enforced_by_service_for_both_paths(self):
        for frames in (23999, 24000, 1440000, 1440001):
            self.decode.return_value = mx.zeros(frames)
            for method in (self.service.speech_bytes, self.service.speech_stream):
                with self.subTest(frames=frames, method=method.__name__):
                    if frames in (23999, 1440001):
                        with mock.patch.object(self.service, "_lock") as lock:
                            with self.assertRaisesRegex(ValueError, "ref_audio"):
                                method("hello", voice=self.voice)
                            lock.__enter__.assert_not_called()
                    else:
                        result = method("hello", voice=self.voice)
                        if method == self.service.speech_stream:
                            result.close()
        self.assert_unlocked()

    def test_streaming_text_and_intervals_are_eager_errors(self):
        for text in (None, "", "  "):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "input"):
                self.service.speech_stream(text, voice=self.voice)
        for field in ("streaming_interval", "streaming_initial_interval"):
            for value in (0, -1, 11, float("nan"), float("inf")):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(ValueError, field),
                ):
                    self.service.speech_stream(
                        "hello", voice=self.voice, **{field: value}
                    )
                service = MossNanoService(
                    self.model, replace(self.service.config, **{field: value})
                )
                with self.assertRaisesRegex(ValueError, field):
                    service.speech_stream("hello", voice=self.voice)

    def test_streaming_is_lazy_and_overrides_do_not_mutate_service_config(self):
        finished = []
        produced = []

        def chunks(*args):
            try:
                produced.append(1)
                yield self.model.audio[:2]
                produced.append(2)
                yield self.model.audio[2:]
            finally:
                finished.append(True)

        with mock.patch.object(
            moss_nano_loop, "synthesize_clone_stream", side_effect=chunks
        ) as run:
            stream = self.service.speech_stream(
                "hello",
                voice=self.voice,
                streaming_interval=0.5,
                streaming_initial_interval=0.16,
            )
            self.addCleanup(stream.close)
            run.assert_not_called()
            self.assert_unlocked()
            self.assertEqual(next(stream), struct.pack("<2h", 0, 16383))
            self.assertEqual(produced, [1])
            self.assertEqual(finished, [])
            self.assertFalse(self.service._lock.acquire(blocking=False))
            self.assertEqual(list(stream), [struct.pack("<h", -32767)])
            self.assertEqual(finished, [True])
            model, config, text, reference = run.call_args.args
            self.assertIs(model, self.model)
            self.assertEqual(text, "hello")
            self.assertIs(reference, self.reference)
            self.assertEqual(
                config,
                replace(
                    self.service.config,
                    streaming_interval=0.5,
                    streaming_initial_interval=0.16,
                ),
            )
            self.assertEqual(self.service.config, MossNanoConfig())
        self.assert_unlocked()

    def test_partial_stream_close_closes_inner_generator_and_allows_next_request(self):
        finished = []

        def chunks(*args):
            try:
                yield self.model.audio
                self.fail("closing a suspended stream must not request another chunk")
            finally:
                finished.append(True)

        with mock.patch.object(
            moss_nano_loop, "synthesize_clone_stream", side_effect=chunks
        ):
            stream = self.service.speech_stream("hello", voice=self.voice)
            self.addCleanup(stream.close)
            next(stream)
            stream.close()
            stream.close()
            self.assertEqual(finished, [True])
            self.assertEqual(list(stream), [])
        self.assert_unlocked()
        pcm, _ = self.service.speech_bytes(
            "again", voice=self.voice, response_format="pcm"
        )
        self.assertEqual(pcm, struct.pack("<3h", 0, 16383, -32767))

    def test_unstarted_stream_close_does_not_generate(self):
        with mock.patch.object(moss_nano_loop, "synthesize_clone_stream") as run:
            stream = self.service.speech_stream("hello", voice=self.voice)
            stream.close()
            self.assertEqual(list(stream), [])
            run.assert_not_called()
        self.assert_unlocked()

    def test_stream_failure_preserves_delivered_chunk_and_releases_lock(self):
        def chunks(*args):
            yield self.model.audio
            raise RuntimeError("codec failed")

        with mock.patch.object(
            moss_nano_loop, "synthesize_clone_stream", side_effect=chunks
        ):
            stream = self.service.speech_stream("hello", voice=self.voice)
            self.addCleanup(stream.close)
            self.assertEqual(next(stream), struct.pack("<3h", 0, 16383, -32767))
            with self.assertRaisesRegex(RuntimeError, "codec failed"):
                next(stream)
        self.assert_unlocked()

    def test_stream_encoding_failure_closes_synthesis_and_releases_lock(self):
        finished = []

        def chunks(*args):
            try:
                yield mx.zeros((1, 2, 3))
            finally:
                finished.append(True)

        with mock.patch.object(
            moss_nano_loop, "synthesize_clone_stream", side_effect=chunks
        ):
            with self.assertRaisesRegex(ValueError, "audio shaped"):
                list(self.service.speech_stream("hello", voice=self.voice))
        self.assertEqual(finished, [True])
        self.assert_unlocked()

    def test_buffered_generation_failure_releases_lock(self):
        with mock.patch.object(
            self.model, "generate", side_effect=RuntimeError("model failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                self.service.speech_bytes("hello", voice=self.voice)
        self.assert_unlocked()

    def test_active_stream_serializes_buffered_request(self):
        # Observe the second caller reaching acquire(), rather than relying on sleeps.
        waiting = threading.Event()
        real_lock = threading.Lock()

        class ObservedLock:
            def __enter__(self):
                if real_lock.locked():
                    waiting.set()
                real_lock.acquire()

            def __exit__(self, *args):
                real_lock.release()

        self.service._lock = ObservedLock()
        result = []
        errors = []

        def buffered():
            try:
                result.append(self.service.speech_bytes("second", voice=self.voice)[0])
            except Exception as exc:
                errors.append(exc)

        def chunks(*args):
            yield self.model.audio

        with mock.patch.object(
            moss_nano_loop, "synthesize_clone_stream", side_effect=chunks
        ):
            stream = self.service.speech_stream("first", voice=self.voice)
            next(stream)
            worker = threading.Thread(target=buffered, daemon=True)
            worker.start()
            try:
                self.assertTrue(
                    waiting.wait(2), "second request never reached the lock"
                )
                self.assertEqual(result, [])
                self.assertIsNone(self.model.kwargs)
            finally:
                stream.close()
                worker.join(5)
            self.assertFalse(worker.is_alive(), "buffered request remained locked")
        self.assertEqual(errors, [])
        self.assertEqual(len(result), 1)
        self.assertEqual(self.model.kwargs["text"], "second")


class NanoLoaderTest(unittest.TestCase):
    @unittest.skipUnless(HAS_MLX_AUDIO, "loader integration needs the [tts] extra")
    def test_delegates_model_loading_to_mlx_audio(self):
        model = FakeModel()
        config = MossNanoConfig(model_ref="/models/nano")
        with mock.patch("mlx_audio.tts.utils.load_model", return_value=model) as load:
            self.assertIs(moss_nano.load_moss_nano_model(config), model)
        load.assert_called_once_with("/models/nano")


if __name__ == "__main__":
    unittest.main()
