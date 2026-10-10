"""Nano service seams: cloning, sample rate and lock lifetime."""

import base64
import importlib.util
import io
import queue
import struct
import threading
import unittest
import wave
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx

from vllm_omni_mlx.tts import moss_nano
from vllm_omni_mlx.tts.moss_nano import (
    MossNanoConfig,
    MossNanoService,
    decode_ref_audio,
)

HAS_MLX_AUDIO = importlib.util.find_spec("mlx_audio") is not None


def reference_wav(sample_rate=48000, channels=1, seconds=0.5):
    """A real small PCM clip so resampling and channel handling are exercised."""
    # Build the wire-format fixture independently of the MLX conversion under test.
    samples = struct.pack("<h", 8192) * (int(sample_rate * seconds) * channels)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeModel:
    sample_rate = 48000
    config = SimpleNamespace(model_type="moss_tts_nano")

    def __init__(self):
        self.kwargs = None
        # Stereo must be averaged, never flattened into twice the duration.
        self.audio = mx.array(
            [[1.0, -1.0], [0.25, 0.75], [-2.0, -2.0]], dtype=mx.float32
        )
        self.finished = False

    def generate(self, **kwargs):
        self.kwargs = kwargs
        try:
            yield SimpleNamespace(audio=self.audio, sample_rate=self.sample_rate)
        finally:
            self.finished = True


@unittest.skipUnless(
    HAS_MLX_AUDIO, "reference decoding needs the optional mlx-audio extra"
)
class NanoReferenceTest(unittest.TestCase):
    def test_resamples_24k_reference_to_48k_without_losing_stereo(self):
        reference = decode_ref_audio(reference_wav(sample_rate=24000, channels=2))
        self.assertEqual(reference.shape, (24000, 2))
        self.assertIsInstance(reference, mx.array)
        self.assertEqual(reference.dtype, mx.float32)
        self.assertTrue(
            mx.allclose(reference[100:-100], mx.array(0.25), atol=0.001).item()
        )

    def test_mono_remains_mono(self):
        reference = decode_ref_audio(reference_wav(channels=1))
        self.assertEqual(reference.shape, (24000,))

    def test_invalid_reference_payloads(self):
        for data in (
            None,
            b"raw audio",
            "",
            "not base64",
            base64.b64encode(b"not audio").decode(),
        ):
            with (
                self.subTest(data=data),
                self.assertRaisesRegex(ValueError, "ref_audio"),
            ):
                decode_ref_audio(data)

    def test_duration_uses_frames_not_channel_count(self):
        # Both limits are inclusive, including when each frame has two channels.
        for frames in (24000, 30 * 48000):
            for channels in (1, 2):
                shape = (frames,) if channels == 1 else (frames, channels)
                samples = mx.zeros(shape, dtype=mx.float32)
                with (
                    self.subTest(frames=frames, channels=channels),
                    mock.patch(
                        "mlx_audio.audio_io.read", return_value=(samples, 48000)
                    ),
                ):
                    self.assertEqual(decode_ref_audio("YQ==").shape, shape)

    def test_duration_rejects_one_frame_outside_either_bound(self):
        for frames in (24000 - 1, 30 * 48000 + 1):
            for channels in (1, 2):
                shape = (frames,) if channels == 1 else (frames, channels)
                samples = mx.zeros(shape, dtype=mx.float32)
                with (
                    self.subTest(frames=frames, channels=channels),
                    mock.patch(
                        "mlx_audio.audio_io.read", return_value=(samples, 48000)
                    ),
                    self.assertRaisesRegex(ValueError, r"must be 0\.5-30s"),
                ):
                    decode_ref_audio("YQ==")

    def test_bad_decoded_audio_is_a_request_error(self):
        for samples, rate in (
            (mx.array([], dtype=mx.float32), 48000),
            (mx.array([float("nan")]), 48000),
            (mx.array([float("inf")]), 48000),
            (mx.array([float("-inf")]), 48000),
            (mx.zeros((10, 3)), 48000),
            (mx.zeros(10), 24000),
        ):
            with self.subTest(shape=samples.shape, rate=rate):
                with mock.patch(
                    "mlx_audio.audio_io.read", return_value=(samples, rate)
                ):
                    with self.assertRaisesRegex(ValueError, "ref_audio"):
                        decode_ref_audio("YQ==")


class NanoPCMTest(unittest.TestCase):
    def test_pcm16_wire_format_for_mlx_arrays(self):
        # Conversion rounds before casting: +/-0.5 maps to +/-16384.
        expected = struct.pack("<6h", -32767, -16384, 0, 16384, 32767, 32767)
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                audio = mx.array([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=dtype)
                self.assertEqual(moss_nano._pcm16(audio), expected)

    def test_pcm16_preserves_strided_sample_order(self):
        audio = mx.array(
            [-0.75, 9.0, -0.5, 9.0, -0.25, 9.0, 0.0, 9.0, 0.25, 9.0, 0.5, 9.0]
        )[::2]
        expected = struct.pack("<6h", -24575, -16384, -8192, 0, 8192, 16384)
        self.assertEqual(moss_nano._pcm16(audio), expected)

    def test_pcm16_rejects_nonfinite_samples(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(RuntimeError, "non-finite"),
            ):
                moss_nano._pcm16(mx.array([value]))


class NanoServiceTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.service = MossNanoService(self.model)
        self.addCleanup(self.service._pool.shutdown)
        self.voice = {"ref_audio": reference_wav()}
        references = {
            self.voice["ref_audio"]: mx.zeros(24000, dtype=mx.float32),
            reference_wav(sample_rate=24000, channels=2): mx.zeros(
                (24000, 2), dtype=mx.float32
            ),
        }

        def fake_decode(data, sample_rate=48000):
            if data not in references:
                raise ValueError("ref_audio is invalid")
            self.assertEqual(sample_rate, 48000)
            return references[data]

        patcher = mock.patch.object(
            moss_nano, "decode_ref_audio", side_effect=fake_decode
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        codec = mock.patch.object(
            moss_nano, "validate_streaming_codec", return_value=3840
        )
        codec.start()
        self.addCleanup(codec.stop)

    def test_cloning_metadata(self):
        self.assertEqual(self.service.voices, [])
        self.assertEqual(self.service.model_type, "moss_tts_nano")
        self.assertEqual((self.service.sample_rate, self.service.channels), (48000, 1))

    def test_buffered_output_is_downmixed_pcm16_and_48k_wav(self):
        pcm, media_type = self.service.speech_bytes(
            "hello", voice=self.voice, response_format="pcm"
        )
        self.assertEqual(media_type, "audio/pcm")
        self.assertEqual(struct.unpack("<3h", pcm), (0, 16384, -32767))
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

    def test_reference_rate_and_nano_sampling_are_forwarded(self):
        config = MossNanoConfig(
            max_new_frames=42,
            max_text_tokens=30,
            do_sample=False,
            audio_temperature=0.2,
            codec_model_ref="/cached/codec",
        )
        service = MossNanoService(self.model, config)
        self.addCleanup(service._pool.shutdown)
        voice = {
            "ref_audio": reference_wav(sample_rate=24000, channels=2),
            "ref_text": "ignored transcript",
        }
        service.speech_bytes("hello", voice=voice)
        kwargs = self.model.kwargs
        self.assertIsInstance(kwargs["ref_audio"], mx.array)
        self.assertEqual(kwargs["ref_audio"].shape, (24000, 2))
        self.assertEqual(kwargs["ref_audio_sample_rate"], 48000)
        self.assertEqual(kwargs["mode"], "voice_clone")
        self.assertFalse(kwargs["stream"])
        self.assertFalse(kwargs["do_sample"])
        self.assertEqual(kwargs["max_tokens"], 42)
        self.assertEqual(kwargs["voice_clone_max_text_tokens"], 30)
        self.assertEqual(kwargs["audio_temperature"], 0.2)
        self.assertEqual(kwargs["audio_tokenizer_source"], "/cached/codec")
        self.assertNotIn("ref_text", kwargs)

    def test_optional_reference_text_is_ignored(self):
        for ref_text in ("", "transcript", "a completely different transcript"):
            self.service.speech_bytes(
                "hello", voice={**self.voice, "ref_text": ref_text}
            )
            self.assertNotIn("ref_text", self.model.kwargs)

    def test_validation_is_eager_for_buffered_and_streaming(self):
        cases = [
            ({"input": " "}, "input"),
            ({"voice": None}, "no preset voices"),
            ({"voice": "Vivian"}, "no preset voices"),
            ({"voice": {}}, "ref_audio"),
            ({"voice": {**self.voice, "unknown": 1}}, "voice object"),
            ({"voice": {**self.voice, "ref_text": 123}}, "ref_text"),
            ({"voice": {"ref_audio": "invalid"}}, "ref_audio"),
            ({"speed": 2.0}, "speed"),
            ({"instructions": "whisper"}, "instructions"),
            ({"language": "English"}, "language"),
        ]
        # Fail immediately if validation moves inside the lock; holding a real
        # non-reentrant lock here would hang the test on that regression.
        with mock.patch.object(self.service, "_lock") as lock:
            lock.__enter__.side_effect = AssertionError(
                "validation entered the generation lock"
            )
            for method in (self.service.speech_bytes, self.service.speech_stream):
                for overrides, hint in cases:
                    with self.subTest(method=method.__name__, overrides=overrides):
                        with self.assertRaisesRegex(ValueError, hint):
                            method(
                                **{"input": "hello", "voice": self.voice, **overrides}
                            )
            lock.__enter__.assert_not_called()
        self.assertIsNone(self.model.kwargs)

    def test_format_and_intervals_are_eager_errors(self):
        with self.assertRaisesRegex(ValueError, "response_format"):
            self.service.speech_bytes("hello", voice=self.voice, response_format="mp3")
        for field in ("streaming_interval", "streaming_initial_interval"):
            for value in (0, -1, 11, float("nan"), float("inf")):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(ValueError, field),
                ):
                    self.service.speech_stream(
                        "hello", voice=self.voice, **{field: value}
                    )

    def test_streaming_emits_pcm_before_generation_finishes(self):
        resume = threading.Event()
        finished = threading.Event()
        self.addCleanup(resume.set)

        def chunks(*args, **kwargs):
            try:
                yield self.model.audio[:2]
                self.assertTrue(resume.wait(5), "consumer never received first chunk")
                yield self.model.audio[2:]
            finally:
                finished.set()

        with mock.patch.object(
            moss_nano, "synthesize_stream", side_effect=chunks
        ) as run:
            stream = self.service.speech_stream(
                "hello",
                voice=self.voice,
                streaming_interval=0.5,
                streaming_initial_interval=0.08,
            )
            self.addCleanup(stream.close)
            run.assert_not_called()
            first = next(stream)
            self.assertEqual(first, struct.pack("<2h", 0, 16384))
            self.assertFalse(finished.is_set())
            self.assertFalse(self.service._lock.acquire(blocking=False))
            resume.set()
            self.assertEqual(list(stream), [struct.pack("<h", -32767)])
            self.assertTrue(finished.is_set())
            call = run.call_args
            self.assertEqual(call.args[:3], (self.model, self.service.config, "hello"))
            self.assertEqual(call.args[3].shape, (24000,))
            self.assertEqual(call.kwargs["streaming_interval"], 0.5)
            self.assertEqual(call.kwargs["streaming_initial_interval"], 0.08)

    def test_partial_stream_close_cancels_producer_and_leaves_service_usable(self):
        finished = threading.Event()

        def chunks(*args, cancel, **kwargs):
            try:
                yield self.model.audio
                self.assertTrue(
                    cancel.wait(5), "stream close did not cancel generation"
                )
            finally:
                finished.set()

        with mock.patch.object(moss_nano, "synthesize_stream", side_effect=chunks):
            stream = self.service.speech_stream("hello", voice=self.voice)
            self.addCleanup(stream.close)
            next(stream)
            stream.close()
            stream.close()
            self.assertTrue(finished.wait(5))
            self.assertTrue(stream._done.wait(5))
            self.assertTrue(self.service._lock.acquire(blocking=False))
            self.service._lock.release()
            self.assertEqual(list(stream), [])
        pcm, _ = self.service.speech_bytes(
            "hello again", voice=self.voice, response_format="pcm"
        )
        self.assertEqual(pcm, struct.pack("<3h", 0, 16384, -32767))

    def test_unstarted_stream_close_does_not_generate(self):
        with mock.patch.object(moss_nano, "synthesize_stream") as run:
            stream = self.service.speech_stream("hello", voice=self.voice)
            stream.close()
            self.assertEqual(list(stream), [])
            run.assert_not_called()

    def test_slow_consumer_has_bounded_buffer_and_can_cancel(self):
        blocked = threading.Event()
        finished = threading.Event()
        produced = []

        def chunks(*args, **kwargs):
            try:
                for index in range(100):
                    produced.append(index)
                    if index == 2:
                        blocked.set()
                    yield self.model.audio
            finally:
                finished.set()

        with mock.patch.object(moss_nano, "synthesize_stream", side_effect=chunks):
            stream = self.service.speech_stream("hello", voice=self.voice)
            self.addCleanup(stream.close)
            next(stream)
            self.assertTrue(blocked.wait(5))
            stream.close()
            self.assertTrue(finished.wait(5))
            self.assertLessEqual(len(produced), 3)  # delivered + queued + in flight

    def test_streaming_failure_propagates_and_releases_lock(self):
        def chunks(*args, **kwargs):
            yield self.model.audio
            raise RuntimeError("codec failed")

        with mock.patch.object(moss_nano, "synthesize_stream", side_effect=chunks):
            stream = self.service.speech_stream("hello", voice=self.voice)
            self.addCleanup(stream.close)
            next(stream)
            with self.assertRaisesRegex(RuntimeError, "codec failed"):
                next(stream)
            self.assertTrue(self.service._lock.acquire(blocking=False))
            self.service._lock.release()

    def test_final_chunk_is_not_lost_when_completion_races_queue_timeout(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                release = threading.Event()

                def chunks(*args, **kwargs):
                    self.assertTrue(release.wait(5))
                    yield self.model.audio
                    if failure:
                        raise RuntimeError("after final chunk")

                with mock.patch.object(
                    moss_nano, "synthesize_stream", side_effect=chunks
                ):
                    stream = self.service.speech_stream("hello", voice=self.voice)
                    self.addCleanup(stream.close)
                    original_get = stream._queue.get
                    first = True

                    def timed_out_get(*args, **kwargs):
                        nonlocal first
                        if first:
                            first = False
                            # Timeout occurs, then the producer writes and
                            # finishes before the consumer checks completion.
                            release.set()
                            self.assertTrue(stream._done.wait(5))
                            raise queue.Empty
                        return original_get(*args, **kwargs)

                    with mock.patch.object(
                        stream._queue, "get", side_effect=timed_out_get
                    ):
                        self.assertEqual(
                            next(stream), struct.pack("<3h", 0, 16384, -32767)
                        )
                        if failure:
                            with self.assertRaisesRegex(
                                RuntimeError, "after final chunk"
                            ):
                                next(stream)
                        else:
                            with self.assertRaises(StopIteration):
                                next(stream)

    def test_empty_stream_is_a_model_failure(self):
        with mock.patch.object(
            moss_nano, "synthesize_stream", return_value=(x for x in ())
        ):
            with self.assertRaisesRegex(RuntimeError, "generated no audio"):
                list(self.service.speech_stream("hello", voice=self.voice))

    def test_streaming_codec_errors_are_eager(self):
        with mock.patch.object(
            moss_nano,
            "validate_streaming_codec",
            side_effect=RuntimeError("codec unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "codec unavailable"):
                self.service.speech_stream("hello", voice=self.voice)

    def test_buffered_and_streamed_generation_use_the_same_worker(self):
        threads = []
        original_generate = self.model.generate

        def buffered(**kwargs):
            threads.append(threading.get_ident())
            yield from original_generate(**kwargs)

        def streamed(*args, **kwargs):
            threads.append(threading.get_ident())
            yield self.model.audio

        with (
            mock.patch.object(self.model, "generate", side_effect=buffered),
            mock.patch.object(moss_nano, "synthesize_stream", side_effect=streamed),
        ):
            self.service.speech_bytes("one", voice=self.voice)
            list(self.service.speech_stream("two", voice=self.voice))
            self.service.speech_bytes("three", voice=self.voice)
        self.assertEqual(len(threads), 3)
        self.assertEqual(len(set(threads)), 1)
        self.assertNotEqual(threads[0], threading.get_ident())

    def test_seed_is_applied_on_worker_for_every_request(self):
        service = MossNanoService(self.model, MossNanoConfig(seed=17))
        self.addCleanup(service._pool.shutdown)
        random_audio = []

        def buffered(**kwargs):
            audio = mx.random.uniform(low=-1, high=1, shape=(16,))
            random_audio.append(audio.tolist())
            yield SimpleNamespace(audio=audio, sample_rate=48000)

        def streamed(*args, **kwargs):
            audio = mx.random.uniform(low=-1, high=1, shape=(16,))
            random_audio.append(audio.tolist())
            yield audio

        with (
            mock.patch.object(self.model, "generate", side_effect=buffered),
            mock.patch.object(moss_nano, "synthesize_stream", side_effect=streamed),
        ):
            first, _ = service.speech_bytes(
                "one", voice=self.voice, response_format="pcm"
            )
            # Resetting the caller's thread must not change worker reproducibility.
            mx.random.seed(999)
            second = b"".join(service.speech_stream("two", voice=self.voice))
            third, _ = service.speech_bytes(
                "three", voice=self.voice, response_format="pcm"
            )
        self.assertEqual(len(random_audio), 3)
        self.assertEqual(random_audio[0], random_audio[1])
        self.assertEqual(random_audio[1], random_audio[2])
        self.assertEqual(first, second)
        self.assertEqual(second, third)

    def test_generation_failure_releases_lock(self):
        self.model.audio = mx.zeros((1, 2, 3))
        with self.assertRaisesRegex(RuntimeError, "audio shape"):
            self.service.speech_bytes("hello", voice=self.voice)
        self.assertTrue(self.model.finished)
        self.assertTrue(self.service._lock.acquire(blocking=False))
        self.service._lock.release()

    def test_empty_generation_is_a_model_failure(self):
        self.model.audio = mx.zeros(0)
        with self.assertRaisesRegex(RuntimeError, "generated no audio"):
            self.service.speech_bytes("hello", voice=self.voice)

    def test_invalid_budget_fails_before_loading(self):
        for name in ("max_new_frames", "max_text_tokens"):
            for value in (0, -1, 1.5):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    MossNanoConfig(**{name: value})


class NanoLoaderTest(unittest.TestCase):
    @unittest.skipUnless(
        HAS_MLX_AUDIO, "loader integration needs the optional mlx-audio extra"
    )
    def test_loads_codec_at_boot_and_accepts_local_codec_source(self):
        model = FakeModel()
        model._ensure_audio_tokenizer = mock.Mock()
        config = MossNanoConfig(
            model_ref="/models/nano", codec_model_ref="/models/codec"
        )
        with mock.patch("mlx_audio.tts.utils.load_model", return_value=model) as load:
            self.assertIs(moss_nano.load_moss_nano_model(config), model)
        load.assert_called_once_with("/models/nano")
        model._ensure_audio_tokenizer.assert_called_once_with(source="/models/codec")

    def test_rejects_wrong_family_or_rate_before_serving(self):
        for model in (
            SimpleNamespace(config=SimpleNamespace(model_type="qwen3_tts")),
            SimpleNamespace(
                config=SimpleNamespace(model_type="moss_tts_nano"), sample_rate=24000
            ),
        ):
            with self.assertRaisesRegex(ValueError, "MOSS Nano"):
                MossNanoService(model)


if __name__ == "__main__":
    unittest.main()
