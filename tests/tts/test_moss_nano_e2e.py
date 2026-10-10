"""Real Nano/codec regression checks, skipped unless BOTH checkpoints are local.

Override MOSS_NANO_MODEL / MOSS_NANO_CODEC with local snapshot directories.
Tests never download weights. The deterministic reference signal exercises
cloning mechanics; these are numerical/streaming tests, not speech-quality tests.
"""

import base64
import contextlib
import io
import os
import threading
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from unittest import mock

import mlx.core as mx
import numpy as np

from vllm_omni_mlx.tts import moss_nano_loop
from vllm_omni_mlx.tts.config import local_snapshot
from vllm_omni_mlx.tts.moss_nano import (
    DEFAULT_CODEC_MODEL,
    DEFAULT_MODEL,
    MossNanoConfig,
    MossNanoService,
    load_moss_nano_model,
)


def _cached_checkpoint(source, *, text_tokenizer=False):
    """Reject partial cache entries as well as entirely missing checkpoints."""
    path = Path(source).expanduser()
    if not path.is_dir():
        snapshot = local_snapshot(source)
        if snapshot is None:
            return None
        path = Path(snapshot)
    required = [path / "config.json"]
    if text_tokenizer:
        required.append(path / "tokenizer.model")
    weights = list(path.glob("*.safetensors"))
    if not weights or not all(
        p.is_file() and p.stat().st_size for p in required + weights
    ):
        return None
    return str(path)


class MossNanoE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_source = os.environ.get("MOSS_NANO_MODEL", DEFAULT_MODEL)
        codec_source = os.environ.get("MOSS_NANO_CODEC", DEFAULT_CODEC_MODEL)
        model_path = _cached_checkpoint(model_source, text_tokenizer=True)
        codec_path = _cached_checkpoint(codec_source)
        if model_path is None or codec_path is None:
            missing = [
                source
                for source, path in (
                    (model_source, model_path),
                    (codec_source, codec_path),
                )
                if path is None
            ]
            raise unittest.SkipTest(
                "Nano E2E needs complete local checkpoints: " + ", ".join(missing)
            )
        try:
            import mlx_audio  # noqa: F401
        except ImportError:
            raise unittest.SkipTest(
                "Nano E2E needs the optional [tts] dependencies"
            ) from None

        offline = mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"})
        offline.start()
        cls.addClassCleanup(offline.stop)
        cls.config = MossNanoConfig(
            model_ref=model_path,
            codec_model_ref=codec_path,
            max_new_frames=16,
            do_sample=False,
        )
        cls.model = load_moss_nano_model(cls.config)
        cls.addClassCleanup(cls._release_model)
        # 0.64 s, above the service's minimum reference duration. No RNG state
        # is consumed, so fixed-seed sampling comparisons start identically.
        t = np.arange(30720, dtype=np.float32) / 48000
        signal = (0.12 * np.sin(2 * np.pi * (180 * t + 70 * t * t))).astype(np.float32)
        cls.reference = mx.array(signal)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            wav.writeframes(np.round(signal * 32767).astype("<i2").tobytes())
        cls.voice = {"ref_audio": base64.b64encode(buffer.getvalue()).decode("ascii")}

    @classmethod
    def _release_model(cls):
        cls.model = None
        cls.reference = None
        mx.clear_cache()

    def _upstream_codes(self, text, config):
        from mlx_audio.tts.models.moss_tts_nano.text import (
            lightweight_normalize_text,
            split_text_into_best_sentences,
        )

        reference_codes = self.model.encode_reference_audio(
            self.reference,
            sample_rate=48000,
            num_quantizers=self.model.config.n_vq,
            source=config.codec_model_ref,
        )
        chunks = split_text_into_best_sentences(
            self.model.tokenizer,
            lightweight_normalize_text(text),
            max_tokens=config.max_text_tokens,
        )
        outputs = []
        for chunk in chunks:
            inputs, mask = self.model.build_inference_input_ids(
                text=chunk,
                tokenizer=self.model.tokenizer,
                mode="voice_clone",
                prompt_audio_codes=reference_codes,
            )
            outputs.append(
                self.model.generate_audio_token_ids(
                    prompt_input_ids=inputs,
                    attention_mask=mask,
                    max_new_frames=config.max_new_frames,
                    **{
                        name: getattr(config, name)
                        for name in (
                            "do_sample",
                            "text_temperature",
                            "text_top_p",
                            "text_top_k",
                            "audio_temperature",
                            "audio_top_p",
                            "audio_top_k",
                            "audio_repetition_penalty",
                        )
                    },
                )[0]
            )
        self.assertTrue(outputs, "upstream text split produced no prompts")
        codes = mx.concatenate(outputs, axis=0)
        mx.eval(codes)
        self.assertGreater(
            codes.shape[0], 1, "need multiple real frames for this regression"
        )
        return codes

    def test_incremental_codes_match_upstream_greedy_and_seeded_sampling(self):
        cases = (
            (False, 0, "Hello, this is a streaming test.", 75),
            (True, 41, "Hello, this is a streaming test.", 75),
            (True, 73, "你好，这是测试。今天我们验证流式合成。", 8),
        )
        for sampled, seed, text, text_budget in cases:
            with self.subTest(sampled=sampled, seed=seed, text=text):
                config = replace(
                    self.config, do_sample=sampled, max_text_tokens=text_budget
                )
                mx.random.seed(seed)
                expected = self._upstream_codes(text, config)
                mx.random.seed(seed)
                frames = list(
                    moss_nano_loop.iter_audio_frames(
                        self.model,
                        config,
                        text,
                        self.reference,
                    )
                )
                self.assertTrue(frames, "incremental generation produced no frames")
                actual = mx.stack(frames)
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_real_codec_matches_whole_decode_across_partitions(self):
        codes = self._upstream_codes("Check the audio decoder.", self.config)
        codec = self.model.audio_tokenizer
        expected = np.asarray(codec.decode_audio_codes(codes))
        hop = moss_nano_loop.validate_streaming_codec(self.model)
        self.assertEqual(hop, 3840)
        self.assertEqual(expected.shape, (codes.shape[0] * hop, 2))
        self.assertTrue(np.isfinite(expected).all())
        for partition in (1, 3, 7):
            with self.subTest(partition=partition):
                decoder = codec.make_streaming_decoder(
                    num_quantizers=self.model.config.n_vq
                )
                pieces = [
                    decoder.decode_frames(codes[i : i + partition])
                    for i in range(0, codes.shape[0], partition)
                ]
                actual = np.asarray(mx.concatenate(pieces, axis=0))
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_first_audio_arrives_before_frame_generation_finishes(self):
        service = MossNanoService(self.model, self.config)
        self.addCleanup(service._pool.shutdown, wait=True, cancel_futures=True)
        proceed = threading.Event()
        paused = threading.Event()
        timed_out = threading.Event()
        exhausted = threading.Event()
        real_frames = moss_nano_loop.iter_audio_frames

        def gated_frames(*args, **kwargs):
            with contextlib.closing(real_frames(*args, **kwargs)) as frames:
                for index, frame in enumerate(frames):
                    yield frame
                    if index == 0:
                        paused.set()
                        if not proceed.wait(10):
                            timed_out.set()
                exhausted.set()

        with mock.patch.object(moss_nano_loop, "iter_audio_frames", gated_frames):
            stream = service.speech_stream(
                "Audio must arrive before the full sentence is complete.",
                voice=self.voice,
                streaming_interval=0.24,
                streaming_initial_interval=0.08,
            )
            try:
                first = next(stream)
                self.assertEqual(len(first), 3840 * 2)
                self.assertTrue(
                    paused.wait(2), "producer did not reach the second-frame gate"
                )
                self.assertFalse(
                    timed_out.is_set(), "first audio waited for subsequent frames"
                )
                self.assertFalse(
                    exhausted.is_set(), "generation finished before the first chunk"
                )
            finally:
                stream.close()
                proceed.set()
        # This runs behind cancellation on the same service's generation
        # worker: leaked locks or unfinished request state fail here.
        next_pcm, _ = service.speech_bytes(
            "A second request works.", voice=self.voice, response_format="pcm"
        )
        self.assertGreater(len(next_pcm), 0)

    def test_sequential_streams_match_buffered_audio_without_cache_leaks(self):
        service = MossNanoService(self.model, self.config)
        self.addCleanup(service._pool.shutdown, wait=True, cancel_futures=True)
        for text in ("First request.", "第二次请求。", "First request."):
            with self.subTest(text=text):
                expected, _ = service.speech_bytes(
                    text, voice=self.voice, response_format="pcm"
                )
                with contextlib.closing(
                    service.speech_stream(
                        text,
                        voice=self.voice,
                        streaming_interval=0.24,
                        streaming_initial_interval=0.08,
                    )
                ) as stream:
                    chunks = list(stream)
                self.assertGreater(len(chunks), 1)
                actual = b"".join(chunks)
                self.assertEqual(len(actual), len(expected))
                # Whole/chunked float kernels may round adjacent PCM16 values
                # differently. Four LSBs correspond to the codec's 1e-4 bound.
                np.testing.assert_allclose(
                    np.frombuffer(actual, dtype="<i2").astype(np.int32),
                    np.frombuffer(expected, dtype="<i2").astype(np.int32),
                    rtol=0,
                    atol=4,
                )

    def test_sampled_service_seed_repeats_on_generation_worker(self):
        config = replace(self.config, do_sample=True, seed=41)
        service = MossNanoService(self.model, config)
        self.addCleanup(service._pool.shutdown, wait=True, cancel_futures=True)
        text = "Seeded requests should repeat."
        buffered = [
            service.speech_bytes(text, voice=self.voice, response_format="pcm")[0]
            for _ in range(2)
        ]
        streamed = []
        for _ in range(2):
            with contextlib.closing(
                service.speech_stream(
                    text,
                    voice=self.voice,
                    streaming_interval=0.24,
                    streaming_initial_interval=0.08,
                )
            ) as stream:
                streamed.append(b"".join(stream))
        self.assertGreater(len(buffered[0]), 0)
        self.assertEqual(
            buffered[0], buffered[1], "buffered worker ignored config.seed"
        )
        self.assertEqual(
            streamed[0], streamed[1], "streaming worker ignored config.seed"
        )
        self.assertEqual(len(buffered[0]), len(streamed[0]))
        np.testing.assert_allclose(
            np.frombuffer(streamed[0], dtype="<i2").astype(np.int32),
            np.frombuffer(buffered[0], dtype="<i2").astype(np.int32),
            rtol=0,
            atol=4,
        )


if __name__ == "__main__":
    unittest.main()
