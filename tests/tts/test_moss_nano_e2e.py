"""Real Nano/codec regression checks, skipped unless BOTH checkpoints are local.

Override MOSS_NANO_MODEL / MOSS_NANO_CODEC with local snapshot directories.
Tests never download weights or fixtures. Numerical/streaming checks use a
deterministic reference signal; the HNR check additionally needs the pinned
VoxCPM2 test_en.wav in the HF cache (only the clip, not VoxCPM2 weights).
HNR detects noise-like decode failures, not intelligibility or speaker fidelity.
"""

import base64
import contextlib
import io
import os
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from unittest import mock

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np

from tests.audio_metrics import int16_pcm_hnr_db
from vllm_omni_mlx.tts import moss_nano_loop
from vllm_omni_mlx.tts.config import local_snapshot
from vllm_omni_mlx.tts.moss_nano import (
    DEFAULT_MODEL,
    MossNanoConfig,
    MossNanoService,
    load_moss_nano_model,
)

SR = 48000
TEXT_EN = "Hello, this is a short voice cloning test."
TEXT_ZH = "这句话用来测试中文语音合成的效果。"
REFERENCE_REPO = "mlx-community/VoxCPM2-4bit"
REFERENCE_REVISION = "dc9e5c187858da5f4a13dc4c247e297339216381"

#: Calibrated catastrophic-decode floor (dB), following the #37/#71 method.
#: M1 Max, mlx 0.32.3 / mlx-audio 0.5.8; MOSS-TTS-Nano-100M@229a9c5,
#: MOSS-Audio-Tokenizer-Nano@edccdfd; pinned test_en.wav above, 48 kHz mono.
#: Two EN + two ZH texts, sampling seeds 7/17/42/73 plus greedy, max_tokens=375,
#: buffered + streamed (initial 0.08 s / interval 0.5 s), 40 outputs total:
#: HNR -0.06..1.56 dB vs seeded white noise -11.39 dB. A -5.0 dB floor
#: separates these outputs from noise with margin for sampling/kernel changes.
#: This is not a perceptual-quality or EOS check: greedy long ZH hit the 30 s
#: frame budget while still passing HNR. The short EN/ZH cases below completed
#: in 2.72..5.04 s, RMS 0.117..0.146; silence must fail the separate RMS guard.
HNR_FLOOR_DB = -5.0


def _assert_speech_pcm(test, pcm):
    test.assertEqual(len(pcm) % 2, 0, "unaligned PCM16 output")
    test.assertGreater(len(pcm), 2 * SR, "expected more than one second of audio")
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(samples**2)))
    # Silence scores 0 dB in the shared HNR helper, so needs its own guard.
    test.assertGreater(rms, 0.01, f"silent/near-silent output: RMS {rms:.4f}")
    hnr = int16_pcm_hnr_db(pcm, sr=SR)
    test.assertGreater(hnr, HNR_FLOOR_DB, f"noise-like output: HNR {hnr:.2f} dB")


class MossNanoHnrGuardTest(unittest.TestCase):
    """Negative controls at Nano's sample rate, requiring no checkpoints."""

    def test_silence_fails_rms_guard(self):
        with self.assertRaisesRegex(AssertionError, "silent/near-silent"):
            _assert_speech_pcm(self, np.zeros(2 * SR, dtype="<i2").tobytes())

    def test_white_noise_fails_hnr_guard(self):
        noise = np.random.default_rng(0).normal(0, 0.1, 2 * SR)
        pcm = (np.clip(noise, -1, 1) * 32767).astype("<i2").tobytes()
        with self.assertRaisesRegex(AssertionError, "noise-like"):
            _assert_speech_pcm(self, pcm)


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
        codec_source = os.environ.get(
            "MOSS_NANO_CODEC", "mlx-community/MOSS-Audio-Tokenizer-Nano"
        )
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
            max_tokens=16,
            do_sample=False,
        )
        cls.model = load_moss_nano_model(cls.config)
        cls.addClassCleanup(cls._release_model)
        # The service loader is now lazy; explicitly preload the local codec
        # through the upstream API so tests never resolve a remote fallback.
        cls.model._ensure_audio_tokenizer(source=codec_path)
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

    def test_cloned_speech_hnr_english_and_chinese_buffered_and_streamed(self):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            reference = hf_hub_download(
                REFERENCE_REPO,
                "test_en.wav",
                revision=REFERENCE_REVISION,
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            self.skipTest(
                f"HNR needs cached {REFERENCE_REPO}@{REFERENCE_REVISION}/test_en.wav"
            )
        voice = {
            "ref_audio": base64.b64encode(Path(reference).read_bytes()).decode("ascii")
        }
        # The short numerical regressions use 16 frames. Quality checks need
        # complete sentences and the normal 375-frame per-segment budget.
        for sampled, seed in ((True, 7), (True, 17), (True, 42), (False, 7)):
            config = replace(
                self.config,
                max_tokens=375,
                do_sample=sampled,
                streaming_interval=0.5,
                streaming_initial_interval=0.08,
            )
            service = MossNanoService(self.model, config)
            self.assertEqual(service.sample_rate, SR)
            for text in (TEXT_EN, TEXT_ZH):
                for mode in ("buffered", "stream"):
                    with self.subTest(sampled=sampled, seed=seed, text=text, mode=mode):
                        mx.random.seed(seed)
                        if mode == "buffered":
                            pcm, _ = service.speech_bytes(
                                text, voice=voice, response_format="pcm"
                            )
                        else:
                            with contextlib.closing(
                                service.speech_stream(text, voice=voice)
                            ) as stream:
                                chunks = list(stream)
                            self.assertGreater(len(chunks), 1)
                            pcm = b"".join(chunks)
                        _assert_speech_pcm(self, pcm)

    def _upstream_codes(self, text, config):
        from mlx_audio.tts.models.moss_tts_nano.text import (
            lightweight_normalize_text,
            split_text_into_best_sentences,
        )

        reference_codes = self.model.encode_reference_audio(
            self.reference,
            sample_rate=48000,
            num_quantizers=self.model.config.n_vq,
        )
        chunks = split_text_into_best_sentences(
            self.model.tokenizer,
            lightweight_normalize_text(text),
            max_tokens=config.voice_clone_max_text_tokens,
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
                    max_new_frames=config.max_tokens,
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
                    self.config,
                    do_sample=sampled,
                    voice_clone_max_text_tokens=text_budget,
                )
                mx.random.seed(seed)
                expected = self._upstream_codes(text, config)
                mx.random.seed(seed)
                captured = []
                codec = self.model.audio_tokenizer
                make_decoder = codec.make_streaming_decoder

                def record_decoder(**kwargs):
                    decoder = make_decoder(**kwargs)
                    decode = decoder.decode_frames

                    def record(codes):
                        captured.append(codes)
                        return decode(codes)

                    decoder.decode_frames = record
                    return decoder

                with mock.patch.object(
                    codec, "make_streaming_decoder", side_effect=record_decoder
                ):
                    list(
                        moss_nano_loop.synthesize_clone_stream(
                            self.model, config, text, self.reference
                        )
                    )
                self.assertTrue(captured, "incremental generation produced no frames")
                actual = mx.concatenate(captured)
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_real_codec_matches_whole_decode_across_partitions(self):
        codes = self._upstream_codes("Check the audio decoder.", self.config)
        codec = self.model.audio_tokenizer
        expected = np.asarray(codec.decode_audio_codes(codes))
        hop = 3840  # Nano codec: 48 kHz / 12.5 frames per second.
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

    def test_first_audio_arrives_after_only_one_generated_frame(self):
        service = MossNanoService(self.model, self.config)
        with mock.patch.object(
            self.model, "_audio_lm_head", wraps=self.model._audio_lm_head
        ) as head:
            with contextlib.closing(
                service.speech_stream(
                    "Audio must arrive before the full sentence is complete.",
                    voice=self.voice,
                    streaming_interval=0.24,
                    streaming_initial_interval=0.08,
                )
            ) as stream:
                first = next(stream)
                self.assertEqual(len(first), 3840 * 2)
                self.assertEqual(head.call_count, self.model.config.n_vq)
                self.assertFalse(service._lock.acquire(blocking=False))
            self.assertEqual(head.call_count, self.model.config.n_vq)
        acquired = service._lock.acquire(timeout=2)
        self.assertTrue(acquired, "closing the stream leaked the service lock")
        service._lock.release()
        next_pcm, _ = service.speech_bytes(
            "A second request works.", voice=self.voice, response_format="pcm"
        )
        self.assertGreater(len(next_pcm), 0)

    def test_sequential_streams_match_buffered_audio_without_cache_leaks(self):
        service = MossNanoService(self.model, self.config)
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

    def test_caller_seed_repeats_buffered_and_streamed_requests(self):
        config = replace(self.config, do_sample=True)
        service = MossNanoService(self.model, config)
        text = "Seeded requests should repeat."
        buffered = []
        for _ in range(2):
            mx.random.seed(41)
            buffered.append(
                service.speech_bytes(text, voice=self.voice, response_format="pcm")[0]
            )
        streamed = []
        for _ in range(2):
            mx.random.seed(41)
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
            buffered[0],
            buffered[1],
            "buffered synthesis did not repeat with the same caller seed",
        )
        self.assertEqual(
            streamed[0],
            streamed[1],
            "streaming synthesis did not repeat with the same caller seed",
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
