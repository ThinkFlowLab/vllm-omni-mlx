"""Weight-free parity and causal decoding tests for the Nano streaming loop."""

import importlib.util
import unittest
from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx

from vllm_omni_mlx.tts import moss_nano_loop as loop
from vllm_omni_mlx.tts.moss_nano import MossNanoConfig

HAS_MLX_AUDIO = importlib.util.find_spec("mlx_audio") is not None


class _TextTokenizer:
    def encode(self, text, **kwargs):
        return [10 + ord(char) % 22 for char in text]


def _tiny_model():
    from mlx_audio.tts.models.moss_tts_nano.config import GPT2Config, ModelConfig
    from mlx_audio.tts.models.moss_tts_nano.moss_tts_nano import Model

    mx.random.seed(9)
    gpt = GPT2Config(vocab_size=32, n_embd=16, n_layer=2, n_head=4, n_inner=24)
    model = Model(
        ModelConfig(
            gpt2_config=gpt,
            n_vq=2,
            audio_vocab_size=8,
            audio_codebook_sizes=[8, 8],
            audio_pad_token_id=8,
        )
    )
    model.tokenizer = _TextTokenizer()
    mx.eval(model.parameters())
    return model


def _tiny_codec(*, causal=True):
    from mlx.utils import tree_flatten
    from mlx_audio.codec.models.moss_audio_tokenizer import (
        AudioTokenizerConfig,
        MossAudioTokenizer,
    )

    mx.random.seed(10)
    codec = MossAudioTokenizer(
        AudioTokenizerConfig(
            sample_rate=8,
            sampling_rate=8,
            downsample_rate=999,
            number_channels=2,
            enable_channel_interleave=True,
            causal_transformer_context_duration=1.0,
            encoder_kwargs=[{"module_type": "PatchedPretransform", "patch_size": 4}],
            decoder_kwargs=[
                {
                    "module_type": "Transformer",
                    "input_dimension": 4,
                    "output_dimension": 4,
                    "d_model": 4,
                    "num_heads": 2,
                    "num_layers": 2,
                    "dim_feedforward": 8,
                    "causal": causal,
                    "positional_embedding": "sin_rope",
                    "max_period": 10000,
                    "conv_layout": True,
                },
                {"module_type": "PatchedPretransform", "patch_size": 4},
            ],
            quantizer_kwargs={
                "input_dim": 4,
                "num_quantizers": 2,
                "codebook_size": 8,
                "codebook_dim": 2,
            },
        )
    )
    # The upstream weight-normalized convolutions expect loaded weights;
    # their zero-initialized directions otherwise divide by zero.
    codec.load_weights(
        [
            (name, mx.random.normal(value.shape) * 0.1)
            for name, value in tree_flatten(codec.parameters())
            if name.endswith("original1")
        ],
        strict=False,
    )
    mx.eval(codec.parameters())
    return codec


@unittest.skipUnless(HAS_MLX_AUDIO, "Nano frame parity requires the [tts] extra")
class NanoFrameParityTest(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_model()
        self.ref_audio = mx.zeros((4, 2))
        self.ref_codes = mx.array([[1, 2], [3, 4]], dtype=mx.int32)
        self.config = MossNanoConfig(
            max_tokens=4,
            text_top_k=2,
            audio_top_k=4,
            audio_top_p=0.85,
            audio_repetition_penalty=1.5,
        )
        self.encode = mock.patch.object(
            self.model,
            "encode_reference_audio",
            return_value=self.ref_codes,
        )
        self.encode_mock = self.encode.start()
        self.addCleanup(self.encode.stop)

    def _force_frames(self):
        head = self.model._text_lm_head
        slot = self.model.config.audio_assistant_slot_token_id
        bias = mx.zeros((1, 32))
        bias[:, slot] = 100.0
        return mock.patch.object(
            self.model, "_text_lm_head", side_effect=lambda x: head(x) + bias
        )

    def _library_tokens(self, text, config, seed):
        captured = []

        def decode(codes, **kwargs):
            captured.append(codes)
            return mx.zeros((codes.shape[1] * 2, 2))

        mx.random.seed(seed)
        with mock.patch.object(
            self.model, "decode_audio_token_ids", side_effect=decode
        ):
            list(
                self.model.generate(
                    text=text,
                    ref_audio=self.ref_audio,
                    mode="voice_clone",
                    max_tokens=config.max_tokens,
                    voice_clone_max_text_tokens=config.voice_clone_max_text_tokens,
                    do_sample=config.do_sample,
                    text_temperature=config.text_temperature,
                    text_top_p=config.text_top_p,
                    text_top_k=config.text_top_k,
                    audio_temperature=config.audio_temperature,
                    audio_top_p=config.audio_top_p,
                    audio_top_k=config.audio_top_k,
                    audio_repetition_penalty=config.audio_repetition_penalty,
                )
            )
        return captured[0][0]

    def _frames(self, text, config, seed):
        # generate_frames now yields waveforms. Capture the actual code groups
        # at the decoder boundary without replacing autoregressive generation.
        captured = []

        def decode(codes):
            captured.append(codes)
            return mx.zeros((codes.shape[0] * 2, 2))

        self.model.audio_tokenizer = SimpleNamespace(
            make_streaming_decoder=lambda **kwargs: SimpleNamespace(
                decode_frames=decode
            )
        )
        mx.random.seed(seed)
        list(loop.synthesize_clone_stream(self.model, config, text, self.ref_audio))
        return (
            mx.concatenate(captured) if captured else mx.zeros((0, 2), dtype=mx.int32)
        )

    def test_greedy_and_seeded_sampling_match_upstream_tokens(self):
        for do_sample in (False, True):
            for seed in (3, 17):
                with self.subTest(do_sample=do_sample, seed=seed), self._force_frames():
                    cfg = replace(self.config, do_sample=do_sample)
                    expected = self._library_tokens("A short phrase.", cfg, seed)
                    actual = self._frames("A short phrase.", cfg, seed)
                    self.assertEqual(actual.shape, (cfg.max_tokens, 2))
                    self.assertTrue(mx.array_equal(actual, expected).item())
                    self.assertEqual(actual.dtype, mx.int32)

    def test_sentence_splitting_resets_lm_cache_and_repetition_history(self):
        from mlx_audio.tts.models.moss_tts_nano import sampling

        cfg = replace(self.config, voice_clone_max_text_tokens=4, max_tokens=3)
        histories = []
        sampler = sampling.sample_next_token

        def record_history(*args, **kwargs):
            # The assistant-text sampler also delegates here, without audio
            # history. Record only codebook calls, including their first frame.
            if "previous_token_ids" in kwargs:
                history = kwargs["previous_token_ids"]
                histories.append(None if history is None else history.shape[1])
            return sampler(*args, **kwargs)

        with self._force_frames():
            expected = self._library_tokens("one.\n two.", cfg, 7)
            with (
                mock.patch.object(
                    sampling, "sample_next_token", side_effect=record_history
                ),
                mock.patch.object(
                    self.model.transformer,
                    "make_cache",
                    wraps=self.model.transformer.make_cache,
                ) as caches,
                mock.patch.object(
                    self.model,
                    "build_inference_input_ids",
                    wraps=self.model.build_inference_input_ids,
                ) as prompts,
            ):
                actual = self._frames("one.\n two.", cfg, 7)
        self.assertEqual(actual.shape, (6, 2))
        self.assertTrue(mx.array_equal(actual, expected).item())
        self.assertEqual(caches.call_count, 2)
        self.assertEqual(histories, [None, None, 1, 1, 2, 2] * 2)
        self.assertEqual(
            [call.kwargs["text"] for call in prompts.call_args_list], ["One.", "two."]
        )

    def test_eos_stops_without_emitting_an_audio_end_frame(self):
        slot = self.model.config.audio_assistant_slot_token_id
        eos = self.model.config.audio_end_token_id
        with mock.patch(
            "mlx_audio.tts.models.moss_tts_nano.sampling.sample_assistant_text_token",
            side_effect=[mx.array([slot]), mx.array([eos])],
        ) as sample:
            frames = self._frames("hello", self.config, 7)
        self.assertEqual(frames.shape, (1, 2))
        self.assertEqual(sample.call_count, 2)

    def test_immediate_eos_produces_no_frame(self):
        with mock.patch(
            "mlx_audio.tts.models.moss_tts_nano.sampling.sample_assistant_text_token",
            return_value=mx.array([self.model.config.audio_end_token_id]),
        ):
            self.assertEqual(self._frames("hello", self.config, 7).shape, (0, 2))

    def test_close_before_iteration_does_not_encode_reference(self):
        audio = loop.synthesize_clone_stream(
            self.model, self.config, "hello", self.ref_audio
        )
        audio.close()
        self.encode_mock.assert_not_called()

    def test_reference_encoding_uses_model_sample_rate(self):
        with self._force_frames():
            self._frames("hello", self.config, 7)
        args, kwargs = self.encode_mock.call_args
        self.assertIs(args[0], self.ref_audio)
        self.assertEqual(kwargs, {"sample_rate": self.model.sample_rate})


@unittest.skipUnless(HAS_MLX_AUDIO, "Nano codec parity requires the [tts] extra")
class NanoCodecStreamingTest(unittest.TestCase):
    def setUp(self):
        self.codec = _tiny_codec()
        self.model = _tiny_model()
        self.model.audio_tokenizer = self.codec
        self.codes = (mx.arange(22, dtype=mx.int32) % 8).reshape(11, 2)

    def _stream(self, *, initial=0.08, interval=0.24, segments=None):
        # Fix sampled codebook tokens to exercise grouping and the real causal
        # decoder independently of random LM outputs. The tiny codec has a
        # two-sample hop; intervals still use Nano's fixed 12.5 Hz frame rate.
        config = MossNanoConfig(
            max_tokens=11 if segments is None else 3,
            streaming_initial_interval=initial,
            streaming_interval=interval,
        )
        slot = self.model.config.audio_assistant_slot_token_id
        with (
            mock.patch.object(
                self.model,
                "encode_reference_audio",
                return_value=mx.zeros((2, 2), dtype=mx.int32),
            ),
            mock.patch(
                "mlx_audio.tts.models.moss_tts_nano.sampling.sample_assistant_text_token",
                return_value=mx.array([slot]),
            ),
            mock.patch(
                "mlx_audio.tts.models.moss_tts_nano.sampling.sample_next_token",
                side_effect=(value.reshape(1) for row in self.codes for value in row),
            ),
            mock.patch(
                "mlx_audio.tts.models.moss_tts_nano.text.split_text_into_best_sentences",
                return_value=segments or ["Text."],
            ),
        ):
            yield from loop.synthesize_clone_stream(
                self.model, config, "text", mx.zeros(4)
            )

    def test_real_causal_codec_matches_full_decode_for_multiple_partitions(self):
        expected = self.codec.decode_audio_codes(self.codes)
        self.assertEqual(expected.shape, (22, 2))
        self.assertGreater(mx.max(mx.abs(expected)).item(), 0.01)
        for sizes in ([1] * 11, [3, 3, 3, 2], [1, 4, 2, 4]):
            with self.subTest(sizes=sizes):
                decoder = self.codec.make_streaming_decoder(num_quantizers=2)
                offset = 0
                chunks = []
                for size in sizes:
                    chunks.append(
                        decoder.decode_frames(self.codes[offset : offset + size])
                    )
                    offset += size
                actual = mx.concatenate(chunks)
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(
                    mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
                )

    def test_initial_steady_and_final_groups_preserve_all_samples(self):
        chunks = list(self._stream())
        self.assertEqual([chunk.shape[0] for chunk in chunks], [2, 6, 6, 6, 2])
        expected = self.codec.decode_audio_codes(self.codes)
        self.assertTrue(
            mx.allclose(mx.concatenate(chunks), expected, atol=1e-5, rtol=1e-5).item()
        )

    def test_initial_interval_is_independent_and_subframe_interval_is_one_frame(self):
        self.assertEqual([x.shape[0] for x in self._stream(initial=0.4)], [10, 6, 6])
        self.assertEqual(
            [x.shape[0] for x in self._stream(initial=0.01, interval=0.01)], [2] * 11
        )

    def test_pending_codes_and_decoder_span_text_segments(self):
        with mock.patch.object(
            self.codec,
            "make_streaming_decoder",
            wraps=self.codec.make_streaming_decoder,
        ) as create:
            chunks = list(self._stream(initial=0.32, segments=["One.", "Two."]))
        self.assertEqual([x.shape[0] for x in chunks], [8, 4])
        create.assert_called_once_with(num_quantizers=2)
        expected = self.codec.decode_audio_codes(self.codes[:6])
        self.assertTrue(
            mx.allclose(mx.concatenate(chunks), expected, atol=1e-5, rtol=1e-5).item()
        )

    def test_requests_have_independent_codec_state(self):
        with mock.patch.object(
            self.codec,
            "make_streaming_decoder",
            wraps=self.codec.make_streaming_decoder,
        ) as create:
            first = mx.concatenate(list(self._stream()))
            second = mx.concatenate(list(self._stream()))
        self.assertEqual(create.call_count, 2)
        self.assertTrue(mx.array_equal(first, second).item())

    def test_close_stops_generation_after_delivered_chunk(self):
        with mock.patch.object(
            self.model, "_audio_lm_head", wraps=self.model._audio_lm_head
        ) as head:
            with closing(self._stream()) as audio:
                self.assertEqual(next(audio).shape, (2, 2))
                self.assertEqual(head.call_count, 2)  # One frame, two codebooks.
            self.assertEqual(list(audio), [])
            self.assertEqual(head.call_count, 2)

    def test_invalid_frame_counts_fail_before_reference_encoding(self):
        for kwargs in (
            {"initial_frames": 0},
            {"chunk_frames": 0},
            {"initial_frames": -1},
        ):
            with (
                self.subTest(kwargs=kwargs),
                mock.patch.object(self.model, "encode_reference_audio") as encode,
            ):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    list(
                        loop.generate_frames(
                            self.model, text="hello", ref_audio=mx.zeros(4), **kwargs
                        )
                    )
                encode.assert_not_called()


class NanoIntervalTest(unittest.TestCase):
    def test_durations_floor_to_whole_80ms_frames_with_minimum_one(self):
        for seconds, frames in (
            (0.001, 1),
            (0.08, 1),
            (0.159, 1),
            (0.16, 2),
            (0.5, 6),
            (2.0, 25),
            (10.0, 125),
        ):
            with self.subTest(seconds=seconds):
                self.assertEqual(loop.frames_for_interval(seconds), frames)


if __name__ == "__main__":
    unittest.main()
