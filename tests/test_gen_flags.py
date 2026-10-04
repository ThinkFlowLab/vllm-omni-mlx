"""--draft-model / --kv-bits plumbing: kwargs reach stream_generate, the prefix
cache stays off while a draft model is set, and the CLI parses the flags."""

import unittest

from vllm_omni_mlx.__main__ import _looks_like_tts, _serve, build_parser, build_serve_parser
from vllm_omni_mlx.schemas import UnifiedRequest

from tests.test_prompt_cache import drain, make_backend, user


def chat(backend, messages):
    return drain(backend.chat(UnifiedRequest(model="", messages=messages)))


class DraftAndKvFlagsTest(unittest.TestCase):
    def test_kv_kwargs_reach_stream_generate(self):
        backend = make_backend()
        backend._kv_kwargs = {"kv_bits": 8, "kv_group_size": 128}
        chat(backend, [user("hi")])
        call = backend._mlx_lm.calls[0]
        self.assertEqual(call["kv_bits"], 8)
        self.assertEqual(call["kv_group_size"], 128)

    def test_draft_model_disables_cache_reuse_and_storage(self):
        backend = make_backend()
        backend._draft_model = object()
        chat(backend, [user("hi")])
        call = backend._mlx_lm.calls[0]
        self.assertIs(call["draft_model"], backend._draft_model)
        self.assertNotIn("prompt_cache", call)  # never passed alongside a draft
        self.assertEqual(backend._cached, (None, None))  # nothing stored

        backend2 = make_backend()
        backend2._draft_model = object()
        chat(backend2, [user("hi")])
        chat(backend2, [user("hi")])
        second = backend2._mlx_lm.calls[1]
        # with a draft model set the cache is never consulted: full prompt both turns
        self.assertEqual(len(second["prompt"]), len(backend2._mlx_lm.calls[0]["prompt"]))
        self.assertNotIn("prompt_cache", second)


class ServeCliTest(unittest.TestCase):
    def test_serve_parses_gen_flags(self):
        args = build_parser().parse_args(
            [
                "serve", "m",
                "--draft-model", "m-draft",
                "--kv-bits", "8",
                "--kv-group-size", "128",
            ]
        )
        self.assertEqual(args.command, "serve")
        self.assertEqual(args.model, "m")
        self.assertEqual(args.draft_model, "m-draft")
        self.assertEqual(args.kv_bits, 8)
        self.assertEqual(args.kv_group_size, 128)

    def test_serve_defaults(self):
        args = build_parser().parse_args(["serve", "m"])
        self.assertIsNone(args.draft_model)
        self.assertIsNone(args.kv_bits)
        self.assertEqual(args.kv_group_size, 64)
        self.assertFalse(args.omni)
        self.assertEqual(args.backend, "auto")
        self.assertIsNone(args.tts_model)

    def test_serve_omni_flag(self):
        args = build_parser().parse_args(["serve", "tts-model", "--omni"])
        self.assertTrue(args.omni)

    def test_tts_subcommand_parses(self):
        args = build_parser().parse_args(["tts", "--voice", "ryan", "--text", "hi", "--out", "o.wav"])
        self.assertEqual(args.command, "tts")
        self.assertEqual(args.voice, "ryan")
        self.assertEqual(args.text, "hi")

    def test_serve_requires_a_model(self):
        with self.assertRaises(SystemExit) as ctx:
            _serve(build_serve_parser().parse_args([]))
        self.assertEqual(ctx.exception.code, 2)

    def test_omni_requires_positional_and_excludes_tts_model(self):
        with self.assertRaises(SystemExit):
            _serve(build_serve_parser().parse_args(["--omni"]))
        with self.assertRaises(SystemExit):
            _serve(build_serve_parser().parse_args(["m", "--omni", "--tts-model", "t"]))


class LooksLikeTtsTest(unittest.TestCase):
    def test_tts_checkpoints(self):
        self.assertTrue(_looks_like_tts({"tts_model_type": "custom_voice", "model_type": "qwen3_tts"}))
        self.assertTrue(_looks_like_tts({"model_type": "qwen3_tts"}))  # VoiceDesign repo without the top-level key

    def test_non_tts_configs(self):
        self.assertFalse(_looks_like_tts({"model_type": "qwen2"}))
        self.assertFalse(_looks_like_tts({"model_type": "qwen2_vl", "vision_config": {}}))
        self.assertFalse(_looks_like_tts({}))


if __name__ == "__main__":
    unittest.main()
