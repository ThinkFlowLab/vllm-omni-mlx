"""--draft-model / --kv-bits plumbing: kwargs reach stream_generate, the prefix
cache stays off while a draft model is set, and the CLI parses the flags."""

import unittest

from vllm_omni_mlx.__main__ import build_parser
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

    def test_cli_parses_new_flags(self):
        args = build_parser().parse_args(
            [
                "--model", "m",
                "--draft-model", "m-draft",
                "--kv-bits", "8",
                "--kv-group-size", "128",
            ]
        )
        self.assertEqual(args.draft_model, "m-draft")
        self.assertEqual(args.kv_bits, 8)
        self.assertEqual(args.kv_group_size, 128)
        defaults = build_parser().parse_args(["--model", "m"])
        self.assertIsNone(defaults.draft_model)
        self.assertIsNone(defaults.kv_bits)
        self.assertEqual(defaults.kv_group_size, 64)


if __name__ == "__main__":
    unittest.main()
