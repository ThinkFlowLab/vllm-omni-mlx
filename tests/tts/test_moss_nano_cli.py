"""MOSS Nano CLI routing and file output, without optional model dependencies."""

import base64
import io
import sys
import tempfile
import unittest
import wave
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from vllm_omni_mlx import __main__ as cli


MODEL = "local-moss-nano"


@dataclass(frozen=True)
class FakeNanoConfig:
    model_ref: str = MODEL
    max_new_frames: int = 375
    audio_temperature: float = 0.8
    seed: int | None = None


def wav_payload(sample_rate=48000, frames=12000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x01\x00" * frames)
    return buffer.getvalue()


def module(name, **attrs):
    result = ModuleType(name)
    result.__dict__.update(attrs)
    return result


class MossNanoCliTest(unittest.TestCase):
    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.reference = Path(self.temp.name, "reference.wav")
        self.reference.write_bytes(wav_payload(sample_rate=24000))
        self.output = Path(self.temp.name, "output.wav")
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.contexts.enter_context(redirect_stdout(self.stdout))
        self.contexts.enter_context(redirect_stderr(self.stderr))

        self.model = object()
        self.payload = wav_payload()
        self.service = SimpleNamespace(
            name=MODEL,
            sample_rate=48000,
            channels=1,
            voices=[],
            speech_bytes=mock.Mock(return_value=(self.payload, "audio/wav")),
        )
        self.load_model = mock.Mock(return_value=self.model)
        self.service_factory = mock.Mock(return_value=self.service)
        self.peek_config = mock.Mock(return_value={"model_type": "moss_tts_nano"})
        self.load_backend = mock.Mock()
        self.create_app = mock.Mock(return_value=object())
        self.run_server = mock.Mock()
        self.seed = mock.Mock()
        core = module("mlx.core", random=SimpleNamespace(seed=self.seed))
        self.contexts.enter_context(
            mock.patch.dict(
                sys.modules,
                {
                    "vllm_omni_mlx.backends": module(
                        "vllm_omni_mlx.backends",
                        _peek_config=self.peek_config,
                        load_backend=self.load_backend,
                    ),
                    "vllm_omni_mlx.tts.moss_nano": module(
                        "vllm_omni_mlx.tts.moss_nano",
                        MossNanoConfig=FakeNanoConfig,
                        MossNanoService=self.service_factory,
                        load_moss_nano_model=self.load_model,
                    ),
                    # Nano must not import Qwen's generation/prewarm pipeline.
                    "vllm_omni_mlx.tts.config": None,
                    "vllm_omni_mlx.tts.generate": None,
                    "vllm_omni_mlx.tts.stream_loop": None,
                    "vllm_omni_mlx.server": module(
                        "vllm_omni_mlx.server", create_app=self.create_app
                    ),
                    "uvicorn": module("uvicorn", run=self.run_server),
                    "mlx": module("mlx", core=core),
                    "mlx.core": core,
                },
            )
        )

    def synthesize(self, *extra, reference=True):
        args = ["tts", "--model", MODEL, "--text", "hello", "--out", str(self.output)]
        if reference:
            args.extend(["--ref-audio", str(self.reference)])
        return cli.main([*args, *extra])

    def test_top_level_model_type_identifies_nano(self):
        self.assertTrue(cli._looks_like_tts({"model_type": "moss_tts_nano"}))
        self.assertFalse(cli._looks_like_tts({"model_type": "moss"}))

    def test_both_serve_forms_load_nano_service_without_qwen_prewarm(self):
        for args in (["serve", MODEL, "--omni"], ["serve", "--tts-model", MODEL]):
            with self.subTest(args=args):
                self.load_model.reset_mock()
                self.service_factory.reset_mock()
                self.create_app.reset_mock()
                self.run_server.reset_mock()
                self.assertEqual(cli.main(args), 0)
                self.load_model.assert_called_once_with(FakeNanoConfig(model_ref=MODEL))
                self.service_factory.assert_called_once_with(
                    self.model, self.load_model.call_args.args[0]
                )
                self.create_app.assert_called_once_with(
                    None, api_key=None, tts_service=self.service, asr_service=None
                )
                self.run_server.assert_called_once_with(
                    self.create_app.return_value,
                    host="127.0.0.1",
                    port=8000,
                    log_level="info",
                )
                self.load_backend.assert_not_called()

    def test_positional_nano_requires_omni(self):
        self.assertEqual(cli.main(["serve", MODEL]), 1)
        self.assertIn("--omni", self.stderr.getvalue())
        self.load_model.assert_not_called()
        self.load_backend.assert_not_called()
        self.run_server.assert_not_called()

    def test_synthesis_maps_options_and_preserves_service_wav(self):
        self.assertEqual(
            self.synthesize(
                "--max-tokens", "42", "--temperature", "0.3", "--seed", "7"
            ),
            0,
        )
        self.load_model.assert_called_once_with(
            FakeNanoConfig(
                model_ref=MODEL, max_new_frames=42, audio_temperature=0.3, seed=7
            )
        )
        # Nano's worker owns its RNG; the CLI passes the seed through config.
        self.seed.assert_not_called()
        call = self.service.speech_bytes.call_args
        self.assertEqual(call.args, ("hello",))
        self.assertEqual(
            base64.b64decode(call.kwargs["voice"]["ref_audio"], validate=True),
            self.reference.read_bytes(),
        )
        self.assertEqual(self.output.read_bytes(), self.payload)
        with wave.open(str(self.output), "rb") as wav:
            self.assertEqual((wav.getframerate(), wav.getnchannels()), (48000, 1))
            self.assertEqual(wav.getnframes(), 12000)
        self.assertIn("0.25s", self.stdout.getvalue())

    def test_synthesis_preserves_defaults_and_accepts_auto_language(self):
        self.assertEqual(self.synthesize("--language", "auto"), 0)
        self.load_model.assert_called_once_with(FakeNanoConfig(model_ref=MODEL))
        self.seed.assert_not_called()
        self.assertTrue(self.output.is_file())

    def test_cloning_requires_reference_before_loading_model(self):
        self.assertEqual(self.synthesize(reference=False), 1)
        self.assertIn("--ref-audio", self.stderr.getvalue())
        self.load_model.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_unsupported_flags_and_invalid_sampling_options_fail_before_loading(self):
        for flag, value in (
            ("--voice", "vivian"),
            ("--instruct", "happy"),
            ("--language", "Chinese"),
            ("--max-tokens", "auto"),
            ("--max-tokens", "1.5"),
            ("--max-tokens", "0"),
            ("--max-tokens", "-1"),
            ("--temperature", "0"),
            ("--temperature", "-0.1"),
            ("--temperature", "nan"),
            ("--temperature", "inf"),
        ):
            with self.subTest(flag=flag, value=value):
                self.assertEqual(self.synthesize(flag, value), 1)
                self.load_model.assert_not_called()
                self.service.speech_bytes.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_missing_reference_file_is_reported(self):
        self.reference = Path(self.temp.name, "missing.wav")
        self.assertEqual(self.synthesize(), 1)
        self.assertIn("error:", self.stderr.getvalue())
        self.service.speech_bytes.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_generation_error_is_reported_without_creating_output(self):
        self.service.speech_bytes.side_effect = RuntimeError(
            "MOSS Nano generated no audio"
        )
        self.assertEqual(self.synthesize(), 1)
        self.assertIn("generated no audio", self.stderr.getvalue())
        self.assertFalse(self.output.exists())

    def test_unreadable_model_config_is_reported_without_loading(self):
        self.peek_config.side_effect = OSError("config unavailable")
        self.assertEqual(self.synthesize(), 1)
        self.assertIn("config unavailable", self.stderr.getvalue())
        self.load_model.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
