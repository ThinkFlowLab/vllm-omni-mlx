"""Benchmark accounting: delivered audio, timing boundaries and warm summaries."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import bench_moss_nano as bench


class MossNanoBenchTest(unittest.TestCase):
    def test_output_cannot_overwrite_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.wav"
            reference.write_bytes(b"reference")
            with (
                patch(
                    "sys.argv",
                    [
                        "bench",
                        "--ref-audio",
                        str(reference),
                        "--output",
                        str(reference),
                    ],
                ),
                patch("sys.stderr"),
                self.assertRaises(SystemExit) as error,
            ):
                bench.main()
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(reference.read_bytes(), b"reference")

    def test_buffered_ttfa_is_total_and_pcm_duration_sets_rtf(self):
        service = SimpleNamespace(
            sample_rate=4, speech_bytes=Mock(return_value=(b"\0" * 16, "audio/pcm"))
        )
        with patch.object(bench.time, "perf_counter", side_effect=[10, 11]):
            result = bench.bench_once(service, "text", {}, "buffered")
        self.assertEqual(result["ttfa_s"], result["total_s"])
        self.assertEqual(result["audio_s"], 2)
        self.assertEqual(result["rtf"], 0.5)
        self.assertIsNone(result["first_gap_s"])
        service.speech_bytes.assert_called_once_with(
            "text", voice={}, response_format="pcm"
        )

    def test_stream_skips_empty_chunks_and_closes(self):
        closed = []

        def stream():
            try:
                yield b""
                yield b"\0" * 4
                yield b""
                yield b"\0" * 12
            finally:
                closed.append(True)

        service = SimpleNamespace(
            sample_rate=4, speech_stream=Mock(side_effect=lambda *a, **kw: stream())
        )
        with patch.object(
            bench.time, "perf_counter", side_effect=[10, 10.25, 10.75, 11]
        ):
            result = bench.bench_once(service, "text", {})
        self.assertEqual(result["ttfa_s"], 0.25)
        self.assertEqual(result["first_gap_s"], 0.5)
        self.assertEqual(result["first_audio_s"], 0.5)
        self.assertEqual(result["chunks"], 2)
        self.assertEqual(result["rtf"], 0.5)
        self.assertEqual(closed, [True])

    def test_empty_output_fails_in_both_modes(self):
        service = SimpleNamespace(
            sample_rate=48000,
            speech_bytes=lambda *a, **kw: (b"", "audio/pcm"),
            speech_stream=lambda *a, **kw: (chunk for chunk in ()),
        )
        for mode in ("buffered", "streaming"):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(RuntimeError, "no audio"),
            ):
                bench.bench_once(service, "text", {}, mode)

    def test_warm_summary_excludes_cold_and_warmup(self):
        rows = [
            dict(mode="buffered", phase=phase, **{key: value for key in bench.METRICS})
            for phase, value in (
                ("cold", 100),
                ("warmup", 50),
                ("warm", 1),
                ("warm", 3),
            )
        ]
        summary = bench.summarize(rows, ["buffered"])["buffered"]
        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["ttfa_s"], dict(min=1, median=2, max=3))


if __name__ == "__main__":
    unittest.main()
