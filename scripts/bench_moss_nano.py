#!/usr/bin/env python3
"""Compare real MOSS Nano buffered/streaming cold and warm serving latency.

Each mode/text pair starts in a fresh process. "cold" is its first request
after loading; model load time is reported separately. Warm requests reuse
the model and service. Peak memory is MLX allocator memory (including weights),
not total process RSS. Reference decoding is included in request latency.

    python scripts/bench_moss_nano.py --ref-audio reference.wav --language both
    python scripts/bench_moss_nano.py --ref-audio reference.wav --text "你好。" --json result.json

Model flags accept cached Hugging Face repositories or explicit local paths.
Uncached repositories may download; use local paths for an offline benchmark.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEXTS = {
    "en": "Welcome to this local speech test. Audio should arrive while the model is still speaking.",
    "zh": "欢迎使用本地语音合成测试。我们希望在完整句子生成之前，就能听到第一段声音。",
}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="mlx-community/MOSS-TTS-Nano-100M")
    parser.add_argument(
        "--codec-model", default="mlx-community/MOSS-Audio-Tokenizer-Nano"
    )
    parser.add_argument(
        "--ref-audio", required=True, type=Path, help="0.5–30 s reference recording"
    )
    parser.add_argument("--text", help="custom text; overrides --language")
    parser.add_argument("--language", choices=("en", "zh", "both"), default="both")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--initial-interval", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-frames", type=int, default=375, help="audio frame limit per text chunk"
    )
    parser.add_argument("--warm-runs", type=int, default=2)
    parser.add_argument("--greedy", action="store_true", help="disable both samplers")
    parser.add_argument(
        "--json", metavar="PATH", help="write JSON; '-' prints only JSON on stdout"
    )
    parser.add_argument(
        "--_worker-mode", choices=("buffered", "stream"), help=argparse.SUPPRESS
    )
    return parser


def _worker(args):
    import mlx.core as mx

    from vllm_omni_mlx.tts.moss_nano import (
        MossNanoConfig,
        MossNanoService,
        load_moss_nano_model,
    )

    voice = {"ref_audio": base64.b64encode(args.ref_audio.read_bytes()).decode("ascii")}
    config = MossNanoConfig(
        model_ref=args.model,
        codec_model_ref=args.codec_model,
        max_new_frames=args.max_frames,
        do_sample=not args.greedy,
        seed=args.seed,
    )
    started = time.perf_counter()
    model = load_moss_nano_model(config)
    service = MossNanoService(model, config)
    load_s = time.perf_counter() - started
    runs = []
    try:
        for index in range(1 + args.warm_runs):
            mx.reset_peak_memory()
            started = time.perf_counter()
            chunks = []
            chunk_times = []
            first_s = None
            if args._worker_mode == "buffered":
                pcm, _ = service.speech_bytes(
                    args.text, voice=voice, response_format="pcm"
                )
                first_s = time.perf_counter() - started
                chunks.append(pcm)
                chunk_times.append(first_s)
            else:
                with contextlib.closing(
                    service.speech_stream(
                        args.text,
                        voice=voice,
                        streaming_interval=args.interval,
                        streaming_initial_interval=args.initial_interval,
                    )
                ) as stream:
                    for chunk in stream:
                        chunk_times.append(time.perf_counter() - started)
                        if first_s is None:
                            first_s = chunk_times[-1]
                        chunks.append(chunk)
            wall_s = time.perf_counter() - started
            pcm = b"".join(chunks)
            if not pcm or len(pcm) % 2:
                raise RuntimeError("service returned empty or misaligned PCM16")
            audio_s = len(pcm) / (2 * service.sample_rate)
            gaps = [right - left for left, right in pairwise(chunk_times)]
            runs.append(
                {
                    "phase": "cold" if index == 0 else "warm",
                    "run": index,
                    "ttfa_s": first_s,
                    "wall_s": wall_s,
                    "audio_s": audio_s,
                    "rtf": wall_s / audio_s,
                    "pcm_bytes": len(pcm),
                    "chunks": len(chunks),
                    "first_chunk_audio_s": len(chunks[0]) / (2 * service.sample_rate),
                    "chunk_arrival_s": chunk_times,
                    "chunk_gap_max_s": max(gaps) if gaps else None,
                    "chunk_gap_median_s": statistics.median(gaps) if gaps else None,
                    "peak_memory_bytes": mx.get_peak_memory(),
                    "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                }
            )
    finally:
        service._pool.shutdown(wait=True, cancel_futures=True)
    return {
        "mode": args._worker_mode,
        "load_s": load_s,
        "repeated_pcm_identical": len({run["pcm_sha256"] for run in runs}) == 1,
        "runs": runs,
    }


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.ref_audio.is_file():
        parser.error("--ref-audio must point to an existing recording")
    if args.max_frames < 1 or args.warm_runs < 1:
        parser.error("--max-frames and --warm-runs must be positive")
    if args.text is not None and not args.text.strip():
        parser.error("--text must not be empty")
    for flag, value in (
        ("--interval", args.interval),
        ("--initial-interval", args.initial_interval),
    ):
        if not math.isfinite(value) or not 0 < value <= 10:
            parser.error(f"{flag} must be in (0, 10]")
    if args._worker_mode:
        # Model loaders sometimes print progress to stdout; keep the child
        # protocol machine-readable and forward those messages to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            result = _worker(args)
        print(json.dumps(result, allow_nan=False))
        return 0

    cases = (
        {"custom": args.text}
        if args.text is not None
        else (
            TEXTS if args.language == "both" else {args.language: TEXTS[args.language]}
        )
    )
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "mlx": version("mlx"),
            "mlx_audio": version("mlx-audio"),
        },
        "model": args.model,
        "codec_model": args.codec_model,
        "reference": str(args.ref_audio.resolve()),
        "seed": args.seed,
        "seed_scope": "MossNanoConfig.seed resets the generation worker for each request",
        "do_sample": not args.greedy,
        "max_frames_per_text_chunk": args.max_frames,
        "interval_s": args.interval,
        "initial_interval_s": args.initial_interval,
        "cold_definition": "first request in a fresh process after model load",
        "memory_definition": "peak MLX allocator bytes, including resident weights",
        "cases": [],
    }
    for language, text in cases.items():
        case = {"language": language, "text": text, "modes": []}
        for mode in ("buffered", "stream"):
            print(
                f"Running {language}/{mode} (cold + {args.warm_runs} warm)...",
                file=sys.stderr,
                flush=True,
            )
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--_worker-mode",
                mode,
                "--model",
                args.model,
                "--codec-model",
                args.codec_model,
                "--ref-audio",
                str(args.ref_audio.resolve()),
                "--text",
                text,
                "--interval",
                str(args.interval),
                "--initial-interval",
                str(args.initial_interval),
                "--seed",
                str(args.seed),
                "--max-frames",
                str(args.max_frames),
                "--warm-runs",
                str(args.warm_runs),
            ]
            if args.greedy:
                command.append("--greedy")
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
            if completed.returncode:
                raise RuntimeError(
                    f"{language}/{mode} benchmark failed (exit {completed.returncode})"
                )
            outcome = json.loads(completed.stdout)
            case["modes"].append(outcome)
            for run in outcome["runs"]:
                print(
                    f"  {run['phase']} {mode}: TTFA {run['ttfa_s']:.3f}s, "
                    f"RTF {run['rtf']:.3f}, audio {run['audio_s']:.2f}s, "
                    f"{run['pcm_bytes']} bytes, peak {run['peak_memory_bytes'] / 2**30:.2f} GiB",
                    file=sys.stderr,
                )
        buffered, streamed = [mode["runs"] for mode in case["modes"]]
        case["comparisons"] = [
            {
                "run": b["run"],
                "phase": b["phase"],
                "ttfa_speedup": b["ttfa_s"] / s["ttfa_s"],
                "equal_pcm_length": b["pcm_bytes"] == s["pcm_bytes"],
                "identical_pcm": b["pcm_sha256"] == s["pcm_sha256"],
            }
            for b, s in zip(buffered, streamed)
        ]
        report["cases"].append(case)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.json == "-":
        print(payload, end="")
    elif args.json:
        Path(args.json).write_text(payload)
        print(f"JSON written to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
