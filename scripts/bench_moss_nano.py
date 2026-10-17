"""MOSS-Nano buffered/streaming A/B through MossNanoService (no HTTP).

python -m scripts.bench_moss_nano --ref-audio reference.wav --output moss-ab.json

Uses local cached checkpoints only. Each path's cold request runs in a fresh
process after model loading; it includes lazy codec loading, but does NOT mean
the OS file cache is cold. Warm measurements alternate A/B order after one
warmup per path. Every request resets the same seed. Timings include reference
decoding/encoding, generation and PCM conversion, excluding model loading,
input file reads, HTTP and playback. Buffered TTFA equals total request time.
"""

import argparse
import base64
import hashlib
import json
import math
import multiprocessing
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx

from vllm_omni_mlx.tts.config import local_snapshot
from vllm_omni_mlx.tts.moss_nano import (
    DEFAULT_MODEL,
    MossNanoConfig,
    MossNanoService,
    load_moss_nano_model,
)

TEXT = "Hello, this is a short voice cloning test."
CODEC = "mlx-community/MOSS-Audio-Tokenizer-Nano"
METRICS = ("ttfa_s", "total_s", "audio_s", "rtf", "first_audio_s", "first_gap_s")


def bench_once(service, text, voice, mode="streaming", seed=7):
    """Measure delivered PCM bytes; MLX work must finish before starting."""
    mx.random.seed(seed)
    mx.synchronize()
    started = time.perf_counter()
    first = first_gap = None
    chunks = []
    if mode == "buffered":
        pcm, _ = service.speech_bytes(text, voice=voice, response_format="pcm")
        total = time.perf_counter() - started
        first = total
        chunks = [pcm] if pcm else []
    elif mode == "streaming":
        with closing(service.speech_stream(text, voice=voice)) as stream:
            for chunk in stream:
                if not chunk:
                    continue
                arrived = time.perf_counter() - started
                if first is None:
                    first = arrived
                elif first_gap is None:
                    first_gap = arrived - first
                chunks.append(chunk)
        total = time.perf_counter() - started
    else:
        raise ValueError(f"unknown mode: {mode}")
    if not chunks:
        raise RuntimeError(f"MOSS-Nano produced no audio ({mode})")
    if any(len(chunk) % 2 for chunk in chunks):
        raise RuntimeError("MOSS-Nano produced unaligned PCM16")
    pcm = b"".join(chunks)
    audio_s = len(pcm) / (2 * service.sample_rate)
    return dict(
        mode=mode,
        seed=seed,
        ttfa_s=first,
        total_s=total,
        audio_s=audio_s,
        rtf=total / audio_s,
        first_audio_s=len(chunks[0]) / (2 * service.sample_rate),
        first_gap_s=first_gap,
        chunks=len(chunks),
        pcm_bytes=len(pcm),
        pcm_sha256=hashlib.sha256(pcm).hexdigest(),
    )


def load_service(config, codec_path):
    started = time.perf_counter()
    model = load_moss_nano_model(config)
    # Keep the codec lazy so first-request timings include its initialization.
    model.config.audio_tokenizer_pretrained_name_or_path = codec_path
    mx.synchronize()
    return MossNanoService(model, config), time.perf_counter() - started


def cold_request(config, codec_path, text, voice, mode, seed):
    service, load_s = load_service(config, codec_path)
    result = bench_once(service, text, voice, mode, seed)
    return dict(model_load_s=load_s, request=dict(result, phase="cold", run=0))


def command_output(*args):
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def checkpoint_info(path):
    root = Path(path)
    return dict(
        path=str(root),
        revision=root.name if root.parent.name == "snapshots" else None,
        config_sha256=hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
    )


def summarize(rows, modes):
    summary = {}
    for mode in modes:
        warm = [row for row in rows if row["mode"] == mode and row["phase"] == "warm"]
        summary[mode] = {"requests": len(warm)}
        for key in METRICS:
            values = [row[key] for row in warm if row[key] is not None]
            summary[mode][key] = (
                dict(median=statistics.median(values), min=min(values), max=max(values))
                if values
                else None
            )
    return summary


def print_row(label, row):
    gap = "-" if row["first_gap_s"] is None else f"{row['first_gap_s']:.3f}"
    print(
        f"{row['mode']:>9} {label:>12} {row['ttfa_s']:9.3f} {row['total_s']:9.3f} "
        f"{row['audio_s']:9.3f} {row['rtf']:8.3f} {row['first_audio_s']:10.3f} {gap:>10}",
        flush=True,
    )


def main():
    defaults = MossNanoConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--text", default=TEXT)
    parser.add_argument(
        "--mode", choices=("both", "buffered", "streaming"), default="both"
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Warm measured requests per path"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--cooldown", type=float, default=1.0, help="Seconds between requests"
    )
    parser.add_argument(
        "--output", type=Path, help="Raw JSON (default: timestamped file in cwd)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=defaults.max_tokens,
        help="Maximum audio frames per text segment",
    )
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--streaming-initial-interval",
        type=float,
        default=defaults.streaming_initial_interval,
    )
    parser.add_argument(
        "--streaming-interval", type=float, default=defaults.streaming_interval
    )
    args = parser.parse_args()
    if not args.ref_audio.is_file():
        parser.error("--ref-audio must point to an existing recording")
    if args.output and args.output.resolve() == args.ref_audio.resolve():
        parser.error("--output must not overwrite --ref-audio")
    if args.runs < 1 or args.max_tokens < 1:
        parser.error("--runs and --max-tokens must be positive")
    if not args.text.strip():
        parser.error("--text must not be empty")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if not math.isfinite(args.cooldown) or args.cooldown < 0:
        parser.error("--cooldown must be finite and non-negative")
    for name in ("streaming_initial_interval", "streaming_interval"):
        if not 0.0 < getattr(args, name) <= 10.0:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 10] seconds")
    model_path, codec_path = local_snapshot(DEFAULT_MODEL), local_snapshot(CODEC)
    if model_path is None or codec_path is None:
        parser.error(f"Cache {DEFAULT_MODEL} and {CODEC} locally before benchmarking")

    reference = args.ref_audio.read_bytes()
    voice = {"ref_audio": base64.b64encode(reference).decode("ascii")}
    config = MossNanoConfig(
        model_ref=model_path,
        max_tokens=args.max_tokens,
        do_sample=not args.greedy,
        streaming_initial_interval=args.streaming_initial_interval,
        streaming_interval=args.streaming_interval,
    )
    modes = ["buffered", "streaming"] if args.mode == "both" else [args.mode]
    stamp = datetime.now(timezone.utc)
    output = args.output or Path(f"moss-ab-{stamp.strftime('%Y%m%dT%H%M%S%fZ')}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        schema_version=1,
        created_utc=stamp.isoformat(),
        status="running",
        command=[sys.executable, "-m", "scripts.bench_moss_nano", *sys.argv[1:]],
        environment=dict(
            platform=platform.platform(),
            machine=platform.machine(),
            cpu=command_output("sysctl", "-n", "machdep.cpu.brand_string"),
            memory_bytes=command_output("sysctl", "-n", "hw.memsize"),
            python=platform.python_version(),
            packages={
                name: version(name)
                for name in ("mlx", "mlx-audio", "numpy", "huggingface-hub")
            },
            git_commit=command_output("git", "rev-parse", "HEAD"),
            git_status=command_output("git", "status", "--short"),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
        model=checkpoint_info(model_path),
        codec=checkpoint_info(codec_path),
        reference=dict(
            path=str(args.ref_audio.resolve()),
            bytes=len(reference),
            sha256=hashlib.sha256(reference).hexdigest(),
        ),
        text=args.text,
        seed=args.seed,
        config=asdict(config),
        sample_rate=48000,
        modes=modes,
        runs=args.runs,
        cooldown_s=args.cooldown,
        timing_scope=__doc__,
        loads=[],
        requests=[],
    )

    def save():
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )

    def record(row):
        report["requests"].append(row)
        save()
        label = f"warm {row['run']}" if row["phase"] == "warm" else row["phase"]
        print_row(label, row)

    save()  # Fail on unwritable output before loading weights.
    print(
        f"Text: {args.text!r}; seed: {args.seed}; JSON: {output.resolve()}", flush=True
    )
    print(
        f"{'mode':>9} {'request':>12} {'TTFA s':>9} {'total s':>9} {'audio s':>9} "
        f"{'RTF':>8} {'first s':>10} {'gap s':>10}",
        flush=True,
    )
    # One task per spawned child: each path gets its own process-cold request.
    with multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        for mode in modes:
            time.sleep(args.cooldown)
            result = pool.apply(
                cold_request, (config, codec_path, args.text, voice, mode, args.seed)
            )
            report["loads"].append(
                dict(mode=mode, phase="cold", seconds=result["model_load_s"])
            )
            record(result["request"])

    service, load_s = load_service(config, codec_path)
    report["loads"].append(dict(phase="warm_session", seconds=load_s))
    for mode in modes:
        time.sleep(args.cooldown)
        record(
            dict(
                bench_once(service, args.text, voice, mode, args.seed),
                phase="warmup",
                run=0,
            )
        )
    for run in range(1, args.runs + 1):
        for mode in modes if run % 2 else modes[::-1]:
            time.sleep(args.cooldown)
            record(
                dict(
                    bench_once(service, args.text, voice, mode, args.seed),
                    phase="warm",
                    run=run,
                )
            )
    report["summary"] = summarize(report["requests"], modes)
    report["status"] = "complete"
    save()
    print(
        "\nA/B summary: cold = fresh-process first request; warm = median (excluding warmup)"
    )
    for mode in modes:
        cold = next(
            row
            for row in report["requests"]
            if row["mode"] == mode and row["phase"] == "cold"
        )
        print_row("cold", cold)
        summary = report["summary"][mode]
        print_row(
            f"warm n={args.runs}",
            dict(
                mode=mode,
                **{
                    key: summary[key]["median"] if summary[key] is not None else None
                    for key in METRICS
                },
            ),
        )
    print(f"Raw requests, warm min/median/max and environment: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
