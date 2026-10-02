#!/usr/bin/env python3
"""M1.0 spike harness (#9): mlx-audio Qwen3-TTS adapt-vs-hand-port evaluation.

Loads the 4-bit CustomVoice checkpoint through mlx_audio and runs streaming +
non-streaming generation per (voice, text) case, reporting model load time,
time-to-first-audio (TTFA), inter-chunk latency, real-time factor (RTF =
wall time / audio duration; lower is better), and peak wired GPU memory.

WAV outputs go to --out-dir (default: a system temp dir) — they are listening
artifacts for the spike, not repo content. Requires the [omni] extra
(mlx-audio), which is already part of this repo's optional dependency set.
"""

from __future__ import annotations

import argparse
import array
import json
import platform
import sys
import time
import wave
from pathlib import Path
from tempfile import gettempdir

import mlx.core as mx

# Default cases: one English voice, one Chinese voice, plus an instructed
# (emotion) variant to exercise the CustomVoice instruct path.
DEFAULT_CASES = [
    {
        "label": "en_vivian",
        "text": "Hello, and welcome to the vllm omni em el ex project. "
        "This spike decides whether we adapt mlx audio or hand port the "
        "text to speech pipeline.",
        "voice": "Vivian",
        "instruct": None,
        "lang_code": "English",
    },
    {
        "label": "zh_dylan",
        "text": "今天天气真不错，我们一起去公园散步吧，顺便聊聊苹果芯片上的推理。",
        "voice": "Dylan",
        "instruct": None,
        "lang_code": "Chinese",
    },
    {
        "label": "en_ryan_instruct",
        "text": "The latency numbers are in, and they look great.",
        "voice": "Ryan",
        "instruct": "Very happy and excited.",
        "lang_code": "English",
    },
]


def write_wav(path: Path, samples: mx.array, sample_rate: int) -> None:
    pcm = (mx.clip(samples, -1.0, 1.0) * 32767.0).astype(mx.int16)
    frames = array.array("h", pcm.tolist()).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)


def run_case(model, case: dict, out_dir: Path, streaming_interval: float) -> dict:
    """Run one case in both modes; returns a result dict for the report."""
    results = {"label": case["label"], "voice": case["voice"]}

    gen_kwargs = dict(
        text=case["text"],
        voice=case["voice"],
        instruct=case["instruct"],
        lang_code=case["lang_code"],
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.05,
        max_tokens=4096,
    )

    # Streaming: consume the generator; TTFA at first chunk with audio,
    # inter-chunk gaps thereafter. One mx.eval on the accumulated audio so
    # timing brackets real GPU work (the chunks themselves are forced by the
    # incremental decoder's own evals inside mlx-audio).
    mx.reset_peak_memory()
    chunk_times, chunk_frames = [], []
    t0 = time.perf_counter()
    stream = model.generate(stream=True, streaming_interval=streaming_interval, **gen_kwargs)
    for chunk in stream:
        if chunk.audio is not None and int(chunk.audio.size) > 0:
            chunk_times.append(time.perf_counter() - t0)
            chunk_frames.append(chunk.audio)
    audio = mx.concatenate(chunk_frames) if chunk_frames else None
    wall = time.perf_counter() - t0
    if audio is not None:
        write_wav(out_dir / f"{case['label']}_stream.wav", audio, 24000)
        gaps = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
        gaps_sorted = sorted(gaps)
        results["stream"] = {
            "ttfa_s": round(chunk_times[0], 3),
            "wall_s": round(wall, 3),
            "audio_s": round(float(audio.size) / 24000, 3),
            "rtf": round(wall / (float(audio.size) / 24000), 3),
            "chunks": len(chunk_times),
            "inter_chunk_p50_s": round(gaps_sorted[len(gaps_sorted) // 2], 3) if gaps else None,
            "inter_chunk_p90_s": round(gaps_sorted[int(len(gaps_sorted) * 0.9)], 3) if gaps else None,
            "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
        }
    else:
        results["stream"] = {"error": "no audio chunks produced"}

    # Non-streaming: single full-utterance result.
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    result = next(iter(model.generate(**gen_kwargs)))
    wall = time.perf_counter() - t0
    mx.eval(result.audio)
    if result.audio is not None and int(result.audio.size) > 0:
        write_wav(out_dir / f"{case['label']}_full.wav", result.audio, 24000)
        results["full"] = {
            "wall_s": round(wall, 3),
            "audio_s": round(float(result.audio.size) / 24000, 3),
            "rtf": round(wall / (float(result.audio.size) / 24000), 3),
            "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
        }
    else:
        results["full"] = {"error": "no audio produced"}
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
        help="HF repo id (must be a Qwen3-TTS CustomVoice variant)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(gettempdir()) / "mlx-audio-qwen3tts-spike",
        help="where WAV artifacts are written (default: system temp)",
    )
    parser.add_argument("--streaming-interval", type=float, default=0.5)
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    from mlx_audio.tts.utils import load_model

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"python {platform.python_version()}  device {mx.default_device()!r}")

    t0 = time.perf_counter()
    model = load_model(args.model)
    load_s = time.perf_counter() - t0
    speakers = model.get_supported_speakers()
    print(f"loaded {args.model} in {load_s:.1f}s; speakers: {speakers}")

    # GPU DVFS ramp: one short generation before measuring (mlx-perf rule).
    if not args.skip_warmup:
        print("warmup generation (clock ramp)...")
        for chunk in model.generate(
            text="Warmup.", voice=speakers[0], lang_code="English", max_tokens=512
        ):
            pass

    report = {"model": args.model, "load_s": round(load_s, 1), "cases": []}
    for case in DEFAULT_CASES:
        if case["voice"].lower() not in {s.lower() for s in speakers}:
            print(f"skip {case['label']}: voice {case['voice']} not in checkpoint", file=sys.stderr)
            continue
        print(f"case {case['label']} ...")
        report["cases"].append(run_case(model, case, args.out_dir, args.streaming_interval))

    print(json.dumps(report, indent=2))
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nartifacts: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
