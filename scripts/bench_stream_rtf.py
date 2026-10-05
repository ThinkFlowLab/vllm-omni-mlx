#!/usr/bin/env python3
"""Sustained-RTF bench for the streaming speech path (#65 A/B rule).

Loads the serving checkpoint offline, prewarms (decoder shapes +, on the
compiled branch, the decode closures), ramps the GPU clock, then runs
``--turns`` full generations of a fixed long text at serving defaults and
reports per-turn wall / audio seconds / RTF plus p50/min/max. The first
audio chunk's arrival (TTFA analog, no HTTP layer) is reported too.

Runs unchanged on main (eager loop) and on the compiled branch; set
VLLM_OMNI_TTS_EAGER_STREAM=1 to force the eager loop on the latter.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx

from vllm_omni_mlx.tts.config import DEFAULT_MODEL, TTSConfig, load_tts_model, local_snapshot
from vllm_omni_mlx.tts.stream_loop import prewarm_streaming, synthesize_stream

TEXT = (
    "Welcome to the sustained streaming benchmark. This paragraph is long "
    "enough that generation reaches steady state well before it ends, so "
    "the real time factor reflects the decode loop rather than startup. "
    "The sampler runs at serving defaults and the voice is a preset, "
    "matching the latency probe's audio mode."
) * 2


def ramp(seconds: float = 1.0) -> None:
    a = mx.random.normal((2048, 2048))
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = a @ a
        mx.eval(a)
        a = a / mx.sqrt((a * a).mean())
    mx.clear_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--label", default="run")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if local_snapshot(DEFAULT_MODEL) is None:
        raise SystemExit(f"{DEFAULT_MODEL} not cached locally")

    t0 = time.perf_counter()
    model = load_tts_model(TTSConfig())
    config = TTSConfig()
    print(f"[{args.label}] loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    shapes = prewarm_streaming(model, 0.5, 0.2)
    print(f"[{args.label}] prewarm shapes {shapes} in {time.perf_counter() - t0:.1f}s", flush=True)

    # discarded warmup generation: traces + allocator steady state, then
    # ~1s sustained load for the GPU clock ramp (docs/profiling.md)
    for _ in synthesize_stream(model, config, TEXT, speaker="vivian", temperature=0.9,
                               seed=args.seed, max_tokens=40):
        pass
    ramp(1.0)

    rtfs, walls, ttfa_ms = [], [], []
    print(f"[{args.label}] {'turn':>4} {'ttfa ms':>8} {'wall s':>7} {'audio s':>8} {'RTF':>6}")
    for turn in range(args.turns):
        t_turn = time.perf_counter()
        samples = 0
        first = None
        for chunk in synthesize_stream(
            model, config, TEXT, speaker="vivian", temperature=0.9,
            seed=args.seed + turn, max_tokens=args.max_tokens,
        ):
            now = time.perf_counter()
            if first is None:
                first = now - t_turn
            samples += chunk.shape[0]
        wall = time.perf_counter() - t_turn
        audio_s = samples / 24000
        rtfs.append(wall / audio_s)
        walls.append(wall)
        ttfa_ms.append(first * 1000)
        print(f"[{args.label}] {turn:>4} {first * 1000:8.0f} {wall:7.2f} {audio_s:8.2f} "
              f"{wall / audio_s:6.3f}", flush=True)

    print(f"[{args.label}] RTF p50 {statistics.median(rtfs):.3f}  min {min(rtfs):.3f}  "
          f"max {max(rtfs):.3f}  |  ttfa p50 {statistics.median(ttfa_ms):.0f} ms  |  "
          f"peak mem {mx.get_peak_memory() / 2**30:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
