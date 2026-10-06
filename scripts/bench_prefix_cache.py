#!/usr/bin/env python3
"""#66 A/B bench: voice-prefix cache OFF vs ON (hits) — TTFA / RTF, interleaved.

The repo A/B rule (same conditions, interleaved, cool machine): each pair
runs one OFF turn (``VLLM_OMNI_TTS_PREFIX_CACHE=0`` — today's batched
per-request prefill) and one ON turn (warm cache — the spliced prefix),
separated by cool-down sleeps, with the pair's order alternating so thermal
drift can't systematically favor either side. Reports per-turn first-chunk
latency (the in-loop TTFA analog), wall/audio/RTF, and p50/min per mode;
plus one cold-cache ON turn (the per-voice miss) for the record.

Usage: scripts/bench_prefix_cache.py [--pairs 4] [--max-tokens 240]
       [--cooldown 8] [--speaker vivian]
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx

from vllm_omni_mlx.tts.config import DEFAULT_MODEL, TTSConfig, load_tts_model, local_snapshot
from vllm_omni_mlx.tts import prefix_cache
from vllm_omni_mlx.tts.stream_loop import prewarm_streaming, synthesize_stream

TEXT = (
    "Welcome to the prefix cache benchmark. This paragraph runs long enough "
    "that generation reaches steady state well before it ends, so the real "
    "time factor reflects the decode loop and the first chunk latency "
    "reflects the prompt path rather than startup noise. The sampler runs "
    "at serving defaults and the voice is a preset."
) * 2


def ramp(seconds: float = 1.0) -> None:
    a = mx.random.normal((2048, 2048))
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = a @ a
        mx.eval(a)
        a = a / mx.sqrt((a * a).mean())
    mx.clear_cache()


def one_turn(model, config, max_tokens, seed):
    t0 = time.perf_counter()
    first = None
    samples = 0
    for chunk in synthesize_stream(
        model, config, TEXT, speaker="vivian", temperature=0.9, seed=seed,
        streaming_interval=0.5, streaming_initial_interval=0.2, max_tokens=max_tokens,
    ):
        if first is None:
            first = (time.perf_counter() - t0) * 1000
        samples += chunk.shape[0]
    wall = time.perf_counter() - t0
    return first, wall, samples / 24000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=240)
    parser.add_argument("--cooldown", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if local_snapshot(DEFAULT_MODEL) is None:
        raise SystemExit(f"{DEFAULT_MODEL} not cached locally")

    print(f"loading {DEFAULT_MODEL} …", flush=True)
    model = load_tts_model(TTSConfig())
    config = TTSConfig()
    prewarm_streaming(model, 0.5, 0.2)

    # warmup: traces + allocator steady state, then populate the prefix for
    # the ON side and ramp the clock
    for _ in synthesize_stream(model, config, TEXT, speaker="vivian", seed=args.seed, max_tokens=40):
        pass
    ramp(1.0)
    stats = prefix_cache.prefix_cache_stats(model)
    print(f"prefix store: {stats['entries']} entries, {stats['bytes'] / 2**20:.2f} MiB", flush=True)

    results = {"off": [], "on": [], "miss": []}

    # one cold-cache ON turn for the record (the per-voice miss cost) —
    # cooled down like every measured turn so the warmup ramp's heat
    # doesn't land on it
    os.environ["VLLM_OMNI_TTS_PREFIX_CACHE"] = "1"
    prefix_cache.clear_prefix_caches()
    time.sleep(args.cooldown)
    r = one_turn(model, config, args.max_tokens, args.seed)
    results["miss"].append(r)
    print(f"{'ON (miss, cold)':>16}: first {r[0]:6.0f} ms  wall {r[1]:5.1f} s  audio {r[2]:5.1f} s  RTF {r[1] / r[2]:.3f}", flush=True)

    for pair in range(args.pairs):
        on_first = pair % 2 == 1  # alternate order within the pair
        order = (("on", "1"), ("off", "0")) if on_first else (("off", "0"), ("on", "1"))
        for mode, flag in order:
            os.environ["VLLM_OMNI_TTS_PREFIX_CACHE"] = flag
            if mode == "on" and prefix_cache.lookup_prefix_kv(model, "vivian", "auto", None) is None:
                raise SystemExit("ON turn found a cold store — warmup incomplete")
            time.sleep(args.cooldown)
            r = one_turn(model, config, args.max_tokens, args.seed)
            results[mode].append(r)
            print(f"pair {pair} {'ON (hit)':>16}: first {r[0]:6.0f} ms  wall {r[1]:5.1f} s  audio {r[2]:5.1f} s  RTF {r[1] / r[2]:.3f}"
                  if mode == "on" else
                  f"pair {pair} {'OFF (uncached)':>16}: first {r[0]:6.0f} ms  wall {r[1]:5.1f} s  audio {r[2]:5.1f} s  RTF {r[1] / r[2]:.3f}",
                  flush=True)

    print()
    for mode, label in (("off", "OFF (uncached)"), ("on", "ON (hit)")):
        firsts = [r[0] for r in results[mode]]
        rtfs = [r[1] / r[2] for r in results[mode]]
        print(f"{label:>16}: first-chunk p50 {statistics.median(firsts):6.0f} ms  min {min(firsts):6.0f} ms   "
              f"RTF p50 {statistics.median(rtfs):.3f}  min {min(rtfs):.3f}")
    off_p50 = statistics.median([r[0] for r in results["off"]])
    on_p50 = statistics.median([r[0] for r in results["on"]])
    print(f"{'ΔTTFA (p50)':>16}: {off_p50 - on_p50:+.0f} ms")
    print(f"\npeak memory: {mx.get_peak_memory() / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
