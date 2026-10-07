"""Image latency/memory benchmark (#101), the #91 metrics doctrine applied to
Qwen-Image-2.1: time-to-image p50/p95-equivalent per cell, per-step cadence +
stability (the step loop is the decode loop), and peak memory (MLX active peak
+ process RSS) across resolutions, solo residency.

One model per process — run the two presets as separate invocations:

    python scripts/bench_image.py --preset base  --out bench_image_base.json
    python scripts/bench_image.py --preset turbo --out bench_image_turbo.json

Cool-down sleeps between runs (thermal doctrine, docs/profiling.md); per-step
times come from mflux's in-loop callback, so each cell's run is fully traced.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import time

# everything this bench needs is cached locally; hub access only adds stalls
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx

from vllm_omni_mlx.diffusion.config import DEFAULT_MODEL, ImageConfig, load_image_model, local_snapshot
from vllm_omni_mlx.diffusion.service import ImageService

# default cells: "SIZE:runs" — three runs up to 1024²-side, two for the
# multi-minute big cells (min/median/max still meaningful at n=2/3)
BASE_CELLS = "512x512:3,768x768:3,1024x1024:3,1280x720:2,1328x1328:2"
TURBO_CELLS = "512x512:3,1024x1024:3"


def turbo_lora_path() -> str:
    """The Viggle turbo adapter, resolved to a local file when cached. The
    root-level diffusers-format files match mflux's key mapping (the peft/
    adapters carry base_model.model prefixes that do not)."""
    from vllm_omni_mlx.diffusion.config import VIGGLE_TURBO_LORA

    candidates = (
        "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors",
        "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors",
    )
    for name in candidates:
        snapshot = local_snapshot(VIGGLE_TURBO_LORA, allow_patterns=(name,))
        if snapshot and os.path.exists(os.path.join(snapshot, name)):
            return os.path.join(snapshot, name)
    return VIGGLE_TURBO_LORA

PROMPT = "A cozy harbor town at dawn, fishing boats on calm water, warm light, photorealistic"


class StepTimer:
    """Per-step cadence via mflux's in-loop callback. Consecutive call_in_loop
    deltas cover mx.eval + the next step — the loop's true steady-state rate."""

    def __init__(self):
        self.times: list[float] = []
        self._t: float | None = None

    def call_before_loop(self, *args, **kwargs):
        self._t = time.perf_counter()

    def call_in_loop(self, *args, **kwargs):
        now = time.perf_counter()
        if self._t is not None:
            self.times.append(now - self._t)
        self._t = now

    def call_after_loop(self, *args, **kwargs):
        self._t = None


def rss_gib() -> float:
    # macOS ru_maxrss is bytes (Linux reports KiB; this repo is macOS-only)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def parse_cells(spec: str) -> list[tuple[int, int, int]]:
    cells = []
    for part in spec.split(","):
        size, _, runs = part.partition(":")
        width_s, height_s = size.lower().split("x")
        cells.append((int(width_s), int(height_s), int(runs or 3)))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("base", "turbo"), default="base")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cells", default=None, help=f"'SIZE:runs,...' (default: preset's, e.g. {BASE_CELLS})")
    parser.add_argument("--lora", default=None, help="LoRA adapter path/repo (turbo preset: the Viggle distilled LoRA)")
    parser.add_argument("--sleep", type=float, default=10.0, help="cool-down seconds between runs")
    parser.add_argument("--steps", type=int, default=None, help="override the preset's step count")
    parser.add_argument("--out", default=None, help="write the raw JSON here")
    args = parser.parse_args()

    if args.preset == "turbo":
        from vllm_omni_mlx.diffusion.config import VIGGLE_TURBO_SCHEDULER, VIGGLE_TURBO_STEPS

        config = ImageConfig(
            model_ref=args.model,
            scheduler=VIGGLE_TURBO_SCHEDULER,
            steps=VIGGLE_TURBO_STEPS,
            lora_paths=(args.lora or turbo_lora_path(),),
        )
    else:
        config = ImageConfig(model_ref=args.model)
    if args.steps is not None:
        config = config.with_overrides(steps=args.steps)

    cells = parse_cells(args.cells or (TURBO_CELLS if args.preset == "turbo" else BASE_CELLS))

    started = time.perf_counter()
    model = load_image_model(config)
    load_s = time.perf_counter() - started
    service = ImageService(model, config)
    timer = StepTimer()
    model.callbacks.register(timer)
    print(
        f"preset={args.preset} steps={config.steps} scheduler={config.scheduler} lora={config.lora_paths or None}\n"
        f"load: {load_s:.1f}s | weights resident: {mx.get_active_memory() / 2**30:.2f} GiB | RSS: {rss_gib():.2f} GiB",
        flush=True,
    )

    meta = {
        "preset": args.preset,
        "model": config.model_ref,
        "family": config.family,
        "steps": config.steps,
        "guidance": config.guidance,
        "scheduler": config.scheduler,
        "lora_paths": list(config.lora_paths) or None,
        "mflux_loaded_bits": getattr(model, "bits", None),
        "load_s": round(load_s, 2),
        "weights_gib": round(mx.get_active_memory() / 2**30, 3),
        "machine": platform.machine(),
        "chip": platform.platform(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    results = {"meta": meta, "cells": []}
    for width, height, runs in cells:
        runs_out = []
        for i in range(runs):
            seed = 1000 + 17 * i
            if args.sleep and (i or runs_out or results["cells"]):
                time.sleep(args.sleep)
            timer.times = []
            mx.reset_peak_memory()
            result = service.generate(PROMPT, size=f"{width}x{height}", seed=seed)[0]
            peak_gib = mx.get_peak_memory() / 2**30
            entry = {
                "run": i,
                "seed": result.seed,
                "tti_s": round(result.elapsed, 2),
                "per_step_s": [round(t, 3) for t in timer.times],
                "per_step_mean_s": round(statistics.fmean(timer.times), 3) if timer.times else None,
                "per_step_stdev_s": round(statistics.stdev(timer.times), 3) if len(timer.times) > 1 else None,
                "peak_mlx_gib": round(peak_gib, 3),
                "active_after_gib": round(mx.get_active_memory() / 2**30, 3),
                "cache_after_gib": round(mx.get_cache_memory() / 2**30, 3),
                "rss_gib": round(rss_gib(), 3),
                "png_kib": round(len(result.png) / 2**10, 1),
            }
            runs_out.append(entry)
            print(
                f"  {width}x{height} run {i}: tti {result.elapsed:.1f}s"
                f" | step {entry['per_step_mean_s']}±{entry['per_step_stdev_s']}s"
                f" | peak {peak_gib:.2f} GiB | RSS {entry['rss_gib']:.2f} GiB | PNG {entry['png_kib']} KiB",
                flush=True,
            )
        ttis = sorted(r["tti_s"] for r in runs_out)
        results["cells"].append(
            {
                "size": f"{width}x{height}",
                "runs": runs_out,
                "tti_min_s": ttis[0],
                "tti_median_s": statistics.median(ttis),
                "tti_max_s": ttis[-1],
            }
        )

    print(f"\nload {load_s:.1f}s, weights {meta['weights_gib']} GiB — time-to-image (steps={config.steps}):")
    for cell in results["cells"]:
        fastest = cell["runs"][0]
        print(
            f"  {cell['size']:>10}  tti {cell['tti_min_s']:>7.1f} / {cell['tti_median_s']:>7.1f} / {cell['tti_max_s']:>7.1f} s"
            f" | step {fastest['per_step_mean_s']}±{fastest['per_step_stdev_s']} s"
            f" | peak {fastest['peak_mlx_gib']:.2f} GiB | RSS {fastest['rss_gib']:.2f} GiB"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
