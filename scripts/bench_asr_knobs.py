#!/usr/bin/env python3
"""ASR knob bench (#68 perf evidence): checkpoint ladder, prefill/chunk knobs.

Benchmarks the knobs the serving path currently leaves at mlx-audio defaults:
checkpoint quant variant, ``prefill_step_size``, ``chunk_duration``,
``batch_size``. Requires the ``[asr]`` extra and locally cached checkpoints;
eval set from ``scripts/build_asr_eval.py`` (defaults to
``~/.cache/vllm-omni-mlx/asr-eval``; ``--eval-dir`` to override).

Two modes:

- ``plan`` runs a list of configs sequentially (JSON list of
  ``{"model", "prefill", "chunk", "batch"}`` dicts) — placement only, one
  model resident at a time;
- ``ab`` interleaves two configs rep-by-rep in one process — the only mode
  comparative claims may rest on. On this hardware sequential sweeps drift
  enough to invert small effects (a ≤1024 prefill "win" of −4% in the sweep
  reversed to +1…3.5% for the default across 6/6 interleaved pairs); treat
  ``plan`` deltas under ~10% as noise until ``ab`` confirms them.

Discipline: GPU-serialized (nothing else on the Metal device), medians over
clips, WER against manifest references as the quality guard, forced
``mx.synchronize()`` on both sides of every timed generate.

    python scripts/bench_asr_knobs.py --mode plan --clips short --configs \
      '[{"model":"mlx-community/Qwen3-ASR-1.7B-4bit","prefill":2048}]'
    python scripts/bench_asr_knobs.py --mode ab --clips short+long --repeats 3 \
      --configs '[{"model":"mlx-community/Qwen3-ASR-1.7B-4bit","prefill":2048},
                  {"model":"mlx-community/Qwen3-ASR-1.7B-4bit","prefill":1024}]'
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx

DEFAULT_EVAL_DIR = Path.home() / ".cache/vllm-omni-mlx/asr-eval"


def normalize(text: str) -> list[str]:
    return re.sub(r"[^a-z' ]+", " ", text.lower()).split()


def wer(reference: str, hypothesis: str) -> float:
    """Same scorer as scripts/asr_roundtrip.py."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / max(len(ref), 1)


@dataclass
class Clip:
    name: str
    text: str
    seconds: float
    samples: Any


def load_clips(eval_dir: Path) -> list[Clip]:
    import numpy as np

    manifest = json.loads((eval_dir / "manifest.json").read_text())
    clips = []
    for c in manifest:
        with wave.open(str(eval_dir / f"{c['name']}.wav"), "rb") as w:
            if w.getframerate() != 16000 or w.getnchannels() != 1:
                raise SystemExit(f"eval clip {c['name']} is not 16 kHz mono")
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        clips.append(Clip(c["name"], c["text"], c["seconds"], pcm.astype(np.float32) / 32768.0))
    return clips


def weights_gib(model) -> float:
    from mlx.utils import tree_flatten

    n = sum(
        v.nbytes
        for _, v in tree_flatten(model.parameters())
        if isinstance(v, mx.array)
    )
    return n / 2**30


def run_once(model, clip: Clip, cfg: dict) -> dict:
    mx.synchronize()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    out = model.generate(
        clip.samples,
        temperature=0.0,
        max_tokens=8192,
        language=None,
        prefill_step_size=cfg.get("prefill", 2048),
        chunk_duration=cfg.get("chunk", 1200.0),
        batch_size=cfg.get("batch", 1),
        verbose=False,
    )
    mx.synchronize()
    wall = time.perf_counter() - t0
    text = (getattr(out, "text", "") or "").strip()
    if isinstance(text, list):
        text = text[0] if text else ""
    return {
        "wall_s": wall,
        "rtf": wall / clip.seconds,
        "wer": wer(clip.text, text),
        "gen_tokens": int(getattr(out, "generation_tokens", 0) or 0),
        "peak_gib": mx.get_peak_memory() / 2**30,
    }


def summarize(rows: list[dict]) -> dict:
    def med(pred, field):
        vals = [r[field] for r in rows if pred(r["clip"])]
        return round(statistics.median(vals), 3) if vals else None

    shorts = [r for r in rows if r["clip"].startswith("short")]
    return {
        "short_wall_med_s": med(lambda c: c.startswith("short"), "wall_s"),
        "short_rtf_med": med(lambda c: c.startswith("short"), "rtf"),
        "short_wer_mean": round(sum(r["wer"] for r in shorts) / max(len(shorts), 1), 4),
        "long60_rtf": med(lambda c: c == "long60", "rtf"),
        "long150_rtf": med(lambda c: c == "long150", "rtf"),
        "peak_gib_max": round(max(r["peak_gib"] for r in rows), 2) if rows else 0,
    }


def bench(tag: str, cfg: dict, clips: list[Clip], model, rep: int, jsonl) -> list[dict]:
    rows = []
    for clip in clips:
        r = run_once(model, clip, cfg)
        r.update(tag=tag, rep=rep, **cfg, clip=clip.name, audio_s=clip.seconds)
        jsonl.write(json.dumps(r) + "\n")
        rows.append(r)
        print(
            f"[{tag}] {clip.name} rep{rep} wall={r['wall_s']:.2f}s rtf={r['rtf']:.3f} "
            f"wer={r['wer']:.3f} tok={r['gen_tokens']} peak={r['peak_gib']:.2f}GiB",
            flush=True,
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["plan", "ab"], default="plan")
    ap.add_argument("--configs", required=True,
                    help="JSON list of config dicts (plan) or exactly [A, B] (ab)")
    ap.add_argument("--tags", default=None, help="comma-separated, one per config")
    ap.add_argument("--clips", default="short",
                    help="short | short+long | comma-separated clip names")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    ap.add_argument("--jsonl-out", default=None, help="append raw per-clip rows here")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    if not (eval_dir / "manifest.json").exists():
        raise SystemExit(f"no eval set at {eval_dir}; run scripts/build_asr_eval.py first")
    all_clips = load_clips(eval_dir)
    if args.clips == "short":
        clips = [c for c in all_clips if c.name.startswith("short")]
    elif args.clips == "short+long":
        clips = all_clips
    else:
        wanted = args.clips.split(",")
        clips = [c for c in all_clips if c.name in wanted]
    if not clips:
        raise SystemExit("no clips selected")

    configs = json.loads(args.configs)
    tags = (args.tags or ",".join(f"c{i}" for i in range(len(configs)))).split(",")
    from mlx_audio.stt import utils as stt_utils

    summaries = []
    warm_clip = min(clips, key=lambda c: c.seconds)

    def load(cfg):
        t0 = time.perf_counter()
        m = stt_utils.load_model(cfg["model"])
        run_once(m, warm_clip, cfg)  # warmup: first hit pays compile/alloc
        return m, time.perf_counter() - t0

    with open(args.jsonl_out, "a") if args.jsonl_out else open(os.devnull, "w") as jsonl:
        if args.mode == "plan":
            if args.repeats != 1:
                print("plan mode ignores --repeats > 1 (use ab for comparisons)", file=sys.stderr)
            for tag, cfg in zip(tags, configs):
                model, load_s = load(cfg)
                rows = bench(tag, cfg, clips, model, 0, jsonl)
                s = summarize(rows)
                s.update(weights_gib=round(weights_gib(model), 2), load_s=round(load_s, 1))
                summaries.append(s)
                del model
                mx.clear_cache()
                time.sleep(1.0)
        else:  # ab: interleaved, one model resident per checkpoint
            a, b = configs
            cur_ref, cur_model = None, None
            for rep in range(args.repeats):
                for tag, cfg in ((f"{tags[0]}#{rep}", a), (f"{tags[1]}#{rep}", b)):
                    if cur_ref != cfg["model"]:
                        if cur_model is not None:
                            del cur_model
                            mx.clear_cache()
                        cur_model, _ = load(cfg)
                        cur_ref = cfg["model"]
                    rows = bench(tag, cfg, clips, cur_model, rep, jsonl)
                    summaries.append({"tag": tag, **summarize(rows)})

    print(json.dumps(summaries, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
