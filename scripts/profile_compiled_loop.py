#!/usr/bin/env python3
"""Post-#67/#66 phase profile of the COMPILED stream loop (#2 goal session).

Wraps the compiled closures (decode_step / predictor_frame / sampler /
embeds_step) with eval brackets via the same factory indirection the loop
uses, so per-phase GPU time attributes to the phase that queued it.
Reports ms + share per phase, per-frame wall, RTF, and the predictor/talker
weight dtypes (the fp16-dequant product decision inputs).

Weight-gated. Usage: scripts/profile_compiled_loop.py [--max-tokens 400]
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx

import vllm_omni_mlx.tts.compiled_steps as cs
import vllm_omni_mlx.tts.stream_loop as sl
from vllm_omni_mlx.tts.config import DEFAULT_MODEL, TTSConfig, load_tts_model, local_snapshot

_PARAGRAPH = (
    "Welcome to the compiled-loop profiling run. This paragraph is deliberately "
    "long so that the frame loop reaches its sustained regime and the phase "
    "split reflects steady state rather than startup. The voice is a preset, "
    "the sampler is the serving default, and every phase bracket syncs so the "
    "split attributes GPU time to the phase that queued it."
)
LONG_TEXT = " ".join([_PARAGRAPH] * 4)


class Timer:
    def __init__(self):
        self.ms = defaultdict(float)

    def bracket(self, phase, *arrays):
        t0 = time.perf_counter()
        mx.eval(*arrays)
        self.ms[phase] += (time.perf_counter() - t0) * 1000.0


def profile(model, *, text, speaker, temperature, max_tokens):
    timer = Timer()

    real_decode, real_pred, real_sampler, real_embeds = (
        cs.make_talker_decode, cs.make_predictor_frame, cs.make_talker_sampler, cs.make_input_embeds
    )
    frames = {"n": 0}

    def decode_factory(talker):
        inner = real_decode(talker)
        def wrapped(input_embeds, pos, keys, values):
            frames["n"] += 1
            out = inner(input_embeds, pos, keys, values)
            timer.bracket("talker_decode", out[0], out[1])
            return out
        return wrapped

    def pred_factory(predictor, **kw):
        inner = real_pred(predictor, **kw)
        def wrapped(hidden, first):
            out = inner(hidden, first)
            timer.bracket("predictor_frame", out)
            return out
        return wrapped

    def sampler_factory(model_, **kw):
        inner = real_sampler(model_, **kw)
        def wrapped(logits, history):
            out = inner(logits, history)
            timer.bracket("talker_sample", out)
            return out
        return wrapped

    def embeds_factory(talker):
        inner = real_embeds(talker)
        def wrapped(codes, text_embed):
            out = inner(codes, text_embed)
            timer.bracket("input_prep", out)
            return out
        return wrapped

    # the loop imports factories into its own namespace at import time
    for mod in (cs, sl):
        mod.make_talker_decode = decode_factory
        mod.make_predictor_frame = pred_factory
        mod.make_talker_sampler = sampler_factory
        mod.make_input_embeds = embeds_factory
    try:
        t0 = time.perf_counter()
        audio = 0
        for chunk in sl.generate_frames(
            model, text=text, speaker=speaker, temperature=temperature,
            top_k=50, top_p=1.0, repetition_penalty=1.05, max_tokens=max_tokens,
        ):
            audio += chunk.shape[0]
        wall = time.perf_counter() - t0
    finally:
        for mod in (cs, sl):
            mod.make_talker_decode, mod.make_predictor_frame = real_decode, real_pred
            mod.make_talker_sampler, mod.make_input_embeds = real_sampler, real_embeds
    return timer, frames["n"], wall, audio / 24000.0


def sustained_load(seconds=1.0):
    a = mx.random.normal((2048, 2048))
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = a @ a
        mx.eval(a)
        a = a / mx.sqrt((a * a).mean())
    mx.clear_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--speaker", default="vivian")
    args = parser.parse_args()

    if local_snapshot(DEFAULT_MODEL) is None:
        raise SystemExit(f"{DEFAULT_MODEL} not cached locally")

    print(f"loading {DEFAULT_MODEL} …", flush=True)
    model = load_tts_model(TTSConfig())
    sl.prewarm_streaming(model, 0.5, 0.2)

    # dtypes for the record (fp16-dequant product decision inputs)
    def weight_dtype(module):
        params = module.parameters()
        while isinstance(params, dict):
            v = next(iter(params.values()), None)
            if v is None:
                return None
            params = v
        return params.dtype
    t = model.talker
    print(f"talker layer0 q_proj: {weight_dtype(t.model.layers[0].self_attn.q_proj)}  "
          f"predictor layer0 q_proj: {weight_dtype(t.code_predictor.model.layers[0].self_attn.q_proj)}  "
          f"text_projection: {weight_dtype(t.text_projection)}  "
          f"predictor lm_head[0]: {weight_dtype(t.code_predictor.lm_head[0])}")

    # warmup then clock ramp
    for _ in sl.generate_frames(model, text=LONG_TEXT, speaker=args.speaker,
                                temperature=args.temperature, max_tokens=48):
        pass
    sustained_load(1.0)

    mx.random.seed(7)
    timer, n, wall, audio_s = profile(
        model, text=LONG_TEXT, speaker=args.speaker,
        temperature=args.temperature, max_tokens=args.max_tokens,
    )
    print(f"\nframes {n} ({audio_s:.1f} s audio), wall {wall:.2f} s, RTF {wall / audio_s:.3f}, "
          f"frame wall {wall / max(1, n) * 1000:.2f} ms")
    total = sum(timer.ms.values())
    print(f"\n{'phase':>16} {'total ms':>10} {'ms/frame':>9} {'share':>7}")
    for phase in ("talker_decode", "predictor_frame", "talker_sample", "input_prep"):
        share = timer.ms[phase] / total * 100 if total else 0
        print(f"{phase:>16} {timer.ms[phase]:10.0f} {timer.ms[phase] / max(1, n):9.2f} {share:6.1f}%")
    residue = wall * 1000 - total
    print(f"{'loop residue':>16} {residue:10.0f} {residue / max(1, n):9.2f} "
          f"{residue / total * 100 if total else 0:6.1f}%   (vocoder + flush + python)")
    print(f"peak memory: {mx.get_peak_memory() / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
