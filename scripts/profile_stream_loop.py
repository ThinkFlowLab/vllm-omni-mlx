#!/usr/bin/env python3
"""Task 1 of #65: per-frame phase split of the vendored stream loop.

Runs an eval-bracket-instrumented copy of ``generate_frames``
(same call order, one extra ``mx.eval`` per phase — the brackets add sync
overhead to the whole, so absolute RTF here reads slightly high; the phase
*shares* are the signal) and reports:

- ms and share per phase: talker forward / talker sampling / predictor
  forwards / predictor sampling / next-input prep / EOS sync / vocoder
  decode (streaming_step) / python residue (frame wall − bracketed phases)
- per-frame wall p50/p90, overall RTF, and decile-binned frame wall across
  the generation (sustained-RTF drift — the O(n²) penalty hypothesis; note
  mlx-audio's repetition penalty only rescans the last
  ``repetition_context_size=64`` tokens, so the hypothesis predicts *no*
  super-linear drift and attention-KV growth is the competing explanation)
- model geometry (layers/hidden/groups) for the record

Weight-gated like the tests: resolves from the local HF cache only.
Usage: scripts/profile_stream_loop.py [--max-tokens 800] [--temperature 0.9]
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx

from vllm_omni_mlx.tts.code_predictor import CodePredictor
from vllm_omni_mlx.tts.config import (
    DEFAULT_MODEL,
    TTSConfig,
    load_tts_model,
    local_snapshot,
)
from vllm_omni_mlx.tts.prompt_embeds import PromptEmbeds
from vllm_omni_mlx.tts.stream_loop import (
    SAMPLES_PER_FRAME,
    generate_frames,
    prewarm_streaming,
)
from vllm_omni_mlx.tts.talker import Talker

_PARAGRAPH = (
    "Welcome to the decode-side profiling run. This paragraph is deliberately "
    "long so that the frame loop reaches its sustained regime and any drift "
    "with generation length becomes visible in the decile table. The voice is "
    "a preset, the sampler is the serving default, and every phase bracket "
    "syncs so the split attributes GPU time to the phase that queued it."
)
# 4x repetition keeps EOS away for ~800+ frames so the drift table has range
LONG_TEXT = " ".join([_PARAGRAPH] * 4)

PHASES = (
    "prefill",
    "talker_fwd",
    "talker_sample",
    "predictor_fwd",
    "predictor_sample",
    "input_prep",
    "eos_sync",
    "decode",
)


class Timer:
    def __init__(self) -> None:
        self.ms: dict[str, float] = defaultdict(float)

    def bracket(self, phase: str, *arrays: mx.array) -> None:
        t0 = time.perf_counter()
        mx.eval(*arrays)
        self.ms[phase] += (time.perf_counter() - t0) * 1000.0


def profiled_frames(model, timer: Timer, *, text: str, speaker: str, temperature: float,
                    top_k: int, top_p: float, repetition_penalty: float, max_tokens: int,
                    initial_frames: int = 2, chunk_frames: int = 6):
    """``generate_frames`` with per-phase eval brackets — the
    call order is step-for-step the production loop's; only the sync points
    are added (one per phase instead of one per frame)."""
    talker = Talker(model.talker)
    predictor = CodePredictor(model.talker)
    t0 = time.perf_counter()
    layout = PromptEmbeds(model).build(text, speaker)
    timer.bracket("prefill", layout.input_embeds, layout.trailing_text_hidden, layout.decode_text_embed)
    input_embeds = layout.input_embeds
    trailing_text_hidden = layout.trailing_text_hidden
    tts_pad_embed = layout.decode_text_embed

    config = model.config.talker_config
    eos_token_id = config.codec_eos_token_id
    suppress_tokens = talker.suppressed_codec_ids()

    cache = model.talker.make_cache()
    code_cache = predictor.make_cache()
    generated_codes: list[mx.array] = []
    generated_token_ids: list[int] = []
    frame_walls: list[float] = []
    trailing_idx = 0
    decoded_frames = 0
    audio_frames = 0

    decoder = model.speech_tokenizer.decoder
    decoder.reset_streaming_state()

    def decode_pending() -> mx.array:
        nonlocal decoded_frames, audio_frames
        t0 = time.perf_counter()
        remainder = len(generated_codes) - decoded_frames
        codes_chunk = mx.stack(generated_codes[decoded_frames:], axis=1)
        codes_for_decoder = mx.transpose(codes_chunk, (0, 2, 1))
        mx.eval(codes_for_decoder)
        audio = decoder.streaming_step(codes_for_decoder).squeeze(1)[0]
        mx.eval(audio)
        decoded_frames = len(generated_codes)
        timer.ms["decode"] += (time.perf_counter() - t0) * 1000.0
        audio_frames += remainder
        return audio[: remainder * SAMPLES_PER_FRAME]

    boundary_initial = initial_frames
    for _ in range(max_tokens):
        t_frame = time.perf_counter()

        logits, hidden = model.talker(input_embeds, cache=cache)
        timer.bracket("talker_fwd", logits, hidden)

        next_token = model._sample_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=generated_token_ids or None,
            suppress_tokens=suppress_tokens,
        )
        is_eos = next_token[0, 0] == eos_token_id
        timer.bracket("talker_sample", next_token, is_eos)

        code_tokens = [next_token]
        code_hidden = hidden[:, -1:, :]
        for c in code_cache:
            c.keys = None
            c.values = None
            c.offset = 0
        step_logits = []
        for code_idx in range(predictor.num_code_groups - 1):
            if code_idx == 0:
                code_0_embed = model.talker.get_input_embeddings()(next_token)
                code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
            else:
                code_input = predictor.residual_embeddings[code_idx - 1](code_tokens[-1])
            code_logits = predictor.step(code_input, code_cache, code_idx)
            step_logits.append(code_logits)
            timer.bracket("predictor_fwd", code_logits)
            sampled = model._sample_token(code_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            code_tokens.append(sampled)
            timer.bracket("predictor_sample", sampled)
        all_codes = mx.concatenate(code_tokens, axis=1)
        _ = step_logits  # kept for symmetry with per-step split above

        if trailing_idx < trailing_text_hidden.shape[1]:
            text_embed = trailing_text_hidden[:, trailing_idx : trailing_idx + 1, :]
            trailing_idx += 1
        else:
            text_embed = tts_pad_embed
        input_embeds = text_embed + talker.codec_embeds(code_tokens)

        t0 = time.perf_counter()
        mx.eval(input_embeds)
        timer.ms["input_prep"] += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        eos = is_eos.item()
        timer.ms["eos_sync"] += (time.perf_counter() - t0) * 1000.0
        if eos:
            frame_walls.append(time.perf_counter() - t_frame)
            break

        generated_token_ids.append(int(next_token[0, 0]))
        generated_codes.append(all_codes)

        boundary = boundary_initial if decoded_frames == 0 else chunk_frames
        if len(generated_codes) - decoded_frames >= boundary:
            decode_pending()

        frame_walls.append(time.perf_counter() - t_frame)

    if len(generated_codes) > decoded_frames:
        yield_last = decode_pending()
    else:
        yield_last = None
    decoder.reset_streaming_state()
    return frame_walls, audio_frames, yield_last


def sustained_load(seconds: float = 1.0) -> None:
    """Clock-ramp prewarm per docs/profiling.md — ~1s of sustained GPU work."""
    a = mx.random.normal((2048, 2048))
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = a @ a
        mx.eval(a)
        a = a / mx.sqrt((a * a).mean())
    mx.clear_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-tokens", type=int, default=800, help="frame budget (default 800 ≈ 64 s audio)")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--speaker", default="vivian")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-tokens", type=int, default=48)
    args = parser.parse_args()

    if local_snapshot(DEFAULT_MODEL) is None:
        raise SystemExit(f"{DEFAULT_MODEL} not cached locally — weight-gated profile needs the snapshot")

    print(f"loading {DEFAULT_MODEL} …", flush=True)
    model = load_tts_model(TTSConfig())
    tc = model.config.talker_config
    pc = tc.code_predictor_config
    print(
        f"talker: {tc.num_hidden_layers} layers, hidden {tc.hidden_size}, "
        f"vocab {tc.vocab_size}; predictor: {pc.num_hidden_layers} layers, "
        f"hidden {pc.hidden_size}, groups {pc.num_code_groups}",
        flush=True,
    )

    prewarm_streaming(model, 0.5, 0.2)

    # warmup: discarded short generation (shader + allocator + JIT), then a
    # sustained-load clock ramp, then the measured run
    print(f"warmup generation ({args.warmup_tokens} frames) …", flush=True)
    for _ in generate_frames(
        model, text=LONG_TEXT, speaker=args.speaker, temperature=args.temperature,
        max_tokens=args.warmup_tokens,
    ):
        pass
    sustained_load(1.0)

    timer = Timer()
    mx.random.seed(args.seed)
    t0 = time.perf_counter()
    frame_walls, audio_frames, _ = profiled_frames(
        model, timer, text=LONG_TEXT, speaker=args.speaker,
        temperature=args.temperature, top_k=50, top_p=1.0,
        repetition_penalty=1.05, max_tokens=args.max_tokens,
    )
    wall = time.perf_counter() - t0

    n = len(frame_walls)
    audio_seconds = audio_frames / 12.5
    print(f"\nframes generated: {n}  ({audio_seconds:.1f} s audio), wall {wall:.2f} s, "
          f"RTF {wall / audio_seconds:.3f}")
    walls_ms = [w * 1000 for w in frame_walls]
    print(f"frame wall ms: p50 {statistics.median(walls_ms):.2f}  "
          f"p90 {statistics.quantiles(walls_ms, n=10)[8]:.2f}  mean {statistics.mean(walls_ms):.2f}")

    loop_ms = sum(timer.ms[p] for p in PHASES if p != "prefill")
    residue = loop_ms and (sum(walls_ms) - loop_ms)
    print(f"\n{'phase':>16} {'total ms':>10} {'ms/frame':>9} {'share':>7}")
    for phase in PHASES:
        share = timer.ms[phase] / loop_ms * 100 if loop_ms else 0.0
        per_frame = timer.ms[phase] / n if n else 0.0
        print(f"{phase:>16} {timer.ms[phase]:10.0f} {per_frame:9.2f} {share:6.1f}%")
    print(f"{'python residue':>16} {residue:10.0f} {residue / n if n else 0:9.2f} "
          f"{residue / loop_ms * 100 if loop_ms else 0:6.1f}%")

    # decile drift: mean frame wall per tenth of the generation
    decile = max(1, n // 10)
    print(f"\n{'decile':>7} {'mean ms':>8}")
    for d in range(10):
        chunk = walls_ms[d * decile : (d + 1) * decile] if d < 9 else walls_ms[9 * decile :]
        if chunk:
            print(f"{d + 1:>7} {statistics.mean(chunk):8.2f}")

    peak = mx.get_peak_memory() / 2**30
    print(f"\npeak memory: {peak:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
