#!/usr/bin/env python3
"""Task 1 of #66: per-request TTFA cost split — prompt build vs prefix
forward vs frame loop — plus the Level-2 splice exactness probe.

For each measured request this reports:

- prompt build, split into tokenizer / text embeds+projection / voice-static
  pieces (the Level-1 cache's savings: tts specials, speaker+codec lookups,
  instruct projection) / assembly concat math — the text rows are the only
  per-request-necessary part;
- the fresh prefill forward (the whole ``input_embeds``: instruct? + role 3 +
  combined n−1 + first-text 1 rows, eager, through ``model.talker``) — the
  per-request cost Level 2 replaces;
- the Level-2 replacement cost: voice-prefix forward (static rows only, once
  per voice key) and the 1-row splice decode at the prefix offset;
- steady-state frame wall for scale;
- exactness: last-row logits/hidden of the fresh prefill vs the splice
  (compiled ``decode_step`` and eager 1-row forward) — max|Δ| and argmax
  equality. max|Δ| == 0 means the splice is bit-identical and greedy parity
  is guaranteed by construction.

Weight-gated like the tests: resolves from the local HF cache only.
Usage: scripts/profile_prefix_cost.py [--voices vivian,ryan] [--iters 5]
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx

from vllm_omni_mlx.tts.config import (
    DEFAULT_MODEL,
    TTSConfig,
    load_tts_model,
    local_snapshot,
)
from vllm_omni_mlx.tts.stream_loop import generate_custom_voice_frames, prewarm_streaming
from vllm_omni_mlx.tts.compiled_steps import make_talker_decode

EN = "The quick brown fox jumps over the lazy dog."
LONG_TEXT = (
    "Welcome to the prefix-cost profiling run. This paragraph is deliberately "
    "long so the frame loop reaches its sustained regime and the per-request "
    "prefill cost can be compared against the steady frame wall."
)


def _timed(fn):
    out = {}
    t0 = time.perf_counter()
    result = fn(out)
    out["total"] = (time.perf_counter() - t0) * 1000.0
    return result, out


def build_split(model, text: str, speaker: str, language: str, instruct: str | None):
    """``_prepare_generation_inputs`` re-run with per-stage timers — the ops
    are mlx-audio's (MIT), in their order, each bracketed with mx.eval so the
    stage owns its GPU time."""
    ms = defaultdict(float)

    def bracket(name, *arrays):
        t0 = time.perf_counter()
        mx.eval(*arrays)
        ms[name] += (time.perf_counter() - t0) * 1000.0

    talker = model.talker
    config = model.config.talker_config

    chat_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    t0 = time.perf_counter()
    input_ids = mx.array(model.tokenizer.encode(chat_text))[None, :]
    ms["tokenizer"] += (time.perf_counter() - t0) * 1000.0
    text_embed = talker.text_projection(talker.get_text_embeddings()(input_ids))
    bracket("text_embed+proj", text_embed)

    tts_tokens = mx.array(
        [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]]
    )
    tts_embeds = talker.text_projection(talker.get_text_embeddings()(tts_tokens))
    bracket("tts_specials", tts_embeds)

    spk_ids = mx.array([[config.spk_id[speaker.lower()]]])
    speaker_embed = talker.get_input_embeddings()(spk_ids)
    language_id = None
    if language.lower() != "auto" and config.codec_language_id:
        language_id = config.codec_language_id.get(language.lower())
    if language_id is None:
        codec_prefill = [config.codec_nothink_id, config.codec_think_bos_id, config.codec_think_eos_id]
    else:
        codec_prefill = [
            config.codec_think_id, config.codec_think_bos_id, language_id, config.codec_think_eos_id,
        ]
    codec_embed = talker.get_input_embeddings()(mx.array([codec_prefill]))
    codec_embed_suffix = talker.get_input_embeddings()(mx.array([[config.codec_pad_id, config.codec_bos_id]]))
    codec_embed = mx.concatenate([codec_embed, speaker_embed.reshape(1, 1, -1), codec_embed_suffix], axis=1)
    bracket("voice_pieces", codec_embed)

    instruct_embed = None
    if instruct:
        instruct_text = f"<|im_start|>user\n{instruct}<|im_end|>\n"
        instruct_ids = mx.array(model.tokenizer.encode(instruct_text))[None, :]
        instruct_embed = talker.text_projection(talker.get_text_embeddings()(instruct_ids))
        bracket("instruct_embed", instruct_embed)

    tts_bos_embed, tts_eos_embed, tts_pad_embed = (tts_embeds[:, i : i + 1, :] for i in range(3))
    role_embed = text_embed[:, :3, :]
    pad_count = codec_embed.shape[1] - 2
    pad_embeds = mx.broadcast_to(tts_pad_embed, (1, pad_count, tts_pad_embed.shape[-1]))
    combined = mx.concatenate([pad_embeds, tts_bos_embed], axis=1) + codec_embed[:, :-1, :]
    pieces = ([instruct_embed] if instruct_embed is not None else []) + [role_embed, combined]
    input_embeds = mx.concatenate(pieces, axis=1)
    first_text = text_embed[:, 3:4, :] + codec_embed[:, -1:, :]
    input_embeds = mx.concatenate([input_embeds, first_text], axis=1)
    trailing = mx.concatenate([text_embed[:, 4:-5, :], tts_eos_embed], axis=1)
    bracket("assembly", input_embeds, trailing)
    ms["build_total"] = sum(v for k, v in ms.items() if k != "build_total")
    return input_embeds, ms


def sustained_load(seconds: float = 1.0) -> None:
    a = mx.random.normal((2048, 2048))
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = a @ a
        mx.eval(a)
        a = a / mx.sqrt((a * a).mean())
    mx.clear_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--cases", default="vivian:auto:,ryan:en:,vivian:auto:Speak with cheerful energy",
        help="speaker:language:instruct cases, comma-separated",
    )
    args = parser.parse_args()

    if local_snapshot(DEFAULT_MODEL) is None:
        raise SystemExit(f"{DEFAULT_MODEL} not cached locally — weight-gated profile needs the snapshot")

    print(f"loading {DEFAULT_MODEL} …", flush=True)
    model = load_tts_model(TTSConfig())
    tc = model.config.talker_config
    decode_step = make_talker_decode(model.talker)
    prewarm_streaming(model, 0.5, 0.2)
    # warmup so the first measured iter sees steady kernels/allocator
    for _ in generate_custom_voice_frames(model, text=LONG_TEXT, speaker="vivian", max_tokens=24):
        pass
    sustained_load(1.0)

    texts = [EN, LONG_TEXT]
    per_case = {}
    for case in args.cases.split(","):
        speaker, language, instruct = (case.split(":") + ["", ""])[:3]
        instruct = instruct or None
        rows = defaultdict(list)
        for i in range(args.iters):
            text = texts[i % len(texts)]
            input_embeds, ms = build_split(model, text, speaker, language, instruct)
            for k, v in ms.items():
                rows[k].append(v)

            # fresh prefill (today's first frame)
            cache = model.talker.make_cache()
            t0 = time.perf_counter()
            logits_fresh, hidden_fresh = model.talker(input_embeds, cache=cache)
            mx.eval(logits_fresh, hidden_fresh)
            rows["prefill_fwd"].append((time.perf_counter() - t0) * 1000.0)
            total = cache[0].offset

            # Level 2: static-prefix forward + 1-row splice at the offset
            static = input_embeds[:, :-1, :]
            c2 = model.talker.make_cache()
            t0 = time.perf_counter()
            model.talker(static, cache=c2)
            prefix_len = c2[0].offset
            keys = [c.keys[..., :prefix_len, :] for c in c2]
            values = [c.values[..., :prefix_len, :] for c in c2]
            mx.eval(keys, values)
            rows["prefix_fwd(once/voice)"].append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            logits_c, hidden_c, _, _ = decode_step(
                input_embeds[:, -1:, :], mx.array([prefix_len], dtype=mx.int32), keys, values
            )
            mx.eval(logits_c, hidden_c)
            rows["splice_decode(compiled)"].append((time.perf_counter() - t0) * 1000.0)

            # eager splice twin: transplant into cache objects, 1-row forward
            c3 = model.talker.make_cache()
            for c, k, v in zip(c3, keys, values):
                c.keys, c.values, c.offset = k, v, prefix_len
            t0 = time.perf_counter()
            logits_e, hidden_e = model.talker(input_embeds[:, -1:, :], cache=c3)
            mx.eval(logits_e, hidden_e)
            rows["splice_decode(eager)"].append((time.perf_counter() - t0) * 1000.0)

            # exactness of the splice vs the fresh prefill's last row
            d_c = float(mx.abs(logits_fresh[:, -1, :] - logits_c[:, -1, :]).max())
            d_e = float(mx.abs(logits_fresh[:, -1, :] - logits_e[:, -1, :]).max())
            am_fresh = int(mx.argmax(logits_fresh[:, -1, :]).item())
            am_c = int(mx.argmax(logits_c[:, -1, :]).item())
            am_e = int(mx.argmax(logits_e[:, -1, :]).item())
            rows["Δlogits_compiled"].append(d_c)
            rows["Δlogits_eager"].append(d_e)
            rows["argmax_match"].append(am_fresh == am_c == am_e)
            rows["_prefix_rows"].append(prefix_len)
            rows["_total_rows"].append(total)

        per_case[(speaker, language, bool(instruct))] = rows

        label = f"{speaker}/{language}" + ("/instruct" if instruct else "")
        print(f"\n=== {label}  (prefix {rows['_prefix_rows'][-1]} of {rows['_total_rows'][-1]} rows static) ===")
        for k in (
            "tokenizer", "text_embed+proj", "tts_specials", "voice_pieces", "instruct_embed",
            "assembly", "build_total", "prefill_fwd", "prefix_fwd(once/voice)",
            "splice_decode(compiled)", "splice_decode(eager)",
        ):
            if rows.get(k):
                v = rows[k]
                print(f"  {k:>24}: mean {statistics.mean(v):7.2f} ms   min {min(v):7.2f} ms")
        print(f"  {'Δlogits compiled':>24}: max {max(rows['Δlogits_compiled']):.3e}   (0 = bit-identical)")
        print(f"  {'Δlogits eager':>24}: max {max(rows['Δlogits_eager']):.3e}")
        print(f"  {'argmax fresh==splice':>24}: {all(rows['argmax_match'])}")

    # steady frame wall for scale
    t0 = time.perf_counter()
    frames = 0
    for chunk in generate_custom_voice_frames(model, text=LONG_TEXT, speaker="vivian", max_tokens=48):
        frames += chunk.shape[0] // 1920
    frame_ms = (time.perf_counter() - t0) * 1000.0 / max(1, frames)
    print(f"\nsteady frame wall (compiled loop): {frame_ms:.2f} ms/frame")

    # KV footprint of one prefix entry
    n_layers, kv_heads, head_dim = tc.num_hidden_layers, tc.num_key_value_heads, tc.head_dim
    for rows_n in (6, 7):
        b = n_layers * 2 * kv_heads * head_dim * rows_n * 4  # fp32 worst case
        print(f"prefix KV @ {rows_n} rows: ≤ {b / 2**20:.2f} MiB (fp32 bound; actual dtype may halve)")
    print(f"peak memory: {mx.get_peak_memory() / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
