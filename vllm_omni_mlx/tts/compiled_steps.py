"""Compiled per-frame decode closures for the vendored stream loops (#65).

The M2.1 profile (issue #65 task 1) put ~61% of frame time in the code
predictor (48.7% forwards + 12% sampling across 15 launch-bound micro-steps)
and ~20% in the talker forward, with another 15.5% pure Python dispatch —
the classic shape ``mx.compile`` removes: fuse the elementwise soup and
collapse per-op dispatch into one call. Two closures:

- ``make_predictor_frame`` — the whole per-frame predictor block (15
  micro-steps + their sampling) as ONE compiled call. The predictor's KV
  cache resets every frame, so all state is frame-local: the growing keys/
  values live entirely inside the traced dataflow and the closure is
  fixed-shape forever (one trace, no retrace risk). This is upstream's
  "fused predictor" (their ``_predict_step_logits``) arrived at from the
  MLX side.
- ``make_talker_decode`` — one decode frame of the talker backbone,
  compiled ``shapeless=True`` with the KV cache passed as plain arrays
  (growing one row per frame without retrace; the growing-kv +
  ``fast.scaled_dot_product_attention`` pattern verified bit-exact against
  eager). Sampling stays outside: the talker draw pays the repetition
  penalty and suppress list on the Python side (measured 0.7 ms/frame —
  not worth baking).

Vendoring note: the forwards below mirror mlx-audio's (MIT)
``TalkerAttention``/``TalkerDecoderLayer``/``Qwen3TTSTalkerModel`` and
``CodePredictorAttention``/``CodePredictorDecoderLayer``/``CodePredictorModel``
``__call__`` bodies step-for-step over the *same module parameters*
(``nn.Linear``/``nn.RMSNorm`` calls are exact), with two deliberate
deviations: the RoPE apply and swiglu are inlined rather than calling
mlx-audio's module-level ``@mx.compile``-decorated helpers (compiled
functions must not nest), and the predictor's per-group ``lm_head`` runs
on the last position only (a Linear over a sequence is per-position
independent, so the sampled logits are bit-identical to the eager
full-sequence head the loop slices anyway).

Numerics: on identical inputs both closures are bit-exact against the
eager paths (verified per-frame: hidden/logits max|Δ| = 0, greedy and
reseeded-sampled tokens 16/16). End-to-end greedy streams can still
diverge rarely — compile fusion's FMA contraction reorders rounding by
~1 ULP, enough to flip an fp16 near-tie argmax (measured 1 frame in 45,
cascading only within that frame's predictor groups) — the loop-level
parity test carries that tolerance and the HNR gate stays the
correctness bar (#65's sanctioned trade).

Closures are cached per model (weakly) so the one-time trace cost amortizes
across requests; ``prewarm_streaming`` traces both up front. Set
``VLLM_OMNI_TTS_EAGER_STREAM=1`` to fall back to the uncompiled loop (A/B
and debugging).
"""

from __future__ import annotations

import os
import weakref
from collections.abc import Callable
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn

EAGER_STREAM = os.environ.get("VLLM_OMNI_TTS_EAGER_STREAM", "") not in ("", "0", "false")

# per-model closure caches: mx.compile traces are global per function
# object, so building a new closure per request would retrace every time.
# nn.Modules are unhashable, so key on id() and keep a weakref to detect a
# freed model (its id may be reused) — the cache never pins a model alive
_CLOSURE_CACHE: dict[int, tuple[weakref.ref[Any], dict[tuple, Callable]]] = {}


def _closures_for(module: Any) -> dict[tuple, Callable]:
    entry = _CLOSURE_CACHE.get(id(module))
    if entry is not None and entry[0]() is module:
        return entry[1]
    sub: dict[tuple, Callable] = {}
    _CLOSURE_CACHE[id(module)] = (weakref.ref(module), sub)
    return sub


def _rotate_half(x: mx.array, half: int) -> mx.array:
    # fixed-shape (predictor) path: plain slices
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def _rotate_half_matrix(dim: int, dtype: mx.Dtype) -> mx.array:
    """rotate_half as a [dim, dim] ±1/0 matrix: row i picks -x[i+dim/2] for
    i < dim/2, else x[i-dim/2]. x @ P == rotate_half(x) bit-exactly for
    finite x (each output is one product with ±1 plus exact-zero terms).
    Needed because Slice/Split cannot infer output shapes under
    shapeless=True, but matmul can."""
    half = dim // 2
    # matmul convention: (x @ P)[j] = Σ_i x[i]·P[i, j] — P's columns select,
    # so column j < half pulls -x[j+half] and column j ≥ half pulls x[j-half]
    row = mx.arange(dim)[:, None]
    col = mx.arange(dim)[None, :]
    p = mx.where(
        (col < half) & (row == col + half), -1.0,
        mx.where((col >= half) & (row == col - half), 1.0, 0.0),
    ).astype(dtype)
    mx.eval(p)
    return p


def _apply_rope(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, half: int, unsqueeze_dim: int = 1
) -> tuple[mx.array, mx.array]:
    # vendored from mlx-audio's apply_rotary_pos_emb / apply_multimodal_
    # rotary_pos_emb (MIT) — MRoPE interleaving is already combined into
    # cos/sin by the position-embedding helpers; both variants share this apply
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q, half) * sin)
    k_embed = (k * cos) + (_rotate_half(k, half) * sin)
    return q_embed, k_embed


def _predictor_position_embeddings(
    rotary: Any, x: mx.array, position_ids: mx.array
) -> tuple[mx.array, mx.array]:
    """cos/sin — vendored from the predictor's ``RotaryEmbedding.__call__``
    (MIT); position_ids is a trace constant per step (fixed frame schedule)."""
    inv_freq = mx.expand_dims(rotary._inv_freq, axis=(0, 2))
    pos = mx.expand_dims(position_ids.astype(mx.float32), axis=1)
    freqs = mx.transpose(inv_freq * pos, (0, 2, 1))
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb).astype(x.dtype), mx.sin(emb).astype(x.dtype)


def _causal_mask_2(dtype: mx.Dtype) -> mx.array:
    # additive causal mask for the predictor's 2-token first step — the only
    # masked call in either closure; == create_additive_causal_mask(2)
    return mx.array([[0.0, float("-inf")], [0.0, 0.0]]).astype(dtype)


def _sample_predictor_token(
    logits_last: mx.array, temperature: float, top_k: int, top_p: float
) -> mx.array:
    """Inlined ``_sample_token`` predictor call path (mlx-audio, MIT): no
    repetition penalty, no suppress list — temperature/top-k/top-p and the
    draw. Branches are per-closure constants, so each sampler configuration
    gets its own (single) trace. The draw is ``mx.random.categorical``,
    which compiles at fixed shapes but not under ``shapeless=True`` — fine
    here, the predictor closure is fixed-shape by construction."""
    if temperature <= 0:
        return mx.argmax(logits_last, axis=-1, keepdims=True)
    if temperature != 1.0:
        logits_last = logits_last / temperature
    vocab = logits_last.shape[-1]
    if 0 < top_k < vocab:
        mask_idx = mx.argpartition(-logits_last, kth=top_k - 1, axis=-1)[..., top_k:]
        logits_last = mx.put_along_axis(
            logits_last, mask_idx, mx.array(float("-inf"), logits_last.dtype), axis=-1
        )
    if 0.0 < top_p < 1.0:
        # vendored _apply_probability_filters -> apply_top_p (mlx_lm, MIT)
        logprobs = nn.log_softmax(logits_last, axis=-1)
        sorted_logprobs = mx.sort(logprobs, axis=-1)
        sorted_probs = mx.exp(sorted_logprobs.astype(mx.float32))
        mass_above = mx.cumsum(sorted_probs, axis=-1, reverse=True, inclusive=False)
        total_mass = mass_above[..., :1] + sorted_probs[..., :1]
        num_dropped = (mass_above >= top_p * total_mass).sum(axis=-1, keepdims=True)
        threshold = mx.take_along_axis(sorted_logprobs, num_dropped, axis=-1)
        neg_inf = mx.array(float("-inf"), logits_last.dtype)
        logits_last = mx.where(
            logprobs == -mx.inf,
            neg_inf,
            mx.where(logprobs < threshold, neg_inf, logits_last),
        )
    token = mx.random.categorical(logits_last)
    return token[:, None]


def make_predictor_frame(
    predictor: Any,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    base_embedding: Any,
) -> Callable[[mx.array, mx.array], mx.array]:
    """Compiled per-frame predictor block: (code_hidden [1,1,talker_hidden],
    first_code_token [1,1]) -> all_codes [1, num_code_groups].

    Mirrors the stream loops' predictor section driving mlx-audio's
    ``Qwen3TTSTalkerCodePredictor`` — same per-group embeddings, same
    5-layer forwards with per-frame reset caches (here: concat-grown arrays
    internal to the trace), same per-group heads, same sampler sequence —
    with the sequential 15-step dataflow fused into one dispatch."""
    cache = _closures_for(predictor)
    key = ("predictor", temperature, top_k, top_p)
    if key in cache:
        return cache[key]

    model = predictor.model
    proj = predictor.small_to_mtp_projection
    layers = model.layers
    final_norm = model.norm
    rotary = model.rotary_emb
    heads = predictor.lm_head
    residual_embeddings = predictor.codec_embedding
    num_groups = predictor.num_code_groups

    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def predictor_frame(code_hidden: mx.array, first_token: mx.array) -> mx.array:
        # step 0 consumes the talker hidden + group-0 token embedding over
        # positions [0, 1]; each later step embeds the previous draw at
        # position step+1 — the frame's position schedule is fixed, so all
        # aranges/masks below are trace constants
        embed_0 = base_embedding(first_token)
        x_in = mx.concatenate([code_hidden, embed_0], axis=1)

        tokens = [first_token]
        keys: list[mx.array | None] = [None] * len(layers)
        values: list[mx.array | None] = [None] * len(layers)

        for step in range(num_groups - 1):
            if step == 0:
                x = proj(x_in) if proj is not None else x_in
                position_ids = mx.arange(0, 2)[None, :]
            else:
                e = residual_embeddings[step - 1](tokens[-1])
                x = proj(e) if proj is not None else e
                position_ids = mx.arange(step + 1, step + 2)[None, :]

            cos, sin = _predictor_position_embeddings(rotary, x, position_ids)
            mask = _causal_mask_2(x.dtype) if step == 0 else None

            h = x
            for i, layer in enumerate(layers):
                a = layer.self_attn
                residual = h
                normed = layer.input_layernorm(h)
                q = a.q_proj(normed).reshape(1, h.shape[1], a.num_heads, a.head_dim)
                k = a.k_proj(normed).reshape(1, h.shape[1], a.num_kv_heads, a.head_dim)
                v = a.v_proj(normed).reshape(1, h.shape[1], a.num_kv_heads, a.head_dim)
                q = a.q_norm(q)
                k = a.k_norm(k)
                q = mx.transpose(q, (0, 2, 1, 3))
                k = mx.transpose(k, (0, 2, 1, 3))
                v = mx.transpose(v, (0, 2, 1, 3))
                q, k = _apply_rope(q, k, cos, sin, a.head_dim // 2)
                k = k if keys[i] is None else mx.concatenate([keys[i], k], axis=2)
                v = v if values[i] is None else mx.concatenate([values[i], v], axis=2)
                keys[i], values[i] = k, v
                out = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale, mask=mask)
                out = mx.transpose(out, (0, 2, 1, 3)).reshape(1, h.shape[1], -1)

                # CodePredictorDecoderLayer.__call__ residual pattern (MIT)
                h = residual + a.o_proj(out)
                residual = h
                normed = layer.post_attention_layernorm(h)
                gate = layer.mlp.gate_proj(normed)
                up = layer.mlp.up_proj(normed)
                h = residual + layer.mlp.down_proj(nn.silu(gate) * up)

            logits = heads[step](final_norm(h)[:, -1:, :])
            tokens.append(_sample_predictor_token(logits[:, -1, :], temperature, top_k, top_p))

        return mx.concatenate(tokens, axis=1)

    cache[key] = predictor_frame
    return predictor_frame


def make_talker_sampler(
    model: Any,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    suppress_tokens: list[int] | None,
) -> Callable[[mx.array, mx.array], mx.array]:
    """Compiled talker-draw sampler: (logits [1,1,vocab], history [64] uint32)
    -> next_token [1,1] uint32.

    Inlines ``_sample_token``'s full talker path (mlx-audio, MIT) with the
    repetition-penalty context as an ARRAY input instead of a Python list —
    the list forced the loop to ``int()`` the token every frame (a hard
    CPU/GPU sync), which is what kept the CPU from running ahead of the
    GPU (#65's residue). Semantics match eager exactly: the history is a
    64-slot ring of the last tokens; eager penalizes the *unique* set of
    ``generated[-64:]``, and duplicate-index penalty writes are idempotent
    (same source logit, same multiplier), so writing per-occurrence is
    value-identical. Unfilled ring slots carry ``vocab_size`` and route to
    a dummy slot appended past the vocab — penalizing a zero by the
    multiplier writes the same zero back, a no-op outside the real vocab.
    """
    cache = _closures_for(model)
    key = (
        "sampler",
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        tuple(suppress_tokens or ()),
    )
    if key in cache:
        return cache[key]

    vocab = model.config.talker_config.vocab_size
    suppress = mx.array(suppress_tokens, dtype=mx.uint32) if suppress_tokens else None

    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def sampler(logits: mx.array, history: mx.array) -> mx.array:
        row = logits[:, -1, :]
        if suppress_tokens:
            row = mx.put_along_axis(
                row, suppress[None, :], mx.array(float("-inf"), row.dtype), axis=-1
            )
        if repetition_penalty != 1.0:
            ext = mx.concatenate([row, mx.zeros((1, 1), row.dtype)], axis=-1)
            selected = mx.take(ext, history, axis=-1)
            penalized = mx.where(
                selected < 0, selected * repetition_penalty, selected / repetition_penalty
            )
            ext = mx.put_along_axis(ext, history[None, :], penalized, axis=-1)
            row = ext[..., :vocab]
        return _sample_predictor_token(row, temperature, top_k, top_p)

    cache[key] = sampler
    return sampler


def make_input_embeds(talker: Any) -> Callable[[mx.array, mx.array], mx.array]:
    """Compiled next-input prep: (all_codes [1, num_code_groups],
    text_embed [1,1,hidden]) -> input_embeds [1,1,hidden].

    ``Talker.codec_embeds``'s channel-summed lookup (mlx-audio, MIT) — base
    embedding for group 0 plus each residual group's predictor embedding —
    with the 16 gathers and 15 adds fused into one kernel instead of 30+
    eager launches per frame."""
    cache = _closures_for(talker)
    if "input_embeds" in cache:
        return cache["input_embeds"]

    base_embedding = talker.get_input_embeddings()
    predictor_embeddings = talker.code_predictor.codec_embedding

    @mx.compile
    def input_embeds_step(all_codes: mx.array, text_embed: mx.array) -> mx.array:
        embed = base_embedding(all_codes[:, 0:1])
        for group in range(1, all_codes.shape[1]):
            embed = embed + predictor_embeddings[group - 1](all_codes[:, group : group + 1])
        return text_embed + embed

    cache["input_embeds"] = input_embeds_step
    return input_embeds_step


def make_talker_decode(talker: Any) -> Callable[..., tuple[mx.array, mx.array, list, list]]:
    """Compiled single-frame talker decode with the KV cache as arrays.

    Inputs: (input_embeds [1,1,hidden], position_ids [3,1,1], keys, values)
    where keys/values are per-layer [1, kv_heads, prefix_len, head_dim]
    arrays (the prefill's cache, transplanted). Returns (logits, hidden,
    keys', values') with one row appended per layer. Compiled
    ``shapeless=True`` so the growing cache length never retraces; decode
    shapes are otherwise constant (single query, no mask — the eager path
    builds a mask only for seq_len > 1)."""
    cache = _closures_for(talker)
    if "decode" in cache:
        return cache["decode"]

    model = talker.model
    layers = model.layers
    final_norm = model.norm
    rotary = model.rotary_emb
    codec_head = talker.codec_head

    # shapeless compile can't infer Slice/Split output shapes, so rotate_half
    # runs as a ±1 permutation matmul on the last axis (closure constant;
    # fp32-built because quantized linears' packed weights aren't the
    # compute dtype — cast to the activation dtype inside the trace)
    head_dim = layers[0].self_attn.head_dim
    rot = _rotate_half_matrix(head_dim, mx.float32)

    @partial(mx.compile, shapeless=True)
    def decode_step(input_embeds, pos_scalar, keys, values):
        # cos/sin from the scalar decode position. At decode the eager path
        # stacks [pos, pos, pos] into MRoPE's three planes — with the planes
        # equal, the interleaved combination is value-identical to plain
        # RoPE over pos, so compute it directly at concrete shapes (the
        # [3,1,1]-id route needs shapes shapeless compile can't infer).
        # pos_scalar: int32 [1]; freqs = inv_freq * pos (TalkerRotaryEmbedding
        # math, MIT — matmul there, broadcast multiply here, same products).
        p = pos_scalar.astype(mx.float32)
        freqs = rotary._inv_freq * p[:, None]
        emb = mx.concatenate([freqs, freqs], axis=-1).reshape(1, 1, -1)
        cos = mx.cos(emb).astype(input_embeds.dtype)
        sin = mx.sin(emb).astype(input_embeds.dtype)

        x = input_embeds
        new_keys: list[mx.array] = []
        new_values: list[mx.array] = []
        for i, layer in enumerate(layers):
            a = layer.self_attn
            residual = x
            normed = layer.input_layernorm(x)
            q = a.q_proj(normed).reshape(1, 1, a.num_heads, a.head_dim)
            k = a.k_proj(normed).reshape(1, 1, a.num_kv_heads, a.head_dim)
            v = a.v_proj(normed).reshape(1, 1, a.num_kv_heads, a.head_dim)
            q = a.q_norm(q)
            k = a.k_norm(k)
            q = mx.transpose(q, (0, 2, 1, 3))
            k = mx.transpose(k, (0, 2, 1, 3))
            v = mx.transpose(v, (0, 2, 1, 3))
            # apply_multimodal_rotary_pos_emb (MIT) with the matmul rotation
            rot_d = rot.astype(q.dtype)
            q_embed = (q * mx.expand_dims(cos, 1)) + ((q @ rot_d) * mx.expand_dims(sin, 1))
            k_embed = (k * mx.expand_dims(cos, 1)) + ((k @ rot_d) * mx.expand_dims(sin, 1))
            k = mx.concatenate([keys[i], k_embed], axis=2)
            v = mx.concatenate([values[i], v], axis=2)
            new_keys.append(k)
            new_values.append(v)
            out = mx.fast.scaled_dot_product_attention(q_embed, k, v, scale=a.scale)
            out = mx.transpose(out, (0, 2, 1, 3)).reshape(1, 1, -1)

            # TalkerDecoderLayer.__call__ residual pattern (MIT)
            x = residual + a.o_proj(out)
            residual = x
            normed = layer.post_attention_layernorm(x)
            gate = layer.mlp.gate_proj(normed)
            up = layer.mlp.up_proj(normed)
            x = residual + layer.mlp.down_proj(nn.silu(gate) * up)

        x = final_norm(x)
        logits = codec_head(x)
        # decode input is always [1, 1, hidden]: the full hidden == the
        # loop's historical [:, -1:, :] slice
        return logits, x, new_keys, new_values

    cache["decode"] = decode_step
    return decode_step
