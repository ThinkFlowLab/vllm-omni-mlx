"""Vendored VoxCPM2 generation loop (#79): the per-patch step, compiled.

mlx-audio's ``generate`` is a Python loop whose step is launch-bound glue:
the phase profile on PR #76 puts CFM sampling at 77% of wall (a 10-step
Euler solver over a CFG-doubled 12-layer DiT per ~20 ms patch), the
per-patch feature encoder at 8%, and both LM steps at 7%. This module
vendors the loop (prompt layout borrowed verbatim from the library's
``generate``) and compiles the step into three closures:

1. ``_solve_cfm`` — the whole solver (z draw → 10 Euler steps, CFG
   doubling batched along axis 0). Every shape is fixed by the patch
   geometry, so this is a fixed-shape trace with the RNG state plumbed
   in/out — the global stream advances exactly as the library's eager
   ``normal`` draw, keeping seeded runs reproducible.
2. ``_encode_patch`` — feat_encoder + enc_to_lm_proj on one patch (fixed
   shapes).
3. ``_lm_step`` / ``_res_step`` — one decode position through the base /
   residual LM with the KV cache as plain array inputs/outputs that grow
   by concat inside a shapeless trace (the library already stores its
   cache as arrays; the objects just never enter a trace). rotate_half's
   slice is banned under shapeless, so RoPE's half-swap rides a ±1
   permutation matmul instead (built as the transpose of the intuitive
   index matrix — see the #67 cookbook).

The library rope's ``.item()`` (a per-call sync inside
``MiniCPMLongRoPE.__call__``) is replaced by cos/sin rows computed
eagerly per position and passed into the step; our sequences never
exceed ``original_max_position_embeddings``, so the short factor is the
only table. Prefill (1% of wall) and the one-shot VAE decode stay on the
library calls; the stop check keeps its per-patch ``.item()`` — the loop
needs the value anyway.

``VLLM_OMNI_VOXCPM2_EAGER=1`` bypasses this module entirely (library
``generate``, nothing compiled) — the A/B baseline.
"""

from __future__ import annotations

import math
import os
import threading
import weakref
from typing import Any, Iterator

import mlx.core as mx
import mlx.nn as mnn

from .voxcpm2 import VoxCPM2Config

#: escape hatch shared with the loader: full library path, no compiling
def eager_escape() -> bool:
    return bool(os.environ.get("VLLM_OMNI_VOXCPM2_EAGER"))


# ---------------------------------------------------------------------------
# rope tables (the library's MiniCPMLongRoPE with the .item() sync removed)
# ---------------------------------------------------------------------------

def _rope_rows(rope: Any, length: int) -> tuple[mx.array, mx.array]:
    """cos/sin for positions [0, length) — mirrors MiniCPMLongRoPE with the
    short factor (all our sequences sit below original_max_position_embeddings)."""
    factors = rope.short_factor
    t = mx.arange(length, dtype=mx.float32)
    freqs = (t[:, None] * (1.0 / factors[None, :])) * rope.inv_freq[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    scale = rope.scaling_factor
    return mx.cos(emb) * scale, mx.sin(emb) * scale


def _rotate_half_matmul(dim: int) -> mx.array:
    """P with out = x @ P equal to rotate_half(x): out[j] = -x[j+dim/2] for
    j < dim/2, x[j-dim/2] otherwise — the block form [[0, I], [-I, 0]], the
    TRANSPOSE of the intuitive [[0, -I], [I, 0]] (P[i, j] selects source row
    i for output column j; the flipped sign negates the rotation)."""
    half = dim // 2
    eye = mx.eye(half)
    zeros = mx.zeros((half, half))
    return mx.concatenate(
        [mx.concatenate([zeros, eye], axis=1), mx.concatenate([-eye, zeros], axis=1)], axis=0
    )


def _apply_rope_matmul(x: mx.array, cos: mx.array, sin: mx.array, p: mx.array) -> mx.array:
    # x: (B, L, H, D); cos/sin: (1, D) rows, broadcasting as (1, 1, 1, D) —
    # value-identical to the library's (1, L, 1, D) layout at L=1
    return x * cos + (x @ p) * sin


# ---------------------------------------------------------------------------
# step closures (cached per model; nn.Module is unhashable, so id-keyed with
# a weakref death callback — the closures capture the layers, and a released
# checkpoint must take its traces with it)
# ---------------------------------------------------------------------------

_CLOSURES: dict[tuple, tuple[weakref.ref, weakref.ref]] = {}


class _StepClosures:
    def __init__(self, model: Any, config: VoxCPM2Config):
        self.model = model
        self.config = config
        est = model.feat_decoder.estimator

        # DiT sequence per solver step: mu tokens (2) + t (1) + cond (P) + x (P)
        self._dit_cos, self._dit_sin = _rope_rows(est.decoder.rope, 2 + 1 + model.patch_size * 2)
        # encoder sequence per patch: special token + P positions
        self._enc_cos, self._enc_sin = _rope_rows(model.feat_encoder.encoder.rope, model.patch_size + 1)
        self._rotate_p = _rotate_half_matmul(
            model.base_lm.layers[0].self_attn.head_dim
        )
        self._t_span = self._swayed_span(config.inference_timesteps)

        self.rotate_p = self._rotate_p
        self.solve_cfm = self._compile_solver()
        self.encode_patch = mx.compile(self._encode_patch)
        self.lm_step = mx.compile(self._lm_step, shapeless=True)
        self.res_step = mx.compile(self._res_step, shapeless=True)

    # -- t span: identical values to the library's linspace + sway, hoisted
    #    out of the trace into constants
    @staticmethod
    def _swayed_span(n: int) -> mx.array:
        t_span = mx.linspace(1, 0, n + 1)
        sway = 1.0
        return t_span + sway * (mx.cos(math.pi / 2 * t_span) - 1 + t_span)

    # -- 1: the CFM solver -------------------------------------------------
    def _solve_cfm(self, mu: mx.array, cond_t: mx.array) -> mx.array:
        """mu (1, 2H) + cond (1, D, P) → pred_feat (1, P, D). Mirrors
        UnifiedCFM.sample / solve_euler with use_cfg_zero_star=True; all
        shapes fixed by the patch geometry."""
        model = self.model
        cfm = model.feat_decoder
        cfg_value = self.config.cfg_value

        z = mx.random.normal((1, cfm.in_channels, model.patch_size))
        t_span = self._t_span
        dt = t_span[0] - t_span[1]
        t = t_span[0]

        zero_init_steps = max(1, int(len(t_span) * 0.04))
        current_x = z
        for step in range(1, len(t_span)):
            if step <= zero_init_steps:
                dphi_dt = mx.zeros_like(current_x)
            else:
                x_in = mx.concatenate([current_x, current_x], axis=0)
                mu_in = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
                t_val = mx.full((x_in.shape[0],), t)
                dt_val_in = mx.zeros((x_in.shape[0],))
                cond_in = mx.concatenate([cond_t, cond_t], axis=0)
                out = self._estimator(x_in, mu_in, t_val, cond_in, dt_val_in)
                dphi_dt = out[:1]
                cfg_dphi_dt = out[1:]
                positive_flat = dphi_dt.reshape(1, -1)
                negative_flat = cfg_dphi_dt.reshape(1, -1)
                dot_prod = mx.sum(positive_flat * negative_flat, axis=1, keepdims=True)
                sq_norm = mx.sum(negative_flat**2, axis=1, keepdims=True) + 1e-8
                st_star = (dot_prod / sq_norm).reshape(1, 1, 1)
                dphi_dt = cfg_dphi_dt * st_star + cfg_value * (dphi_dt - cfg_dphi_dt * st_star)
            current_x = current_x - dt * dphi_dt
            t = t - dt
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]
        return current_x  # (1, C, P), the library sample()'s output form

    def _estimator(self, x_in, mu_in, t_val, cond_in, dt_val_in):
        """VoxCPMLocDiTV2.__call__ with the rope table injected (its .item()
        cannot be traced) — everything else is the library's own math."""
        est = self.model.feat_decoder.estimator
        x = x_in.transpose(0, 2, 1)
        x_proj = est.in_proj(x)
        cond = cond_in.transpose(0, 2, 1)
        cond_proj = est.cond_proj(cond)
        prefix = cond.shape[1]

        t_emb = est.time_mlp(est.time_embeddings(t_val))
        dt_emb = est.delta_time_mlp(est.time_embeddings(dt_val_in))
        t_comb = t_emb + dt_emb

        H = x_proj.shape[-1]
        mu_tokens = mu_in.reshape(x.shape[0], -1, H)
        num_mu_tokens = mu_tokens.shape[1]

        hidden = mx.concatenate([mu_tokens, t_comb[:, None, :], cond_proj, x_proj], axis=1)
        hidden = self._decoder_forward(est.decoder, hidden, self._dit_cos, self._dit_sin)
        hidden = hidden[:, num_mu_tokens + 1 + prefix :, :]
        hidden = est.out_proj(hidden)
        return hidden.transpose(0, 2, 1)

    @staticmethod
    def _decoder_forward(decoder: Any, h: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """MiniCPMModel.__call__ without cache/mask, cos/sin provided."""
        cos = cos[None, :, :]
        sin = sin[None, :, :]
        for layer in decoder.layers:
            h, _ = layer(h, cos, sin, mask=None, cache=None)
        return decoder.norm(h)

    def _compile_solver(self):
        return mx.compile(self._solve_cfm, inputs=mx.random.state, outputs=mx.random.state)

    # -- 2: the per-patch encoder -------------------------------------------
    def _encode_patch(self, pred_feat: mx.array) -> mx.array:
        """(1, P, D) feat → (1, 1, H) embed: feat_encoder + enc_to_lm_proj,
        with the encoder's rope table injected."""
        model = self.model
        enc = model.feat_encoder
        B, T, P = 1, 1, pred_feat.shape[1]
        x = enc.in_proj(pred_feat[:, None, :, :])
        special = mx.broadcast_to(enc.special_token, (B, T, 1, enc.config.hidden_size))
        x = mx.concatenate([special, x], axis=2).reshape(B * T, P + 1, -1)
        outputs, _ = self._encoder_forward(x)
        cls_output = outputs[:, 0, :]
        return model.enc_to_lm_proj(cls_output.reshape(B, T, -1))

    def _encoder_forward(self, x: mx.array) -> tuple[mx.array, None]:
        enc = self.model.feat_encoder.encoder
        cos = self._enc_cos[None, :, :]
        sin = self._enc_sin[None, :, :]
        h = x
        for layer in enc.layers:
            h, _ = layer(h, cos, sin, mask=None, cache=None)
        return enc.norm(h), None

    # -- 3: the decode-position LM steps -------------------------------------
    def _lm_step(self, embed: mx.array, cos_row: mx.array, sin_row: mx.array, rotate_p: mx.array, *kv: mx.array):
        """One position through the base LM → (fsq'd hidden row, residual
        input row, grown KV). ``kv`` is per-layer (k, v) pairs; shapes grow
        by one each call, carried symbolically (shapeless)."""
        model = self.model
        lm = model.base_lm
        cfg = lm.config
        scale = cfg.scale_depth / math.sqrt(cfg.num_hidden_layers) if cfg.use_mup else 1.0

        h = embed
        outs = []
        for i, layer in enumerate(lm.layers):
            k_cache, v_cache = kv[2 * i], kv[2 * i + 1]
            r = h
            x = layer.input_layernorm(h)
            attn = layer.self_attn
            B, L, _ = x.shape
            q = attn.q_proj(x).reshape(B, L, attn.num_heads, attn.head_dim)
            k = attn.k_proj(x).reshape(B, L, attn.num_kv_heads, attn.head_dim)
            v = attn.v_proj(x).reshape(B, L, attn.num_kv_heads, attn.head_dim)
            q = _apply_rope_matmul(q, cos_row, sin_row, rotate_p)
            k = _apply_rope_matmul(k, cos_row, sin_row, rotate_p)
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)
            outs.extend([k, v])
            q = q.transpose(0, 2, 1, 3)
            kt = k.transpose(0, 2, 1, 3)
            vt = v.transpose(0, 2, 1, 3)
            o = mx.fast.scaled_dot_product_attention(q, kt, vt, scale=1 / math.sqrt(attn.head_dim), mask=None)
            o = attn.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))
            h = r + o * scale
            r = h
            x = layer.post_attention_layernorm(h)
            h = r + layer.mlp(x) * scale
        row = lm.norm(h)[:, -1, :]
        lm_hidden = model.fsq_layer(row)
        res_input = model.fusion_concat_proj(mx.concatenate([lm_hidden[:, None, :], embed], axis=-1))
        return lm_hidden, res_input, *outs

    def _res_step(self, embed: mx.array, *kv: mx.array):
        """One position through the rope-less residual LM → (hidden row,
        grown KV)."""
        res = self.model.residual_lm
        cfg = res.config
        scale = cfg.scale_depth / math.sqrt(cfg.num_hidden_layers) if cfg.use_mup else 1.0
        h = embed
        outs = []
        for i, layer in enumerate(res.layers):
            k_cache, v_cache = kv[2 * i], kv[2 * i + 1]
            r = h
            x = layer.input_layernorm(h)
            attn = layer.self_attn
            B, L, _ = x.shape
            q = attn.q_proj(x).reshape(B, L, attn.num_heads, attn.head_dim)
            k = attn.k_proj(x).reshape(B, L, attn.num_kv_heads, attn.head_dim)
            v = attn.v_proj(x).reshape(B, L, attn.num_kv_heads, attn.head_dim)
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)
            outs.extend([k, v])
            q = q.transpose(0, 2, 1, 3)
            kt = k.transpose(0, 2, 1, 3)
            vt = v.transpose(0, 2, 1, 3)
            o = mx.fast.scaled_dot_product_attention(q, kt, vt, scale=1 / math.sqrt(attn.head_dim), mask=None)
            o = attn.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))
            h = r + o * scale
            r = h
            x = layer.post_attention_layernorm(h)
            h = r + layer.mlp(x) * scale
        row = res.norm(h)[:, -1, :]
        return row, *outs


def closures_for(model: Any, config: VoxCPM2Config) -> _StepClosures:
    """Per-(model, thread) closure cache.

    MLX compiled functions are thread-bound — a closure traced on one
    thread refuses to evaluate on another ("There is no Stream(gpu, …) in
    current thread") — so traces are keyed by thread as well as model.
    The solver's t_span and cfg strength are BAKED into the trace at
    build, so they join the key: changing ``inference_timesteps`` or
    ``cfg_value`` at runtime rebuilds instead of silently serving stale
    traces. The entry validates model identity through a weakref: a
    released and garbage-collected checkpoint's id can be REUSED by the
    next load (the weight-gated batteries hit exactly that), and a stale
    entry would hand the new model traces bound to a dead thread.
    """
    key = (id(model), threading.get_ident(), config.inference_timesteps, config.cfg_value)
    entry = _CLOSURES.get(key)
    if entry is not None:
        model_ref, fresh_ref = entry
        if model_ref() is model and fresh_ref() is not None:
            return fresh_ref()

    fresh = _StepClosures(model, config)
    model_ref = weakref.ref(model)
    fresh_ref = weakref.ref(fresh)
    _CLOSURES[key] = (model_ref, fresh_ref)

    def _drop(key=key, mine=fresh):
        held = _CLOSURES.get(key)
        if held is not None and held[1]() is mine:
            del _CLOSURES[key]

    weakref.finalize(model, _drop)
    return fresh


# ---------------------------------------------------------------------------
# the vendored loop
# ---------------------------------------------------------------------------

def generate_frames(
    model: Any,
    config: VoxCPM2Config,
    text: str,
    instruct: str | None = None,
    ref_audio: mx.array | None = None,
    ref_text: str | None = None,
    compiled: bool = True,
) -> Iterator[mx.array]:
    """The library's zero-shot / instruct / reference-clone generate, with the
    per-patch step running through compiled closures. Prompt layouts are the
    library's own (``_tokenize`` / ``_encode_wav`` / ``_make_ref_prefix``),
    prefill and the VAE decode are eager library calls, and the RNG stream is
    consumed exactly as the library consumes it (one ``normal`` draw per
    patch, plumbed through the solver trace). Yields one 48 kHz audio chunk,
    like mlx-audio's single-yield generate."""
    if model.tokenizer is None:
        raise ValueError("Tokenizer not loaded")
    if instruct:
        text = f"({instruct}){text}"
        warmup_patches = min(config.warmup_patches, 1)
    else:
        warmup_patches = config.warmup_patches

    scale_emb = model.args.lm_config.scale_emb if model.args.lm_config.use_mup else 1.0
    latent_dim = model.audio_vae.latent_dim

    # prompt build — verbatim from Model.generate (modes 1 and 3)
    text_ids = model._tokenize(text)
    text_token = mx.array(text_ids + [model.audio_start_token], dtype=mx.int32)
    text_length = text_token.shape[0]

    if ref_audio is not None:
        ref_feat = model._encode_wav(ref_audio, padding_mode="right")
        ref_tokens, ref_feats, ref_t_mask, ref_a_mask = model._make_ref_prefix(ref_feat)
        text_pad_feat = mx.zeros((text_length, model.patch_size, latent_dim))
        text_token = mx.concatenate([ref_tokens, text_token])
        audio_feat = mx.concatenate([ref_feats, text_pad_feat], axis=0)
        text_mask = mx.concatenate([ref_t_mask, mx.ones(text_length, dtype=mx.float32)])
        audio_mask = mx.concatenate([ref_a_mask, mx.zeros(text_length, dtype=mx.float32)])
    else:
        audio_feat = mx.zeros((text_length, model.patch_size, latent_dim))
        text_mask = mx.ones(text_length, dtype=mx.float32)
        audio_mask = mx.zeros(text_length, dtype=mx.float32)

    text_token = text_token[None, :]
    audio_feat = audio_feat[None, :, :, :]
    text_mask = text_mask[None, :]
    audio_mask = audio_mask[None, :]

    # prefill (eager, library modules — 1% of wall)
    feat_embed = model.enc_to_lm_proj(model.feat_encoder(audio_feat))
    text_embed = model.base_lm.embed_tokens(text_token) * scale_emb
    combined = text_mask[:, :, None] * text_embed + audio_mask[:, :, None] * feat_embed
    prefix_feat_cond = audio_feat[:, -1, :, :]

    enc_outputs, lm_cache = model.base_lm(combined)
    enc_outputs = model.fsq_layer(enc_outputs) * audio_mask[:, :, None] + enc_outputs * text_mask[:, :, None]
    lm_hidden = enc_outputs[:, -1, :]
    residual_input = model.fusion_concat_proj(
        mx.concatenate([enc_outputs, audio_mask[:, :, None] * feat_embed], axis=-1)
    )
    residual_outputs, res_cache = model.residual_lm(residual_input)
    residual_hidden = residual_outputs[:, -1, :]

    steps = closures_for(model, config) if compiled else None
    cos_rows, sin_rows = _rope_rows(model.base_lm.rope, text_token.shape[1] + config.max_tokens + warmup_patches + 1)

    pred_feat_seq: list[mx.array] = []
    real_steps = -1
    for i in range(config.max_tokens + warmup_patches):
        dit_h1 = model.lm_to_dit_proj(lm_hidden)
        dit_h2 = model.res_to_dit_proj(residual_hidden)
        dit_h = mx.concatenate([dit_h1, dit_h2], axis=-1)

        pos = text_token.shape[1] + i  # decode position for this patch
        if steps is not None:
            pred_feat = steps.solve_cfm(dit_h, prefix_feat_cond.transpose(0, 2, 1))
            pred_feat = pred_feat.transpose(0, 2, 1)
            if i >= warmup_patches:
                pred_feat_seq.append(pred_feat[:, None, :, :])
            curr_embed = steps.encode_patch(pred_feat)
        else:
            pred_feat = model.feat_decoder.sample(
                mu=dit_h,
                n_timesteps=config.inference_timesteps,
                patch_size=model.patch_size,
                cond=prefix_feat_cond.transpose(0, 2, 1),
                cfg_value=config.cfg_value,
            ).transpose(0, 2, 1)
            if i >= warmup_patches:
                pred_feat_seq.append(pred_feat[:, None, :, :])
            curr_embed = model.enc_to_lm_proj(model.feat_encoder(pred_feat[:, None, :, :]))

        # stop prediction — after warmup + min_tokens of real output (the
        # library's exact gate; the .item() is the loop's control flow)
        stop_logits = model.stop_head(mnn.silu(model.stop_proj(lm_hidden)))
        stop_flag = mx.argmax(stop_logits, axis=-1).item()
        real_steps = i - warmup_patches
        if real_steps > 2 and stop_flag == 1:
            break

        if steps is not None:
            lm_hidden, residual_input, *grown = steps.lm_step(
                curr_embed,
                cos_rows[pos : pos + 1],
                sin_rows[pos : pos + 1],
                steps.rotate_p,
                *_flatten(lm_cache),
            )
            lm_cache = _pair_up(grown)
            residual_hidden, *res_grown = steps.res_step(residual_input, *_flatten(res_cache))
            res_cache = _pair_up(res_grown)
        else:
            new_lm_out, lm_cache = model.base_lm(inputs_embeds=curr_embed, cache=lm_cache)
            lm_hidden = model.fsq_layer(new_lm_out[:, -1, :])
            residual_input = model.fusion_concat_proj(
                mx.concatenate([lm_hidden[:, None, :], curr_embed], axis=-1)
            )
            new_res_out, res_cache = model.residual_lm(inputs_embeds=residual_input, cache=res_cache)
            residual_hidden = new_res_out[:, -1, :]

        prefix_feat_cond = pred_feat

    all_feats = mx.concatenate(pred_feat_seq, axis=1)
    all_feats_flat = all_feats.reshape(1, -1, model.feat_dim)
    audio = model.audio_vae.decode(all_feats_flat).flatten()
    if audio.size:
        yield audio


def _flatten(cache: list[tuple[mx.array, mx.array]]) -> list[mx.array]:
    out = []
    for k, v in cache:
        out.append(k)
        out.append(v)
    return out


def _pair_up(flat: list[mx.array]) -> list[tuple[mx.array, mx.array]]:
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
