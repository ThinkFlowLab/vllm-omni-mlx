"""Per-voice prefix caches for TTS synthesis (#66): batch-1 TTFA under the
single-stream lock.

Two levels, both keyed by the *voice identity* ``(speaker, language,
instruct)`` — everything the static prompt rows depend on; per-request text
never enters the key:

- **Level 1 — prompt pieces.** ``_prepare_generation_inputs``'s
  voice-dependent work (tts special-token projections, speaker + codec
  prefix lookups, the instruct projection) computed once per key and reused;
  the per-request tokenizer + text embedding pass stays fresh. The assembled
  prompt is bit-identical to mlx-audio's (same ops, same order — the cached
  arrays are the very arrays a fresh call would recompute), so Level 1 alone
  cannot perturb a stream. (Measured on the 4-bit 1.7B: the cached pieces
  are ~2–3 ms of a 4–17 ms build.)
- **Level 2 — prefix KV reuse.** The static prompt rows — instruct? +
  role(3) + codec prefix(n−1), 8–17 rows — prefill to the *same* KV every
  request for a voice. ``stream_loop`` stores the prefill's K/V rows for the
  static prefix on the first request (a slice of the cache it already
  built — zero extra forward) and splices them into later requests, so each
  request forwards only its first-text row at the prefix offset instead of
  re-running the whole eager prefill. Per row, K/V depend only on that
  row's inputs (causal attention, per-row residuals), so the splice carries
  the same information; the *numerics* of the first-text row move from the
  multi-row prefill kernels to the single-row decode kernels (mlx-audio's
  own M=9 vs M=1 batching), which is why parity is asserted at the draw
  level, not bitwise. (Measured: 86–148 ms prefill → ~22 ms one-row decode
  per request — the mask-path eager prefill is launch-bound.)

Design constraints honored from the issue: no paging, no block tables (the
conflict that makes upstream disable prefix caching cannot arise); a small
LRU cap bounds memory (an entry is ~1.3–1.6 MiB — layers × 2 × kv_heads ×
head_dim × prefix rows); ``VLLM_OMNI_TTS_PREFIX_CACHE=0`` disables both
levels for A/B and debugging.

Upstream precedent: vllm-omni #8259 reuses "static CustomVoice prompt
pieces" — the same idea on the CUDA/ROCm stack.
"""

from __future__ import annotations

import os
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx

# per-model stores, weakly keyed like compiled_steps' closure cache: models
# are released by tests and reloads, and the entries pin their K/V arrays
_STORES: dict[int, tuple[weakref.ref[Any], "_ModelPrefixState"]] = {}
# Level-1 pieces share the same per-model lifetime
_PIECES: dict[int, tuple[weakref.ref[Any], "OrderedDict[tuple, VoicePromptPieces]"]] = {}


def prefix_cache_enabled() -> bool:
    """Kill switch — read per call so benches and tests can flip it."""
    return os.environ.get("VLLM_OMNI_TTS_PREFIX_CACHE", "") not in ("0", "false")


def voice_key(speaker: Optional[str], language: str, instruct: Optional[str]) -> tuple:
    """Cache key for a voice: the normalized (speaker, language, instruct).

    ``language`` and ``instruct`` normalize through the same lowercase/empty
    rules mlx-audio's prompt build applies, so two spellings of one voice
    share one entry."""
    return (
        (speaker or "").lower(),
        (language or "auto").lower(),
        (instruct or "").strip() or "",
    )


@dataclass(frozen=True)
class VoicePromptPieces:
    """The voice-static half of ``_prepare_generation_inputs`` (Level 1).

    ``codec_embed`` is the fully assembled codec-track prefix — think rows
    (+ language id unless Auto, + dialect override) + speaker row (when the
    speaker resolves) + [pad, bos] suffix — exactly mlx-audio's array. The
    three ``tts_*`` embeds are the projected special tokens; the padding
    row reused by every decode step is ``tts_pad_embed``."""

    codec_embed: mx.array
    tts_bos_embed: mx.array
    tts_eos_embed: mx.array
    tts_pad_embed: mx.array
    instruct_embed: Optional[mx.array]
    has_speaker: bool

    @property
    def codec_prefix_len(self) -> int:
        return self.codec_embed.shape[1]


def build_prompt_pieces(model: Any, speaker: Optional[str], language: str, instruct: Optional[str]) -> VoicePromptPieces:
    """Vendored static half of mlx-audio's ``_prepare_generation_inputs``
    (MIT) — same ops, same order — factored out so it can be computed once
    per voice key. The dynamic half (text tokenization → embedding →
    projection, then concat with these pieces) lives in
    :func:`prompt_embeds.assemble_prompt`; together they reproduce
    ``_prepare_generation_inputs`` bit-for-bit.

    Semantics mirror mlx-audio: a speaker that does not resolve through
    ``spk_id`` simply contributes no row (the preset paths validate voices
    before reaching here), and the dialect override applies when one is
    configured for the speaker."""
    config = model.config.talker_config
    talker = model.talker

    tts_tokens = mx.array(
        [
            [
                model.config.tts_bos_token_id,
                model.config.tts_eos_token_id,
                model.config.tts_pad_token_id,
            ]
        ]
    )
    tts_embeds = talker.text_projection(talker.get_text_embeddings()(tts_tokens))

    speaker_embed = None
    if speaker and speaker.lower() in (config.spk_id or {}):
        spk_ids = mx.array([[config.spk_id[speaker.lower()]]])  # [1, 1]
        speaker_embed = talker.get_input_embeddings()(spk_ids)  # [1, 1, hidden]

    language_id = None
    if language.lower() != "auto" and config.codec_language_id:
        if language.lower() in config.codec_language_id:
            language_id = config.codec_language_id[language.lower()]
    if (
        language.lower() in ["chinese", "auto"]
        and speaker
        and speaker.lower() in (config.spk_is_dialect or {})
        and config.spk_is_dialect[speaker.lower()]
    ):
        dialect = config.spk_is_dialect[speaker.lower()]
        if dialect in config.codec_language_id:
            language_id = config.codec_language_id[dialect]

    if language_id is None:
        codec_prefill = [
            config.codec_nothink_id,
            config.codec_think_bos_id,
            config.codec_think_eos_id,
        ]
    else:
        codec_prefill = [
            config.codec_think_id,
            config.codec_think_bos_id,
            language_id,
            config.codec_think_eos_id,
        ]
    codec_embed = talker.get_input_embeddings()(mx.array([codec_prefill]))
    codec_embed_suffix = talker.get_input_embeddings()(
        mx.array([[config.codec_pad_id, config.codec_bos_id]])
    )
    if speaker_embed is not None:
        # the speaker encoder runs in float32 while the talker may be
        # lower precision — mlx-audio's cast avoids promoting the prefill
        # (and the KV cache that follows); input-embedding lookups are
        # already the talker dtype, making this a no-op on the preset path
        speaker_embed = speaker_embed.astype(codec_embed.dtype)
        codec_embed = mx.concatenate(
            [codec_embed, speaker_embed.reshape(1, 1, -1), codec_embed_suffix], axis=1
        )
    else:
        codec_embed = mx.concatenate([codec_embed, codec_embed_suffix], axis=1)

    instruct_embed = None
    if instruct:
        instruct_text = f"<|im_start|>user\n{instruct}<|im_end|>\n"
        instruct_ids = mx.array(model.tokenizer.encode(instruct_text))[None, :]
        instruct_embed = talker.text_projection(talker.get_text_embeddings()(instruct_ids))

    pieces = VoicePromptPieces(
        codec_embed=codec_embed,
        tts_bos_embed=tts_embeds[:, 0:1, :],
        tts_eos_embed=tts_embeds[:, 1:2, :],
        tts_pad_embed=tts_embeds[:, 2:3, :],
        instruct_embed=instruct_embed,
        has_speaker=speaker_embed is not None,
    )
    mx.eval(
        pieces.codec_embed,
        pieces.tts_bos_embed,
        pieces.tts_eos_embed,
        pieces.tts_pad_embed,
    )
    if instruct_embed is not None:
        mx.eval(instruct_embed)
    return pieces


def prompt_pieces(model: Any, speaker: Optional[str], language: str = "auto", instruct: Optional[str] = None) -> VoicePromptPieces:
    """Level 1: :func:`build_prompt_pieces` cached per (model, voice key).

    Subsequent calls with the same key return the identical arrays — what a
    fresh ``_prepare_generation_inputs`` would have recomputed — so the
    assembled prompt is bit-identical to the uncached build."""
    if not prefix_cache_enabled():
        return build_prompt_pieces(model, speaker, language, instruct)

    key = voice_key(speaker, language, instruct)
    entry = _PIECES.get(id(model))
    if entry is not None and entry[0]() is not model:
        entry = None
    if entry is None:
        def _drop(_ref: Any, mid: int = id(model)) -> None:
            current = _PIECES.get(mid)
            if current is not None and current[0]() is None:
                del _PIECES[mid]

        entry = (weakref.ref(model, _drop), OrderedDict())
        _PIECES[id(model)] = entry
    cache = entry[1]
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    pieces = build_prompt_pieces(model, speaker, language, instruct)
    cache[key] = pieces
    cache.move_to_end(key)
    # pieces are tiny (≤ ~20 rows); a generous per-model cap keeps an
    # unbounded voice/instruct space from growing without limit
    while len(cache) > 32:
        cache.popitem(last=False)
    return pieces


@dataclass(frozen=True)
class PrefixKV:
    """Level 2 entry: the K/V rows of one voice's static prompt prefix."""

    keys: list  # per-layer [1, kv_heads, length, head_dim]
    values: list
    length: int

    @property
    def nbytes(self) -> int:
        return sum(k.nbytes + v.nbytes for k, v in zip(self.keys, self.values))


class _ModelPrefixState:
    """LRU of :class:`PrefixKV` entries for one loaded model. Entries are
    never mutated after store — consumers concatenate onto them (functional
    updates), so one entry safely serves concurrent-in-time requests under
    the service lock."""

    def __init__(self, cap: int = 8):
        self.entries: "OrderedDict[tuple, PrefixKV]" = OrderedDict()
        self.cap = cap

    def lookup(self, key: tuple) -> Optional[PrefixKV]:
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
        return entry

    def store(self, key: tuple, keys: list, values: list, length: int) -> PrefixKV:
        entry = PrefixKV(keys=list(keys), values=list(values), length=length)
        self.entries[key] = entry
        self.entries.move_to_end(key)
        while len(self.entries) > self.cap:
            self.entries.popitem(last=False)
        return entry

    def bytes_resident(self) -> int:
        return sum(e.nbytes for e in self.entries.values())


def _state_for(model: Any) -> _ModelPrefixState:
    key = id(model)
    entry = _STORES.get(key)
    if entry is not None and entry[0]() is model:
        return entry[1]
    state = _ModelPrefixState()

    def _drop(_ref: Any, mid: int = key, state: _ModelPrefixState = state) -> None:
        current = _STORES.get(mid)
        if current is not None and current[1] is state:
            # only drop OUR entry: a replacement model may have reused the id
            del _STORES[mid]

    _STORES[key] = (weakref.ref(model, _drop), state)
    return state


def lookup_prefix_kv(model: Any, speaker: Optional[str], language: str, instruct: Optional[str]) -> Optional[PrefixKV]:
    """Level 2 lookup — None when disabled, cold, or evicted."""
    if not prefix_cache_enabled():
        return None
    return _state_for(model).lookup(voice_key(speaker, language, instruct))


def store_prefix_kv(model: Any, speaker: Optional[str], language: str, instruct: Optional[str], keys: list, values: list, length: int) -> Optional[PrefixKV]:
    """Record the static-prefix K/V rows for a voice (the caller slices them
    off a prefill it already ran — no extra forward) and return the entry.
    The slices are materialized as compact copies: they cut into mlx-audio's
    step-256 preallocated cache buffers, and keeping a lazy slice would pin
    the whole buffer — ~32× the entry's real footprint."""
    if not prefix_cache_enabled() or length <= 0:
        return None
    keys = [mx.array(k) for k in keys]
    values = [mx.array(v) for v in values]
    mx.eval(*keys, *values)
    return _state_for(model).store(voice_key(speaker, language, instruct), keys, values, length)


def prefix_cache_stats(model: Any) -> dict:
    """Resident entries + bytes — the memory accounting the PR reports."""
    key = id(model)
    entry = _STORES.get(key)
    if entry is None or entry[0]() is not model:
        return {"entries": 0, "bytes": 0}
    state = entry[1]
    return {"entries": len(state.entries), "bytes": state.bytes_resident()}


def clear_prefix_caches() -> None:
    """Drop every Level-1/Level-2 entry (tests, benches, explicit eviction)."""
    _STORES.clear()
    _PIECES.clear()
