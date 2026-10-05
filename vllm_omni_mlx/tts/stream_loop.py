"""First-chunk fast-path generation loops (M2, #39 tasks 2+3; ICL twin #50).

Vendors mlx-audio's CustomVoice decode loop (MIT) — the escape hatch the
M1.0 spike sanctioned for exactly this case: mlx-audio exposes a single
fixed ``streaming_interval``, so first audio always waits for a full chunk
(25 frames / 2 s at the old default; 6 frames / 0.5 s now). The loops are
step-for-step mlx-audio's (0.5.7) — ``generate_frames`` mirrors
``_generate_with_instruct`` (qwen3_tts.py:2516) and ``generate_icl_frames``
mirrors ``_generate_icl`` (qwen3_tts.py:2204; the two share their AR
skeleton) — same sampler call, same caches, same single ``mx.eval`` sync per
frame, driving mlx-audio's own components through our M1 seams (PromptEmbeds,
Talker, CodePredictor, Code2Wav), with two additions:

- an independent *initial* chunk boundary: the first chunk covers
  ``initial_frames`` frames (default 2 ≈ 160 ms), steady chunks cover
  ``chunk_frames`` — upstream's ``initial_codec_chunk_frames`` trick for
  TTFA without flooding the vocoder with tiny windows;
- a bounded compiled-shape set: the initial boundary is quantized to a
  power of two ≤ ``chunk_frames`` and the final remainder is padded to the
  nearest bucket then trimmed, so the mx.compile'd ``streaming_step``
  traces at most two shapes per configuration instead of one per distinct
  remainder (each new shape costs a seconds-long retrace);
- compiled per-frame decode closures (#65): the 15 predictor micro-steps
  (+ their sampling) run as ONE mx.compile'd call — the predictor's cache
  resets every frame, so its state is frame-local and the closure is
  fixed-shape forever — and the talker decode runs compiled shapeless with
  the KV cache as arrays (the prompt prefill stays eager, then its cache
  is transplanted). Both are bit-exact against the eager paths on
  identical inputs; end-to-end greedy streams may still diverge on rare
  fp16 near-ties under fusion (see tts.compiled_steps).
  ``VLLM_OMNI_TTS_EAGER_STREAM=1`` keeps the uncompiled
  mlx-audio-mirror loop for A/B and debugging;
- voice-prefix KV splice (#66): the prompt's voice-static rows (instruct?
  + role + codec prefix, everything but the first-text row) prefill to the
  same K/V for every request with one voice, so they are forwarded once per
  ``(speaker, language, instruct)`` key — one batched forward whose K/V is
  cached — and every request feeds only its first-text row through the
  compiled decode at the prefix offset; a miss computes exactly what a hit
  replays (same arrays, same closures → bitwise-identical streams, so
  repeated requests are reproducible) while the ~90 ms multi-row eager
  prefill becomes a ~22 ms single-row decode. The first-text row's
  numerics move between mlx-audio's own kernel batchings (multi-row
  prefill vs single-row decode), so cached streams are NOT draw-identical
  to the uncached path — fp16 chaos amplifies the rounding difference
  within a few frames; the eager loop (never cached) and
  ``VLLM_OMNI_TTS_PREFIX_CACHE=0`` keep the uncached reference for parity
  and A/B.
"""

from __future__ import annotations

from typing import Any, Iterator

import mlx.core as mx

from .code_predictor import CodePredictor
from .compiled_steps import (
    EAGER_STREAM,
    make_input_embeds,
    make_predictor_frame,
    make_talker_decode,
    make_talker_sampler,
)
from .config import TTSConfig
from .prefix_cache import lookup_prefix_kv, prefix_cache_enabled, store_prefix_kv
from .prompt_embeds import PromptEmbeds
from .talker import Talker
from .variants import VOICE_DESIGN, ensure_served, model_variant, require_served

FRAME_RATE = 12.5  # codec frames per second of audio (12 Hz tokenizer)
SAMPLES_PER_FRAME = 1920  # 24000 Hz / 12.5


def frames_for_interval(seconds: float) -> int:
    """Seconds of audio → codec frames (mlx-audio's chunk-size formula)."""
    return max(1, int(seconds * FRAME_RATE))


def initial_frames_bucket(requested: int, chunk_frames: int) -> int:
    """Largest power of two ≤ min(requested, chunk_frames), at least 1.

    Quantizing the initial boundary keeps the compiled-shape set closed:
    requests can ask for any sub-second first chunk without each value
    paying its own ``mx.compile`` trace.
    """
    capped = max(1, min(requested, chunk_frames))
    bucket = 1
    while bucket * 2 <= capped:
        bucket *= 2
    return bucket


def _pad_target(remainder: int, initial_frames: int, chunk_frames: int) -> int:
    """Compiled shape for a remainder chunk: the smallest bucket it fits."""
    return initial_frames if remainder <= initial_frames else chunk_frames


def _flush_pending(
    pending_flags: list, input_embeds: mx.array, generated_codes: list
) -> bool:
    """Batched EOS check for the compiled path (#65): evaluate the batch's
    ``is_eos`` flags — and the current ``input_embeds``, bounding the lazy
    graph depth — at the chunk boundary instead of every frame, so the CPU
    stays ahead of the GPU across frames. Frames generated past an EOS were
    computed from dead feedback and are truncated before any decode (≤
    chunk_frames of them, and only in the final chunk of a stream). Returns
    False when generation must stop."""
    mx.eval(*pending_flags, input_embeds)
    for i, flag in enumerate(pending_flags):
        if flag.item():
            keep = len(generated_codes) - (len(pending_flags) - i)
            del generated_codes[keep:]
            pending_flags.clear()
            return False
    pending_flags.clear()
    return True


def generate_frames(
    model: Any,
    *,
    text: str,
    speaker: str | None,
    language: str = "auto",
    instruct: str | None = None,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    max_tokens: int = 4096,
    initial_frames: int = 2,
    chunk_frames: int = 6,
) -> Iterator[mx.array]:
    """Yield audio chunks ([samples] float, 24 kHz mono) — CustomVoice with
    a preset ``speaker``, or VoiceDesign speakerless with the voice
    description in ``instruct`` (#52: same layout minus the spk row; the AR
    loop, chunk scheduling and compiled closures are shared) — decoding the
    first ``initial_frames`` frames as soon as they exist and every
    ``chunk_frames`` frames thereafter."""
    talker = Talker(model.talker)
    predictor = CodePredictor(model.talker)
    layout = PromptEmbeds(model).build(text, speaker, language, instruct)
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
    trailing_idx = 0
    decoded_frames = 0
    # compiled-path state: the cache as arrays after the eager prefill
    keys = values = None
    position = 0
    # compiled-path sampler state: on-device token ring (dummy = vocab) and
    # the batched EOS flags flushed at chunk boundaries
    history = pending_flags = None
    frame_idx = 0

    # compiled fast path (#65): closures are cached per model, so building
    # them here reuses the trace across requests
    compiled = not EAGER_STREAM
    decode_step = make_talker_decode(model.talker) if compiled else None
    predictor_frame = (
        make_predictor_frame(
            model.talker.code_predictor,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            base_embedding=model.talker.get_input_embeddings(),
        )
        if compiled
        else None
    )
    sampler_c = (
        make_talker_sampler(
            model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            suppress_tokens=suppress_tokens,
        )
        if compiled
        else None
    )
    embeds_step = make_input_embeds(model.talker) if compiled else None
    if compiled:
        history = mx.full((64,), config.vocab_size, dtype=mx.uint32)
        pending_flags = []

    initial_frames = initial_frames_bucket(initial_frames, chunk_frames)
    decoder = model.speech_tokenizer.decoder
    decoder.reset_streaming_state()

    # #66 voice-prefix splice — compiled mode only (the eager loop stays the
    # exact mlx-audio-mirror reference). Every entry — miss-built, hit, or
    # boot-warmed — originates from the same canonical build
    # (:func:`ensure_voice_prefix`), so all requests with a voice splice
    # bitwise-identical static K/V and feed only their first-text row
    # through the compiled decode at the prefix offset: the same (voice,
    # text, seed) is reproducible across requests. The numerics of the
    # first-text row move from mlx-audio's multi-row prefill kernels to the
    # single-row decode kernels — a valid decode of the same prompt, but
    # NOT draw-identical to the uncached path (fp16 chaos amplifies the
    # kernel-rounding difference within a few frames); the eager loop and
    # ``VLLM_OMNI_TTS_PREFIX_CACHE=0`` keep the uncached reference for A/B.
    use_prefix = compiled and prefix_cache_enabled()
    prefix = ensure_voice_prefix(model, speaker, language, instruct) if use_prefix else None
    if prefix is not None:
        keys = list(prefix.keys)
        values = list(prefix.values)
        position = prefix.length
        input_embeds = layout.input_embeds[:, -1:, :]

    def decode_pending(padded_to: int | None = None) -> mx.array:
        """streaming_step over generated-but-undecoded frames, optionally
        padded to a compiled bucket shape; padded tail audio is trimmed
        (the decoder is causal, so pad frames only affect their own tail)."""
        nonlocal decoded_frames
        remainder = len(generated_codes) - decoded_frames
        codes_chunk = mx.stack(generated_codes[decoded_frames:], axis=1)
        codes_for_decoder = mx.transpose(codes_chunk, (0, 2, 1))
        if padded_to is not None and remainder < padded_to:
            pad = mx.broadcast_to(
                codes_for_decoder[..., -1:], (1, codes_for_decoder.shape[1], padded_to)
            )
            codes_for_decoder = mx.concatenate([codes_for_decoder, pad], axis=-1)
        mx.eval(codes_for_decoder)
        audio = decoder.streaming_step(codes_for_decoder).squeeze(1)[0]
        mx.eval(audio)
        decoded_frames = len(generated_codes)
        return audio[: remainder * SAMPLES_PER_FRAME]

    try:
        for _ in range(max_tokens):
            if keys is None:
                # first frame doubles as the prompt prefill through
                # mlx-audio's eager path — the uncached paths' prefill
                # (eager reference loop; compiled with the prefix cache
                # disabled); the compiled splice above bypasses this by
                # construction
                logits, hidden = model.talker(input_embeds, cache=cache)
                if compiled:
                    position = cache[0].offset
                    keys = [c.keys[..., :position, :] for c in cache]
                    values = [c.values[..., :position, :] for c in cache]
            else:
                logits, hidden, keys, values = decode_step(
                    input_embeds, mx.array([position], dtype=mx.int32), keys, values
                )
                position += 1

            if compiled:
                next_token = sampler_c(logits, history)
            else:
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

            all_codes = predictor_frame(hidden[:, -1:, :], next_token) if compiled else None
            if not compiled:
                code_tokens = [next_token]
                code_hidden = hidden[:, -1:, :]
                for c in code_cache:  # reset in place, as mlx-audio's loop does
                    c.keys = None
                    c.values = None
                    c.offset = 0
                for code_idx in range(predictor.num_code_groups - 1):
                    if code_idx == 0:
                        code_0_embed = model.talker.get_input_embeddings()(next_token)
                        code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
                    else:
                        code_input = predictor.residual_embeddings[code_idx - 1](code_tokens[-1])
                    code_logits = predictor.step(code_input, code_cache, code_idx)
                    code_tokens.append(
                        model._sample_token(code_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                    )
                all_codes = mx.concatenate(code_tokens, axis=1)

            if trailing_idx < trailing_text_hidden.shape[1]:
                text_embed = trailing_text_hidden[:, trailing_idx : trailing_idx + 1, :]
                trailing_idx += 1
            else:
                text_embed = tts_pad_embed
            if compiled:
                input_embeds = embeds_step(all_codes, text_embed)
            else:
                input_embeds = text_embed + talker.codec_embeds(code_tokens)

            if compiled:
                # on-device ring update for the repetition penalty —
                # functional (concat), so no materialization/sync per frame
                slot = frame_idx % 64
                history = mx.concatenate(
                    [history[:slot], next_token.reshape(1), history[slot + 1 :]]
                )
                frame_idx += 1
                pending_flags.append(is_eos)
                generated_codes.append(all_codes)
            else:
                # single sync point per frame, as mlx-audio's loop does
                mx.eval(input_embeds, is_eos)
                if is_eos.item():
                    break

                generated_token_ids.append(int(next_token[0, 0]))
                generated_codes.append(all_codes)

            boundary = initial_frames if decoded_frames == 0 else chunk_frames
            if len(generated_codes) - decoded_frames >= boundary:
                if compiled and not _flush_pending(pending_flags, input_embeds, generated_codes):
                    break
                yield decode_pending()

        if compiled and pending_flags:
            # EOS inside the final, never-flushed batch
            _flush_pending(pending_flags, input_embeds, generated_codes)
        if len(generated_codes) > decoded_frames:
            remainder = len(generated_codes) - decoded_frames
            yield decode_pending(padded_to=_pad_target(remainder, initial_frames, chunk_frames))
    finally:
        decoder.reset_streaming_state()
        mx.clear_cache()


def synthesize_stream(model: Any, config: TTSConfig, text: str, **overrides) -> Iterator[mx.array]:
    """Config-driven wrapper mirroring generate.synthesize's contract, on the
    fast-path loop, routing by checkpoint type (#52): CustomVoice passes the
    preset speaker, VoiceDesign passes ``speaker=None`` with the voice
    description in ``instruct`` (required there). `seed` reseeds MLX's RNG
    for reproducibility; overrides follow TTSConfig.with_overrides semantics."""
    variant = model_variant(model)
    require_served(variant, path="design" if variant == VOICE_DESIGN else "preset")
    cfg = config.with_overrides(**overrides)
    if variant == VOICE_DESIGN and not (cfg.instruct or "").strip():
        raise ValueError(
            "VoiceDesign synthesis needs `instruct` — a voice description "
            "like 'A cheerful young female voice with high pitch and "
            "energetic tone' (#46)"
        )
    if overrides.get("seed") is not None:
        mx.random.seed(int(overrides["seed"]))
    yield from generate_frames(
        model,
        text=text,
        speaker=cfg.speaker if variant != VOICE_DESIGN else None,
        language=cfg.language,
        instruct=cfg.instruct,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
        max_tokens=cfg.max_tokens,
        initial_frames=frames_for_interval(cfg.streaming_initial_interval),
        chunk_frames=frames_for_interval(cfg.streaming_interval),
    )


#: canonical text for prefix entries (#66): the role rows pass through
#: ``text_projection``, whose GEMM rounding depends on the prompt's
#: tokenized length, so "the static rows" of two texts with the same voice
#: differ in low bits. Building every entry from ONE fixed text pins those
#: bits: miss-built, hit, and boot-warmed entries are bitwise identical.
_PREFIX_TEXT = "warm"


def ensure_voice_prefix(model: Any, speaker: "str | None", language: str = "auto", instruct: str | None = None):
    """The voice's prefix entry — looked up, or built now from the canonical
    text (one batched forward over the static rows, stored per voice key).
    Returns None when the cache is disabled."""
    if not prefix_cache_enabled():
        return None
    entry = lookup_prefix_kv(model, speaker, language, instruct)
    if entry is not None:
        return entry
    layout = PromptEmbeds(model).build(_PREFIX_TEXT, speaker, language, instruct)
    n = layout.prefix_rows
    if n <= 0:
        return None
    cache = model.talker.make_cache()
    model.talker(layout.input_embeds[:, :-1, :], cache=cache)
    entry = store_prefix_kv(
        model, speaker, language, instruct,
        keys=[c.keys[..., :n, :] for c in cache],
        values=[c.values[..., :n, :] for c in cache],
        length=n,
    )
    mx.clear_cache()
    return entry


def warm_voice_prefix(model: Any, speaker: "str | None", language: str = "auto", instruct: str | None = None) -> bool:
    """Populate the #66 prefix store for one voice up front — the static-row
    forward only, no generation — so a server's first request on its default
    voice is a cache hit. Thin wrapper over :func:`ensure_voice_prefix`;
    failures are the caller's to tolerate (a request would simply build the
    entry on demand)."""
    return ensure_voice_prefix(model, speaker, language, instruct) is not None


def generate_icl_frames(
    model: Any,
    *,
    text: str,
    ref_audio: "mx.array",
    ref_text: str,
    language: str = "auto",
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    max_tokens: int = 4096,
    initial_frames: int = 2,
    chunk_frames: int = 6,
) -> Iterator[mx.array]:
    """Yield audio chunks for Base ICL voice cloning (#50) — the same
    fast-path loop as :func:`generate_frames`, vendoring
    mlx-audio's ``_generate_icl`` (qwen3_tts.py:2204): identical AR body
    (their two loops share the skeleton), ICL prefill instead of the preset
    prompt, and the repetition penalty floored to 1.5 — mlx-audio's guard
    against "code degeneration with long reference audio prefills".

    Decode context follows mlx-audio's streaming (decision A on #50):
    chunks decode generated codes only, without the reference-code prefix
    the buffered path concatenates — chunked audio is boundary-sensitive by
    nature (see the vocoder-chunking calibration); parity is claimed at the
    sampler level, not bitwise audio.
    """
    talker = Talker(model.talker)
    predictor = CodePredictor(model.talker)
    input_embeds, trailing_text_hidden, tts_pad_embed, _ref_codes = (
        model._prepare_icl_generation_inputs(
            text=text, ref_audio=ref_audio, ref_text=ref_text, language=language
        )
    )
    repetition_penalty = max(repetition_penalty, 1.5)

    config = model.config.talker_config
    eos_token_id = config.codec_eos_token_id
    suppress_tokens = talker.suppressed_codec_ids()

    cache = model.talker.make_cache()
    code_cache = predictor.make_cache()
    generated_codes: list[mx.array] = []
    generated_token_ids: list[int] = []
    trailing_idx = 0
    decoded_frames = 0
    # compiled-path state: the cache as arrays after the eager prefill
    keys = values = None
    position = 0
    # compiled-path sampler state: on-device token ring (dummy = vocab) and
    # the batched EOS flags flushed at chunk boundaries
    history = pending_flags = None
    frame_idx = 0

    # compiled fast path (#65) — same closures as the CustomVoice loop
    compiled = not EAGER_STREAM
    decode_step = make_talker_decode(model.talker) if compiled else None
    predictor_frame = (
        make_predictor_frame(
            model.talker.code_predictor,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            base_embedding=model.talker.get_input_embeddings(),
        )
        if compiled
        else None
    )
    sampler_c = (
        make_talker_sampler(
            model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            suppress_tokens=suppress_tokens,
        )
        if compiled
        else None
    )
    embeds_step = make_input_embeds(model.talker) if compiled else None
    if compiled:
        history = mx.full((64,), config.vocab_size, dtype=mx.uint32)
        pending_flags = []

    initial_frames = initial_frames_bucket(initial_frames, chunk_frames)
    decoder = model.speech_tokenizer.decoder
    decoder.reset_streaming_state()

    def decode_pending(padded_to: int | None = None) -> mx.array:
        nonlocal decoded_frames
        remainder = len(generated_codes) - decoded_frames
        codes_chunk = mx.stack(generated_codes[decoded_frames:], axis=1)
        codes_for_decoder = mx.transpose(codes_chunk, (0, 2, 1))
        if padded_to is not None and remainder < padded_to:
            pad = mx.broadcast_to(
                codes_for_decoder[..., -1:], (1, codes_for_decoder.shape[1], padded_to)
            )
            codes_for_decoder = mx.concatenate([codes_for_decoder, pad], axis=-1)
        mx.eval(codes_for_decoder)
        audio = decoder.streaming_step(codes_for_decoder).squeeze(1)[0]
        mx.eval(audio)
        decoded_frames = len(generated_codes)
        return audio[: remainder * SAMPLES_PER_FRAME]

    try:
        for _ in range(max_tokens):
            if keys is None:
                # first frame doubles as the ICL prefill (reference audio +
                # text) through mlx-audio's eager path; transplant its cache
                # into the compiled decode's array state on the way out
                logits, hidden = model.talker(input_embeds, cache=cache)
                if compiled:
                    position = cache[0].offset
                    keys = [c.keys[..., :position, :] for c in cache]
                    values = [c.values[..., :position, :] for c in cache]
            else:
                logits, hidden, keys, values = decode_step(
                    input_embeds, mx.array([position], dtype=mx.int32), keys, values
                )
                position += 1

            if compiled:
                next_token = sampler_c(logits, history)
            else:
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

            all_codes = predictor_frame(hidden[:, -1:, :], next_token) if compiled else None
            if not compiled:
                code_tokens = [next_token]
                code_hidden = hidden[:, -1:, :]
                for c in code_cache:
                    c.keys = None
                    c.values = None
                    c.offset = 0
                for code_idx in range(predictor.num_code_groups - 1):
                    if code_idx == 0:
                        code_0_embed = model.talker.get_input_embeddings()(next_token)
                        code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
                    else:
                        code_input = predictor.residual_embeddings[code_idx - 1](code_tokens[-1])
                    code_logits = predictor.step(code_input, code_cache, code_idx)
                    code_tokens.append(
                        model._sample_token(code_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                    )
                all_codes = mx.concatenate(code_tokens, axis=1)

            if trailing_idx < trailing_text_hidden.shape[1]:
                text_embed = trailing_text_hidden[:, trailing_idx : trailing_idx + 1, :]
                trailing_idx += 1
            else:
                text_embed = tts_pad_embed
            if compiled:
                input_embeds = embeds_step(all_codes, text_embed)
            else:
                input_embeds = text_embed + talker.codec_embeds(code_tokens)

            if compiled:
                # on-device ring update for the repetition penalty —
                # functional (concat), so no materialization/sync per frame
                slot = frame_idx % 64
                history = mx.concatenate(
                    [history[:slot], next_token.reshape(1), history[slot + 1 :]]
                )
                frame_idx += 1
                pending_flags.append(is_eos)
                generated_codes.append(all_codes)
            else:
                mx.eval(input_embeds, is_eos)
                if is_eos.item():
                    break

                generated_token_ids.append(int(next_token[0, 0]))
                generated_codes.append(all_codes)

            boundary = initial_frames if decoded_frames == 0 else chunk_frames
            if len(generated_codes) - decoded_frames >= boundary:
                if compiled and not _flush_pending(pending_flags, input_embeds, generated_codes):
                    break
                yield decode_pending()

        if compiled and pending_flags:
            # EOS inside the final, never-flushed batch
            _flush_pending(pending_flags, input_embeds, generated_codes)
        if len(generated_codes) > decoded_frames:
            remainder = len(generated_codes) - decoded_frames
            yield decode_pending(padded_to=_pad_target(remainder, initial_frames, chunk_frames))
    finally:
        decoder.reset_streaming_state()
        mx.clear_cache()


def synthesize_clone_stream(
    model: Any, config: TTSConfig, text: str, ref_audio: "mx.array", ref_text: str, **overrides
) -> Iterator[mx.array]:
    """Config-driven wrapper for the ICL fast path — the streaming twin of
    :func:`generate.synthesize_clone`. `ref_audio` is a decoded 24 kHz mono
    waveform; `seed` reseeds MLX's RNG for reproducibility."""
    ensure_served(model, path="clone_stream")
    cfg = config.with_overrides(**overrides)
    if overrides.get("seed") is not None:
        mx.random.seed(int(overrides["seed"]))
    yield from generate_icl_frames(
        model,
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        language=cfg.language,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,  # floored to 1.5 inside
        max_tokens=cfg.max_tokens,
        initial_frames=frames_for_interval(cfg.streaming_initial_interval),
        chunk_frames=frames_for_interval(cfg.streaming_interval),
    )


def prewarm_streaming(
    model: Any,
    interval: float,
    initial_interval: float,
    *,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
) -> "tuple[int, ...]":
    """Trace the compiled streaming_step shapes this stream configuration
    will hit (initial bucket + steady chunk), then leave the decoder state
    clean; returns the traced shapes. Also traces the compiled decode
    closures on dummy state (serving-default sampler params; other
    configurations trace on first use). First real request then pays no
    compile; failures are the caller's to tolerate (first request degrades
    to tracing on demand). Per-request interval overrides beyond this pair
    still trace on first use."""
    chunk = frames_for_interval(interval)
    initial = initial_frames_bucket(frames_for_interval(initial_interval), chunk)
    frame_counts = (initial, chunk) if initial != chunk else (chunk,)
    decoder = model.speech_tokenizer.decoder
    num_quantizers = decoder.config.num_quantizers
    for frames in frame_counts:
        mx.eval(decoder.streaming_step(mx.zeros((1, num_quantizers, frames), dtype=mx.int32)))
    decoder.reset_streaming_state()

    if not EAGER_STREAM:
        _prewarm_decode_closures(model, temperature, top_k, top_p)
    mx.clear_cache()
    return frame_counts


def _prewarm_decode_closures(model: Any, temperature: float, top_k: int, top_p: float) -> None:
    """Trace the compiled talker decode + predictor frame on dummy state
    (outputs are garbage and discarded). One 1-token eager forward supplies
    real k/v arrays so the talker closure traces in its serving dtype;
    dtypes matter — mx.compile retraces on dtype change even shapeless."""
    hidden_size = model.config.talker_config.hidden_size
    embed_dtype = model.talker.get_input_embeddings()(
        mx.zeros((1, 1), dtype=mx.uint32)
    ).dtype

    decode_step = make_talker_decode(model.talker)
    dummy_cache = model.talker.make_cache()
    model.talker(mx.zeros((1, 1, hidden_size), dtype=embed_dtype), cache=dummy_cache)
    length = dummy_cache[0].offset
    keys = [c.keys[..., :length, :] for c in dummy_cache]
    values = [c.values[..., :length, :] for c in dummy_cache]
    logits, hidden, _, _ = decode_step(
        mx.zeros((1, 1, hidden_size), dtype=embed_dtype),
        mx.array([length], dtype=mx.int32),
        keys,
        values,
    )
    mx.eval(logits, hidden)

    predictor_frame = make_predictor_frame(
        model.talker.code_predictor,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        base_embedding=model.talker.get_input_embeddings(),
    )
    codes = predictor_frame(
        mx.zeros((1, 1, hidden_size), dtype=embed_dtype), mx.zeros((1, 1), dtype=mx.uint32)
    )
    mx.eval(codes)

    # sampler (penalty ring) + next-input embeds closures
    vocab = model.config.talker_config.vocab_size
    suppress = Talker(model.talker).suppressed_codec_ids()
    sampler = make_talker_sampler(
        model,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.05,
        suppress_tokens=suppress,
    )
    token = sampler(
        mx.zeros((1, 1, vocab), dtype=embed_dtype), mx.full((64,), vocab, dtype=mx.uint32)
    )
    mx.eval(token)

    embeds_step = make_input_embeds(model.talker)
    groups = model.config.talker_config.code_predictor_config.num_code_groups
    embeds = embeds_step(
        mx.zeros((1, groups), dtype=mx.uint32),
        mx.zeros((1, 1, hidden_size), dtype=embed_dtype),
    )
    mx.eval(embeds)
