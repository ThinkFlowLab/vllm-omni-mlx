"""First-chunk fast-path generation loop (M2, #39 tasks 2+3).

Vendors mlx-audio's CustomVoice decode loop (MIT) — the escape hatch the
M1.0 spike sanctioned for exactly this case: mlx-audio exposes a single
fixed ``streaming_interval``, so first audio always waits for a full chunk
(25 frames / 2 s at the old default; 6 frames / 0.5 s now). This loop is
step-for-step ``_generate_with_instruct`` (mlx-audio 0.5.7,
qwen3_tts.py:2516) — same sampler call, same caches, same single
``mx.eval`` sync per frame — driving mlx-audio's own components through
our M1 seams (PromptEmbeds, Talker, CodePredictor, Code2Wav), with two
additions:

- an independent *initial* chunk boundary: the first chunk covers
  ``initial_frames`` frames (default 2 ≈ 160 ms), steady chunks cover
  ``chunk_frames`` — upstream's ``initial_codec_chunk_frames`` trick for
  TTFA without flooding the vocoder with tiny windows;
- a bounded compiled-shape set: the initial boundary is quantized to a
  power of two ≤ ``chunk_frames`` and the final remainder is padded to the
  nearest bucket then trimmed, so the mx.compile'd ``streaming_step``
  traces at most two shapes per configuration instead of one per distinct
  remainder (each new shape costs a seconds-long retrace).
"""

from __future__ import annotations

from typing import Any, Iterator

import mlx.core as mx

from .code_predictor import CodePredictor
from .config import TTSConfig
from .prompt_embeds import PromptEmbeds
from .talker import Talker

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


def generate_custom_voice_frames(
    model: Any,
    *,
    text: str,
    speaker: str,
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
    """Yield audio chunks ([samples] float, 24 kHz mono) for CustomVoice
    synthesis, decoding the first ``initial_frames`` frames as soon as they
    exist and every ``chunk_frames`` frames thereafter."""
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

    initial_frames = initial_frames_bucket(initial_frames, chunk_frames)
    decoder = model.speech_tokenizer.decoder
    decoder.reset_streaming_state()

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
            logits, hidden = model.talker(input_embeds, cache=cache)

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
            input_embeds = text_embed + talker.codec_embeds(code_tokens)

            # single sync point per frame, as mlx-audio's loop does
            mx.eval(input_embeds, is_eos)
            if is_eos.item():
                break

            generated_token_ids.append(int(next_token[0, 0]))
            generated_codes.append(all_codes)

            boundary = initial_frames if decoded_frames == 0 else chunk_frames
            if len(generated_codes) - decoded_frames >= boundary:
                yield decode_pending()

        if len(generated_codes) > decoded_frames:
            remainder = len(generated_codes) - decoded_frames
            yield decode_pending(padded_to=_pad_target(remainder, initial_frames, chunk_frames))
    finally:
        decoder.reset_streaming_state()
        mx.clear_cache()


def synthesize_stream(model: Any, config: TTSConfig, text: str, **overrides) -> Iterator[mx.array]:
    """Config-driven wrapper mirroring generate.synthesize's contract, on the
    fast-path loop. `seed` reseeds MLX's RNG for reproducibility; overrides
    follow TTSConfig.with_overrides semantics."""
    cfg = config.with_overrides(**overrides)
    if overrides.get("seed") is not None:
        mx.random.seed(int(overrides["seed"]))
    yield from generate_custom_voice_frames(
        model,
        text=text,
        speaker=cfg.speaker,
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


def prewarm_streaming(model: Any, interval: float, initial_interval: float) -> "tuple[int, ...]":
    """Trace the compiled streaming_step shapes this stream configuration
    will hit (initial bucket + steady chunk), then leave the decoder state
    clean; returns the traced shapes. First real request then pays no
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
    mx.clear_cache()
    return frame_counts
