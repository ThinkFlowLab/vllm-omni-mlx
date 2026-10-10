"""Incremental MOSS Nano generation, adapting mlx-audio 0.5.7 (MIT).

The eager frame loop mirrors ``Model.generate_audio_token_ids``: the global
GPT2 KV cache spans a sentence, and the local transformer predicts each
codebook in sequence. Preprocessing jobs such as prompt construction, text splitting
and sampling remain upstream. Three main pieces of the functionality implementation:

1. iter_audio_frames(): Encode the reference audio, split the input text into
manageable segments, and prepare each segment for the model.

2. _sentence_frames(): Generate audio codes one frame at a time. Each frame
represents about 80 ms of audio and must be decoded before playback.

3. synthesize_stream(): Decode audio codes into waveform chunks and yield
each chunk as it becomes ready, while generation continues.
"""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Any, Iterator

import mlx.core as mx

if TYPE_CHECKING:
    from .moss_nano import MossNanoConfig


def _cancelled(cancel: threading.Event | None) -> bool:
    return cancel is not None and cancel.is_set()


def validate_streaming_codec(model: Any) -> int:
    """Validate the loaded codec and return samples per frame per channel.

    ``downsample_rate`` controls encoder input padding, so derive the output
    hop from decoder expansion and stereo interleaving instead. Configuration
    errors must be discovered before starting an HTTP streaming response.
    """
    codec = getattr(model, "audio_tokenizer", None)
    if codec is None or not callable(getattr(codec, "make_streaming_decoder", None)):
        raise RuntimeError(
            "MOSS Nano requires a loaded codec with streaming decode support"
        )
    if getattr(codec, "sample_rate", None) != model.sample_rate:
        raise RuntimeError("MOSS Nano codec and model sample rates must match")
    stages = getattr(codec, "decoder", None)
    if not stages:
        raise RuntimeError("MOSS Nano codec has no decoder stages")
    expansion = 1
    for stage in stages:
        ratio = getattr(stage, "downsample_ratio", None)
        if not isinstance(ratio, int) or isinstance(ratio, bool) or ratio < 1:
            raise RuntimeError("MOSS Nano codec has an invalid decoder expansion ratio")
        expansion *= ratio
        transformer = getattr(stage, "transformer", None)
        if transformer is not None:
            if not callable(getattr(stage, "make_step_cache", None)):
                raise RuntimeError("MOSS Nano codec transformer lacks streaming caches")
            for layer in transformer.layers:
                if not layer.self_attn.causal:
                    raise RuntimeError("MOSS Nano streaming requires a causal codec")
    channels = getattr(codec, "channels", None)
    if channels not in (1, 2):
        raise RuntimeError("MOSS Nano codec must output mono or stereo audio")
    channel_factor = channels if codec.enable_channel_interleave else 1
    if expansion % channel_factor:
        raise RuntimeError("MOSS Nano codec frame is not aligned to its channels")
    return expansion // channel_factor


def _sentence_frames(
    model: Any,
    config: MossNanoConfig,
    input_ids: mx.array,
    attention_mask: mx.array,
    cancel: threading.Event | None,
) -> Iterator[mx.array]:
    # Imports stay lazy so importing the service does not require [tts].
    from mlx_audio.tts.models.moss_tts_nano.sampling import (
        sample_assistant_text_token,
        sample_next_token,
    )

    effective_nq = model._resolve_nq(None)
    cache = model.transformer.make_cache()
    current_model_input_ids = input_ids
    current_attention_mask = attention_mask.astype(mx.bool_)
    generated_frames: list[mx.array] = []

    for _ in range(config.max_new_frames):
        if _cancelled(cancel):
            return
        global_inputs_embeds = model._build_inputs_embeds(current_model_input_ids)
        global_outputs = model.transformer(
            inputs_embeds=global_inputs_embeds,
            attention_mask=current_attention_mask,
            cache=cache,
        )
        global_hidden = global_outputs[:, -1, :]
        if _cancelled(cancel):
            return
        local_inputs_embeds = global_hidden[:, None, :]
        local_outputs = model.local_transformer(inputs_embeds=local_inputs_embeds)
        local_hidden = local_outputs[:, -1, :]
        text_logits = model._text_lm_head(local_hidden)
        next_text_token = sample_assistant_text_token(
            text_logits,
            audio_assistant_slot_token_id=model.config.audio_assistant_slot_token_id,
            audio_end_token_id=model.config.audio_end_token_id,
            do_sample=config.do_sample,
            temperature=config.text_temperature,
            top_k=config.text_top_k,
            top_p=config.text_top_p,
        )
        mx.eval(next_text_token)
        if _cancelled(cancel):
            return
        if int(next_text_token.item()) != model.config.audio_assistant_slot_token_id:
            return

        current_local_input = model.transformer.wte(next_text_token)
        frame_tokens: list[mx.array] = []
        history = mx.stack(generated_frames, axis=1) if generated_frames else None
        for channel_index in range(effective_nq):
            if _cancelled(cancel):
                return
            local_inputs_embeds = mx.concatenate(
                [local_inputs_embeds, current_local_input[:, None, :]], axis=1
            )
            local_outputs = model.local_transformer(inputs_embeds=local_inputs_embeds)
            local_hidden = local_outputs[:, -1, :]
            channel_logits = model._audio_lm_head(local_hidden, channel_index)
            previous_tokens = None if history is None else history[:, :, channel_index]
            channel_token = sample_next_token(
                channel_logits,
                do_sample=config.do_sample,
                temperature=config.audio_temperature,
                top_k=config.audio_top_k,
                top_p=config.audio_top_p,
                previous_token_ids=previous_tokens,
                repetition_penalty=config.audio_repetition_penalty,
            )
            frame_tokens.append(channel_token)
            current_local_input = model.audio_embeddings[channel_index](channel_token)

        frame = mx.stack(frame_tokens, axis=-1)
        if effective_nq < model.config.n_vq:
            frame = mx.concatenate(
                [
                    frame,
                    mx.full(
                        (frame.shape[0], model.config.n_vq - effective_nq),
                        model.config.audio_pad_token_id,
                        dtype=mx.int32,
                    ),
                ],
                axis=-1,
            )
        generated_frames.append(frame)
        text_column = mx.full(
            (frame.shape[0], 1, 1),
            model.config.audio_assistant_slot_token_id,
            dtype=mx.int32,
        )
        next_row = mx.concatenate([text_column, frame[:, None, :]], axis=-1)
        current_model_input_ids = next_row
        current_attention_mask = mx.concatenate(
            [current_attention_mask, mx.ones((frame.shape[0], 1), dtype=mx.bool_)],
            axis=1,
        )
        mx.eval(frame)
        if _cancelled(cancel):
            return
        yield frame[0].astype(mx.int32)


def iter_audio_frames(
    model: Any,
    config: MossNanoConfig,
    text: str,
    ref_audio: mx.array,
    *,
    cancel: threading.Event | None = None,
) -> Iterator[mx.array]:
    """Yield complete ``[n_vq]`` code frames with upstream eager semantics."""
    from mlx_audio.tts.models.moss_tts_nano.text import (
        lightweight_normalize_text,
        split_text_into_best_sentences,
    )

    if _cancelled(cancel):
        return
    if model.tokenizer is None:
        raise RuntimeError("MOSS Nano text tokenizer is not initialized")
    prompt_audio_codes = model.encode_reference_audio(
        ref_audio,
        sample_rate=model.sample_rate,
        num_quantizers=model.config.n_vq,
        source=config.codec_model_ref,
    )
    if _cancelled(cancel):
        return
    chunks = split_text_into_best_sentences(
        model.tokenizer,
        lightweight_normalize_text(text),
        max_tokens=config.max_text_tokens,
    )
    for chunk in chunks:
        if _cancelled(cancel):
            return
        input_ids, attention_mask = model.build_inference_input_ids(
            text=chunk,
            tokenizer=model.tokenizer,
            mode="voice_clone",
            prompt_audio_codes=prompt_audio_codes,
        )
        # Fresh sentence-local cache/history, matching upstream generate().
        yield from _sentence_frames(model, config, input_ids, attention_mask, cancel)


def synthesize_stream(
    model: Any,
    config: MossNanoConfig,
    text: str,
    ref_audio: mx.array,
    *,
    streaming_interval: float,
    streaming_initial_interval: float,
    cancel: threading.Event | None = None,
) -> Iterator[mx.array]:
    """Decode arriving frames into sample-major audio with request-local state."""
    for name, value in (
        ("streaming_interval", streaming_interval),
        ("streaming_initial_interval", streaming_initial_interval),
    ):
        if not math.isfinite(value) or not 0 < value <= 10:
            raise ValueError(f"{name} must be in (0, 10] seconds")
    if _cancelled(cancel):
        return
    samples_per_frame = validate_streaming_codec(model)
    codec = model.audio_tokenizer
    decoder = codec.make_streaming_decoder(num_quantizers=model.config.n_vq)
    if not callable(getattr(decoder, "decode_frames", None)):
        raise RuntimeError("MOSS Nano codec streaming decoder lacks decode_frames")
    steady_frames = max(
        1, int(streaming_interval * model.sample_rate / samples_per_frame)
    )
    initial_frames = max(
        1, int(streaming_initial_interval * model.sample_rate / samples_per_frame)
    )
    target = min(initial_frames, steady_frames)
    pending: list[mx.array] = []
    frames = iter_audio_frames(model, config, text, ref_audio, cancel=cancel)
    try:
        for frame in frames:
            if _cancelled(cancel):
                return
            pending.append(frame)
            if len(pending) >= target:
                audio = decoder.decode_frames(mx.stack(pending, axis=0))
                pending.clear()
                if _cancelled(cancel):
                    return
                yield audio
                target = steady_frames
        # There is no decoder lookahead/tail: only the final code group needs
        # flushing. A cancellation must not synthesize or send that remainder.
        if pending and not _cancelled(cancel):
            audio = decoder.decode_frames(mx.stack(pending, axis=0))
            if not _cancelled(cancel):
                yield audio
    finally:
        frames.close()
