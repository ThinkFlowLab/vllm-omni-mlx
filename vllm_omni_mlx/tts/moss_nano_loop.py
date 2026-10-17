"""Streaming MOSS-Nano audio using mlx-audio's model, samplers, and codec."""

from collections.abc import Iterator
from typing import Any

import mlx.core as mx

from .moss_nano import MossNanoConfig


FRAME_RATE = 12.5  # 48000 samples/second / 3840 samples/frame.


def frames_for_interval(seconds: float) -> int:
    """Seconds of audio → codec frames, with at least one frame per chunk."""
    return max(1, int(seconds * FRAME_RATE))


def generate_frames(
    model: Any,
    *,
    text: str,
    ref_audio: str | mx.array,
    do_sample: bool = True,
    text_temperature: float = 1.0,
    text_top_k: int = 50,
    text_top_p: float = 1.0,
    audio_temperature: float = 0.8,
    audio_top_k: int = 25,
    audio_top_p: float = 0.95,
    audio_repetition_penalty: float = 1.2,
    max_tokens: int = 375,
    voice_clone_max_text_tokens: int = 75,
    initial_frames: int = 2,
    chunk_frames: int = 6,
) -> Iterator[mx.array]:
    """Yield float32 waveform chunks shaped [samples, channels].

    ``ref_audio`` is a file path or decoded waveform at the model's sample rate.
    Set ``do_sample=False`` for greedy generation.
    ``max_tokens`` caps audio frames per text segment; each segment gets fresh state.
    Decode the first ``initial_frames`` frames, then every ``chunk_frames`` frames.
    The decoder and pending codes span text segments; AR state resets per segment.
    """
    from mlx_audio.tts.models.moss_tts_nano.sampling import (
        sample_assistant_text_token,
        sample_next_token,
    )
    from mlx_audio.tts.models.moss_tts_nano.text import (
        lightweight_normalize_text,
        split_text_into_best_sentences,
    )

    if initial_frames < 1 or chunk_frames < 1:
        raise ValueError("initial_frames and chunk_frames must be positive")

    reference_codes = model.encode_reference_audio(
        ref_audio, sample_rate=model.sample_rate
    )
    chunks = split_text_into_best_sentences(
        model.tokenizer,
        lightweight_normalize_text(text),
        max_tokens=voice_clone_max_text_tokens,
    )
    decoder = model.audio_tokenizer.make_streaming_decoder(num_quantizers=model.config.n_vq)
    pending: list[mx.array] = []
    first_chunk = True
    for chunk in chunks:
        prompt_input_ids, attention_mask = model.build_inference_input_ids(
            text=chunk,
            tokenizer=model.tokenizer,
            mode="voice_clone",
            prompt_audio_codes=reference_codes,
        )

        cache = model.transformer.make_cache()
        current_model_input_ids = prompt_input_ids
        current_attention_mask = attention_mask.astype(mx.bool_)
        generated_frames: list[mx.array] = []

        for _ in range(max_tokens):
            # 1. Read the full prompt once, then only the previous frame with the same cache.
            inputs_embeds = model._build_inputs_embeds(current_model_input_ids)
            global_outputs = model.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=current_attention_mask,
                cache=cache,
            )
            global_hidden = global_outputs[:, -1, :]

            # 2. Start fresh local context for this frame and choose continue or stop.
            local_inputs_embeds = global_hidden[:, None, :]
            local_outputs = model.local_transformer(inputs_embeds=local_inputs_embeds)
            text_logits = model._text_lm_head(local_outputs[:, -1, :])
            next_text_token = sample_assistant_text_token(
                text_logits,
                audio_assistant_slot_token_id=model.config.audio_assistant_slot_token_id,
                audio_end_token_id=model.config.audio_end_token_id,
                do_sample=do_sample,
                temperature=text_temperature,
                top_k=text_top_k,
                top_p=text_top_p,
            )
            mx.eval(next_text_token)
            if next_text_token.item() != model.config.audio_assistant_slot_token_id:
                break

            # 3. Generate each codebook token, penalizing its history from earlier frames.
            current_local_input = model.transformer.wte(next_text_token)
            frame_tokens = []
            history = mx.stack(generated_frames, axis=1) if generated_frames else None
            for channel_index in range(model.config.n_vq):
                local_inputs_embeds = mx.concatenate(
                    [local_inputs_embeds, current_local_input[:, None, :]], axis=1
                )
                local_outputs = model.local_transformer(inputs_embeds=local_inputs_embeds)
                channel_logits = model._audio_lm_head(local_outputs[:, -1, :], channel_index)
                previous_tokens = None if history is None else history[:, :, channel_index]
                channel_token = sample_next_token(
                    channel_logits,
                    do_sample=do_sample,
                    temperature=audio_temperature,
                    top_k=audio_top_k,
                    top_p=audio_top_p,
                    previous_token_ids=previous_tokens,
                    repetition_penalty=audio_repetition_penalty,
                )
                frame_tokens.append(channel_token)
                current_local_input = model.audio_embeddings[channel_index](channel_token)

            frame = mx.stack(frame_tokens, axis=-1)  # [1, n_vq]
            mx.eval(frame)
            generated_frames.append(frame)

            # 4. Feed [continue, codebook 0, ..., codebook 15] into the next global step.
            text_column = next_text_token[:, None, None]
            current_model_input_ids = mx.concatenate([text_column, frame[:, None, :]], axis=-1)
            current_attention_mask = mx.concatenate(
                [current_attention_mask, mx.ones((frame.shape[0], 1), dtype=mx.bool_)], axis=1
            )
            # 5. Decode only new codes, keeping the same decoder for this request.
            pending.append(frame)
            boundary = initial_frames if first_chunk else chunk_frames
            if len(pending) >= boundary:
                waveform = decoder.decode_frames(mx.concatenate(pending, axis=0))
                pending.clear()
                first_chunk = False
                yield waveform

    if pending:
        yield decoder.decode_frames(mx.concatenate(pending, axis=0))


def synthesize_clone_stream(
    model: Any,
    config: MossNanoConfig,
    text: str,
    ref_audio: mx.array,
) -> Iterator[mx.array]:
    """Apply config, convert chunk durations to frames, and yield waveform chunks."""
    yield from generate_frames(
        model,
        text=text,
        ref_audio=ref_audio,
        max_tokens=config.max_tokens,
        do_sample=config.do_sample,
        text_temperature=config.text_temperature,
        text_top_k=config.text_top_k,
        text_top_p=config.text_top_p,
        audio_temperature=config.audio_temperature,
        audio_top_k=config.audio_top_k,
        audio_top_p=config.audio_top_p,
        audio_repetition_penalty=config.audio_repetition_penalty,
        voice_clone_max_text_tokens=config.voice_clone_max_text_tokens,
        initial_frames=frames_for_interval(config.streaming_initial_interval),
        chunk_frames=frames_for_interval(config.streaming_interval),
    )
