"""Generation engine (M1.6, #15) — adapting mlx-audio (MIT).

Drives mlx-audio's CustomVoice generation (prefill → AR decode with talker +
MTP code predictor per frame → codes → code2wav) behind our config type,
yielding 24 kHz mono audio chunks. EOS is the checkpoint's
``codec_eos_token_id`` (mlx-audio's loop); sampling defaults live on
:class:`TTSConfig` and match the vllm-omni deploy-yaml defaults
(temp 0.9 / top-k 50 / top-p 1.0 / rep-penalty 1.05); pass
``temperature=0`` for greedy.
"""

from __future__ import annotations

import io
import wave
from typing import Any, Iterator

import mlx.core as mx

from .config import TTSConfig


def synthesize(model: Any, config: TTSConfig, text: str, **overrides) -> Iterator[mx.array]:
    """Yield audio chunks (mx.array, 24 kHz mono float) for `text`.

    Overrides follow TTSConfig.with_overrides semantics; `seed` sets the
    global MLX RNG before generation for reproducibility.
    """
    cfg = config.with_overrides(**overrides)
    if overrides.get("seed") is not None:
        mx.random.seed(int(overrides["seed"]))
    stream = cfg.max_tokens > 0  # always stream; final chunk carries the tail
    for result in model.generate_custom_voice(
        text=text,
        speaker=cfg.speaker,
        language=cfg.language,
        instruct=cfg.instruct,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
        max_tokens=cfg.max_tokens,
        stream=stream,
        streaming_interval=cfg.streaming_interval,
    ):
        if result.audio is not None and result.audio.size:
            yield result.audio


def wav_bytes(chunks: Iterator[mx.array], sample_rate: int = 24000) -> bytes:
    """Assemble float chunks into a 16-bit PCM mono WAV file."""
    from array import array

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for chunk in chunks:
            samples = mx.clip(chunk.reshape(-1), -1.0, 1.0)
            wav.writeframes(array("h", (samples * 32767.0).astype(mx.int16).tolist()).tobytes())
    return buffer.getvalue()
