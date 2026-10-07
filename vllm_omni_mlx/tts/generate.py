"""Generation engine (M1.6, #15) — adapting mlx-audio (MIT).

Drives mlx-audio's CustomVoice generation (prefill → AR decode with talker +
MTP code predictor per frame → codes → code2wav) behind our config type,
yielding 24 kHz mono audio chunks. EOS is the checkpoint's
``codec_eos_token_id`` (mlx-audio's loop); sampling defaults live on
:class:`TTSConfig` and match the vllm-omni deploy-yaml defaults
(temp 0.9 / top-k 50 / top-p 1.0 / rep-penalty 1.05); pass
``temperature=0`` for greedy.

#49 adds the Base voice-cloning entry (:func:`synthesize_clone`) on
mlx-audio's ICL path — buffered only; the streaming clone loop is #50.
#51 adds the VoiceDesign entry (:func:`synthesize_design`) on
``generate_voice_design`` — the description rides ``instruct``; buffered
only, the streaming design loop is #52.
"""

from __future__ import annotations

import base64
import binascii
import io
import wave
from typing import Any, Iterator

import mlx.core as mx

from ..audio_io import decode_audio
from .config import TTSConfig
from .variants import ensure_served

# reference-audio cap: mlx-audio's ICL prefill pays per ref frame; beyond
# this the prompt grows without cloning benefit (upstream suggests seconds,
# not minutes)
MAX_REF_SECONDS = 30.0


def synthesize(model: Any, config: TTSConfig, text: str, **overrides) -> Iterator[mx.array]:
    """Yield audio chunks (mx.array, 24 kHz mono float) for `text`.

    Overrides follow TTSConfig.with_overrides semantics; `seed` sets the
    global MLX RNG before generation for reproducibility.
    """
    ensure_served(model)
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


def synthesize_design(model: Any, config: TTSConfig, text: str, **overrides) -> Iterator[mx.array]:
    """Yield audio chunks (mx.array, 24 kHz mono float) for `text`, in the
    voice described by ``instruct`` — VoiceDesign checkpoints only, on
    mlx-audio's ``generate_voice_design`` (the same instruct-track loop as
    CustomVoice with the speaker row absent — qwen3_tts.py:2143 delegates
    to ``_generate_with_instruct(speaker=None)``). Buffered: chunking
    follows :class:`TTSConfig` as in :func:`synthesize`.
    """
    ensure_served(model, path="design")
    cfg = config.with_overrides(**overrides)
    if not (cfg.instruct or "").strip():
        raise ValueError(
            "VoiceDesign synthesis needs `instruct` — a voice description "
            "like 'A cheerful young female voice with high pitch and "
            "energetic tone' (#46)"
        )
    if overrides.get("seed") is not None:
        mx.random.seed(int(overrides["seed"]))
    stream = cfg.max_tokens > 0  # always stream; final chunk carries the tail
    for result in model.generate_voice_design(
        text=text,
        instruct=cfg.instruct,
        language=cfg.language,
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


def decode_ref_audio(data: str | bytes) -> mx.array:
    """Base64 (or raw) audio bytes → 24 kHz mono float32 waveform.

    Uses mlx-audio's own decode+resample (``audio_io.read``, miniaudio/ffmpeg
    backed) so a reference clip is preprocessed exactly as
    ``load_audio(path)`` would — mlx-audio passes mx.arrays through as-is,
    so the resample is on us. Raises ValueError on non-audio payloads.
    """
    if isinstance(data, str):
        try:
            data = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"ref_audio must be base64-encoded audio: {exc}") from None
    return decode_audio(data, 24000, "ref_audio")


def synthesize_clone(
    model: Any,
    config: TTSConfig,
    text: str,
    ref_audio: mx.array,
    ref_text: str,
    **overrides,
) -> Iterator[mx.array]:
    """Yield audio chunks (mx.array, 24 kHz mono float) for `text`, in the
    voice cloned from (`ref_audio`, `ref_text`) — Base checkpoints only, on
    mlx-audio's ICL path (`generate(ref_audio=…, ref_text=…)` →
    `_generate_icl`). Buffered: one chunk carries the full clip. mlx-audio
    raises the floor on repetition penalty to 1.5 on this path itself.
    """
    ensure_served(model, path="clone")
    cfg = config.with_overrides(**overrides)
    duration = ref_audio.size / 24000
    if duration > MAX_REF_SECONDS:
        raise ValueError(f"ref_audio is {duration:.1f}s; the cap is {MAX_REF_SECONDS:.0f}s")
    if overrides.get("seed") is not None:
        mx.random.seed(int(overrides["seed"]))
    for result in model.generate(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
        max_tokens=cfg.max_tokens,
        verbose=False,
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
