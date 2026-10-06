"""Shared audio decode for request payloads (TTS reference clips, ASR uploads).

One decode+resample path so every endpoint preprocesses audio the same way:
mlx-audio's own ``audio_io.read`` (miniaudio/ffmpeg backed), downmixed to mono
float32 at the rate the consuming model expects.
"""

from __future__ import annotations

import io

import mlx.core as mx


def decode_audio(data: bytes, sample_rate: int, label: str = "audio") -> mx.array:
    """Encoded audio bytes → mono float32 waveform at ``sample_rate``.

    ``label`` names the request field in error messages. Raises ValueError on
    non-audio payloads or a clip that decodes to zero samples.
    """
    try:
        from mlx_audio.audio_io import read as audio_read

        samples, _ = audio_read(io.BytesIO(data), dtype="float32", sample_rate=sample_rate, nchannels=1)
    except Exception as exc:
        raise ValueError(f"{label} could not be decoded as audio: {exc}") from None
    if samples.size == 0:
        raise ValueError(f"{label} decoded to zero samples")
    return mx.array(samples, dtype=mx.float32)
