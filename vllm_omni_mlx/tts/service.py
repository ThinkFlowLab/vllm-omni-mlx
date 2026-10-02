"""TTS serving service (M1.7, #16): serialized speech synthesis for the app.

Wraps a loaded TTS model with the single-user lock (batch-1 target) and
request-level validation: unknown voices, empty input, and speed ≠ 1.0 are
ValueErrors the server maps to 400s. Output formats match the OpenAI audio
API: `wav` (16-bit PCM mono RIFF) and `pcm` (raw 16-bit LE mono, 24 kHz).
"""

from __future__ import annotations

import threading
from array import array
from typing import Any, Iterator, Optional

import mlx.core as mx

from .config import TTSConfig
from .generate import synthesize, wav_bytes
from .prompt_embeds import PromptEmbeds


def _pcm16(chunk: mx.array) -> bytes:
    """One float chunk → raw little-endian 16-bit mono bytes (OpenAI 'pcm')."""
    samples = mx.clip(chunk.reshape(-1), -1.0, 1.0)
    return array("h", (samples * 32767.0).astype(mx.int16).tolist()).tobytes()


class TTSService:
    def __init__(self, model: Any, config: TTSConfig | None = None):
        self._model = model
        self.config = config or TTSConfig()
        self._embeds = PromptEmbeds(model)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.config.model_ref

    @property
    def voices(self) -> list[str]:
        return self._embeds.speakers

    def speech_bytes(
        self,
        input: str,
        voice: Optional[str] = None,
        response_format: str = "wav",
        speed: float = 1.0,
        instructions: Optional[str] = None,
        language: Optional[str] = None,
    ) -> tuple[bytes, str]:
        """Synthesize `input` to (payload, content_type). Raises ValueError on
        invalid requests; generation is serialized under the service lock."""
        overrides = self._validated_overrides(input, voice, speed, instructions, language)
        with self._lock:
            chunks = synthesize(self._model, self.config, input, **overrides)
            if response_format == "wav":
                return wav_bytes(chunks), "audio/wav"
            return b"".join(_pcm16(chunk) for chunk in chunks), "audio/pcm"

    def speech_stream(
        self,
        input: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        instructions: Optional[str] = None,
        language: Optional[str] = None,
        streaming_interval: Optional[float] = None,
    ) -> "Iterator[bytes]":
        """Yield 16-bit PCM mono chunks (24 kHz) as they are generated.

        The service lock is held for the whole stream (batch-1); a client
        disconnect closes the generator and releases it at the next chunk.
        streaming_interval defaults to 0.5 s — the measured knee where first
        audio lands under ~0.5 s; smaller buys faster first audio at the cost
        of choppier cadence (total RTF is interval-independent, ~0.9 on the
        4-bit model — see #39's sweep).
        """
        overrides = self._validated_overrides(input, voice, speed, instructions, language)
        interval = 0.5 if streaming_interval is None else streaming_interval
        if not 0.0 < interval <= 10.0:
            raise ValueError("streaming_interval must be in (0, 10] seconds")
        overrides["streaming_interval"] = interval

        def stream() -> Iterator[bytes]:
            with self._lock:
                for chunk in synthesize(self._model, self.config, input, **overrides):
                    yield _pcm16(chunk)

        return stream()

    def _validated_overrides(self, input, voice, speed, instructions, language) -> dict:
        if not input or not input.strip():
            raise ValueError("input must be a non-empty string")
        if speed != 1.0:
            raise ValueError("speed != 1.0 is not supported yet")
        speaker = (voice or self.config.speaker).lower()
        if speaker not in self.voices:
            raise ValueError(f"voice '{voice}' is not one of the preset voices")
        overrides = {"speaker": speaker}
        if instructions:
            overrides["instruct"] = instructions
        if language:
            overrides["language"] = language
        return overrides
