"""TTS serving service (M1.7, #16): serialized speech synthesis for the app.

Wraps a loaded TTS model with the single-user lock (batch-1 target) and
request-level validation: unknown voices, empty input, and speed ≠ 1.0 are
ValueErrors the server maps to 400s. Output formats match the OpenAI audio
API: `wav` (16-bit PCM mono RIFF) and `pcm` (raw 16-bit LE mono, 24 kHz).
"""

from __future__ import annotations

import threading
from array import array
from typing import Any, Optional

import mlx.core as mx

from .config import TTSConfig
from .generate import synthesize, wav_bytes
from .prompt_embeds import PromptEmbeds


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
        if not input or not input.strip():
            raise ValueError("input must be a non-empty string")
        if response_format not in ("wav", "pcm"):
            raise ValueError(f"response_format must be 'wav' or 'pcm', got '{response_format}'")
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

        with self._lock:
            chunks = synthesize(self._model, self.config, input, **overrides)
            if response_format == "wav":
                return wav_bytes(chunks), "audio/wav"
            # raw little-endian 16-bit mono at 24 kHz (OpenAI 'pcm' shape)
            parts = []
            for chunk in chunks:
                samples = mx.clip(chunk.reshape(-1), -1.0, 1.0)
                parts.append(array("h", (samples * 32767.0).astype(mx.int16).tolist()).tobytes())
            return b"".join(parts), "audio/pcm"
