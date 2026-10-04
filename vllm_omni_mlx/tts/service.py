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
from .generate import (
    MAX_REF_SECONDS,
    decode_ref_audio,
    synthesize,
    synthesize_clone,
    synthesize_design,
    wav_bytes,
)
from .prompt_embeds import PromptEmbeds
from .stream_loop import synthesize_clone_stream, synthesize_stream
from .variants import BASE, SMALL_SIZE, VOICE_DESIGN, model_size, model_variant, require_served

# stream-path chunk default (#40's measured knee; the buffered path keeps
# TTSConfig.streaming_interval — chunking is irrelevant when joining)
DEFAULT_STREAM_INTERVAL = 0.5


def _pcm16(chunk: mx.array) -> bytes:
    """One float chunk → raw little-endian 16-bit mono bytes (OpenAI 'pcm')."""
    samples = mx.clip(chunk.reshape(-1), -1.0, 1.0)
    return array("h", (samples * 32767.0).astype(mx.int16).tolist()).tobytes()


class TTSService:
    def __init__(self, model: Any, config: TTSConfig | None = None):
        self._model = model
        self.config = config or TTSConfig()
        self._variant = model_variant(model)  # unknown types fail at boot
        self._model_size = model_size(model)
        self._embeds = PromptEmbeds(model)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.config.model_ref

    @property
    def model_type(self) -> str:
        """The checkpoint's ``tts_model_type`` (see tts/variants.py)."""
        return self._variant

    @property
    def voices(self) -> list[str]:
        return self._embeds.speakers

    def speech_bytes(
        self,
        input: str,
        voice: "str | dict | None" = None,
        response_format: str = "wav",
        speed: float = 1.0,
        instructions: Optional[str] = None,
        language: Optional[str] = None,
    ) -> tuple[bytes, str]:
        """Synthesize `input` to (payload, content_type). `voice` is a preset
        speaker name or — on Base checkpoints — a cloning object
        ``{"ref_audio": <base64>, "ref_text": "..."}`` (#49). On VoiceDesign
        checkpoints the voice comes from `instructions` and `voice` is
        rejected (#51). Raises ValueError on invalid requests; generation is
        serialized under the service lock."""
        if isinstance(voice, dict):
            ref_audio, ref_text = self._clone_inputs(input, voice, speed)
            with self._lock:
                chunks = synthesize_clone(self._model, self.config, input, ref_audio, ref_text)
                if response_format == "wav":
                    return wav_bytes(chunks), "audio/wav"
                return b"".join(_pcm16(chunk) for chunk in chunks), "audio/pcm"
        overrides = self._validated_overrides(input, voice, speed, instructions, language)
        entry = synthesize_design if self._variant == VOICE_DESIGN else synthesize
        with self._lock:
            chunks = entry(self._model, self.config, input, **overrides)
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
        streaming_initial_interval: Optional[float] = None,
    ) -> "Iterator[bytes]":
        """Yield 16-bit PCM mono chunks (24 kHz) as they are generated —
        preset speakers or (on Base checkpoints, #50) a cloning ``voice``
        object.

        The service lock is held for the whole stream (batch-1); a client
        disconnect closes the generator and releases it at the next chunk.
        streaming_interval defaults to 0.5 s — the measured knee where first
        audio lands under ~0.5 s; smaller buys faster first audio at the cost
        of choppier cadence (total RTF is interval-independent, ~0.9 on the
        4-bit model — see #39's sweep). streaming_initial_interval (default
        0.2 s) is the #39 fast path: the first chunk is emitted as soon as
        that much audio exists, independent of the steady chunk size, so
        TTFA is not floored by streaming_interval. For cloning, the
        reference clip is decoded and validated before the stream starts —
        request errors never wait on the lock.
        """
        if isinstance(voice, dict):
            ref_audio, ref_text = self._clone_inputs(input, voice, speed)
            interval = DEFAULT_STREAM_INTERVAL if streaming_interval is None else streaming_interval
            if not 0.0 < interval <= 10.0:
                raise ValueError("streaming_interval must be in (0, 10] seconds")
            overrides = {"streaming_interval": interval}
            if streaming_initial_interval is not None:
                if not 0.0 < streaming_initial_interval <= 10.0:
                    raise ValueError("streaming_initial_interval must be in (0, 10] seconds")
                overrides["streaming_initial_interval"] = streaming_initial_interval

            def clone_stream() -> Iterator[bytes]:
                with self._lock:
                    for chunk in synthesize_clone_stream(
                        self._model, self.config, input, ref_audio, ref_text, **overrides
                    ):
                        yield _pcm16(chunk)

            return clone_stream()
        if self._variant == VOICE_DESIGN:
            raise ValueError(
                "streaming VoiceDesign synthesis is not supported yet — #52; "
                "the buffered path (stream absent/false) serves it"
            )
        overrides = self._validated_overrides(input, voice, speed, instructions, language)
        interval = DEFAULT_STREAM_INTERVAL if streaming_interval is None else streaming_interval
        if not 0.0 < interval <= 10.0:
            raise ValueError("streaming_interval must be in (0, 10] seconds")
        overrides["streaming_interval"] = interval
        if streaming_initial_interval is not None:
            if not 0.0 < streaming_initial_interval <= 10.0:
                raise ValueError("streaming_initial_interval must be in (0, 10] seconds")
            overrides["streaming_initial_interval"] = streaming_initial_interval

        def stream() -> Iterator[bytes]:
            with self._lock:
                for chunk in synthesize_stream(self._model, self.config, input, **overrides):
                    yield _pcm16(chunk)

        return stream()

    @staticmethod
    def _require_valid_request(input: str, speed: float) -> None:
        """Checks shared by every generation path (#54 review nit: one copy,
        three callers)."""
        if not input or not input.strip():
            raise ValueError("input must be a non-empty string")
        if speed != 1.0:
            raise ValueError("speed != 1.0 is not supported yet")

    def _clone_inputs(self, input: str, voice: dict, speed: float) -> tuple["mx.array", str]:
        """Validate a cloning ``voice`` object and decode the reference clip
        (base64 → 24 kHz mono) outside the service lock — decode errors and
        caps are request errors, not generation failures."""
        self._require_valid_request(input, speed)
        require_served(self._variant, path="clone")
        unknown = set(voice) - {"ref_audio", "ref_text"}
        if unknown:
            raise ValueError(f"voice object supports ref_audio and ref_text, got {sorted(unknown)}")
        ref_text = voice.get("ref_text")
        if not isinstance(ref_text, str) or not ref_text.strip():
            raise ValueError("voice.ref_text (transcript of the reference clip) is required for cloning")
        if "ref_audio" not in voice:
            raise ValueError("voice.ref_audio (base64 audio) is required for cloning")
        ref_audio = decode_ref_audio(voice["ref_audio"])
        duration = ref_audio.size / 24000
        if duration > MAX_REF_SECONDS:
            raise ValueError(f"ref_audio is {duration:.1f}s; the cap is {MAX_REF_SECONDS:.0f}s")
        if duration < 0.5:
            raise ValueError(f"ref_audio is {duration:.2f}s; a cloning reference needs at least 0.5s of speech")
        return ref_audio, ref_text

    def _validated_overrides(self, input, voice, speed, instructions, language) -> dict:
        self._require_valid_request(input, speed)
        if self._variant == VOICE_DESIGN:
            # the description IS the voice: no preset speakers exist on this
            # checkpoint type (#51; mlx-audio maps instructions → instruct)
            if voice is not None:
                raise ValueError(
                    "VoiceDesign checkpoints have no preset voices — the "
                    "voice comes from `instructions` (a text description)"
                )
            if not instructions or not instructions.strip():
                raise ValueError(
                    "`instructions` is required on VoiceDesign checkpoints — "
                    "a voice description like 'A cheerful young female voice "
                    "with high pitch and energetic tone'"
                )
            overrides = {"instruct": instructions.strip()}
        elif self._variant == BASE:
            require_served(self._variant)  # → guidance to the cloning shape
        else:
            speaker = (voice or self.config.speaker).lower()
            if speaker not in self.voices:
                raise ValueError(f"voice '{voice}' is not one of the preset voices")
            if instructions and self._model_size == SMALL_SIZE:
                # mlx-audio's own 0.6B instruct guard is dead code; rejecting
                # here keeps the behavior deterministic instead of hoping the
                # small model handles a prompt it wasn't trained for
                raise ValueError(
                    "instructions (emotion/style) need a 1.7B CustomVoice model; "
                    "the 0.6B model was not trained for them"
                )
            overrides = {"speaker": speaker}
            if instructions:
                overrides["instruct"] = instructions
        if language:
            overrides["language"] = language
        return overrides
