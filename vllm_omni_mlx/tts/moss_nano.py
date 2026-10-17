import base64
import binascii
import threading
from array import array
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, replace
from typing import Any

import mlx.core as mx

from ..audio_io import decode_audio
from .generate import wav_bytes as _wav_bytes

DEFAULT_MODEL = "mlx-community/MOSS-TTS-Nano-100M"
MIN_REF_SECONDS = 0.5
MAX_REF_SECONDS = 30.0


@dataclass(frozen=True)
class MossNanoConfig:
    """MOSS-Nano generation and streaming options."""

    model_ref: str = DEFAULT_MODEL
    max_tokens: int = 375  # Maximum new audio frames per text segment.
    do_sample: bool = True
    text_temperature: float = 1.0
    text_top_k: int = 50
    text_top_p: float = 1.0
    audio_temperature: float = 0.8
    audio_top_k: int = 25
    audio_top_p: float = 0.95
    audio_repetition_penalty: float = 1.2
    voice_clone_max_text_tokens: int = 75  # Text-token budget used for splitting.
    # Target audio chunk durations in seconds for synthesize_clone_stream.
    streaming_interval: float = 2.0
    streaming_initial_interval: float = 0.08


def decode_ref_audio(data: str | bytes) -> mx.array:
    """Base64 or encoded audio bytes → mono float32 at 48 kHz.
    Strings are base64 payloads, not paths. Invalid inputs raise ValueError.
    """
    if isinstance(data, str):
        try:
            data = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"ref_audio must be base64-encoded audio: {exc}") from None
    elif not isinstance(data, bytes):
        raise ValueError("ref_audio must be a base64 string or encoded audio bytes")

    return decode_audio(data, 48000, label="ref_audio")


def synthesize(
    model: Any,
    config: MossNanoConfig,
    text: str,
    ref_audio: str | mx.array,
) -> Iterator[mx.array]:
    results = model.generate(
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
    )

    for result in results:
        yield result.audio


def _mono(chunk: mx.array) -> mx.array:
    """Normalize [samples] or [samples, channels] audio to [samples]."""
    if chunk.ndim == 1:
        return chunk
    if chunk.ndim == 2 and chunk.shape[1] > 0:
        return mx.mean(chunk, axis=1)
    raise ValueError(
        f"expected audio shaped [samples] or [samples, channels], got {chunk.shape}"
    )


def _pcm16(chunk: mx.array) -> bytes:
    """Encode one float audio chunk as mono PCM16 (little-endian on Apple Silicon)."""
    samples = mx.clip(_mono(chunk), -1.0, 1.0)
    return array("h", (samples * 32767.0).astype(mx.int16).tolist()).tobytes()


def wav_bytes(chunks: Iterator[mx.array], sample_rate: int) -> bytes:
    """Assemble audio chunks into a PCM16 mono WAV at ``sample_rate``."""
    return _wav_bytes((_mono(chunk) for chunk in chunks), sample_rate)


def load_moss_nano_model(config: MossNanoConfig) -> Any:
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError as exc:
        raise RuntimeError(
            "TTS needs mlx-audio; install it with: pip install 'vllm-omni-mlx[tts]'"
        ) from exc
    return load_model(config.model_ref)


class MossNanoService:
    """Voice cloning: request validation, serialized generation, and audio encoding."""

    def __init__(self, model: Any, config: MossNanoConfig | None = None):
        self._model = model
        self.config = config or MossNanoConfig()
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.config.model_ref

    @property
    def model_type(self) -> str:
        return self._model.model_type

    @property
    def voices(self) -> list[str]:
        return []  # #74: Cloning only; no preset speakers.

    @property
    def sample_rate(self) -> int:
        return int(self._model.sample_rate)

    def speech_bytes(
        self,
        input: str,
        voice: str | dict | None = None,
        response_format: str = "wav",
        speed: float = 1.0,
        instructions: str | None = None,
        language: str | None = None,
    ) -> tuple[bytes, str]:
        """Return a complete mono PCM16 WAV or raw PCM and its content type.

        ``voice`` must contain ``ref_audio`` (base64 or encoded audio bytes).
        #74: Optional ``ref_text`` is accepted but ignored: voice cloning uses
        the reference audio alone. ``instructions`` and ``language`` are also
        accepted but ignored. The caller must supply non-empty text and a
        ``response_format`` of ``wav`` or ``pcm``.
        """
        if speed != 1.0:
            raise ValueError("speed != 1.0 is not supported yet")
        ref_audio = self._clone_inputs(voice)
        with self._lock:
            chunks = synthesize(self._model, self.config, input, ref_audio)
            if response_format == "wav":
                return wav_bytes(chunks, self.sample_rate), "audio/wav"
            return b"".join(_pcm16(chunk) for chunk in chunks), "audio/pcm"

    def speech_stream(
        self,
        input: str,
        voice: str | dict | None = None,
        speed: float = 1.0,
        instructions: str | None = None,
        language: str | None = None,
        streaming_interval: float | None = None,
        streaming_initial_interval: float | None = None,
    ) -> Iterator[bytes]:
        """Return mono PCM16 chunks at 48 kHz, validating before iteration starts.

        The generation lock is held until the stream ends or is closed.
        As with speech_bytes, ref_text, instructions, and language are ignored.
        """
        from .moss_nano_loop import synthesize_clone_stream

        if not isinstance(input, str) or not input.strip():
            raise ValueError("input must be a non-empty string")
        if speed != 1.0:
            raise ValueError("speed != 1.0 is not supported yet")
        overrides = {}
        for name, value in (
            ("streaming_interval", streaming_interval),
            ("streaming_initial_interval", streaming_initial_interval),
        ):
            value = getattr(self.config, name) if value is None else value
            if not 0.0 < value <= 10.0:
                raise ValueError(f"{name} must be in (0, 10] seconds")
            overrides[name] = value
        config = replace(self.config, **overrides)
        ref_audio = self._clone_inputs(voice)

        def stream() -> Iterator[bytes]:
            with self._lock:
                with closing(synthesize_clone_stream(self._model, config, input, ref_audio)) as chunks:
                    for chunk in chunks:
                        yield _pcm16(chunk)

        return stream()

    def _clone_inputs(self, voice: str | dict | None) -> mx.array:
        """Decode the cloning reference and enforce 0.5–30 seconds before locking."""
        if not isinstance(voice, dict):
            raise ValueError(
                "MOSS-Nano supports voice cloning only, with no preset voices; "
                "provide a voice object containing ref_audio"
            )
        if "ref_audio" not in voice:
            raise ValueError("voice.ref_audio is required for cloning")
        ref_audio = decode_ref_audio(voice["ref_audio"])
        duration = ref_audio.shape[0] / 48000
        if duration < MIN_REF_SECONDS:
            raise ValueError(
                f"ref_audio is {duration:.2f}s; clone's minimum is {MIN_REF_SECONDS}s"
            )
        if duration > MAX_REF_SECONDS:
            raise ValueError(
                f"ref_audio is {duration:.2f}s; clone's maximum is {MAX_REF_SECONDS}s"
            )
        return ref_audio
