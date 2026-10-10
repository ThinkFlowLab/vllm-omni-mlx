"""
MOSS-TTS-Nano voice cloning through mlx-audio (#73).

Nano has its own GPT2/codebook pipeline and 48 kHz stereo codec.
The public audio contract is explicitly downmixed
48 kHz mono PCM16. Reference waveforms retain their channels and are
resampled to the codec rate before being passed to mlx-audio.
"""

from __future__ import annotations

import base64
import concurrent.futures
import contextlib
import io
import queue
import sys
import threading
import wave
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import mlx.core as mx
import numpy as np

from .moss_nano_loop import synthesize_stream, validate_streaming_codec

DEFAULT_MODEL = "mlx-community/MOSS-TTS-Nano-100M"
DEFAULT_CODEC_MODEL = "mlx-community/MOSS-Audio-Tokenizer-Nano"
SAMPLE_RATE = 48000

#: HF ``config.json`` → ``architecture`` value identifying the family
ARCH = "moss_tts_nano"

#: mlx-audio module prefix of the loaded model class
_MODULE_PREFIX = "mlx_audio.tts.models.moss_tts_nano"

MAX_REF_SECONDS = 30.0
MIN_REF_SECONDS = 0.5

# PCM16 wire format is little-endian; mx.array exposes native-endian buffers.
if sys.byteorder != "little":
    raise ImportError("MOSS Nano PCM16 output requires a little-endian host")


@dataclass(frozen=True)
class MossNanoConfig:
    """Nano defaults match mlx-audio 0.5.7, including both samplers.

    Streaming intervals measure audio duration and are rounded down to whole
    codec frames (at least one); geometry comes from the loaded codec. The
    generation budget applies per text segment, as in the library generator.
    """

    model_ref: str = DEFAULT_MODEL
    codec_model_ref: str | None = None
    max_new_frames: int = 375
    max_text_tokens: int = 75
    do_sample: bool = True
    text_temperature: float = 1.0
    text_top_p: float = 1.0
    text_top_k: int = 50
    audio_temperature: float = 0.8
    audio_top_p: float = 0.95
    audio_top_k: int = 25
    audio_repetition_penalty: float = 1.2
    seed: int | None = None

    streaming_interval: float = 0.5
    streaming_initial_interval: float = 0.08

    def __post_init__(self) -> None:
        for name in ("max_new_frames", "max_text_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def _all_finite(x: mx.array) -> bool:
    """True when x has no NaN or ±Inf values."""
    bad = mx.logical_or(mx.isnan(x), mx.isinf(x))
    return not mx.any(bad).item()


def _validate_model(model: Any) -> None:
    variant = getattr(getattr(model, "config", None), "model_type", None)
    if variant != "moss_tts_nano":
        raise ValueError(f"MOSS Nano needs model_type='moss_tts_nano', got {variant!r}")
    if model.sample_rate != SAMPLE_RATE:
        raise ValueError(f"MOSS Nano needs a 48000 Hz codec, got {model.sample_rate!r}")


def load_moss_nano_model(config: MossNanoConfig) -> Any:
    """Load the language model, tokenizer and MLX codec before serving. Requires the [tts] extra."""
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError as exc:
        raise RuntimeError(
            "MOSS Nano needs mlx-audio 0.5.7; install 'vllm-omni-mlx[tts]'"
        ) from exc
    model = load_model(config.model_ref)
    _validate_model(model)

    # Preload the codec
    model._ensure_audio_tokenizer(source=config.codec_model_ref)
    return model


def decode_ref_audio(data: str, sample_rate: int = SAMPLE_RATE) -> mx.array:
    """Decode base64 audio at the codec rate, preserving mono/stereo layout.

    Independent of Qwen's 24 kHz reference helper. Duration counts frames,
    not interleaved channel values.
    """
    # Outside the try: a missing [tts] extra is a server config error, not a bad request.
    from mlx_audio.audio_io import read as audio_read

    if not isinstance(data, str) or not data:
        raise ValueError("voice.ref_audio must be non-empty base64-encoded audio")
    try:
        payload = base64.b64decode(data, validate=True)
        raw, rate = audio_read(
            io.BytesIO(payload), dtype="float32", sample_rate=sample_rate
        )
        samples = mx.array(raw, dtype=mx.float32)
    except Exception as exc:
        raise ValueError(
            f"ref_audio could not be decoded as base64 audio: {exc}"
        ) from None

    if rate != sample_rate:
        raise ValueError(
            f"ref_audio decoder returned {rate} Hz, expected {sample_rate} Hz"
        )
    if samples.ndim not in (1, 2) or (
        samples.ndim == 2 and samples.shape[1] not in (1, 2)
    ):
        raise ValueError("ref_audio must contain mono or stereo audio")
    if not _all_finite(samples):
        raise ValueError("ref_audio contains non-finite samples")
    duration = samples.shape[0] / sample_rate
    if not MIN_REF_SECONDS <= duration <= MAX_REF_SECONDS:
        raise ValueError(
            f"ref_audio is {duration:.2f}s; must be {MIN_REF_SECONDS:g}-{MAX_REF_SECONDS:g}s"
        )
    return samples


def _pcm16(audio: Any) -> bytes:
    """Sample-major float audio in [-1, 1] → mono little-endian PCM16 (stereo is averaged)."""
    samples = (
        audio.astype(mx.float32)
        if isinstance(audio, mx.array)
        else mx.array(audio, dtype=mx.float32)
    )
    if samples.ndim == 2 and samples.shape[1] in (1, 2):
        samples = mx.mean(samples, axis=1)
    elif samples.ndim != 1:
        raise RuntimeError(
            f"MOSS Nano returned an unsupported audio shape {tuple(samples.shape)}"
        )
    if not _all_finite(samples):
        raise RuntimeError("MOSS Nano returned non-finite audio samples")
    pcm = mx.round(mx.clip(samples, -1.0, 1.0) * 32767.0).astype(mx.int16)
    return np.asarray(pcm).astype("<i2", copy=False).tobytes()


class _PCMStream(Iterator[bytes]):
    """Lazy, bounded delivery from the service's single generation worker.

    Only the worker touches MLX or closes the synthesis generator. Consumers
    may cancel concurrently with ``next``; cancellation also unblocks a
    producer waiting for a slow client to drain the one-chunk queue.
    """

    def __init__(
        self,
        pool: concurrent.futures.ThreadPoolExecutor,
        generate: Callable[[threading.Event], Iterator[bytes]],
    ):
        self._pool = pool
        self._generate = generate
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._state_lock = threading.Lock()
        self._future: concurrent.futures.Future | None = None
        self._error: Exception | None = None

    def _produce(self) -> None:
        try:
            with contextlib.closing(self._generate(self._cancel)) as chunks:
                for chunk in chunks:
                    while not self._cancel.is_set():
                        try:
                            self._queue.put(chunk, timeout=0.05)
                            break
                        except queue.Full:
                            continue
                    if self._cancel.is_set():
                        break
        except Exception as exc:
            self._error = exc
        finally:
            self._done.set()

    def __next__(self) -> bytes:
        with self._state_lock:
            if self._cancel.is_set():
                raise StopIteration
            if self._future is None:
                self._future = self._pool.submit(self._produce)
        while not self._cancel.is_set():
            try:
                chunk = self._queue.get(timeout=0.05)
                if self._cancel.is_set():
                    raise StopIteration
                return chunk
            except queue.Empty:
                if self._done.is_set():
                    # A final put can race with get() timing out. Once done is
                    # set there are no more writes, so drain before end/error.
                    try:
                        chunk = self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        if self._cancel.is_set():
                            raise StopIteration
                        return chunk
                    error, self._error = self._error, None
                    self.close()
                    if error is not None:
                        raise error
                    raise StopIteration
        raise StopIteration

    def close(self) -> None:
        with self._state_lock:
            self._cancel.set()
            if self._future is None or self._future.cancel():
                self._done.set()

    def cancel(self) -> None:
        """Nonblocking cancellation hook for the HTTP audio adapter."""
        self.close()


class MossNanoService:
    """Batch-1, cloning-only speech service; request validation is eager."""

    sample_rate = SAMPLE_RATE
    channels = 1
    model_type = "moss_tts_nano"

    def __init__(self, model: Any, config: MossNanoConfig | None = None):
        _validate_model(model)
        self._model = model
        self.config = config or MossNanoConfig()
        self._lock = threading.Lock()
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="moss-nano-gen"
        )

    @property
    def name(self) -> str:
        return self.config.model_ref

    @property
    # clone-only so far
    def voices(self) -> list[str]:
        return []

    def _clone_inputs(
        self,
        input: str,
        voice: Any,
        speed: float,
        instructions: str | None,
        language: str | None,
    ) -> mx.array:
        if not isinstance(input, str) or not input.strip():
            raise ValueError("input must be a non-empty string")
        if speed != 1.0:
            raise ValueError("speed != 1.0 is not supported yet")
        if instructions:
            raise ValueError(
                "instructions are not supported by MOSS Nano voice cloning"
            )
        if language not in (None, "", "auto"):
            raise ValueError(
                "language is selected from the input text by MOSS Nano; use 'auto'"
            )
        if not isinstance(voice, dict):
            raise ValueError(
                "MOSS Nano has no preset voices; voice must be a cloning object "
                "with ref_audio (base64 audio); ref_text is optional and ignored"
            )
        unknown = set(voice) - {"ref_audio", "ref_text"}
        if unknown:
            raise ValueError(
                f"voice object supports ref_audio and ref_text, got {sorted(unknown)}"
            )
        if voice.get("ref_text") is not None and not isinstance(voice["ref_text"], str):
            raise ValueError(
                "voice.ref_text must be a string when provided (MOSS Nano ignores it)"
            )
        if "ref_audio" not in voice:
            raise ValueError(
                "voice.ref_audio (base64 audio) is required for MOSS Nano cloning"
            )
        # Parsing happens outside the generation lock and before HTTP headers.
        return decode_ref_audio(voice["ref_audio"], sample_rate=self.sample_rate)

    def _require_open(self) -> Any:
        if self._model is None:
            raise RuntimeError("MOSS Nano service is closed")
        return self._model

    def _buffered_chunks(self, text: str, ref_audio: mx.array) -> Iterator[bytes]:
        cfg = self.config
        results = self._require_open().generate(
            text=text,
            ref_audio=ref_audio,
            ref_audio_sample_rate=self.sample_rate,
            mode="voice_clone",
            stream=False,
            max_tokens=cfg.max_new_frames,
            voice_clone_max_text_tokens=cfg.max_text_tokens,
            do_sample=cfg.do_sample,
            text_temperature=cfg.text_temperature,
            text_top_p=cfg.text_top_p,
            text_top_k=cfg.text_top_k,
            audio_temperature=cfg.audio_temperature,
            audio_top_p=cfg.audio_top_p,
            audio_top_k=cfg.audio_top_k,
            audio_repetition_penalty=cfg.audio_repetition_penalty,
            audio_tokenizer_source=cfg.codec_model_ref or DEFAULT_CODEC_MODEL,
        )
        try:
            for result in results:
                if result.sample_rate != self.sample_rate:
                    raise RuntimeError(
                        f"MOSS Nano returned {result.sample_rate} Hz audio, expected {self.sample_rate} Hz"
                    )
                if result.audio is not None and result.audio.size:
                    yield _pcm16(result.audio)
        finally:
            close = getattr(results, "close", None)
            if close is not None:
                close()

    def speech_bytes(
        self,
        input: str,
        voice: str | dict | None = None,
        response_format: str = "wav",
        speed: float = 1.0,
        instructions: str | None = None,
        language: str | None = None,
    ) -> tuple[bytes, str]:
        if response_format not in ("wav", "pcm"):
            raise ValueError("response_format must be 'wav' or 'pcm'")
        ref_audio = self._clone_inputs(input, voice, speed, instructions, language)

        def generate() -> bytes:
            with self._lock:
                if self.config.seed is not None:
                    mx.random.seed(self.config.seed)
                return b"".join(self._buffered_chunks(input, ref_audio))

        pcm = self._pool.submit(generate).result()
        if not pcm:
            raise RuntimeError("MOSS Nano generated no audio")
        if response_format == "pcm":
            return pcm, "audio/pcm"
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm)
        return buffer.getvalue(), "audio/wav"

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
        """Yield PCM while Nano is generating, with independent codec state.

        Validation is eager. Generation and conversion run on the same worker
        as buffered requests, preserving batch-1 semantics. Closing the returned
        iterator cancels work at the next frame boundary and releases the lock.
        """

        ref_audio = self._clone_inputs(input, voice, speed, instructions, language)
        interval = (
            self.config.streaming_interval
            if streaming_interval is None
            else streaming_interval
        )
        initial = (
            self.config.streaming_initial_interval
            if streaming_initial_interval is None
            else streaming_initial_interval
        )
        for name, value in (
            ("streaming_interval", interval),
            ("streaming_initial_interval", initial),
        ):
            if not 0.0 < value <= 10.0:
                raise ValueError(f"{name} must be in (0, 10] seconds")

        validate_streaming_codec(self._model)

        def generate(cancel: threading.Event) -> Iterator[bytes]:
            if cancel.is_set():
                return
            with self._lock:
                if cancel.is_set():
                    return
                if self.config.seed is not None:
                    mx.random.seed(self.config.seed)
                emitted = False
                with contextlib.closing(
                    synthesize_stream(
                        self._model,
                        self.config,
                        input,
                        ref_audio,
                        streaming_interval=interval,
                        streaming_initial_interval=initial,
                        cancel=cancel,
                    )
                ) as chunks:
                    for audio in chunks:
                        if cancel.is_set():
                            return
                        if audio is not None and audio.size:
                            pcm = _pcm16(audio)
                            emitted = True
                            yield pcm
                if not emitted and not cancel.is_set():
                    raise RuntimeError("MOSS Nano generated no audio")

        return _PCMStream(self._pool, generate)
