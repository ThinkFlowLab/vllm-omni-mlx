"""VoxCPM2 serving seam (#71): presets-as-zero-shot + ref-audio cloning.

VoxCPM2 (OpenBMB, Apache-2.0; mlx-audio 0.5.7 port) is a 2.5B tokenizer-free
AR TTS — MiniCPM4 base LM over scalar-quantized latents, a residual LM, a
LocDiT CFM solver, and a causal 48 kHz AudioVAE. mlx-audio's ``generate`` is
**single-yield**: the whole AR loop runs, then all patches decode at once,
so this seam serves buffered audio and a stream that delivers the finished
buffer in interval-sized chunks — TTFA ≈ full synthesis time is the honest
#71 baseline; incremental per-patch decode (the AudioVAE decoder is causal,
so it is architecturally possible) is the follow-up loop work, as is
compiled decode (the issue keeps both out of scope).

Voice semantics differ from Qwen3-TTS: there are no speaker presets. The
model speaks zero-shot in its intrinsic voice (`voice: "default"`, matching
upstream vllm-omni's recipe), clones any uploaded clip via ``ref_audio``
(no transcript needed — mlx-audio's reference mode ignores ``ref_text``;
upstream drives cloning entirely by ref_audio too), and designs voices from
`instructions` (mlx-audio maps ``instruct`` onto a ``(description)text``
prompt). Sampling params (temperature/top-k/top-p/rep-penalty) don't apply:
the AR path embeds continuous features, and the only stochasticity is the
CFM noise; the knobs are ``inference_timesteps`` and ``cfg_value``.
"""

from __future__ import annotations

import base64
import binascii
import io
import threading
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import mlx.core as mx

from .generate import wav_bytes as _wav_bytes

DEFAULT_MODEL = "mlx-community/VoxCPM2-4bit"

#: HF ``config.json`` → ``architecture`` value identifying the family
ARCH = "voxcpm2"

#: mlx-audio module prefix of the loaded model class (ModelArgs carries no
#: architecture field, so post-load detection goes through the module)
_MODULE_PREFIX = "mlx_audio.tts.models.voxcpm2"

#: reference-clip caps, shared with the Qwen3-TTS clone path: mlx-audio
#: VAE-encodes the clip into the prompt (seconds, not minutes)
MAX_REF_SECONDS = 30.0
MIN_REF_SECONDS = 0.5

#: chunk size the stream path slices the finished buffer into (#40's
#: measured knee; irrelevant to generation itself)
DEFAULT_STREAM_INTERVAL = 0.5


@dataclass(frozen=True)
class VoxCPM2Config:
    """Serving defaults for VoxCPM2 — the checkpoint's own generate defaults
    (inference_timesteps 10, cfg 2.0, max 2000 patches ≈ 40 s of audio at
    ~20 ms per patch)."""

    model_ref: str = DEFAULT_MODEL
    instruct: str | None = None
    inference_timesteps: int = 10
    cfg_value: float = 2.0
    max_tokens: int = 2000
    warmup_patches: int = 0
    streaming_interval: float = DEFAULT_STREAM_INTERVAL

    def with_overrides(self, **overrides: Any) -> VoxCPM2Config:
        known = {
            k: v
            for k, v in overrides.items()
            if v is not None and k in VoxCPM2Config.__dataclass_fields__ and k != "model_ref"
        }
        return replace(self, **known) if known else self


def config_is_voxcpm2(config: dict) -> bool:
    """Pre-load peek: an HF ``config.json`` dict belongs to VoxCPM2."""
    return config.get("architecture") == ARCH


def is_voxcpm2_model(model: Any) -> bool:
    """Post-load check: the model class came from mlx-audio's voxcpm2 package."""
    return type(model).__module__.startswith(_MODULE_PREFIX)


def load_voxcpm2_model(config: VoxCPM2Config) -> Any:
    """Load VoxCPM2 through mlx-audio. Requires the [tts] extra."""
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError as exc:
        raise RuntimeError(
            "TTS needs mlx-audio; install it with: pip install 'vllm-omni-mlx[tts]'"
        ) from exc
    model = load_model(config.model_ref)
    _unpin_cpu_buffers(model)
    return model


def _unpin_cpu_buffers(model: Any) -> None:
    """Re-materialize non-parameter arrays left CPU-pinned by the load path.

    ``mx.load`` returns memory-mapped CPU arrays; the loader's
    ``mx.eval(model.parameters())`` moves every parameter to the GPU, but
    mlx-audio's voxcpm2 ``sanitize`` pops ``audio_vae.decoder._sr_boundaries``
    out of the weight dict as a module buffer *before* that — it stays a lazy
    CPU array only the loading thread can evaluate ("There is no
    Stream(cpu, 1) in current thread" the first time a server worker thread
    decodes audio). Rebuilding it here on the loading thread pins it to the
    GPU like every other tensor.
    """
    decoder = getattr(getattr(model, "audio_vae", None), "decoder", None)
    boundaries = getattr(decoder, "_sr_boundaries", None)
    if boundaries is not None:
        decoder._sr_boundaries = mx.array(boundaries.tolist(), dtype=boundaries.dtype)
        mx.eval(decoder._sr_boundaries)


def local_snapshot(model_ref: str) -> str | None:
    """Re-export of the weight-gate helper (see tts.config.local_snapshot)."""
    from .config import local_snapshot as _local_snapshot

    return _local_snapshot(model_ref)


def decode_ref_audio(data: str | bytes, sample_rate: int) -> mx.array:
    """Base64 (or raw) audio bytes → float32 mono waveform at ``sample_rate``.

    mlx-audio's ``_encode_wav`` treats a bare array as being at the model's
    output rate (48 kHz) and resamples down to its 16 kHz encode rate, so a
    request clip is decoded to the output rate here — the same
    decode+resample path (``audio_io.read``) the Qwen3-TTS clone seam uses.
    """
    if isinstance(data, str):
        try:
            data = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"ref_audio must be base64-encoded audio: {exc}") from None
    try:
        from mlx_audio.audio_io import read as audio_read

        samples, _ = audio_read(io.BytesIO(data), dtype="float32", sample_rate=sample_rate, nchannels=1)
    except Exception as exc:
        raise ValueError(f"ref_audio could not be decoded as audio: {exc}") from None
    if samples.size == 0:
        raise ValueError("ref_audio decoded to zero samples")
    return samples


def synthesize(
    model: Any,
    config: VoxCPM2Config,
    text: str,
    instruct: str | None = None,
    ref_audio: mx.array | None = None,
    ref_text: str | None = None,
) -> Iterator[mx.array]:
    """Yield audio chunks (mx.array, 48 kHz mono float) for `text`.

    One chunk: mlx-audio's generate is single-yield (see the module doc).
    ``ref_text`` is accepted for API parity and ignored — the reference
    mode conditions on the clip alone. ``instruct`` selects voice design
    (mlx-audio prepends ``(description)`` to the text).
    """
    cfg = config.with_overrides(instruct=instruct)
    for result in model.generate(
        text=text,
        instruct=cfg.instruct,
        ref_audio=ref_audio,
        ref_text=ref_text,
        max_tokens=cfg.max_tokens,
        inference_timesteps=cfg.inference_timesteps,
        cfg_value=cfg.cfg_value,
        warmup_patches=cfg.warmup_patches,
    ):
        if result.audio is not None and result.audio.size:
            yield result.audio


def _pcm16(chunk: mx.array) -> bytes:
    """One float chunk → raw little-endian 16-bit mono bytes."""
    from array import array

    samples = mx.clip(chunk.reshape(-1), -1.0, 1.0)
    return array("h", (samples * 32767.0).astype(mx.int16).tolist()).tobytes()


def wav_bytes(chunks: Iterator[mx.array], sample_rate: int) -> bytes:
    """Assemble float chunks into a 16-bit PCM mono WAV file at ``sample_rate``."""
    return _wav_bytes(chunks, sample_rate)


class VoxCPM2Service:
    """Serving surface for /v1/audio/* — duck-types :class:`TTSService`
    (name / model_type / voices / sample_rate / speech_bytes / speech_stream)
    so the server takes either without a family branch of its own."""

    def __init__(self, model: Any, config: VoxCPM2Config | None = None):
        if not is_voxcpm2_model(model):
            raise ValueError(
                f"expected a mlx-audio VoxCPM2 model, got {type(model).__module__}.{type(model).__name__}"
            )
        self._model = model
        self.config = config or VoxCPM2Config()
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.config.model_ref

    @property
    def model_type(self) -> str:
        return ARCH

    @property
    def voices(self) -> list[str]:
        # no speaker presets: "default" is the zero-shot intrinsic voice,
        # matching upstream vllm-omni's `voice: "default"` recipe usage
        return ["default"]

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
        """Synthesize `input` to (payload, content_type). `voice` is "default"
        (zero-shot) or a cloning object ``{"ref_audio": <base64>, "ref_text":
        optional}``; `instructions` designs a voice from a text description.
        Raises ValueError on invalid requests; generation is serialized
        under the service lock."""
        if isinstance(voice, dict):
            ref_audio, ref_text = self._clone_inputs(input, voice, speed)
            with self._lock:
                chunks = synthesize(self._model, self.config, input, ref_audio=ref_audio, ref_text=ref_text)
                if response_format == "wav":
                    return wav_bytes(chunks, self.sample_rate), "audio/wav"
                return b"".join(_pcm16(chunk) for chunk in chunks), "audio/pcm"
        instruct = self._validated(input, voice, speed, instructions, language)
        with self._lock:
            chunks = synthesize(self._model, self.config, input, instruct=instruct)
            if response_format == "wav":
                return wav_bytes(chunks, self.sample_rate), "audio/wav"
            return b"".join(_pcm16(chunk) for chunk in chunks), "audio/pcm"

    def speech_stream(
        self,
        input: str,
        voice: str | None = None,
        speed: float = 1.0,
        instructions: str | None = None,
        language: str | None = None,
        streaming_interval: float | None = None,
        streaming_initial_interval: float | None = None,
    ) -> Iterator[bytes]:
        """Yield 16-bit PCM mono chunks (48 kHz) at ``streaming_interval``.

        The chunks come from the finished buffer — mlx-audio's generate is
        single-yield, so first audio lands when synthesis completes (#71's
        honest baseline; incremental streaming is follow-up loop work).
        The service lock is held for the whole stream (batch-1).
        """
        interval = self.config.streaming_interval if streaming_interval is None else streaming_interval
        if not 0.0 < interval <= 10.0:
            raise ValueError("streaming_interval must be in (0, 10] seconds")
        # accepted for API parity with the Qwen3-TTS stream; the first
        # chunk is already everything, so an initial interval has nothing
        # to speed up here
        if streaming_initial_interval is not None and not 0.0 < streaming_initial_interval <= 10.0:
            raise ValueError("streaming_initial_interval must be in (0, 10] seconds")

        if isinstance(voice, dict):
            ref_audio, ref_text = self._clone_inputs(input, voice, speed)
        else:
            ref_audio = None
            instruct = self._validated(input, voice, speed, instructions, language)

        def stream() -> Iterator[bytes]:
            with self._lock:
                if ref_audio is not None:
                    chunks = synthesize(self._model, self.config, input, ref_audio=ref_audio, ref_text=ref_text)
                else:
                    chunks = synthesize(self._model, self.config, input, instruct=instruct)
                for audio in chunks:
                    step = max(1, int(interval * self.sample_rate))
                    flat = audio.reshape(-1)
                    for start in range(0, flat.size, step):
                        yield _pcm16(flat[start : start + step])

        return stream()

    @staticmethod
    def _require_valid_request(input: str, speed: float) -> None:
        if not input or not input.strip():
            raise ValueError("input must be a non-empty string")
        if speed != 1.0:
            raise ValueError("speed != 1.0 is not supported yet")

    def _clone_inputs(self, input: str, voice: dict, speed: float) -> tuple[mx.array, str | None]:
        """Validate a cloning ``voice`` object and decode the reference clip
        (base64 → 48 kHz mono) outside the service lock — decode errors and
        caps are request errors, not generation failures."""
        self._require_valid_request(input, speed)
        unknown = set(voice) - {"ref_audio", "ref_text"}
        if unknown:
            raise ValueError(f"voice object supports ref_audio and ref_text, got {sorted(unknown)}")
        if "ref_audio" not in voice:
            raise ValueError("voice.ref_audio (base64 audio) is required for cloning")
        ref_text = voice.get("ref_text")
        if ref_text is not None and (not isinstance(ref_text, str) or not ref_text.strip()):
            raise ValueError("voice.ref_text must be a non-empty string when provided")
        ref_audio = decode_ref_audio(voice["ref_audio"], self.sample_rate)
        duration = ref_audio.size / self.sample_rate
        if duration > MAX_REF_SECONDS:
            raise ValueError(f"ref_audio is {duration:.1f}s; the cap is {MAX_REF_SECONDS:.0f}s")
        if duration < MIN_REF_SECONDS:
            raise ValueError(
                f"ref_audio is {duration:.2f}s; a cloning reference needs at least {MIN_REF_SECONDS}s of speech"
            )
        return ref_audio, ref_text

    def _validated(
        self, input: str, voice: str | None, speed: float, instructions: str | None, language: str | None
    ) -> str | None:
        """Shared request checks for the zero-shot and voice-design paths →
        the ``instruct`` override (None = zero-shot)."""
        self._require_valid_request(input, speed)
        if language:
            raise ValueError(
                "language is not supported on VoxCPM2 — the model is multilingual by default"
            )
        if voice is not None and not isinstance(voice, str):
            raise ValueError("voice must be 'default', a cloning object, or omitted")
        if voice is not None and voice.lower() not in self.voices:
            raise ValueError(
                f"voice '{voice}' is not one of the preset voices; VoxCPM2 speaks zero-shot "
                "as 'default', clones a voice from a voice object with ref_audio, or designs "
                "one from `instructions`"
            )
        return instructions.strip() if instructions and instructions.strip() else None
