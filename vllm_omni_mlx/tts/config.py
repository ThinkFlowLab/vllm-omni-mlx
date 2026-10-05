"""Typed configuration and loader for the TTS stage (M1.1, #10).

Adapts mlx-audio's Qwen3-TTS implementation (MIT) — the adapt-vs-port verdict
and evidence live on #1. mlx-audio owns the checkpoint→MLX weight mapping
(conv `(C_out, K, C_in)` transposes, EMA-normalized RVQ codebooks, quantized
variants), so loading the full model is itself the mapping validation: the
weight-dependent test runs against the local snapshot when cached and skips
elsewhere (CI has no weights).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"
DEFAULT_SPEAKER = "Vivian"


@dataclass(frozen=True)
class TTSConfig:
    """Serving defaults for /v1/audio/speech (values are Qwen3-TTS defaults)."""

    model_ref: str = DEFAULT_MODEL
    speaker: str = DEFAULT_SPEAKER
    language: str = "auto"
    instruct: Optional[str] = None
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    max_tokens: int = 4096
    streaming_interval: float = 2.0
    # first-chunk size for the streaming path (#39 fast path): seconds of
    # audio emitted as soon as they exist, quantized to a power-of-two frame
    # bucket ≤ streaming_interval's chunk (see tts/stream_loop.py). Default
    # 0.08 s = one codec frame: measured −35 ms time-to-first-audio vs 0.2 s
    # at unchanged RTF (#77) — the choppier first chunk is the trade
    streaming_initial_interval: float = 0.08

    def with_overrides(self, **overrides: Any) -> "TTSConfig":
        known = {k: v for k, v in overrides.items() if v is not None and k in TTSConfig.__dataclass_fields__ and k != "model_ref"}
        return replace(self, **known) if known else self


def load_tts_model(config: TTSConfig) -> Any:
    """Load the Qwen3-TTS model through mlx-audio. Requires the [tts] extra."""
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError as exc:
        raise RuntimeError(
            "TTS needs mlx-audio; install it with: pip install 'vllm-omni-mlx[tts]'"
        ) from exc
    return load_model(config.model_ref)


def local_snapshot(model_ref: str) -> Optional[str]:
    """Path to the locally cached HF snapshot, or None. Gates weight-dependent
    tests so they run where the checkpoint exists and skip in CI."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_ref, local_files_only=True)
    except Exception:
        return None
