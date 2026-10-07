"""Typed configuration and loader for the ASR stage (#68).

mlx-audio owns the model (Conv2d-frontend audio encoder + Qwen3 text decoder
for ``qwen3_asr``) and the checkpoint mapping; we wrap it. Weight-dependent
tests gate on :func:`local_snapshot` like the TTS ones.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..tts.config import local_snapshot  # noqa: F401  (re-exported: same HF-cache gate)

#: development default (#68): best WER of the size ladder that still fits a
#: 16 GB machine next to a chat model; the 0.6B-4bit is the small-Mac config
DEFAULT_MODEL = "mlx-community/Qwen3-ASR-1.7B-4bit"

#: the model's input rate (Qwen3-ASR's feature extractor is 16 kHz)
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class ASRConfig:
    """Serving defaults for /v1/audio/transcriptions."""

    model_ref: str = DEFAULT_MODEL
    language: str | None = None  # None → the model detects it
    temperature: float = 0.0  # greedy: transcription is not a creative task
    max_tokens: int = 8192

    def with_overrides(self, **overrides: Any) -> "ASRConfig":
        known = {
            k: v
            for k, v in overrides.items()
            if v is not None and k in ASRConfig.__dataclass_fields__ and k != "model_ref"
        }
        return replace(self, **known) if known else self


def load_asr_model(config: ASRConfig) -> Any:
    """Load the ASR model through mlx-audio. Requires the [asr] extra."""
    try:
        from mlx_audio.stt.utils import load_model
    except ImportError as exc:
        raise RuntimeError(
            "ASR needs mlx-audio; install it with: pip install 'vllm-omni-mlx[asr]'"
        ) from exc
    return load_model(config.model_ref)
