"""Image config + loader (#91 seam, #101 Qwen-Image-2.1 backend).

One `ImageConfig` per served image model, mirroring tts/config.py: loading
resolves a model_ref (an mflux-format HF repo or local directory) into a
loaded model. Zero native model code — everything below the seam is mflux,
pinned by the [image] extra.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

DEFAULT_MODEL = "mlx-community/Qwen-Image-2.1-mflux-q4"

# #101 license verdict, recorded before any download: Qwen-Image-2.1 and its
# fast variants (Viggle turbo LoRA, Pruna) all ship under the Qwen Research
# License — non-commercial, research/evaluation only. Commercial use needs a
# separate license from Qwen. Surfaced at load time so operators see it.
QWEN_IMAGE_21_LICENSE = "Qwen Research License — non-commercial (research/evaluation only)"

# Fast-variant verdict (#101): Viggle turbo — a DMD-distilled LoRA over the
# base DiT, sampled on its 6 trained sigma nodes via mflux's built-in
# `viggle_turbo` scheduler (6 steps, no CFG). Pruna ships merged dense
# transformers with no mflux loader path → not selected (follow-up).
VIGGLE_TURBO_LORA = "Viggle/Qwen-Image-2.1-viggle-turbo"
VIGGLE_TURBO_SCHEDULER = "viggle_turbo"
VIGGLE_TURBO_STEPS = 6


@dataclass
class ImageConfig:
    model_ref: str = DEFAULT_MODEL
    family: str = "qwen-image-2.1"
    steps: int = 40  # measured before fixing (see issue #101); the card default
    guidance: float = 1.0  # guidance-free: 1.0 disables CFG unless a negative prompt is sent
    size: str = "1024x1024"
    negative_prompt: Optional[str] = None
    scheduler: str = "linear"
    lora_paths: tuple[str, ...] = ()
    lora_scales: tuple[float, ...] = ()
    quantize: Optional[int] = None  # on-the-fly quantization for dense (non-pre-quantized) repos
    license: str = QWEN_IMAGE_21_LICENSE

    def with_overrides(self, **overrides: Any) -> "ImageConfig":
        known = {
            k: v for k, v in overrides.items() if v is not None and k in ImageConfig.__dataclass_fields__ and k != "model_ref"
        }
        return replace(self, **known) if known else self


def load_image_model(config: ImageConfig) -> Any:
    """Load an image model through mflux. Requires the [image] extra.

    Pre-quantized mflux-format repos (mlx-community/*-mflux-q4) load as-is —
    the safetensors metadata carries the quantization level; `quantize` only
    applies when serving a dense repo."""
    if config.family != "qwen-image-2.1":
        raise ValueError(
            f"unsupported image family '{config.family}' (Z-Image-Turbo is #99, FLUX.2-klein is #100)"
        )
    try:
        from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
    except ImportError as exc:
        raise RuntimeError(
            "image generation needs mflux; install it with: pip install 'vllm-omni-mlx[image]'"
        ) from exc
    return QwenImage21(
        quantize=config.quantize,
        model_path=config.model_ref,
        lora_paths=list(config.lora_paths) or None,
        lora_scales=list(config.lora_scales) or None,
    )


def local_snapshot(model_ref: str, allow_patterns: Optional[tuple[str, ...]] = None) -> Optional[str]:
    """Path to the locally cached HF snapshot, or None. Gates weight-dependent
    tests so they run where the checkpoint exists and skip in CI. With
    allow_patterns, a partially-downloaded snapshot still resolves when the
    matching subtree is complete (hub otherwise refuses partial snapshots)."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_ref, local_files_only=True, allow_patterns=allow_patterns)
    except Exception:
        return None
