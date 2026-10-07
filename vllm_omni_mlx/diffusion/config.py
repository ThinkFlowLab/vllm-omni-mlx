"""Typed configuration and family registry for image generation (#91, #99).

The diffusion package is a seam over library backends (mflux first, the
vendored mlx-examples SD family later) — no native model code lives here
(same rule as tts/ over mlx-audio, see docs/architecture.md). This module
resolves a served model ref to a family and the weights source that goes
with it. It imports no backend at module scope so CI (which installs the
package without extras) can exercise resolution and request validation.

Z-Image-Turbo notes (mflux 0.21): guidance-distilled — guidance is forced
to 0.0 and a negative prompt is never encoded (mflux's own CLI ignores
both with a warning; we reject guidance with a 400 instead, the same
early-clear-signal choice as instruct-on-0.6B). The default source is the
pre-quantized 4-bit mirror (~5.5 GiB on disk vs ~12.3 GiB fp16 for the
canonical repo, whose stored quantization level is honored on load).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Pre-quantized 4-bit mirror of Tongyi-MAI/Z-Image-Turbo in mflux's save
# format — the default source (the canonical fp16 repo does not fit the
# dev machine's disk budget).
DEFAULT_MODEL = "filipstrand/Z-Image-Turbo-mflux-4bit"


@dataclass(frozen=True)
class Family:
    """A model family behind the mflux seam: what a request may set and
    which refs identify it. Weight routing lives in ResolvedModel."""

    name: str
    aliases: tuple[str, ...]
    canonical_repo: str
    steps: int
    supports_guidance: bool = False


@dataclass(frozen=True)
class ResolvedModel:
    family: Family
    weights_ref: str  # repo id or local path handed to mflux as model_path
    quantize: Optional[int] = None  # None → honor the checkpoint's stored level


_FAMILIES: tuple[Family, ...] = (
    Family(
        name="z-image-turbo",
        aliases=("z-image-turbo", "zimage-turbo"),
        canonical_repo="Tongyi-MAI/Z-Image-Turbo",
        steps=9,
    ),
)

FAMILIES = _FAMILIES  # re-exported for tests/introspection


def resolve_model(model_ref: str) -> ResolvedModel:
    ref = (model_ref or "").strip()
    low = ref.lower()
    for family in _FAMILIES:
        if low in family.aliases:
            # the alias is the family's disk/memory-friendly default: the
            # pre-quantized mirror with its stored 4-bit level
            return ResolvedModel(family, weights_ref=DEFAULT_MODEL)
        if low == family.canonical_repo.lower():
            # canonical fp16 repo: quantize 4-bit on load (memory + parity
            # with the mirror; the fp16 download is also ~12 GiB of disk)
            return ResolvedModel(family, weights_ref=ref, quantize=4)
    if low == DEFAULT_MODEL.lower():
        return ResolvedModel(_FAMILIES[0], weights_ref=DEFAULT_MODEL)
    known = ", ".join(sorted({alias for f in _FAMILIES for alias in f.aliases}))
    raise ValueError(
        f"unknown image model '{model_ref}'; supported: {known}, "
        f"canonical repos ({', '.join(f.canonical_repo for f in _FAMILIES)}), "
        f"or the pre-quantized mirror ({DEFAULT_MODEL})"
    )


@dataclass(frozen=True)
class DiffusionConfig:
    """Serving defaults for /v1/images/generations."""

    model_ref: str = DEFAULT_MODEL
    width: int = 1024
    height: int = 1024


def load_image_service(model_ref: str = DEFAULT_MODEL) -> "MFluxService":  # noqa: F821
    """Resolve ``model_ref`` and load it via the mflux seam.

    Imports mflux lazily: the [image] extra is optional, and config/test
    imports must not require it (CI installs no extras).
    """
    from .mflux_service import MFluxService

    return MFluxService.load(resolve_model(model_ref))
