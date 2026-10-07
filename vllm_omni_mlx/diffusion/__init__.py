"""Image generation seam (diffusion, #91/#99): config + family registry,
service protocol, and the mflux adapter. See docs/architecture.md for the
layer map and the #91 design record for the API/metrics doctrine."""

from .config import DEFAULT_MODEL, DiffusionConfig, Family, FAMILIES, ResolvedModel, load_image_service, resolve_model
from .service import DiffusionService, ImageResult

__all__ = [
    "DEFAULT_MODEL",
    "DiffusionConfig",
    "DiffusionService",
    "FAMILIES",
    "Family",
    "ImageResult",
    "ResolvedModel",
    "load_image_service",
    "resolve_model",
]
