"""DiffusionService protocol + result type (#91).

One long blocking call per request, not a token stream — so image
generation is a sibling service to tts/ rather than a Backend (whose
protocol is chat-shaped, see backends.py). The server injects whichever
service load_image_service() picked and runs generate() on a worker
thread; a per-service lock enforces one generation at a time (batch-1
doctrine, same as the chat backend lock).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ImageResult:
    png: bytes
    seed: int
    width: int
    height: int
    steps: int
    generation_time: float  # seconds, measured around the model call
    peak_memory_gib: float


@runtime_checkable
class DiffusionService(Protocol):
    """What the /v1/images/generations route needs from a backend seam."""

    @property
    def name(self) -> str: ...

    @property
    def model_type(self) -> str: ...

    def generate(
        self,
        prompt: str,
        n: int = 1,
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
    ) -> list[ImageResult]:
        """Generate ``n`` images sequentially; seeds are ``seed + i`` when a
        seed is given (deterministic per index), fresh randoms otherwise.
        Raises ValueError for family-unsupported parameters (mapped to a
        400 by the route, the audio_speech pattern)."""
        ...


def png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
