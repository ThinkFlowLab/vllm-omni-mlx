"""Image serving service (#101): serialized image generation for the app.

Wraps a loaded mflux model with the single-user lock (batch-1 target, same
doctrine as TTSService) and request-level validation. The OpenAI images API
surface lives in server.py; this module owns generation: prompt → PNG bytes.

Latency convention (#91 metrics doctrine): time-to-image is the wall clock
from `generate()` entry to encoded bytes — prompt encoding included, the
TTFA analog. Peak memory accounting is the caller's job (mx.reset_peak_memory()
bracket), which keeps this class free of profiling concerns.
"""

from __future__ import annotations

import io
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from .config import VIGGLE_TURBO_SCHEDULER, VIGGLE_TURBO_STEPS, ImageConfig

MIN_SIDE, MAX_SIDE = 256, 2048
MIN_STEPS, MAX_STEPS = 1, 100
MIN_GUIDANCE, MAX_GUIDANCE = 1.0, 10.0


class ImageError(ValueError):
    """Request validation failure the server maps to a 400."""


@dataclass
class ImageResult:
    png: bytes
    width: int
    height: int
    seed: int
    steps: int
    elapsed: float  # time-to-image: generate() entry → encoded bytes


def parse_size(size: str) -> tuple[int, int]:
    """OpenAI `size` ("WIDTHxHEIGHT") → (width, height). Sides must be
    multiples of 16 (the latent factor mflux rounds to) within serving bounds;
    explicit rejection beats silent rounding for an API surface."""
    if not isinstance(size, str):
        raise ImageError(f"size must be 'WIDTHxHEIGHT' (e.g. '1024x1024'), got {size!r}")
    normalized = size.strip().lower()
    if "x" not in normalized:
        raise ImageError(f"size must be 'WIDTHxHEIGHT' (e.g. '1024x1024'), got {size!r}")
    try:
        width_s, height_s = normalized.split("x", 1)
        width, height = int(width_s), int(height_s)
    except ValueError as exc:
        raise ImageError(f"size must be 'WIDTHxHEIGHT' (e.g. '1024x1024'), got {size!r}") from exc
    for side, name in ((width, "width"), (height, "height")):
        if not MIN_SIDE <= side <= MAX_SIDE:
            raise ImageError(f"size {name} must be within [{MIN_SIDE}, {MAX_SIDE}], got {side}")
        if side % 16 != 0:
            raise ImageError(f"size {name} must be a multiple of 16 (got {side}); mflux rounds silently and we refuse to")
    return width, height


class ImageService:
    def __init__(self, model: Any, config: ImageConfig | None = None):
        self._model = model
        self.config = config or ImageConfig()
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.config.model_ref

    @property
    def license(self) -> str:
        return self.config.license

    def generate(
        self,
        prompt: str,
        *,
        size: Optional[str] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = "__unset__",
        n: int = 1,
    ) -> list[ImageResult]:
        """Generate n images; sequential by design (batch-1 doctrine, and the
        prompt encode amortizes only inside mflux's own loop anyway). Seeds run
        seed, seed+1, … when given — deterministic per request — else random.
        The lock spans the whole request: one image at a time on this box."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ImageError("prompt must be a non-empty string")
        cfg = self.config
        width, height = parse_size(size or cfg.size)
        steps = cfg.steps if steps is None else self._int(steps, "steps")
        guidance = cfg.guidance if guidance is None else self._float(guidance, "guidance")
        if negative_prompt == "__unset__":
            negative_prompt = cfg.negative_prompt
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            raise ImageError("negative_prompt must be a string")
        if not MIN_STEPS <= steps <= MAX_STEPS:
            raise ImageError(f"steps must be within [{MIN_STEPS}, {MAX_STEPS}], got {steps}")
        if not MIN_GUIDANCE <= guidance <= MAX_GUIDANCE:
            raise ImageError(
                f"guidance must be within [{MIN_GUIDANCE}, {MAX_GUIDANCE}], got {guidance}"
                f" ({cfg.family} is guidance-free at 1.0)"
            )
        if cfg.scheduler == VIGGLE_TURBO_SCHEDULER and steps != VIGGLE_TURBO_STEPS:
            raise ImageError(
                f"the {VIGGLE_TURBO_SCHEDULER} scheduler samples the distilled LoRA on its"
                f" {VIGGLE_TURBO_STEPS} trained sigma nodes; steps must be {VIGGLE_TURBO_STEPS}"
            )
        if not isinstance(n, int) or not 1 <= n <= 4:
            raise ImageError("n must be an integer within [1, 4]")
        if seed is not None:
            seed = self._int(seed, "seed")

        results: list[ImageResult] = []
        with self._lock:
            for i in range(n):
                seed_i = seed if seed is not None else random.randint(0, 2**31 - 1)
                seed_i += i
                started = time.perf_counter()
                generated = self._model.generate_image(
                    seed=seed_i,
                    prompt=prompt,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance,
                    negative_prompt=negative_prompt,
                    scheduler=cfg.scheduler,
                )
                buf = io.BytesIO()
                generated.image.save(buf, format="PNG")
                results.append(
                    ImageResult(
                        png=buf.getvalue(),
                        width=width,
                        height=height,
                        seed=seed_i,
                        steps=steps,
                        elapsed=time.perf_counter() - started,
                    )
                )
        return results

    @staticmethod
    def _int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ImageError(f"{name} must be an integer, got {value!r}")
        return value

    @staticmethod
    def _float(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ImageError(f"{name} must be a number, got {value!r}")
        return float(value)
