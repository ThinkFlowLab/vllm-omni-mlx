"""mflux seam: the adapter over the pip-installed mflux library (#91/#99).

Everything model-specific stays inside mflux — this module maps a Family
onto mflux's loader (ZImage + ModelConfig + model_path for pre-quantized
mirrors), validates family capabilities, and wraps the blocking
generate_image call with the timings/metrics the image doctrine wants.
mflux is pinned like mlx-audio (see CONTRIBUTING.md): the classes this
module touches live under mflux.models.*, so a minor bump is an
adaptation PR, not a free upgrade.
"""

from __future__ import annotations

import secrets
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx

from .config import Family, ResolvedModel
from .service import ImageResult, png_bytes


class MFluxService:
    """All MLX work — load included — runs on one dedicated worker thread:
    splitting load (uvicorn main thread) from generate (`asyncio.to_thread`)
    trips MLX's per-thread stream context ("There is no Stream(cpu, 0) in
    current thread" — same class as the VoxCPM2 serve-smoke bug, #71). The
    single-thread executor doubles as the batch-1 serializer."""

    def __init__(self, family: Family, resolved: ResolvedModel):
        self.family = family
        self.weights_ref = resolved.weights_ref
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mflux")
        self._model = None
        self.bits = None

    @classmethod
    def load(cls, resolved: ResolvedModel) -> "MFluxService":
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.z_image import ZImage

        service = cls(resolved.family, resolved)
        started = time.perf_counter()
        # model_path routes weights (pre-quantized mirrors keep their stored
        # level when quantize=None); quantize applies on-load for fp16 repos.
        service._model = service._executor.submit(
            ZImage,
            model_config=ModelConfig.z_image_turbo(),
            quantize=resolved.quantize,
            model_path=resolved.weights_ref,
        ).result()
        service.bits = getattr(service._model, "bits", resolved.quantize)
        print(
            f"image model '{resolved.family.name}' loaded from '{resolved.weights_ref}' in "
            f"{time.perf_counter() - started:.1f}s ({service.bits}-bit)",
            flush=True,
        )
        return service

    @property
    def name(self) -> str:
        return self.weights_ref

    @property
    def model_type(self) -> str:
        return "image"

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
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not 1 <= n <= 4:
            raise ValueError(f"n must be between 1 and 4, got {n}")
        if guidance is not None and not self.family.supports_guidance:
            raise ValueError(f"guidance is not supported on {self.family.name} (guidance-distilled)")
        steps = self.family.steps if steps is None else steps
        if not 1 <= steps <= 50:
            raise ValueError(f"steps must be between 1 and 50, got {steps}")
        base_seed = seed if seed is not None else secrets.randbelow(2**32)
        return self._executor.submit(self._generate_many, prompt, n, width, height, steps, guidance, base_seed).result()

    def _generate_many(
        self, prompt: str, n: int, width: int, height: int, steps: int, guidance, base_seed: int
    ) -> list[ImageResult]:
        return [self._generate_one(prompt, width, height, steps, guidance, base_seed + i) for i in range(n)]

    def _generate_one(self, prompt: str, width: int, height: int, steps: int, guidance, seed: int) -> ImageResult:
        mx.reset_peak_memory()
        started = time.perf_counter()
        image = self._model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
        )
        elapsed = time.perf_counter() - started
        return ImageResult(
            png=png_bytes(image.image),
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            generation_time=elapsed,
            peak_memory_gib=mx.get_peak_memory() / 2**30,
        )
