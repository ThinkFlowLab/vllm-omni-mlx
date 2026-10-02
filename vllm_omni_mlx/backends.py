"""Model backends.

One model instance is loaded per server process and generation is serialized
with a lock — MLX models are not safe to drive concurrently, and this keeps
the server tiny. Requests wait their turn instead of us building a scheduler.

- TextBackend: any chat LLM via mlx-lm.
- OmniBackend: vision/audio/video models via mlx-vlm (optional dependency).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Iterator, Optional, Protocol

from .schemas import UnifiedRequest


class Backend(Protocol):
    """What the server needs from a model backend."""

    name: str

    def chat(self, req: UnifiedRequest) -> Iterator[Chunk]: ...


@dataclass
class Chunk:
    """One streaming unit from a backend. The final Chunk carries usage and
    a non-None finish_reason; its text is usually empty (except stop-sequence
    truncation tails)."""

    text: str = ""
    finish_reason: Optional[str] = None  # "stop" | "length" | "stop_sequence"
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


# --------------------------------------------------------------------------
# stop-sequence filtering
# --------------------------------------------------------------------------

def _stop_filter(pending: str, stops: tuple[str, ...]) -> tuple[str, str, Optional[str]]:
    """Given the accumulated unfiltered text, return (emittable, remaining, hit).

    Text that could still be the prefix of a stop sequence is held back.
    """
    best_index, best_stop = None, None
    for s in stops:
        index = pending.find(s)
        if index != -1 and (best_index is None or index < best_index):
            best_index, best_stop = index, s
    if best_index is not None:
        return pending[:best_index], "", best_stop
    hold = max(len(s) for s in stops) - 1
    cut = max(len(pending) - hold, 0)
    return pending[:cut], pending[cut:], None


# --------------------------------------------------------------------------
# media helpers
# --------------------------------------------------------------------------

_MEDIA_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
}


def _write_temp_media(sink: list, data: bytes, media_type: Optional[str], prefix: str) -> str:
    ext = _MEDIA_EXT.get((media_type or "").lower(), "")
    fd, path = tempfile.mkstemp(suffix=ext, prefix=f"vllm-omni-mlx-{prefix}-")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    sink.append(path)
    return path


# --------------------------------------------------------------------------
# text backend (mlx-lm)
# --------------------------------------------------------------------------

class TextBackend:
    """Chat generation for text LLMs through mlx-lm."""

    def __init__(self, model_ref: str):
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        self._mlx_lm = mlx_lm
        self._make_sampler = make_sampler
        self.model, self.tokenizer = mlx_lm.load(model_ref)
        self.name = model_ref
        self._lock = threading.Lock()

    def _render_prompt(self, req: UnifiedRequest) -> str:
        messages = [
            {"role": m.role, "content": "".join(p.text for p in m.parts if p.kind == "text")}
            for m in req.messages
        ]
        if req.system:
            messages.insert(0, {"role": "system", "content": req.system})
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def chat(self, req: UnifiedRequest) -> Iterator[Chunk]:
        prompt = self._render_prompt(req)
        sampler = self._make_sampler(
            temp=req.temperature if req.temperature is not None else 0.0,
            top_p=req.top_p or 0.0,
            top_k=req.top_k or 0,
        )
        with self._lock:
            yield from self._generate(prompt, req, sampler)

    def _generate(self, prompt: str, req: UnifiedRequest, sampler) -> Iterator[Chunk]:
        prompt_tokens: Optional[int] = None
        finish: Optional[str] = None
        completion_tokens = 0
        pending = ""
        for resp in self._mlx_lm.stream_generate(
            self.model, self.tokenizer, prompt, max_tokens=req.max_tokens, sampler=sampler
        ):
            completion_tokens += 1
            if resp.prompt_tokens:
                prompt_tokens = resp.prompt_tokens
            if resp.finish_reason:
                finish = resp.finish_reason
            if not req.stop:
                if resp.text:
                    yield Chunk(text=resp.text)
                continue
            pending += resp.text
            emit, pending, hit = _stop_filter(pending, req.stop)
            if emit:
                yield Chunk(text=emit)
            if hit:
                finish = "stop_sequence"
                break
        if pending:
            # text held back while watching for a stop sequence that never came
            yield Chunk(text=pending)
        yield Chunk(
            finish_reason=finish or "stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


# --------------------------------------------------------------------------
# omni backend (mlx-vlm): image / audio input
# --------------------------------------------------------------------------

class OmniBackend:
    """Chat generation for multimodal models (image, audio, video) through
    mlx-vlm. Requires the optional dependency: pip install 'vllm-omni-mlx[omni]'."""

    def __init__(self, model_ref: str):
        try:
            import mlx_vlm
        except ImportError as exc:
            raise RuntimeError(
                "OmniBackend needs mlx-vlm; install it with: pip install 'vllm-omni-mlx[omni]'"
            ) from exc
        self._vlm = mlx_vlm
        self.model, self.processor = mlx_vlm.load(model_ref)
        self.config = getattr(self.model, "config", None)
        self.name = model_ref
        self._lock = threading.Lock()

    def chat(self, req: UnifiedRequest) -> Iterator[Chunk]:
        messages: list[dict] = []
        if req.system:
            messages.append({"role": "system", "content": [{"type": "text", "text": req.system}]})

        temp_files: list[str] = []
        images: list[str] = []
        audios: list[str] = []
        try:
            for m in req.messages:
                content: list[dict] = []
                for part in m.parts:
                    if part.kind == "text":
                        content.append({"type": "text", "text": part.text})
                    elif part.kind == "image":
                        images.append(_write_temp_media(temp_files, part.data, part.media_type, "image"))
                        content.append({"type": "image"})
                    elif part.kind == "audio":
                        audios.append(_write_temp_media(temp_files, part.data, part.media_type, "audio"))
                        content.append({"type": "audio"})
                messages.append({"role": m.role, "content": content})

            prompt = self._vlm.apply_chat_template(
                self.processor,
                self.config,
                messages,
                add_generation_prompt=True,
                num_images=len(images),
                num_audios=len(audios),
            )
            kwargs: dict = {"max_tokens": req.max_tokens}
            if req.temperature is not None:
                kwargs["temperature"] = req.temperature
            if req.top_p is not None:
                kwargs["top_p"] = req.top_p
            if req.top_k is not None:
                kwargs["top_k"] = req.top_k

            with self._lock:
                yield from self._generate(prompt, images, audios, req, kwargs)
        finally:
            for path in temp_files:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _generate(self, prompt, images, audios, req, kwargs) -> Iterator[Chunk]:
        stream = self._vlm.stream_generate(
            self.model,
            self.processor,
            prompt,
            image=images or None,
            audio=audios or None,
            **kwargs,
        )
        prompt_tokens: Optional[int] = None
        finish: Optional[str] = None
        completion_tokens = 0
        pending = ""
        for resp in stream:
            completion_tokens += 1
            if resp.prompt_tokens:
                prompt_tokens = resp.prompt_tokens
            if resp.finish_reason:
                finish = resp.finish_reason
            if not req.stop:
                if resp.text:
                    yield Chunk(text=resp.text)
                continue
            pending += resp.text
            emit, pending, hit = _stop_filter(pending, req.stop)
            if emit:
                yield Chunk(text=emit)
            if hit:
                finish = "stop_sequence"
                break
        if pending:
            # text held back while watching for a stop sequence that never came
            yield Chunk(text=pending)
        yield Chunk(
            finish_reason=finish or "stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


# --------------------------------------------------------------------------
# backend selection
# --------------------------------------------------------------------------

_OMNI_CONFIG_KEYS = ("vision_config", "audio_config", "vision_encoder_config", "multimodal_projector_config")


def _peek_config(model_ref: str) -> dict:
    if os.path.isdir(model_ref):
        with open(os.path.join(model_ref, "config.json")) as f:
            return json.load(f)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_ref, "config.json")
    with open(path) as f:
        return json.load(f)


def load_backend(model_ref: str, preferred: str = "auto"):
    """Load a backend: 'text', 'omni', or 'auto' (sniffs config.json)."""
    if preferred == "text":
        return TextBackend(model_ref)
    if preferred == "omni":
        return OmniBackend(model_ref)
    config = _peek_config(model_ref)
    wants_omni = any(key in config for key in _OMNI_CONFIG_KEYS)
    if wants_omni:
        return OmniBackend(model_ref)
    return TextBackend(model_ref)
