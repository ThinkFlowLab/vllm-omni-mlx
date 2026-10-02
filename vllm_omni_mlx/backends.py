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


def _usage_first_emitter():
    """Chunk factory that carries prompt_tokens on the first emitted chunk so
    streaming consumers can report input usage at message start; later chunks
    omit it. mlx-lm/mlx-vlm set prompt_tokens on every response, so it is
    already known when the first text is emitted."""
    state = {"sent": False}

    def make(text: str, prompt_tokens: Optional[int]) -> Chunk:
        pt = None if state["sent"] else prompt_tokens
        if pt is not None:
            state["sent"] = True
        return Chunk(text=text, prompt_tokens=pt)

    return make


# --------------------------------------------------------------------------
# text backend (mlx-lm)
# --------------------------------------------------------------------------

class TextBackend:
    """Chat generation for text LLMs through mlx-lm."""

    def __init__(self, model_ref: str, draft_model_ref: Optional[str] = None, kv_bits: Optional[int] = None, kv_group_size: int = 64):
        import mlx_lm
        from mlx_lm.generate import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        self._mlx_lm = mlx_lm
        self._make_sampler = make_sampler
        self._make_cache = make_prompt_cache
        self.model, self.tokenizer = mlx_lm.load(model_ref)
        self.name = model_ref
        # speculative decoding: draft model must share the main model's tokenizer
        self._draft_model = mlx_lm.load(draft_model_ref)[0] if draft_model_ref else None
        self._kv_kwargs: dict = {}
        if kv_bits:
            self._kv_kwargs = {"kv_bits": kv_bits, "kv_group_size": kv_group_size}
        self._lock = threading.Lock()
        # single-entry prefix cache: (token ids whose KV entries the cache holds,
        # the live KV cache). Last conversation wins — matches the single-user target.
        # Unused while a draft model is set: speculative batching makes the
        # fed-tokens invariant across turns uncertain, so we re-prefill instead.
        self._cached: tuple[Optional[list], Optional[list]] = (None, None)

    def _render_prompt(self, req: UnifiedRequest) -> str:
        messages = [
            {"role": m.role, "content": "".join(p.text for p in m.parts if p.kind == "text")}
            for m in req.messages
        ]
        if req.system:
            messages.insert(0, {"role": "system", "content": req.system})
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def _encode(self, prompt) -> list:
        # apply_chat_template returns token ids (transformers >= 5); encode only
        # plain strings, mirroring mlx-lm's own handling so cached token ids
        # line up with what a full re-prefill would have processed
        if not isinstance(prompt, str):
            return list(prompt)
        add_special = self.tokenizer.bos_token is None or not prompt.startswith(self.tokenizer.bos_token)
        return self.tokenizer.encode(prompt, add_special_tokens=add_special)

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
        tokens = self._encode(prompt)
        cached_tokens, cache = self._cached
        reusable = (
            self._draft_model is None
            and cache is not None
            and cached_tokens is not None
            and len(tokens) > len(cached_tokens)
            and tokens[: len(cached_tokens)] == cached_tokens
        )
        prompt_tokens = len(tokens)  # full count even when only a suffix is prefilled
        finish: Optional[str] = None
        completion_tokens = 0
        gen_ids: list = []
        pending = ""
        make = _usage_first_emitter()
        gen_kwargs: dict = dict(max_tokens=req.max_tokens, sampler=sampler, **self._kv_kwargs)
        if self._draft_model is not None:
            # mlx-lm's speculative path builds its own model+draft caches and
            # splices any prompt_cache passed in — never supply one alongside a draft
            gen_kwargs["draft_model"] = self._draft_model
            feed = tokens
        elif reusable:
            # the new conversation extends the cached one: prefill only the suffix
            feed = tokens[len(cached_tokens) :]
            gen_kwargs["prompt_cache"] = cache
        else:
            feed = tokens
            gen_kwargs["prompt_cache"] = self._make_cache(self.model)
        for resp in self._mlx_lm.stream_generate(self.model, self.tokenizer, feed, **gen_kwargs):
            completion_tokens += 1
            gen_ids.append(resp.token)
            if resp.finish_reason:
                finish = resp.finish_reason
            if not req.stop:
                if resp.text:
                    yield make(resp.text, prompt_tokens)
                continue
            pending += resp.text
            emit, pending, hit = _stop_filter(pending, req.stop)
            if emit:
                yield make(emit, prompt_tokens)
            if hit:
                finish = "stop_sequence"
                break
        # store only after generation ran to completion: a consumer that drops
        # the generator mid-stream leaves the previous (consistent) pair in place
        if self._draft_model is None:
            self._cached = (tokens + gen_ids, gen_kwargs["prompt_cache"])
        if pending:
            # text held back while watching for a stop sequence that never came
            yield make(pending, prompt_tokens)
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
    mlx-vlm. Requires the optional dependency: pip install 'vllm-omni-mlx[omni]'.
    No cross-turn prompt caching yet; mlx-vlm's PromptCacheState is the follow-up."""

    def __init__(self, model_ref: str, kv_bits: Optional[int] = None, kv_group_size: int = 64):
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
        self._kv_kwargs: dict = {}
        if kv_bits:
            self._kv_kwargs = {"kv_bits": kv_bits, "kv_group_size": kv_group_size}
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
            kwargs.update(self._kv_kwargs)

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
        make = _usage_first_emitter()
        for resp in stream:
            completion_tokens += 1
            if resp.prompt_tokens:
                prompt_tokens = resp.prompt_tokens
            if resp.finish_reason:
                finish = resp.finish_reason
            if not req.stop:
                if resp.text:
                    yield make(resp.text, prompt_tokens)
                continue
            pending += resp.text
            emit, pending, hit = _stop_filter(pending, req.stop)
            if emit:
                yield make(emit, prompt_tokens)
            if hit:
                finish = "stop_sequence"
                break
        if pending:
            # text held back while watching for a stop sequence that never came
            yield make(pending, prompt_tokens)
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


def load_backend(
    model_ref: str,
    preferred: str = "auto",
    draft_model: Optional[str] = None,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
):
    """Load a backend: 'text', 'omni', or 'auto' (sniffs config.json)."""
    if preferred == "text":
        return TextBackend(model_ref, draft_model_ref=draft_model, kv_bits=kv_bits, kv_group_size=kv_group_size)
    if preferred == "omni":
        return OmniBackend(model_ref, kv_bits=kv_bits, kv_group_size=kv_group_size)
    config = _peek_config(model_ref)
    wants_omni = any(key in config for key in _OMNI_CONFIG_KEYS)
    if wants_omni:
        return OmniBackend(model_ref, kv_bits=kv_bits, kv_group_size=kv_group_size)
    return TextBackend(model_ref, draft_model_ref=draft_model, kv_bits=kv_bits, kv_group_size=kv_group_size)
