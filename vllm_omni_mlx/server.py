"""HTTP server: OpenAI- and Anthropic-compatible chat endpoints over one backend."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import AsyncIterator, Iterator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .backends import Backend, Chunk
from .schemas import ApiError, UnifiedRequest, normalize_anthropic, normalize_openai

_STARTED_AT = int(time.time())


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _openai_error(exc: ApiError) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": exc.message, "type": exc.err_type, "param": None, "code": None}},
        status_code=exc.status,
    )


def _anthropic_error(exc: ApiError) -> JSONResponse:
    return JSONResponse(
        {"type": "error", "error": {"type": exc.err_type, "message": exc.message}},
        status_code=exc.status,
    )


async def _json_body(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception:
        raise ApiError(400, "request body must be valid JSON")
    if not isinstance(payload, dict):
        raise ApiError(400, "request body must be a JSON object")
    return payload


def _check_auth(request: Request, api_key: str | None) -> None:
    if not api_key:
        return
    auth = request.headers.get("authorization", "")
    supplied = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key")
    if supplied != api_key:
        raise ApiError(401, "invalid or missing API key", err_type="authentication_error")


def _collect(generator: Iterator[Chunk]) -> tuple[str, str, int, int]:
    """Drain a backend generator (runs in a worker thread)."""
    text_parts: list[str] = []
    finish, prompt_tokens, completion_tokens = "stop", 0, 0
    for chunk in generator:
        if chunk.text:
            text_parts.append(chunk.text)
        if chunk.finish_reason:
            finish = chunk.finish_reason
        if chunk.prompt_tokens is not None:
            prompt_tokens = chunk.prompt_tokens
        if chunk.completion_tokens is not None:
            completion_tokens = chunk.completion_tokens
    return "".join(text_parts), finish, prompt_tokens, completion_tokens


async def _bridge(generator: Iterator[Chunk], cancel: threading.Event) -> AsyncIterator[Chunk]:
    """Iterate a synchronous generator in a worker thread and yield chunks on
    the event loop. Client disconnect sets `cancel`; the worker stops and closes
    the generator (releasing the backend lock) at the next token boundary."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Chunk | None | Exception] = asyncio.Queue()

    def produce() -> None:
        try:
            for chunk in generator:
                if cancel.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as exc:  # surface backend failures to the consumer
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            try:
                generator.close()
            except Exception:
                pass
            loop.call_soon_threadsafe(queue.put_nowait, None)

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        cancel.set()
        await asyncio.to_thread(worker.join, 5.0)


# --------------------------------------------------------------------------
# OpenAI: /v1/chat/completions
# --------------------------------------------------------------------------

def _openai_finish(finish: str) -> str:
    return {"stop": "stop", "length": "length", "stop_sequence": "stop"}.get(finish, "stop")


def _openai_sse(req: UnifiedRequest, generator: Iterator[Chunk], model_name: str) -> AsyncIterator[bytes]:
    async def stream() -> AsyncIterator[bytes]:
        cancel = threading.Event()
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        first = True
        try:
            async for chunk in _bridge(generator, cancel):
                delta: dict = {}
                if first:
                    delta["role"] = "assistant"
                    first = False
                if chunk.text:
                    delta["content"] = chunk.text
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": _openai_finish(chunk.finish_reason) if chunk.finish_reason else None}],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        except ApiError as exc:
            body = json.dumps({"error": {"message": exc.message, "type": exc.err_type}})
            yield f"data: {body}\n\ndata: [DONE]\n\n".encode()

    return stream()


# --------------------------------------------------------------------------
# Anthropic: /v1/messages
# --------------------------------------------------------------------------

def _anthropic_stop(finish: str) -> str:
    return {"stop": "end_turn", "length": "max_tokens", "stop_sequence": "stop_sequence"}.get(finish, "end_turn")


def _sse_event(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _anthropic_sse(req: UnifiedRequest, generator: Iterator[Chunk], model_name: str) -> AsyncIterator[bytes]:
    async def stream() -> AsyncIterator[bytes]:
        cancel = threading.Event()
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        prompt_tokens = 0
        try:
            started = False
            async for chunk in _bridge(generator, cancel):
                if not started:
                    if chunk.prompt_tokens is not None:
                        prompt_tokens = chunk.prompt_tokens
                    yield _sse_event(
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": message_id,
                                "type": "message",
                                "role": "assistant",
                                "model": model_name,
                                "content": [],
                                "stop_reason": None,
                                "stop_sequence": None,
                                "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
                            },
                        },
                    )
                    yield _sse_event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
                    yield _sse_event("ping", {"type": "ping"})
                    started = True
                if chunk.text:
                    yield _sse_event(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": chunk.text}},
                    )
                if chunk.finish_reason:
                    yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
                    yield _sse_event(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": _anthropic_stop(chunk.finish_reason), "stop_sequence": None},
                            "usage": {"output_tokens": chunk.completion_tokens or 0},
                        },
                    )
                    yield _sse_event("message_stop", {"type": "message_stop"})
        except ApiError as exc:
            yield _sse_event("error", {"type": "error", "error": {"type": exc.err_type, "message": exc.message}})

    return stream()


# --------------------------------------------------------------------------
# app factory
# --------------------------------------------------------------------------

def create_app(backend: Backend, api_key: str | None = None) -> Starlette:
    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def models(request: Request) -> Response:
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": backend.name,
                        "object": "model",
                        "created": _STARTED_AT,
                        "owned_by": "vllm-omni-mlx",
                    }
                ],
            }
        )

    async def chat_completions(request: Request) -> Response:
        try:
            _check_auth(request, api_key)
            req = normalize_openai(await _json_body(request))
            generator = backend.chat(req)
            model_name = req.model or backend.name
            if not req.stream:
                text, finish, prompt_tokens, completion_tokens = await asyncio.to_thread(_collect, generator)
                return JSONResponse(
                    {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": _openai_finish(finish),
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                )
            return StreamingResponse(
                _openai_sse(req, generator, model_name),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        except ApiError as exc:
            return _openai_error(exc)

    async def messages(request: Request) -> Response:
        try:
            _check_auth(request, api_key)
            req = normalize_anthropic(await _json_body(request))
            generator = backend.chat(req)
            model_name = req.model or backend.name
            if not req.stream:
                text, finish, prompt_tokens, completion_tokens = await asyncio.to_thread(_collect, generator)
                return JSONResponse(
                    {
                        "id": f"msg_{uuid.uuid4().hex[:24]}",
                        "type": "message",
                        "role": "assistant",
                        "model": model_name,
                        "content": [{"type": "text", "text": text}],
                        "stop_reason": _anthropic_stop(finish),
                        "stop_sequence": None,
                        "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
                    }
                )
            return StreamingResponse(
                _anthropic_sse(req, generator, model_name),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        except ApiError as exc:
            return _anthropic_error(exc)

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/models", models),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/messages", messages, methods=["POST"]),
        ]
    )
