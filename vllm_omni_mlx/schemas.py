"""Normalize OpenAI and Anthropic request payloads into one internal shape.

Media (images, audio) is decoded eagerly to bytes here so the backends can
stay dumb: they only ever see text parts and ready-to-use media bytes.
"""

from __future__ import annotations

import base64
import binascii
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

MAX_REMOTE_MEDIA = 25 * 1024 * 1024  # 25 MiB cap for http(s) media fetches
DEFAULT_MAX_TOKENS = 1024


class ApiError(Exception):
    """Request validation failure carrying an HTTP status."""

    def __init__(self, status: int, message: str, err_type: str = "invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type


@dataclass
class Part:
    kind: str  # "text" | "image" | "audio"
    text: Optional[str] = None
    data: Optional[bytes] = None
    media_type: Optional[str] = None


@dataclass
class Message:
    role: str
    parts: list[Part] = field(default_factory=list)


@dataclass
class UnifiedRequest:
    model: str
    messages: list[Message]
    system: Optional[str] = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: tuple[str, ...] = ()
    stream: bool = False


# --------------------------------------------------------------------------
# media decoding helpers
# --------------------------------------------------------------------------

def _b64decode(s: str) -> bytes:
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        raise ApiError(400, "invalid base64 media payload")


def _decode_data_url(url: str) -> tuple[bytes, str]:
    try:
        head, payload = url.split(",", 1)
    except ValueError:
        raise ApiError(400, "malformed data: URL")
    media_type = head[5:].split(";")[0] or "application/octet-stream"
    if "base64" in head.lower():
        return _b64decode(payload), media_type
    return urllib.parse.unquote_to_bytes(payload), media_type


def _fetch_url(url: str) -> tuple[bytes, str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vllm-omni-mlx"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            media_type = resp.headers.get("content-type", "application/octet-stream").split(";")[0].strip()
            data = resp.read(MAX_REMOTE_MEDIA + 1)
    except Exception as exc:
        raise ApiError(400, f"failed to fetch remote media: {exc}")
    if len(data) > MAX_REMOTE_MEDIA:
        raise ApiError(400, f"remote media exceeds {MAX_REMOTE_MEDIA // (1024 * 1024)} MiB limit")
    return data, media_type


def _image_from_url(url: str) -> Part:
    if not isinstance(url, str) or not url:
        raise ApiError(400, "image url must be a non-empty string")
    if url.startswith("data:"):
        data, media_type = _decode_data_url(url)
    elif url.startswith(("http://", "https://")):
        data, media_type = _fetch_url(url)
    else:
        raise ApiError(400, "unsupported image url: expected a data: or http(s):// URL")
    return Part("image", data=data, media_type=media_type)


# --------------------------------------------------------------------------
# OpenAI /v1/chat/completions
# --------------------------------------------------------------------------

def _openai_content_parts(content, role: str) -> list[Part]:
    if isinstance(content, str):
        return [Part("text", text=content)]
    if not isinstance(content, list):
        raise ApiError(400, f"message content for role '{role}' must be a string or an array of parts")
    parts: list[Part] = []
    for item in content:
        if not isinstance(item, dict):
            raise ApiError(400, "content parts must be objects")
        ptype = item.get("type")
        if ptype == "text":
            text = item.get("text")
            if not isinstance(text, str):
                raise ApiError(400, "text part requires a 'text' string")
            parts.append(Part("text", text=text))
        elif ptype == "image_url":
            image_url = item.get("image_url")
            if not isinstance(image_url, dict):
                raise ApiError(400, "image_url part requires an 'image_url' object")
            parts.append(_image_from_url(image_url.get("url")))
        elif ptype == "input_audio":
            audio = item.get("input_audio")
            if not isinstance(audio, dict) or not isinstance(audio.get("data"), str):
                raise ApiError(400, "input_audio part requires 'input_audio.data' base64 string")
            fmt = audio.get("format") or "wav"
            media_type = {"wav": "audio/wav", "mp3": "audio/mpeg"}.get(fmt, f"audio/{fmt}")
            parts.append(Part("audio", data=_b64decode(audio["data"]), media_type=media_type))
        else:
            raise ApiError(400, f"unsupported content part type: {ptype!r}")
    return parts


def normalize_openai(payload: dict) -> UnifiedRequest:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "'messages' must be a non-empty array")

    parsed: list[Message] = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise ApiError(400, "each message must be an object")
        role = msg.get("role")
        if role not in ("system", "user", "assistant"):
            raise ApiError(400, f"unsupported message role: {role!r}")
        parts = _openai_content_parts(msg.get("content"), role)
        parsed.append(Message(role, parts))

    system = None
    if parsed[0].role == "system":
        system = "".join(p.text for p in parsed[0].parts if p.kind == "text") or None
        parsed = parsed[1:]
    if not parsed:
        raise ApiError(400, "'messages' must contain at least one non-system message")
    if not any(m.role == "user" for m in parsed):
        raise ApiError(400, "'messages' must contain at least one user message")

    max_tokens = payload.get("max_tokens", payload.get("max_completion_tokens"))
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ApiError(400, "'max_tokens' must be a positive integer")

    stop = payload.get("stop", [])
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or not all(isinstance(s, str) for s in stop):
        raise ApiError(400, "'stop' must be a string or an array of strings")

    return UnifiedRequest(
        model=payload.get("model") or "",
        messages=parsed,
        system=system,
        max_tokens=max_tokens,
        temperature=_opt_float(payload.get("temperature"), "temperature"),
        top_p=_opt_float(payload.get("top_p"), "top_p"),
        top_k=_opt_int(payload.get("top_k"), "top_k"),
        stop=tuple(s for s in stop if s),
        stream=bool(payload.get("stream", False)),
    )


# --------------------------------------------------------------------------
# Anthropic /v1/messages
# --------------------------------------------------------------------------

def _anthropic_system(payload: dict) -> Optional[str]:
    system = payload.get("system")
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        if not all(isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str) for b in system):
            raise ApiError(400, "'system' must be a string or an array of text blocks")
        return "".join(b["text"] for b in system) or None
    raise ApiError(400, "'system' must be a string or an array of text blocks")


def _anthropic_content_parts(content, role: str) -> list[Part]:
    if isinstance(content, str):
        return [Part("text", text=content)]
    if not isinstance(content, list):
        raise ApiError(400, f"message content for role '{role}' must be a string or an array of blocks")
    parts: list[Part] = []
    for block in content:
        if not isinstance(block, dict):
            raise ApiError(400, "content blocks must be objects")
        btype = block.get("type")
        if btype == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ApiError(400, "text block requires a 'text' string")
            parts.append(Part("text", text=text))
        elif btype == "image":
            source = block.get("source")
            if not isinstance(source, dict):
                raise ApiError(400, "image block requires a 'source' object")
            stype = source.get("type")
            if stype == "base64":
                media_type = source.get("media_type")
                if not isinstance(media_type, str) or not media_type.startswith("image/"):
                    raise ApiError(400, "base64 image source requires an 'image/*' media_type")
                parts.append(Part("image", data=_b64decode(source.get("data", "")), media_type=media_type))
            elif stype == "url":
                parts.append(_image_from_url(source.get("url")))
            else:
                raise ApiError(400, f"unsupported image source type: {stype!r}")
        else:
            raise ApiError(400, f"unsupported content block type: {btype!r} (tools are not supported yet)")
    return parts


def normalize_anthropic(payload: dict) -> UnifiedRequest:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "'messages' must be a non-empty array")

    parsed: list[Message] = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise ApiError(400, "each message must be an object")
        role = msg.get("role")
        if role not in ("user", "assistant"):
            raise ApiError(400, f"unsupported message role: {role!r}")
        parts = _anthropic_content_parts(msg.get("content"), role)
        parsed.append(Message(role, parts))

    max_tokens = payload.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ApiError(400, "'max_tokens' is required and must be a positive integer")

    stop = payload.get("stop_sequences", [])
    if not isinstance(stop, list) or not all(isinstance(s, str) for s in stop):
        raise ApiError(400, "'stop_sequences' must be an array of strings")

    return UnifiedRequest(
        model=payload.get("model") or "",
        messages=parsed,
        system=_anthropic_system(payload),
        max_tokens=max_tokens,
        temperature=_opt_float(payload.get("temperature"), "temperature"),
        top_p=_opt_float(payload.get("top_p"), "top_p"),
        top_k=_opt_int(payload.get("top_k"), "top_k"),
        stop=tuple(s for s in stop if s),
        stream=bool(payload.get("stream", False)),
    )


# --------------------------------------------------------------------------
# shared scalar validation
# --------------------------------------------------------------------------

def _opt_float(value, name: str) -> Optional[float]:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ApiError(400, f"'{name}' must be a number")
    return float(value)


def _opt_int(value, name: str) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ApiError(400, f"'{name}' must be an integer")
    return value
