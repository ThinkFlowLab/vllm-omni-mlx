"""ASR serving service (#68): serialized, validated transcription.

Mirrors ``tts/service.py``: the single-user lock (batch-1 target), and
request-level validation where every bad input is a ValueError the server
maps to a 400 *before* the lock is taken (decode errors and caps are request
errors, not generation failures).

The OpenAI ``prompt`` field rides mlx-audio's ``system_prompt`` — the static
text head of the decoder prompt, which is also the prefix-cache key later
(task 5). ``hotwords`` is an extension field folded into the same head by
mlx-audio itself.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from ..audio_io import decode_audio
from .config import SAMPLE_RATE, ASRConfig
from .capabilities import family_of, require_served

#: formats this build renders; srt/vtt need segment timestamps the decoder
#: family does not emit (forced-aligner sibling, later)
RESPONSE_FORMATS = ("json", "text", "verbose_json")

#: upload cap, same as the OpenAI API's
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

#: ISO-639-1 → the language names Qwen3-ASR's prompt template uses
_ISO_TO_NAME = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German",
    "fr": "French", "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
    "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "ja": "Japanese",
    "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech", "fil": "Filipino",
    "fa": "Persian", "el": "Greek", "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian",
}


@dataclass
class Transcription:
    text: str
    language: Optional[str] = None
    duration: float = 0.0  # input audio seconds
    segments: list = field(default_factory=list)
    prompt_tokens: int = 0
    generation_tokens: int = 0


class ASRService:
    def __init__(self, model: Any, config: ASRConfig | None = None):
        self._model = model
        self.config = config or ASRConfig()
        self._model_type = family_of({"model_type": getattr(model.config, "model_type", None)})
        require_served(self._model_type)  # wrong family fails at boot
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.config.model_ref

    @property
    def model_type(self) -> str:
        return self._model_type

    def transcribe(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        hotwords: Optional[list[str]] = None,
        temperature: Optional[float] = None,
    ) -> Transcription:
        """Transcribe encoded audio bytes. Raises ValueError on invalid
        requests; generation is serialized under the service lock."""
        if not audio:
            raise ValueError("file must be non-empty audio")
        if len(audio) > MAX_UPLOAD_BYTES:
            raise ValueError(f"file is {len(audio) / 2**20:.1f} MiB; the cap is {MAX_UPLOAD_BYTES // 2**20} MiB")
        if temperature is not None and not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        lang = self._resolve_language(language)
        waveform = decode_audio(audio, SAMPLE_RATE, "file")
        duration = waveform.size / SAMPLE_RATE
        cfg = self.config.with_overrides(temperature=temperature)
        kwargs: dict[str, Any] = {
            "language": lang if lang is not None else cfg.language,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "verbose": False,
        }
        if prompt and prompt.strip():
            kwargs["system_prompt"] = prompt.strip()
        if hotwords:
            kwargs["hotwords"] = hotwords
        with self._lock:
            out = self._model.generate(waveform, **kwargs)
        detected = getattr(out, "language", None)
        if isinstance(detected, (list, tuple)):  # mlx-audio reports a list for single input
            detected = detected[0] if detected else None
        return Transcription(
            text=(out.text or "").strip(),
            language=detected,
            duration=duration,
            segments=list(getattr(out, "segments", None) or []),
            prompt_tokens=int(getattr(out, "prompt_tokens", 0) or 0),
            generation_tokens=int(getattr(out, "generation_tokens", 0) or 0),
        )

    def _resolve_language(self, language: Optional[str]) -> Optional[str]:
        """ISO-639-1 code or language name → the name the model's prompt
        template uses; None → auto-detect. Unsupported → ValueError listing
        what the checkpoint supports."""
        if language is None or not language.strip():
            return None
        raw = language.strip()
        wanted = _ISO_TO_NAME.get(raw.lower(), raw)
        supported = list(getattr(self._model.config, "support_languages", None) or [])
        if not supported:
            return wanted
        by_lower = {name.lower(): name for name in supported}
        if wanted.lower() not in by_lower:
            raise ValueError(f"language {language!r} is not supported; supported: {', '.join(supported)}")
        return by_lower[wanted.lower()]


def render(result: Transcription, response_format: str) -> tuple[Any, str]:
    """(payload, kind): ``kind`` is "json" (payload is a dict) or "text"."""
    if response_format == "text":
        return result.text, "text"
    if response_format == "json":
        return {"text": result.text}, "json"
    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": result.language,
            "duration": round(result.duration, 3),
            "text": result.text,
            "segments": result.segments,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.generation_tokens,
            },
        }, "json"
    raise ValueError(
        f"response_format must be one of {', '.join(RESPONSE_FORMATS)}, got '{response_format}'"
    )
