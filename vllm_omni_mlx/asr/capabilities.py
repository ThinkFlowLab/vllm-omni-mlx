"""ASR capability matrix (#68): which checkpoint families this build serves.

Two classes, because they have different metric frames and different
optimization surfaces (the same role ``tts/variants.py`` plays for TTS):

- ``decoder`` — the audio adapter conditions a Qwen3-class text decoder, so
  the output is autoregressive text and the batch-1 chat doctrine transfers:
  TPOT/ITL are the sustained metrics, TTFT (incl. audio-adapter prefill) the
  entry metric, and the static text prompt is a prefix-cache key.
- ``encoder_decoder`` — whisper / parakeet / sensevoice style: RTF per window,
  no cacheable text prefix. Explicitly secondary (#68 non-goals).

Keyed on ``config.json`` ``model_type`` so a wrong checkpoint is rejected at
boot with guidance, not with a stack trace from inside generation.
"""

from __future__ import annotations

from dataclasses import dataclass

DECODER = "decoder"
ENCODER_DECODER = "encoder_decoder"

#: class → (headline metrics, prefix-cacheable)
CLASS_TRAITS = {
    DECODER: (("TTFT", "TPOT", "ITL"), True),
    ENCODER_DECODER: (("RTF per window",), False),
}


@dataclass(frozen=True)
class Family:
    model_type: str
    style: str
    served: bool
    note: str = ""


FAMILIES = {
    f.model_type: f
    for f in (
        Family("qwen3_asr", DECODER, True),
        Family("qwen2_audio", DECODER, False, "queued behind qwen3_asr via this matrix (#68)"),
        Family("voxtral", DECODER, False, "queued behind qwen3_asr via this matrix (#68)"),
        Family("whisper", ENCODER_DECODER, False, "encoder-decoder families are a separate, later issue"),
        Family("parakeet", ENCODER_DECODER, False, "encoder-decoder families are a separate, later issue"),
        Family("parakeet_tdt", ENCODER_DECODER, False, "encoder-decoder families are a separate, later issue"),
        Family("sensevoice", ENCODER_DECODER, False, "encoder-decoder families are a separate, later issue"),
    )
}

SERVED = tuple(f.model_type for f in FAMILIES.values() if f.served)


def family_of(config: dict) -> str:
    """The checkpoint's ``model_type`` ("" when absent)."""
    raw = config.get("model_type")
    return raw.strip().lower() if isinstance(raw, str) else ""


def require_served(model_type: str) -> Family:
    """Return the :class:`Family` or raise ValueError carrying guidance."""
    family = FAMILIES.get(model_type)
    if family is None:
        raise ValueError(
            f"unknown ASR model type {model_type!r}; this build serves: {', '.join(SERVED)}"
        )
    if not family.served:
        raise ValueError(
            f"ASR model type {model_type!r} ({family.style}) is not served yet — "
            f"{family.note}; this build serves: {', '.join(SERVED)}"
        )
    return family
