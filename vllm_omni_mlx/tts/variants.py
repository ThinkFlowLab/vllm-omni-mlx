"""Qwen3-TTS variant seam (#47): which ``tts_model_type`` a checkpoint is,
and which of them this build can synthesize.

The family ships three types (HF ``config.json`` → ``tts_model_type``):

- ``base`` — no ``spk_id`` presets; zero-shot voice cloning via
  ``ref_audio``/``ref_text`` only (mlx-audio builds the ECAPA speaker
  encoder solely for base checkpoints — qwen3_tts.py:180; its README table
  "Fast, predefined voices" for Base is stale, see #45's correction).
  Serving cloning: #49 buffered, #50 streaming.
- ``custom_voice`` — ``spk_id`` presets + emotion ``instruct`` — the type
  this build serves.
- ``voice_design`` — any voice from a text description (#46).
"""

from __future__ import annotations

from typing import Any

BASE = "base"
CUSTOM_VOICE = "custom_voice"
VOICE_DESIGN = "voice_design"

#: every ``tts_model_type`` mlx-audio's qwen3_tts knows
KNOWN = (BASE, CUSTOM_VOICE, VOICE_DESIGN)
#: the types this build can synthesize
SERVED = (CUSTOM_VOICE,)

_TRACKING = {
    BASE: "voice cloning (ref_audio/ref_text) — #49 buffered, #50 streaming",
    VOICE_DESIGN: "text-described voices — #46",
}


def model_variant(model: Any) -> str:
    """The checkpoint's ``tts_model_type``, validated against :data:`KNOWN`.

    A novel or corrupt config fails here with the raw value in the message,
    not deep inside generation where the cause is opaque.
    """
    raw = getattr(model.config, "tts_model_type", None)
    variant = raw.strip().lower() if isinstance(raw, str) else ""
    if variant not in KNOWN:
        raise ValueError(
            f"unknown TTS model type {raw!r}; expected one of {', '.join(KNOWN)}"
        )
    return variant


def require_served(variant: str) -> None:
    """Raise ValueError unless this build synthesizes ``variant``.

    The message names the tracking issue so a 400 carries guidance instead
    of mlx-audio's generic error from inside the generation loop.
    """
    if variant not in SERVED:
        raise ValueError(
            f"Qwen3-TTS '{variant}' models are not supported by this build yet "
            f"({_TRACKING[variant]})"
        )


def ensure_served(model: Any) -> str:
    """``model_variant`` + :func:`require_served` — the entry guard for both
    generation paths (:func:`generate.synthesize`,
    :func:`stream_loop.synthesize_stream`)."""
    variant = model_variant(model)
    require_served(variant)
    return variant
