"""Qwen3-TTS variant seam (#47): which ``tts_model_type`` a checkpoint is,
and which of them this build can synthesize — **per generation path**.

The family ships three types (HF ``config.json`` → ``tts_model_type``):

- ``base`` — no ``spk_id`` presets; zero-shot voice cloning via
  ``ref_audio``/``ref_text`` only (mlx-audio builds the ECAPA speaker
  encoder solely for base checkpoints — qwen3_tts.py:180; its README table
  "Fast, predefined voices" for Base is stale, see #45's correction).
- ``custom_voice`` — ``spk_id`` presets + emotion ``instruct``.
- ``voice_design`` — any voice from a text description (#46): the
  description rides ``instruct`` (mlx-audio's own ``generate()`` maps it
  that way), the speaker row is simply absent from the prompt.

Served-ness is a path × type question, not a type question (#49):

=================  ==============  =====================================
path               served types    entry
=================  ==============  =====================================
preset (buffered)  custom_voice    :func:`generate.synthesize`
preset (streaming) custom_voice    :func:`stream_loop.synthesize_stream`
clone (buffered)   base            :func:`generate.synthesize_clone`
clone (streaming)  base            :func:`stream_loop.synthesize_clone_stream`
design (buffered)  voice_design    :func:`generate.synthesize_design`
design (streaming) voice_design    :func:`stream_loop.synthesize_stream` (#52)
=================  ==============  =====================================
"""

from __future__ import annotations

from typing import Any

BASE = "base"
CUSTOM_VOICE = "custom_voice"
VOICE_DESIGN = "voice_design"

#: ``tts_model_size`` tags the family ships ("0b6"/"1b7"); the small
#: CustomVoice model was not trained for ``instruct`` — mlx-audio's own
#: 0.6B guard is dead code (tests ``!= custom_voice`` inside the
#: ``== custom_voice`` branch), so the rejection has to live here
SMALL_SIZE = "0b6"

#: every ``tts_model_type`` mlx-audio's qwen3_tts knows
KNOWN = (BASE, CUSTOM_VOICE, VOICE_DESIGN)

#: generation path name → the types this build synthesizes on it
SERVED_BY_PATH = {
    "preset": (CUSTOM_VOICE,),
    "clone": (BASE,),  # buffered
    "clone_stream": (BASE,),  # streaming (#50)
    "design": (VOICE_DESIGN,),  # buffered (#51) + streaming (#52)
}

#: kept from #47 for the preset paths — ``SERVED_BY_PATH["preset"]``
SERVED = SERVED_BY_PATH["preset"]

_TRACKING = {
    (BASE, "preset"): (
        "Base models have no preset voices — send voice as an object with "
        "ref_audio/ref_text to clone a voice (buffered and streaming, #49/#50)"
    ),
    (CUSTOM_VOICE, "clone"): (
        "voice cloning needs a Base checkpoint; CustomVoice serves preset "
        "voices (send voice as a speaker string)"
    ),
    (CUSTOM_VOICE, "clone_stream"): (
        "voice cloning needs a Base checkpoint; CustomVoice serves preset "
        "voices (send voice as a speaker string)"
    ),
    (VOICE_DESIGN, "preset"): (
        "VoiceDesign checkpoints have no preset voices — the voice comes from "
        "`instructions` (a text description); preset `voice` is not accepted"
    ),
    (VOICE_DESIGN, "clone"): (
        "voice cloning needs a Base checkpoint; VoiceDesign takes a text "
        "description in `instructions` (#46)"
    ),
    (VOICE_DESIGN, "clone_stream"): (
        "voice cloning needs a Base checkpoint; VoiceDesign takes a text "
        "description in `instructions` (#46)"
    ),
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


def require_served(variant: str, path: str = "preset") -> None:
    """Raise ValueError unless this build synthesizes ``variant`` on ``path``.

    The message names the tracking issue or the correct request shape so a
    400 carries guidance instead of mlx-audio's generic error from inside
    the generation loop.
    """
    if path not in SERVED_BY_PATH:
        raise ValueError(f"unknown generation path {path!r}")
    if variant in SERVED_BY_PATH[path]:
        return
    raise ValueError(_TRACKING.get((variant, path), f"Qwen3-TTS '{variant}' models are not served on the {path} path yet"))


def model_size(model: Any) -> str:
    """The checkpoint's ``tts_model_size`` tag ("" when absent — older or
    novel checkpoints simply aren't small)."""
    raw = getattr(model.config, "tts_model_size", None)
    return raw.strip().lower() if isinstance(raw, str) else ""


def ensure_served(model: Any, path: str = "preset") -> str:
    """:func:`model_variant` + :func:`require_served` — the entry guard for
    every generation path (synthesize, synthesize_stream, synthesize_clone)."""
    variant = model_variant(model)
    require_served(variant, path)
    return variant
