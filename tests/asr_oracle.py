"""ASR round-trip oracle (#88): transcribe synthesized speech and score it
against the input text — the "same audio quality" gate for optimization PRs.

The oracle is a local whisper-base dir built by ``scripts/build_asr_oracle.py``
(hybrid: mlx-community npz weights converted to safetensors + openai/whisper-base
processor files — no single small repo carries both in mlx-audio's stt layout).
It resolves from ``~/.cache/vllm-omni-mlx/asr-oracle`` or
``$VLLM_OMNI_ASR_ORACLE``; where absent, :func:`load_oracle` returns None and
callers skip (the HNR floors stay the always-on gate; this is the stronger,
machine-dependent one).
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest.mock
import wave
from pathlib import Path
from typing import Any

DEFAULT_ORACLE_DIR = Path.home() / ".cache/vllm-omni-mlx/asr-oracle"

_MODEL: Any = None
_LOADED = False


def oracle_dir() -> Path | None:
    env = os.environ.get("VLLM_OMNI_ASR_ORACLE")
    if env:
        p = Path(env)
        return p if (p / "weights.safetensors").exists() else None
    return DEFAULT_ORACLE_DIR if (DEFAULT_ORACLE_DIR / "weights.safetensors").exists() else None


def load_oracle():
    """The stt model, or None where the oracle isn't built (skip, not fail)."""
    global _MODEL, _LOADED
    if not _LOADED:
        _LOADED = True
        path = oracle_dir()
        if path is not None:
            try:
                from mlx_audio.stt.utils import load_model

                _MODEL = load_model(str(path))
            except Exception:
                _MODEL = None
    return _MODEL


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def round_trip_similarity(model: Any, audio, sample_rate: int, reference: str) -> float:
    """Transcribe `audio` (mx.array float waveform) and score the transcript
    against `reference` (the text that was synthesized), in [0, 1].

    The waveform goes through a temp 16-bit WAV — mlx-audio's stt path does
    its own resampling to 16 kHz mel features from a file.
    """
    import mlx.core as mx

    from mlx_audio.stt.generate import generate_transcription

    samples = mx.clip(audio.reshape(-1), -1.0, 1.0)
    pcm = (samples * 32767.0).astype(mx.int16).tolist()
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        with wave.open(tmp.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
        with unittest.mock.patch("sys.stderr"):  # silence the frames/s bar
            result = generate_transcription(model=model, audio=tmp.name, output_path="/dev/null", format="txt", verbose=False)
    transcript = getattr(result, "text", "") or ""
    return _similarity(transcript, reference)


def requires_oracle(test):
    """Skip decorator for tests that need the local oracle."""
    return unittest.skipUnless(oracle_dir() is not None, "needs the ASR oracle (scripts/build_asr_oracle.py)")
