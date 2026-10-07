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


def _similarity(transcript: str, reference: str) -> float:
    """Transcript-vs-reference score in [0, 1].

    Two whisper behaviors would otherwise punish good audio: short clips
    get padded to 30 s and hallucinate past the speech ("如果如果…",
    "Thanks for watching…" — trailing noise unrelated to quality), and
    Chinese comes out in traditional characters while the reference is
    often simplified. The transcript is therefore trimmed to the span
    that could plausibly contain the reference, and the caller may pass
    several reference variants (scripts) and keep the best.
    """
    from difflib import SequenceMatcher

    ref = _normalize(reference)
    trimmed = _normalize(transcript)[: 2 * len(ref) + 24]
    return SequenceMatcher(None, trimmed, ref).ratio()


def round_trip_similarity(model: Any, audio, sample_rate: int, reference) -> float:
    """Transcribe `audio` (mx.array float waveform) and score the transcript
    against `reference` (the synthesized text, or a list of acceptable
    variants — e.g. simplified and traditional Chinese), in [0, 1].

    The waveform goes through a temp 16-bit WAV — mlx-audio's stt path does
    its own resampling to 16 kHz mel features from a file. A CJK reference
    pins whisper's language to zh: its auto-detection wobbles on Mandarin
    and is the dominant source of score variance.
    """
    import mlx.core as mx

    from mlx_audio.stt.generate import generate_transcription

    samples = mx.clip(audio.reshape(-1), -1.0, 1.0)
    pcm = (samples * 32767.0).astype(mx.int16).tolist()
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = str(Path(tmpdir) / "clip.wav")
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
        with unittest.mock.patch("sys.stderr"):  # silence the frames/s bar
            kwargs = {}
            first = reference if isinstance(reference, str) else reference[0]
            if any("\u4e00" <= ch <= "\u9fff" for ch in first):
                kwargs["language"] = "zh"
            result = generate_transcription(
                model=model, audio=wav_path, output_path=str(Path(tmpdir) / "out"), format="txt", verbose=False, **kwargs
            )
    transcript = getattr(result, "text", "") or ""
    references = reference if isinstance(reference, (list, tuple)) else [reference]
    return max(_similarity(transcript, ref) for ref in references)


def requires_oracle(test):
    """Skip decorator for tests that need the local oracle.

    Applies the decorator to `test` — returning it unapplied compiles fine
    and unittest then "passes" the test vacuously with only a
    DeprecationWarning (caught in review on #98: the equivalence gate had
    never actually run).
    """
    return unittest.skipUnless(oracle_dir() is not None, "needs the ASR oracle (scripts/build_asr_oracle.py)")(test)
