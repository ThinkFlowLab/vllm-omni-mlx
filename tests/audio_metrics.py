"""Audio-correctness metrics for TTS tests, ported from vllm-omni's
`tests/helpers/assertions.py` (Apache-2.0): the harmonics-to-noise ratio
check that catches catastrophic decode failures (noise where speech should
be) without any models — pure numpy.
"""

from __future__ import annotations

import numpy as np

SPEECH_SAMPLE_RATE_HZ = 24000
MIN_SPEECH_HNR_DB = 1.0  # upstream's floor for clean full-precision codecs
# our calibrated floors for the 4-bit CustomVoice model (issue #37): white
# noise measures ~-10 dB; clean voices (vivian) span 0.95–2.44 dB across
# greedy/seeded draws (kernel nondeterminism shifts HNR ~1 dB), noisier
# timbres (aiden 0.47–0.84, ryan -0.84–+2.29) sit below upstream's 1.0 dB
# intrinsically
CLEAN_VOICE_HNR_DB = 0.0
CATASTROPHIC_HNR_DB = -5.0


def pcm_hnr_db(samples: np.ndarray, sr: int = SPEECH_SAMPLE_RATE_HZ) -> float:
    """Mean per-frame HNR estimate (dB) via normalized autocorrelation.

    30 ms frames, 15 ms hop; for each non-silent frame the autocorrelation
    peak within the 80–400 Hz pitch-lag band gives peak/(1−peak) in dB.
    Periodic speech scores high; noise centers near or below 0 dB.
    """
    frame_len = int(0.03 * sr)
    hop = frame_len // 2
    hnr_values: list[float] = []
    for start in range(0, len(samples) - frame_len, hop):
        frame = samples[start : start + frame_len].astype(np.float32, copy=False)
        frame = frame - np.mean(frame)
        if np.max(np.abs(frame)) < 0.01:
            continue
        ac = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        ac = ac / (ac[0] + 1e-10)
        min_lag = int(sr / 400)
        max_lag = min(int(sr / 80), len(ac))
        if min_lag >= max_lag:
            continue
        peak = float(np.max(ac[min_lag:max_lag]))
        if 0 < peak < 1:
            hnr_values.append(10 * np.log10(peak / (1 - peak + 1e-10)))
    return float(np.mean(hnr_values)) if hnr_values else 0.0


def int16_pcm_hnr_db(pcm_bytes: bytes, sr: int = SPEECH_SAMPLE_RATE_HZ) -> float:
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm_hnr_db(samples, sr)


def wav_hnr_db(wav_bytes: bytes, sr: int = SPEECH_SAMPLE_RATE_HZ) -> float:
    import io
    import wave

    with wave.open(io.BytesIO(wav_bytes)) as wav:
        assert wav.getframerate() == sr and wav.getnchannels() == 1
        pcm = wav.readframes(wav.getnframes())
    return int16_pcm_hnr_db(pcm, sr)
