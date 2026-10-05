"""A/B bench for Qwen3-TTS voice cloning (#49): Base ICL vs CustomVoice
presets on the same text, plus the ref-encode cache effect.

Fresh numbers per the house rule — run it, don't quote old comments:

    .venv/bin/python scripts/bench_clone.py ["text ..."]

Reports buffered totals (streaming clone numbers live on #50): wall time,
audio duration, RTF, peak unified memory. The donor reference clip is
synthesized deterministically from the CustomVoice checkpoint (seed 7), so
preset and clone sides speak comparable content. Wall time varies with
unseeded generation length — compare RTF, not raw wall.
"""

import argparse
import base64
import sys
import time

import mlx.core as mx

from vllm_omni_mlx.tts.config import TTSConfig, load_tts_model
from vllm_omni_mlx.tts.generate import synthesize, wav_bytes
from vllm_omni_mlx.tts.service import TTSService

CUSTOM = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"
BASE = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"
REF_TEXT = "This is the voice we are cloning today."


def bench(label: str, runs: int, fn) -> None:
    for i in range(runs):
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        payload = fn()
        dt = time.perf_counter() - t0
        audio_s = (len(payload) - 44) / 2 / 24000  # 16-bit mono past the RIFF header
        peak = mx.get_peak_memory() / 2**30
        print(f"{label} run{i + 1}: {dt:.2f}s wall, {audio_s:.2f}s audio, RTF {dt / audio_s:.3f}, peak {peak:.2f} GiB", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text", nargs="?", default=(
        "Voice cloning lets any speaker be reproduced from a short reference clip, "
        "and the same sentence serves both sides of this comparison."
    ))
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    text = args.text

    cv_model = load_tts_model(TTSConfig(model_ref=CUSTOM))
    cv = TTSService(cv_model, TTSConfig(model_ref=CUSTOM))
    bench("preset-vivian", args.runs, lambda: cv.speech_bytes(text, voice="vivian")[0])

    clip = wav_bytes(synthesize(cv_model, TTSConfig(), REF_TEXT, seed=7))
    print(f"reference clip: {len(clip) / 2 / 24000:.2f}s (deterministic, seed 7)", flush=True)

    base_model = load_tts_model(TTSConfig(model_ref=BASE))
    base = TTSService(base_model, TTSConfig(model_ref=BASE))
    print("has_encoder:", base_model.speech_tokenizer.has_encoder,
          "| speaker_encoder:", base_model.speaker_encoder is not None, flush=True)
    voice = {"ref_audio": base64.b64encode(clip).decode(), "ref_text": REF_TEXT}
    bench("clone", args.runs, lambda: base.speech_bytes(text, voice=voice)[0])

    # ref-encode cache effect in isolation: the ICL input prep pays the
    # speech-tokenizer encode once per (ref_text, audio) pair, then hits
    # mlx-audio's _icl_cache
    from vllm_omni_mlx.tts.generate import decode_ref_audio

    audio = decode_ref_audio(voice["ref_audio"])
    for label in ("ref prep cold (encode)", "ref prep warm (cache)"):
        t0 = time.perf_counter()
        base_model._prepare_icl_generation_inputs(text=text, ref_audio=audio, ref_text=REF_TEXT)
        mx.eval(mx.array(0))
        print(f"{label}: {time.perf_counter() - t0:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
