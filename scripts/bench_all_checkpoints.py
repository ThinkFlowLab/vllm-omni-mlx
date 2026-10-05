#!/usr/bin/env python3
"""Per-checkpoint streaming bench on the merged optimization stack.

One checkpoint per process (16 GB rule): load, prewarm (0.5 s / 0.08 s —
the serving defaults), one discarded warmup generation (warms the prefix
cache where the path has one), then N measured turns of first-chunk
latency + RTF with cool-down sleeps. Base runs the clone stream against a
reference clip synthesized deterministically by the CustomVoice model
(stage 1 writes /tmp/qwen3tts_ref.wav; stage 2 consumes it).

Usage: scripts/bench_all_checkpoints.py cv17|cv06|vd17|make-ref|base17
"""

from __future__ import annotations

import os
import statistics
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx

from vllm_omni_mlx.tts.config import TTSConfig, load_tts_model, local_snapshot
from vllm_omni_mlx.tts.stream_loop import prewarm_streaming, synthesize_clone_stream, synthesize_stream

CV17 = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"
CV06 = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
VD17 = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"
BASE17 = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"

TEXT = (
    "Welcome to the per checkpoint benchmark. This paragraph runs long enough "
    "that generation reaches steady state before it ends, so the real time "
    "factor reflects each model's decode loop rather than startup noise."
) * 2

VD_INSTRUCT = "A cheerful young female voice with high pitch and energetic tone"
REF_TEXT = "This is the voice we are cloning today."
REF_WAV = "/tmp/qwen3tts_ref.wav"


def ramp(seconds=1.0):
    a = mx.random.normal((2048, 2048))
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = a @ a
        mx.eval(a)
        a = a / mx.sqrt((a * a).mean())
    mx.clear_cache()


def measure(label, gen, turns=3, cooldown=6.0):
    firsts, rtfs = [], []
    for _ in range(turns):
        time.sleep(cooldown)
        t0 = time.perf_counter()
        first = None
        samples = 0
        for chunk in gen():
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            samples += chunk.shape[0]
        wall = time.perf_counter() - t0
        firsts.append(first)
        rtfs.append(wall / (samples / 24000.0))
        print(f"  {label}: first {first:6.0f} ms   RTF {rtfs[-1]:.3f}", flush=True)
    print(f"== {label}: first-chunk p50 {statistics.median(firsts):6.0f} ms (min {min(firsts):.0f})   "
          f"RTF p50 {statistics.median(rtfs):.3f} (min {min(rtfs):.3f})   peak {mx.get_peak_memory() / 2**30:.2f} GiB", flush=True)


def run_speech(model_ref, voice_kwargs):
    if local_snapshot(model_ref) is None:
        raise SystemExit(f"{model_ref} not cached locally")
    print(f"loading {model_ref} …", flush=True)
    model = load_tts_model(TTSConfig(model_ref=model_ref))
    cfg = TTSConfig(model_ref=model_ref)
    prewarm_streaming(model, 0.5, 0.08)
    for _ in synthesize_stream(model, cfg, TEXT, seed=7, max_tokens=40, **voice_kwargs):
        pass  # warmup: traces + prefix cache where applicable
    ramp(1.0)
    measure(model_ref.split("Hz-")[1], lambda: synthesize_stream(
        model, cfg, TEXT, seed=7, max_tokens=240, streaming_interval=0.5, **voice_kwargs))


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv17"
    if mode == "cv17":
        run_speech(CV17, {"speaker": "vivian"})
    elif mode == "cv06":
        run_speech(CV06, {"speaker": "vivian"})
    elif mode == "vd17":
        run_speech(VD17, {"instruct": VD_INSTRUCT})
    elif mode == "make-ref":
        from vllm_omni_mlx.tts.generate import synthesize, wav_bytes
        model = load_tts_model(TTSConfig(model_ref=CV17))
        wav = wav_bytes(synthesize(model, TTSConfig(), REF_TEXT, seed=7))
        open(REF_WAV, "wb").write(wav)
        print(f"reference clip written: {len(wav) / 2 / 24000:.2f}s", flush=True)
    elif mode == "base17":
        import array
        raw = open(REF_WAV, "rb").read()
        samples = array.array("h", raw[44:])
        ref_audio = mx.array([s / 32767.0 for s in samples])
        model = load_tts_model(TTSConfig(model_ref=BASE17))
        cfg = TTSConfig(model_ref=BASE17)
        prewarm_streaming(model, 0.5, 0.08)
        for _ in synthesize_clone_stream(model, cfg, TEXT, ref_audio, REF_TEXT, seed=7, max_tokens=40):
            pass
        ramp(1.0)
        measure("clone-base17", lambda: synthesize_clone_stream(
            model, cfg, TEXT, ref_audio, REF_TEXT, seed=7, max_tokens=240, streaming_interval=0.5))
    else:
        raise SystemExit(f"unknown mode {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
