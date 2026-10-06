#!/usr/bin/env python3
"""Build the local ASR round-trip oracle (#88): a whisper-base model dir
mlx-audio's stt loader can use, assembled from two sources because no single
small repo carries both parts:

  - weights: mlx-community/whisper-base-mlx (mlx-whisper layout, npz) —
    converted to weights.safetensors, the format mlx_audio.stt globs
  - processor files: openai/whisper-base (preprocessor/tokenizer/generation
    configs) — mlx_audio's post-load hook runs WhisperProcessor.from_pretrained
    on the model dir

Serves the "same audio quality" gate for optimization PRs: transcribe the
synthesized audio and score it against the input text, so quality changes are
a number instead of an opinion (upstream vllm-omni tests use the same
round-trip idea, cosine > 0.9). Output: ~/.cache/vllm-omni-mlx/asr-oracle
(~145 MB). huggingface.co is unreachable from this network — downloads go
through hf-mirror.com with stall detection.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

MIRROR = "https://hf-mirror.com"
WEIGHTS_REPO = "mlx-community/whisper-base-mlx"
PROCESSOR_REPO = "openai/whisper-base"
PROCESSOR_FILES = ("preprocessor_config.json", "generation_config.json", "normalizer.json", "tokenizer.json")


def fetch(url: str, out: pathlib.Path, min_speed: int = 20000) -> None:
    for attempt in range(1, 4):
        result = subprocess.run(
            [
                "curl", "-sS", "-L", "--max-time", "600",
                "--speed-limit", str(min_speed), "--speed-time", "25",
                "-o", str(out), url,
            ],
        )
        if result.returncode == 0 and out.stat().st_size > 0:
            return
        print(f"retry {attempt} for {url}", file=sys.stderr)
    raise RuntimeError(f"could not fetch {url}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(pathlib.Path.home() / ".cache/vllm-omni-mlx/asr-oracle"))
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    fetch(f"{MIRROR}/{WEIGHTS_REPO}/resolve/main/config.json", out / "config.json")
    for name in PROCESSOR_FILES:
        fetch(f"{MIRROR}/{PROCESSOR_REPO}/resolve/main/{name}", out / name)
    npz = out / "weights.npz"
    if not (out / "weights.safetensors").exists():
        fetch(f"{MIRROR}/{WEIGHTS_REPO}/resolve/main/weights.npz", npz)
        weights = dict(mx.load(str(npz)))
        mx.save_safetensors(str(out / "weights.safetensors"), weights)
        npz.unlink()
    print(f"oracle ready at {out} ({(out / 'weights.safetensors').stat().st_size / 2**20:.0f} MiB weights)")

    # functional check: load through mlx-audio and transcribe a bundled clip
    from mlx_audio.stt.generate import generate_transcription
    from mlx_audio.stt.utils import load_model

    model = load_model(str(out))
    clips = sorted(
        (pathlib.Path.home() / ".cache/huggingface/hub/models--mlx-community--VoxCPM2-4bit/snapshots").glob("*/test_en.wav")
    )
    if clips:
        result = generate_transcription(model=model, audio=str(clips[0]), output_path="/tmp/asr_oracle_check", format="txt", verbose=False)
        text = getattr(result, "text", "")
        print(f"check transcript ({len(text)} chars): {text[:120]!r}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
