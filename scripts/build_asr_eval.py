#!/usr/bin/env python3
"""Build the local ASR knob-bench eval set (#68 perf evidence).

Downloads the ``clean/validation`` split of
``hf-internal-testing/librispeech_asr_dummy`` (LibriSpeech dev-clean samples,
CC-BY-4.0) and writes, under ``--out-dir`` (default
``~/.cache/vllm-omni-mlx/asr-eval``):

- ``short_XX.wav`` — 10 real-speech clips, 1–10 s, one per row (with pauses
  trimmed by the 1 s floor), plus their reference texts in the manifest
- ``long60.wav`` / ``long150.wav`` — the same rows concatenated with 0.25 s
  silence gaps to ~60 s / ~150 s, so long-audio knobs
  (``prefill_step_size``, ``chunk_duration``) are exercised at prompt lengths
  where they actually bite (≥ 4k audio tokens)
- ``manifest.json`` — ``[{name, text, seconds}]``, consumed by
  ``bench_asr_knobs.py``

Needs ``pyarrow`` (not a project dep; ``pip install pyarrow``) and network on
first build. ``HF_ENDPOINT`` is honored for restricted networks. GPU-free.

    python scripts/build_asr_eval.py
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import miniaudio
import numpy as np

DATASET = "hf-internal-testing/librispeech_asr_dummy"
PARQUET = "clean/validation-00000-of-00001.parquet"
SR = 16000
N_SHORT = 10
LONG_SPECS = {"long60": 60.0, "long150": 150.0}


def decode_to_f32(data: bytes) -> np.ndarray:
    dec = miniaudio.decode(
        data, nchannels=1, sample_rate=SR,
        output_format=miniaudio.SampleFormat.FLOAT32,
    )
    return np.array(dec.samples, dtype=np.float32)


def save_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def load_rows() -> list[dict]:
    from huggingface_hub import hf_hub_download

    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("build needs pyarrow: pip install pyarrow")
    path = hf_hub_download(DATASET, PARQUET, repo_type="dataset")
    return pq.read_table(path).to_pylist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-dir",
        default=str(Path.home() / ".cache/vllm-omni-mlx/asr-eval"),
    )
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    print(f"{len(rows)} rows from {DATASET}")

    manifest: list[dict] = []
    for r in rows:
        if len(manifest) >= N_SHORT:
            break
        try:
            x = decode_to_f32(r["audio"]["bytes"])
        except Exception as e:  # noqa: BLE001 — one bad row shouldn't kill the build
            print("skip (decode):", e)
            continue
        if not SR <= len(x) <= 10 * SR:  # 1–10 s clips only
            continue
        name = f"short_{len(manifest):02d}"
        save_wav(out / f"{name}.wav", x)
        manifest.append({"name": name, "text": r["text"].strip(), "seconds": len(x) / SR})

    pool: list[tuple[np.ndarray, str]] = []
    for r in rows:
        try:
            x = decode_to_f32(r["audio"]["bytes"])
        except Exception:
            continue
        if len(x) >= SR:
            pool.append((x, r["text"].strip()))

    gap = np.zeros(int(0.25 * SR), dtype=np.float32)
    used = 0
    for name, target in LONG_SPECS.items():
        parts: list[np.ndarray] = []
        texts: list[str] = []
        secs = 0.0
        while secs < target and pool:
            x, txt = pool[used % len(pool)]
            parts.extend(x)
            parts.extend(gap)
            texts.append(txt)
            secs += len(x) / SR
            used += 1
        save_wav(out / f"{name}.wav", np.array(parts, dtype=np.float32))
        manifest.append(
            {"name": name, "text": " ".join(texts), "seconds": len(parts) / SR}
        )

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} clips ({sum(c['seconds'] for c in manifest):.1f}s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
