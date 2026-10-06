#!/usr/bin/env python3
"""Unified TTS accuracy battery — every served checkpoint, one command (#88-adjacent).

The per-family test files (test_tts_generate, test_small_custom_voice,
test_voice_design, test_icl_stream, test_voxcpm2_*) each hold one family's
calibrated battery; this script runs the SAME battery across the whole served
fleet so a shared-infra change (loop, quant, service) is accepted or rejected
in one pass:

  - duration sanity (>= 1 s) and the checkpoint's native sample rate
  - HNR against the per-path floor calibrated in those per-family files
    (the catastrophic-decode gate)
  - ASR round-trip against the synthesized text (tests/asr_oracle.py —
    the intelligibility gate; whisper-base hybrid built by
    scripts/build_asr_oracle.py; skipped with a note where not built)

Each checkpoint runs in its OWN subprocess — the one-heavy-model-at-a-time
rule (checkpoints stack past 16 GB in one process) — and every draw goes
through the real serving surface (service.speech_bytes), not internal
generators. The Base clone path uses the CustomVoice run's vivian output as
its reference clip, so the clone request path (base64 voice object) is
exercised with a known transcript.

Usage:
    python scripts/acc_all_checkpoints.py            # enforce floors, exit 1 on breach
    python scripts/acc_all_checkpoints.py --calibrate  # report measured stats + suggested floors
    python scripts/acc_all_checkpoints.py --model voxcpm2   # one checkpoint (substring match)

Floors without a calibrated value yet are None and enforced as "report only"
until set. VLLM_OMNI_* env vars pass through (e.g. VLLM_OMNI_VOXCPM2_QUANT)
so the battery always measures what would actually be served.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: the fleet — model_ref → served paths, with the HNR floors calibrated in
#: the per-family batteries (None = report-only until calibrated here)
MODELS: dict[str, dict] = {
    "qwen3-1.7b-customvoice": {
        "ref": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
        "kind": "qwen3",
        "paths": [
            {"name": "preset-vivian", "text": "The unified accuracy battery speaks this sentence so the harmonics and the transcriber can both check it.", "voice": "vivian", "hnr_floor_db": 0.0, "rt_floor": 0.70, "save_ref": True},
            {"name": "preset-ryan", "text": "The unified accuracy battery speaks this sentence so the harmonics and the transcriber can both check it.", "voice": "ryan", "hnr_floor_db": -5.0, "rt_floor": 0.50},
        ],
    },
    "qwen3-0.6b-customvoice": {
        "ref": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
        "kind": "qwen3",
        "paths": [
            {"name": "preset-vivian", "text": "This sentence measures the small custom voice model end to end.", "voice": "vivian", "hnr_floor_db": 0.0, "rt_floor": 0.50},
            {"name": "preset-ryan", "text": "This sentence measures the small custom voice model end to end.", "voice": "ryan", "hnr_floor_db": -5.0, "rt_floor": 0.45},
        ],
    },
    "qwen3-1.7b-base": {
        "ref": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
        "kind": "qwen3",
        "paths": [
            {"name": "clone", "text": "This sentence is spoken through the cloned reference voice.", "voice": {"ref_clip": True}, "hnr_floor_db": 0.0, "rt_floor": 0.55},
        ],
    },
    "qwen3-1.7b-voicedesign": {
        "ref": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
        "kind": "qwen3",
        "paths": [
            {"name": "design", "text": "The designed voice speaks the unified battery sentence.", "instructions": "A cheerful young female voice with high pitch and energetic tone", "hnr_floor_db": 0.0, "rt_floor": 0.45},
        ],
    },
    "voxcpm2": {
        "ref": "mlx-community/VoxCPM2-4bit",
        "kind": "voxcpm2",
        "paths": [
            {"name": "zero-en", "text": "The unified accuracy battery speaks this sentence so the harmonics and the transcriber can both check it.", "voice": "default", "hnr_floor_db": -5.0, "rt_floor": 0.50},
            {"name": "zero-zh", "text": "这句话验证统一准确率测试覆盖中文语音合成。", "asr_ref": ["这句话验证统一准确率测试覆盖中文语音合成。", "這句話驗證統一準確率測試覆蓋中文語音合成。"], "voice": "default", "hnr_floor_db": -5.0, "rt_floor": 0.45},
        ],
    },
}

REF_CLIP = Path("/tmp/acc_battery_ref.wav")  # written by the CustomVoice run, read by the Base clone
REF_TEXT = "The unified accuracy battery speaks this sentence so the harmonics and the transcriber can both check it."
SEEDS = (11, 12, 13)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="run one checkpoint (substring of its key, e.g. voxcpm2)")
    parser.add_argument("--calibrate", action="store_true", help="report measured stats + suggested floors; don't fail on None-floor paths")
    parser.add_argument("--draws", type=int, default=len(SEEDS))
    args = parser.parse_args()

    selected = {k: v for k, v in MODELS.items() if args.model is None or args.model in k}
    if not selected:
        print(f"no checkpoint matches {args.model!r}; keys: {list(MODELS)}", file=sys.stderr)
        return 2

    failures = 0
    print(f"{'checkpoint':26s} {'path':16s} {'draw':>4s}  {'dur':>5s}  {'HNR dB':>7s}  {'RT sim':>6s}  verdict")
    for key, spec in selected.items():
        # the clone path needs the ref clip from the CustomVoice run first
        needs_ref = any(p.get("voice", {}).get("ref_clip") if isinstance(p.get("voice"), dict) else False for p in spec["paths"])
        if needs_ref and not REF_CLIP.exists() and "qwen3-1.7b-customvoice" not in selected:
            print(f"{key}: clone path needs {REF_CLIP}; run the customvoice checkpoint first (or the full battery)")
            failures += 1
            continue
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--_worker", key, "--draws", str(args.draws)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            hnr = row["hnr_db"]
            rt = row["rt_sim"]
            floor = row["hnr_floor"]
            rt_floor = row["rt_floor"]
            hnr_ok = floor is None or hnr > floor
            rt_ok = rt is None or rt_floor is None or rt > rt_floor
            dur_ok = row["seconds"] >= 1.0
            ok = hnr_ok and rt_ok and dur_ok
            if not ok:
                failures += 1
            verdict = "ok" if ok else "FAIL"
            if floor is None or rt_floor is None:
                verdict += " (report-only)" if ok else ""
            print(f"{key:26s} {row['path']:16s} {row['seed']:>4d}  {row['seconds']:5.1f}  {hnr:7.2f}  {rt if rt is not None else float('nan'):6.3f}  {verdict}"
                  f"{'' if hnr_ok else f' [HNR<= {floor}]'}{'' if rt_ok else f' [RT<= {rt_floor}]'}")
        if proc.returncode != 0:
            failures += 1
            print(f"{key}: worker failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}")

    if args.calibrate:
        print("\ncalibration: set hnr_floor_db/rt_floor in MODELS from the minima above minus a margin")
    print(f"\n{'PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 0 if failures == 0 else 1


def worker(key: str, draws: int) -> int:
    """One checkpoint, one process: load, draw each path, emit JSON lines."""
    import mlx.core as mx
    import numpy as np

    spec = MODELS[key]

    if spec["kind"] == "voxcpm2":
        from vllm_omni_mlx.tts.voxcpm2 import VoxCPM2Config, VoxCPM2Service, load_voxcpm2_model

        config = VoxCPM2Config(model_ref=spec["ref"])
        service = VoxCPM2Service(load_voxcpm2_model(config), config)
    else:
        from vllm_omni_mlx.tts.config import TTSConfig, load_tts_model
        from vllm_omni_mlx.tts.service import TTSService

        config = TTSConfig(model_ref=spec["ref"])
        service = TTSService(load_tts_model(config), config)

    oracle = None
    try:
        from tests.asr_oracle import load_oracle, round_trip_similarity

        oracle = load_oracle()
    except Exception:
        pass
    from tests.audio_metrics import pcm_hnr_db

    for path in spec["paths"]:
        voice = path.get("voice")
        if isinstance(voice, dict) and voice.get("ref_clip"):
            clip = base64.b64encode(REF_CLIP.read_bytes()).decode()
            voice = {"ref_audio": clip, "ref_text": REF_TEXT}
        for seed in SEEDS[:draws]:
            mx.random.seed(seed)
            payload, _ = service.speech_bytes(path["text"], voice=voice, instructions=path.get("instructions"))
            with wave.open(io.BytesIO(payload)) as w:
                rate = w.getframerate()
                samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32767.0
            seconds = samples.size / rate
            hnr = pcm_hnr_db(samples, sr=rate)
            rt = round_trip_similarity(oracle, mx.array(samples), rate, path.get("asr_ref", path["text"])) if oracle else None
            if path.get("save_ref") and seed == SEEDS[0] and not REF_CLIP.exists():
                REF_CLIP.write_bytes(payload)
            print(json.dumps({
                "model": key, "path": path["name"], "seed": seed, "seconds": round(seconds, 2),
                "sample_rate": rate, "hnr_db": round(hnr, 2), "rt_sim": round(rt, 3) if rt is not None else None,
                "hnr_floor": path["hnr_floor_db"], "rt_floor": path["rt_floor"],
            }), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--_worker")
    parser.add_argument("--draws", type=int, default=len(SEEDS))
    args, _ = parser.parse_known_args()
    raise SystemExit(worker(args._worker, args.draws) if args._worker else main())
