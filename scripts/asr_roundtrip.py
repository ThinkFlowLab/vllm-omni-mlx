#!/usr/bin/env python3
"""TTS → ASR round-trip gate (#68 task 3): our speech in, our transcript out.

No external fixtures: each sentence is synthesized by the serving TTS
checkpoint, transcribed by the serving ASR checkpoint, and scored by
word-error-rate against the input text (lowercased, punctuation stripped,
digits spelled out in the text so normalization stays trivial). Catches
catastrophic breakage on either side — a broken decode loop, a resample
mismatch, a mangled prompt — not fine WER differences between checkpoints.

Both models stay resident in one process (≈3 GB on the 4-bit defaults); run it
on its own, not next to another heavy battery on a 16 GB machine.

    python scripts/asr_roundtrip.py --voices vivian ryan --max-wer 0.15
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please turn off the lights before you leave the office tonight.",
    "Apple silicon runs large language models with unified memory.",
    "Peter Piper picked a peck of pickled peppers.",
    "Tomorrow morning we will review the results of the experiment.",
    "The server answers every request in under half a second.",
]


def normalize(text: str) -> list[str]:
    return re.sub(r"[^a-z' ]+", " ", text.lower()).split()


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / max(len(ref), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tts-model", default=None)
    parser.add_argument("--asr-model", default=None)
    parser.add_argument("--voices", nargs="+", default=["vivian"])
    parser.add_argument("--max-wer", type=float, default=0.15, help="fail if the mean WER exceeds this")
    args = parser.parse_args()

    import mlx.core as mx

    from vllm_omni_mlx.asr.config import ASRConfig, load_asr_model
    from vllm_omni_mlx.asr.service import ASRService
    from vllm_omni_mlx.tts.config import TTSConfig, load_tts_model
    from vllm_omni_mlx.tts.service import TTSService

    tts_cfg = TTSConfig(model_ref=args.tts_model) if args.tts_model else TTSConfig()
    asr_cfg = ASRConfig(model_ref=args.asr_model) if args.asr_model else ASRConfig()
    tts = TTSService(load_tts_model(tts_cfg), tts_cfg)
    asr = ASRService(load_asr_model(asr_cfg), asr_cfg)
    print(f"tts={tts.name}\nasr={asr.name}", file=sys.stderr)

    rows = []
    for voice in args.voices:
        for text in SENTENCES:
            t0 = time.perf_counter()
            wav, _ = tts.speech_bytes(text, voice=voice)
            t1 = time.perf_counter()
            out = asr.transcribe(wav)
            t2 = time.perf_counter()
            score = wer(text, out.text)
            rows.append((voice, text, out.text, score, t1 - t0, t2 - t1, out.duration))
            print(f"[{voice}] WER {score:.2f}  tts {t1 - t0:.1f}s  asr {t2 - t1:.2f}s ({out.duration:.1f}s audio)\n"
                  f"   ref: {text}\n   hyp: {out.text}")
    mx.clear_cache()

    scores = [r[3] for r in rows]
    mean = statistics.mean(scores)
    print(f"\nutterances {len(rows)}  mean WER {mean:.3f}  max {max(scores):.2f}  "
          f"exact {sum(s == 0 for s in scores)}/{len(scores)}  peak {mx.get_peak_memory() / 2**30:.2f} GiB")
    if mean > args.max_wer:
        print(f"FAIL: mean WER {mean:.3f} > {args.max_wer}", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
