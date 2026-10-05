#!/usr/bin/env python3
"""VoxCPM2 bench (#71): solo + chat-coexistence latency, the honest eager
baseline (compiled/incremental decode are follow-up loop work).

Drives the serving path (VoxCPM2Service.speech_stream — what
/v1/audio/speech stream:true runs) in ONE process:

  solo:  zero-shot + cloning RTF / time-to-first-chunk / peak memory
         (TTFA ≈ total synthesis: mlx-audio's generate is single-yield,
         so first audio lands when the buffer finishes)
  mixed: chat (default Qwen2.5-0.5B) interleaved — chat TTFT/tok/s alone
         vs during speech, speech RTF solo vs mixed, peak with both resident

Usage:
    python scripts/bench_voxcpm2.py [--chat-model mlx-community/Qwen2.5-0.5B-Instruct-4bit]
"""

from __future__ import annotations

import argparse
import base64
import os
import queue
import threading
import time
from pathlib import Path

import mlx.core as mx

SPEECH_TEXT = (
    "Welcome to the VoxCPM2 benchmark. This paragraph is long enough that "
    "the speech path runs for many seconds, which makes the real time "
    "factor meaningful over a sustained generation."
)
CHAT_PROMPT = "Describe the sea in two sentences."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="mlx-community/VoxCPM2-4bit")
    parser.add_argument("--chat-model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--chat-turns", type=int, default=4)
    parser.add_argument("--clone", action="store_true", help="also bench cloning from the bundled test_en.wav")
    parser.add_argument("--solo-only", action="store_true", help="skip the chat-coexistence phase")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="library generate path, nothing compiled (VLLM_OMNI_VOXCPM2_EAGER=1) — the A/B baseline",
    )
    parser.add_argument(
        "--mode",
        choices=("compiled", "eager", "library"),
        default=None,
        help="generation path: compiled = vendored compiled loop (default), eager/library = library path, nothing compiled",
    )
    args = parser.parse_args()
    if args.eager or args.mode in ("eager", "library"):
        os.environ["VLLM_OMNI_VOXCPM2_EAGER"] = "1"

    from vllm_omni_mlx.tts.voxcpm2 import VoxCPM2Config, VoxCPM2Service, load_voxcpm2_model, local_snapshot

    print(f"loading voxcpm2 {args.model} ...", flush=True)
    config = VoxCPM2Config(model_ref=args.model)
    service = VoxCPM2Service(load_voxcpm2_model(config), config)
    sr = service.sample_rate

    def speech_once(voice=None, label="zero-shot"):
        t0 = time.perf_counter()
        ttfa = None
        total_bytes = 0
        for chunk in service.speech_stream(SPEECH_TEXT, voice=voice, streaming_interval=args.interval):
            now = time.perf_counter()
            if ttfa is None:
                ttfa = now - t0
            total_bytes += len(chunk)
        wall = time.perf_counter() - t0
        audio_s = total_bytes / 2 / sr
        rtf = wall / audio_s if audio_s else float("inf")
        print(
            f"  {label}: {audio_s:.2f}s audio in {wall:.2f}s — TTFA {ttfa:.2f}s, RTF {rtf:.2f}, "
            f"peak {mx.get_peak_memory() / 2**30:.2f} GiB",
            flush=True,
        )
        return ttfa, rtf

    print("warmup (first request; includes any lazy init) ...", flush=True)
    speech_once(label="warmup")

    print("solo:", flush=True)
    speech_once(label="zero-shot")
    if args.clone:
        snapshot = Path(local_snapshot(args.model))
        ref = base64.b64encode((snapshot / "test_en.wav").read_bytes()).decode()
        speech_once(voice={"ref_audio": ref}, label="clone(test_en.wav)")

    if args.solo_only:
        return 0

    from vllm_omni_mlx.backends import load_backend
    from vllm_omni_mlx.schemas import Message, Part, UnifiedRequest

    print(f"loading chat {args.chat_model} ...", flush=True)
    backend = load_backend(args.chat_model)
    print(f"both resident: peak {mx.get_peak_memory() / 2**30:.2f} GiB", flush=True)

    def chat_once():
        req = UnifiedRequest(model="", messages=[Message(role="user", parts=[Part(kind="text", text=CHAT_PROMPT)])], max_tokens=64)
        t0 = time.perf_counter()
        ttft = first = last = None
        tokens = 0
        for chunk in backend.chat(req):
            now = time.perf_counter()
            if chunk.text:
                if ttft is None:
                    ttft = now - t0
                if first is None:
                    first = now
                last = now
                tokens += 1
        rate = (tokens - 1) / (last - first) if last and first and tokens > 1 else 0.0
        return ttft, rate

    # 1) chat alone (warm)
    chat_once()
    ttfts = [chat_once() for _ in range(3)]
    ttft_alone = sum(t for t, _ in ttfts) / len(ttfts)
    tps_alone = sum(r for _, r in ttfts) / len(ttfts)
    print(f"  chat alone: TTFT {ttft_alone:.2f}s, {tps_alone:.1f} tok/s", flush=True)

    # 2) speech + interleaved chat
    results: queue.Queue = queue.Queue()

    def chat_loop():
        for _ in range(args.chat_turns):
            results.put(chat_once())
            time.sleep(0.4)

    worker = threading.Thread(target=chat_loop)
    worker.start()
    t0 = time.perf_counter()
    ttfa_mixed = None
    total_bytes = 0
    for chunk in service.speech_stream(SPEECH_TEXT, streaming_interval=args.interval):
        now = time.perf_counter()
        if ttfa_mixed is None:
            ttfa_mixed = now - t0
        total_bytes += len(chunk)
    wall_mixed = time.perf_counter() - t0
    worker.join()
    chat_mixed = []
    while not results.empty():
        chat_mixed.append(results.get())
    audio_s = total_bytes / 2 / sr
    ttft_mixed = sum(t for t, _ in chat_mixed) / len(chat_mixed)
    tps_mixed = sum(r for _, r in chat_mixed) / len(chat_mixed)
    print(
        f"  mixed: speech RTF {wall_mixed / audio_s:.2f} (TTFA {ttfa_mixed:.2f}s, {audio_s:.2f}s audio); "
        f"chat TTFT {ttft_mixed:.2f}s, {tps_mixed:.1f} tok/s over {len(chat_mixed)} turns; "
        f"peak {mx.get_peak_memory() / 2**30:.2f} GiB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
