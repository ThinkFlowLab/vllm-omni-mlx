#!/usr/bin/env python3
"""Coexistence benchmark (#39): chat + TTS in ONE process — the single-user
serving shape this project targets.

Loads a chat backend and the TTS service side by side, then measures:
  - speech sustained RTF / TTFA / cadence while chat requests interleave
  - chat TTFT and tok/s alone vs during speech
  - peak unified memory with both models resident

Usage:
    python scripts/coexistence_bench.py [--chat-model mlx-community/Qwen2.5-0.5B-Instruct-4bit]
"""

from __future__ import annotations

import argparse
import queue
import threading
import time

import mlx.core as mx

SPEECH_TEXT = (
    "Welcome to the coexistence benchmark. This paragraph is long enough that "
    "the speech stream runs for many seconds while chat requests interleave, "
    "which is exactly the contention this benchmark exists to measure."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chat-model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--tts-model", default="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--chat-turns", type=int, default=6)
    args = parser.parse_args()

    from vllm_omni_mlx.backends import load_backend
    from vllm_omni_mlx.schemas import UnifiedRequest, Message, Part
    from vllm_omni_mlx.tts.config import TTSConfig, load_tts_model
    from vllm_omni_mlx.tts.generate import synthesize

    print(f"loading chat {args.chat_model} ...", flush=True)
    backend = load_backend(args.chat_model)
    print(f"loading tts {args.tts_model} ...", flush=True)
    tts = load_tts_model(TTSConfig(model_ref=args.tts_model))
    config = TTSConfig()
    resident_gb = mx.get_peak_memory() / 2**30

    def chat_once():
        req = UnifiedRequest(model="", messages=[Message(role="user", parts=[Part(kind="text", text="Describe the sea in two sentences.")])], max_tokens=64)
        t0 = time.perf_counter()
        ttft = None
        last = None
        tokens = 0
        for chunk in backend.chat(req):
            now = time.perf_counter()
            if chunk.text:
                if ttft is None:
                    ttft = now - t0
                last = now
                tokens += 1
        return ttft, tokens / (last - t0 + 1e-9) if last else 0.0

    def speech_once():
        t0 = time.perf_counter()
        ttfa = None
        arrivals = []
        sizes = []
        for chunk in synthesize(tts, config, SPEECH_TEXT, speaker="vivian", streaming_interval=args.interval, seed=7):
            now = time.perf_counter()
            if ttfa is None:
                ttfa = now - t0
            arrivals.append(now - t0)
            sizes.append(chunk.size)
        return t0, ttfa, arrivals, sizes

    # warm both paths (compile)
    chat_once()
    for _ in synthesize(tts, config, "Warmup.", speaker="vivian"):
        pass

    # 1) chat alone
    ttfts = [chat_once() for _ in range(3)]
    chat_ttft_alone = sum(t for t, _ in ttfts) / len(ttfts)
    chat_tps_alone = sum(r for _, r in ttfts) / len(ttfts)

    # 2) speech alone
    _, ttfa_alone, arrivals, sizes = speech_once()
    audio = sum(sizes) / 24000
    alone_wall = arrivals[-1]

    # 3) speech + interleaved chat
    results: queue.Queue = queue.Queue()

    def chat_loop():
        for _ in range(args.chat_turns):
            results.put(chat_once())
            time.sleep(0.4)

    worker = threading.Thread(target=chat_loop)
    start = time.perf_counter()
    worker.start()
    t0, ttfa_mixed, arrivals, sizes = speech_once()
    worker.join()
    mixed_wall = time.perf_counter() - start
    chat_mixed = []
    while not results.empty():
        chat_mixed.append(results.get())
    audio_mixed = sum(sizes) / 24000

    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    chunk_secs = [s / 24000 for s in sizes]
    sustained = max((g / max(c, 1e-9) for g, c in zip(gaps, chunk_secs[1:])), default=float("nan"))

    print(f"\npeak memory (both resident): {mx.get_peak_memory()/2**30:.2f} GB (after load {resident_gb:.2f} GB)")
    print(f"chat alone : TTFT {chat_ttft_alone*1000:6.0f} ms | {chat_tps_alone:5.1f} tok/s")
    if chat_mixed:
        m_ttft = sum(t for t, _ in chat_mixed) / len(chat_mixed)
        m_tps = sum(r for _, r in chat_mixed) / len(chat_mixed)
        print(f"chat mixed : TTFT {m_ttft*1000:6.0f} ms | {m_tps:5.1f} tok/s  ({len(chat_mixed)} turns during speech)")
    print(f"speech alone: TTFA {ttfa_alone*1000:6.0f} ms | audio {audio:5.2f}s | RTF {alone_wall / audio:.3f}")
    print(f"speech mixed: TTFA {ttfa_mixed*1000:6.0f} ms | audio {audio_mixed:5.2f}s | wall {arrivals[-1]:5.2f}s | RTF {arrivals[-1]/audio_mixed:.3f} | sustained {sustained:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
