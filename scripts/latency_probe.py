#!/usr/bin/env python3
"""TTFT / inter-token latency probe against a running vllm-omni-mlx server.

Streams from /v1/chat/completions and measures, per turn: time-to-first-token
(TTFT), inter-token latency (p50 / p95 / max), tokens/s, and total time.
Turn 2+ extend the same conversation, so with the server's prefix cache they
measure the cached-prefix path while turn 1 measures the cold one.

Stdlib only. Run e.g.:

    python scripts/latency_probe.py --url http://127.0.0.1:8000 \
        --system-words 500 --turns 3 --max-tokens 128

Prints one table row per turn; a summary line follows. Against a server
started with --api-key, pass --api-key (or set VLLM_OMNI_MLX_KEY).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

# bypass any system HTTP proxy: probes target localhost, and macOS system
# proxy settings otherwise leak into urllib and 502 the request
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="server base URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_OMNI_MLX_KEY"), help="API key if the server requires one")
    parser.add_argument("--turns", type=int, default=3, help="conversation turns to probe (turn 1 = cold prefix, later turns = cached; default: 3)")
    parser.add_argument("--max-tokens", type=int, default=128, help="generation budget per turn (default: 128)")
    parser.add_argument("--system-words", type=int, default=500, help="system-prompt length in words, sizing the prefilled prefix (default: 500)")
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (default: 0)")
    parser.add_argument("--audio", action="store_true", help="probe POST /v1/audio/speech instead of chat: reports TTFB, total, RTF per turn")
    parser.add_argument("--voice", default="vivian", help="preset voice for --audio mode (default: vivian)")
    parser.add_argument("--audio-text", default="Welcome to the speech latency probe. This sentence is long enough to make the real time factor meaningful.", help="text synthesized in --audio mode")
    return parser


def probe_audio(args) -> int:
    """--audio mode: POST /v1/audio/speech per turn, reporting time-to-first-
    byte (server + prefill overhead; chunked-audio TTFA arrives with M2
    streaming), total wall, audio seconds, and RTF."""
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    print(f"probing {args.url} /v1/audio/speech — voice {args.voice}, {args.turns} turns")
    print(f"{'turn':>4} {'TTFB ms':>9} {'total s':>8} {'audio s':>8} {'RTF':>6}")
    for turn in range(1, args.turns + 1):
        body = json.dumps({"input": args.audio_text, "voice": args.voice}).encode()
        request = urllib.request.Request(f"{args.url}/v1/audio/speech", data=body, headers=headers)
        start = time.perf_counter()
        ttfb = None
        with _OPENER.open(request, timeout=600) as response:
            first = response.read(1)
            ttfb = time.perf_counter() - start
            rest = response.read()
        total = time.perf_counter() - start
        audio_seconds = (len(first) + len(rest) - 44) / 2 / 24000  # 16-bit mono
        if ttfb is None or not (first or rest):
            print(f"{turn:>4}  no audio returned", flush=True)
            continue
        print(f"{turn:>4} {ttfb*1000:9.0f} {total:8.2f} {audio_seconds:8.2f} {total/max(audio_seconds, 1e-9):6.2f}", flush=True)
    return 0


def probe_turn(messages: list, args) -> dict:
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    body = json.dumps(
        {
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(f"{args.url}/v1/chat/completions", data=body, headers=headers)

    starts = time.perf_counter()
    ttft = None
    content_chunks = 0
    deltas: list[float] = []
    text_parts: list[str] = []
    finish_reason = None
    with _OPENER.open(request, timeout=300) as response:
        last = None
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            now = time.perf_counter()
            payload = json.loads(line[6:])
            choices = payload.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                if ttft is None:
                    ttft = now - starts
                elif last is not None:
                    deltas.append(now - last)
                last = now
                content_chunks += 1
                text_parts.append(delta["content"])
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
    total = time.perf_counter() - starts
    return {
        "ttft": ttft or float("nan"),
        "itl_p50": statistics.median(deltas) if deltas else float("nan"),
        "itl_p95": _p95(deltas),
        "itl_max": max(deltas) if deltas else float("nan"),
        "tok_s": content_chunks / total if total else float("nan"),
        "chunks": content_chunks,
        "total": total,
        "text": "".join(text_parts),
        "finish_reason": finish_reason,
    }


def _p95(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))
    return ordered[k]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    system = " ".join(["You are a careful assistant; keep the full context in mind."] * max(1, args.system_words // 9))
    if args.audio:
        return probe_audio(args)

    messages = [{"role": "system", "content": system}]

    print(f"probing {args.url} — {args.turns} turns, ~{args.system_words}-word system prefix, {args.max_tokens}-token budget")
    print(f"{'turn':>4} {'prefix':>7} {'TTFT ms':>9} {'ITL p50':>8} {'ITL p95':>8} {'ITL max':>8} {'tok/s':>7} {'chunks':>6}")
    rows = []
    for turn in range(1, args.turns + 1):
        label = "cold" if turn == 1 else "cached"
        messages = messages + [{"role": "user", "content": f"Turn {turn}: tell me a short story about the sea."}]
        result = probe_turn(messages, args)
        rows.append((label, result))
        print(
            f"{turn:>4} {label:>7} {result['ttft']*1000:9.1f} {result['itl_p50']*1000:8.1f}"
            f" {result['itl_p95']*1000:8.1f} {result['itl_max']*1000:8.1f} {result['tok_s']:7.1f} {result['chunks']:6d}",
            flush=True,
        )
        messages = messages + [{"role": "assistant", "content": result["text"]}]

    if len(rows) >= 2 and rows[0][1]["ttft"] > 0:
        cold, cached = rows[0][1]["ttft"], min(r["ttft"] for _, r in rows[1:])
        print(f"\nTTFT cold {cold*1000:.0f} ms vs best cached {cached*1000:.0f} ms ({cold/cached:.1f}x)" if cached > 0 else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
