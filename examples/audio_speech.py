#!/usr/bin/env python3
"""Python client for POST /v1/audio/speech (stdlib only, #16).

Start the server first:
    vllm-omni-mlx --tts-model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit --api-key demo
"""

import argparse
import json
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # bypass system proxies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="demo")
    parser.add_argument("--input", required=True)
    parser.add_argument("--voice", default="vivian")
    parser.add_argument("--language", default=None)
    parser.add_argument("--instructions", default=None)
    parser.add_argument("--format", choices=("wav", "pcm"), default="wav")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    body = {"input": args.input, "voice": args.voice, "response_format": args.format}
    if args.language:
        body["language"] = args.language
    if args.instructions:
        body["instructions"] = args.instructions
    request = urllib.request.Request(
        f"{args.url}/v1/audio/speech",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"},
    )
    with OPENER.open(request, timeout=300) as response:
        data = response.read()
    with open(args.out, "wb") as f:
        f.write(data)
    print(f"wrote {args.out}: {len(data)} bytes ({args.format})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
