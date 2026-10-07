#!/usr/bin/env python3
"""Python client for POST /v1/images/generations (stdlib + nothing else, #101).

Start the server first:
    vllm-omni-mlx serve mlx-community/Qwen-Image-2.1-mflux-q4 --api-key demo
"""

import argparse
import base64
import json
import time
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # bypass system proxies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="demo")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--out", required=True, help="output path; --n>1 appends -1, -2, … before the extension")
    args = parser.parse_args()

    payload = {"prompt": args.prompt, "size": args.size, "n": args.n}
    if args.steps is not None:
        payload["steps"] = args.steps
    if args.seed is not None:
        payload["seed"] = args.seed

    request = urllib.request.Request(
        f"{args.url}/v1/images/generations",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"},
    )
    started = time.perf_counter()
    with OPENER.open(request) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - started

    stem, dot, ext = args.out.rpartition(".")
    for i, item in enumerate(body["data"]):
        path = args.out if len(body["data"]) == 1 else f"{stem}-{i + 1}{dot}{ext}"
        with open(path, "wb") as f:
            f.write(base64.b64decode(item["b64_json"]))
        print(f"wrote {path}")
    print(f"{len(body['data'])} image(s) in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
