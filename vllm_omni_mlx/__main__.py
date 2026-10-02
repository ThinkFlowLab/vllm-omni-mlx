"""CLI: serve one model.

    vllm-omni-mlx --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-omni-mlx",
        description="Lightweight OpenAI- and Anthropic-compatible omni-modality server for Apple Silicon.",
    )
    parser.add_argument("--model", "-m", required=True, help="Hugging Face repo or local path of the model to serve")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    parser.add_argument(
        "--backend",
        choices=("auto", "text", "omni"),
        default="auto",
        help="force a backend; 'auto' sniffs the model config (default: auto)",
    )
    parser.add_argument("--api-key", default=None, help="require this API key (Bearer or x-api-key header)")
    parser.add_argument(
        "--draft-model",
        default=None,
        help="speculative decoding: HF repo or path of a smaller model sharing the main model's tokenizer (text backend)",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        help="quantize the KV cache to this many bits (e.g. 8) to cut memory on long contexts",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help="group size for KV-cache quantization (default: 64)",
    )
    parser.add_argument("--log-level", default="info", help="uvicorn log level (default: info)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import uvicorn

    from .backends import load_backend
    from .server import create_app

    try:
        backend = load_backend(
            args.model,
            preferred=args.backend,
            draft_model=args.draft_model,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: failed to load model '{args.model}': {exc}", file=sys.stderr)
        return 1

    print(f"serving {backend.name} ({type(backend).__name__}) on http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(create_app(backend, api_key=args.api_key), host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
