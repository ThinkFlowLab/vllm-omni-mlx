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
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "tts":
        return _tts(argv[1:])
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


def _tts(argv: list[str]) -> int:
    """`vllm-omni-mlx tts --voice vivian --text "..." --out out.wav` (#15)."""
    parser = argparse.ArgumentParser(prog="vllm-omni-mlx tts", description="Synthesize speech to a WAV file.")
    parser.add_argument("--model", default=None, help="TTS model repo or path (default: the [tts] default)")
    parser.add_argument("--voice", default=None, help="preset CustomVoice speaker (e.g. vivian, ryan)")
    parser.add_argument("--language", default=None, help="spoken language hint (default: auto)")
    parser.add_argument("--instruct", default=None, help="emotion/style instruction")
    parser.add_argument("--text", required=True, help="text to synthesize")
    parser.add_argument("--out", required=True, help="output WAV path")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args(argv)

    import time

    from .tts.config import TTSConfig, load_tts_model
    from .tts.generate import synthesize, wav_bytes

    config = TTSConfig()
    if args.model:
        config = TTSConfig(model_ref=args.model)
    overrides = {k: getattr(args, k) for k in ("voice", "language", "instruct", "temperature", "seed", "max_tokens")}
    overrides = {("speaker" if k == "voice" else k): v for k, v in overrides.items()}
    try:
        model = load_tts_model(config)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: failed to load TTS model '{config.model_ref}': {exc}", file=sys.stderr)
        return 1

    start = time.perf_counter()
    data = wav_bytes(synthesize(model, config, args.text, **overrides))
    elapsed = time.perf_counter() - start
    if not data:
        print("error: model produced no audio", file=sys.stderr)
        return 1
    with open(args.out, "wb") as f:
        f.write(data)
    frames = (len(data) - 44) // 2
    print(f"wrote {args.out}: {frames / 24000:.2f}s of audio in {elapsed:.2f}s (RTF {elapsed / (frames / 24000):.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
