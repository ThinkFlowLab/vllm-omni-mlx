"""CLI, shaped like upstream vllm-omni's `vllm serve <model> --omni`.

    vllm-mlx serve mlx-community/Qwen2.5-0.5B-Instruct-4bit
    vllm-mlx serve mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit --omni
"""

from __future__ import annotations

import argparse
import sys


def build_serve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-mlx serve",
        description="Serve a model on the OpenAI- and Anthropic-compatible API.",
        add_help=False,
    )
    parser.add_argument("model", nargs="?", default=None, help="HF repo or local path to serve (optional when --tts-model is set)")
    parser.add_argument(
        "--omni",
        action="store_true",
        help="serve the model omni-modally: a Qwen3-TTS checkpoint serves /v1/audio/*, anything else forces the mlx-vlm backend",
    )
    parser.add_argument(
        "--tts-model",
        default=None,
        help="also serve speech synthesis on /v1/audio/* (Qwen3-TTS via the [tts] extra); the only model when <model> is omitted",
    )
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


def build_tts_parser() -> argparse.ArgumentParser:
    """`vllm-mlx tts --voice vivian --text "..." --out out.wav` (#15)."""
    parser = argparse.ArgumentParser(prog="vllm-mlx tts", description="Synthesize speech to a WAV file.", add_help=False)
    parser.add_argument("--model", default=None, help="TTS model repo or path (default: the [tts] default)")
    parser.add_argument("--voice", default=None, help="preset CustomVoice speaker (e.g. vivian, ryan)")
    parser.add_argument("--language", default=None, help="spoken language hint (default: auto)")
    parser.add_argument("--instruct", default=None, help="emotion/style instruction")
    parser.add_argument("--text", required=True, help="text to synthesize")
    parser.add_argument("--out", required=True, help="output WAV path")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-mlx",
        description="Lightweight OpenAI- and Anthropic-compatible omni-modality server for Apple Silicon.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{serve,tts}")
    sub.add_parser("serve", parents=[build_serve_parser()], help="serve a model")
    sub.add_parser("tts", parents=[build_tts_parser()], help="one-shot speech synthesis to a WAV file")
    return parser


def _looks_like_tts(config: dict) -> bool:
    """Qwen3-TTS checkpoints announce themselves via ``tts_model_type`` (the
    same key tts/variants.py dispatches on after load) or ``model_type``."""
    return "tts_model_type" in config or config.get("model_type") == "qwen3_tts"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    return _tts_synthesize(args)


def _serve(args) -> int:
    if not args.model and not args.tts_model:
        build_serve_parser().error("a model is required (or --tts-model to serve speech alone)")
    if args.omni and not args.model:
        build_serve_parser().error("--omni applies to the served model")
    if args.omni and args.tts_model:
        build_serve_parser().error("--omni and --tts-model are mutually exclusive: --omni already serves the model as TTS when it is a Qwen3-TTS checkpoint")

    import uvicorn

    from .server import create_app

    backend = None
    tts_service = None
    if args.model:
        from .backends import _peek_config, load_backend

        try:
            if args.omni:
                if _looks_like_tts(_peek_config(args.model)):
                    tts_service = _load_tts(args.model)
                else:
                    backend = load_backend(args.model, preferred="omni", kv_bits=args.kv_bits, kv_group_size=args.kv_group_size)
            elif _looks_like_tts(_peek_config(args.model)):
                print(f"error: '{args.model}' is a Qwen3-TTS checkpoint; pass --omni to serve speech synthesis", file=sys.stderr)
                return 1
            else:
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

    if args.tts_model:
        try:
            tts_service = _load_tts(args.tts_model)
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"error: failed to load TTS model '{args.tts_model}': {exc}", file=sys.stderr)
            return 1

    names = [n for n in (getattr(backend, "name", None), getattr(tts_service, "name", None)) if n]
    print(f"serving {', '.join(names)} on http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(create_app(backend, api_key=args.api_key, tts_service=tts_service), host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _load_tts(model_ref: str):
    """Load a Qwen3-TTS checkpoint for serving — shared by the --omni and
    --tts-model paths: load, reject variants this build can't synthesize at
    startup (a 400 per request is the late signal otherwise), prewarm, wrap."""
    import time

    from .tts.config import TTSConfig, load_tts_model
    from .tts.service import DEFAULT_STREAM_INTERVAL, TTSService
    from .tts.stream_loop import prewarm_streaming
    from .tts.variants import ensure_served

    config = TTSConfig(model_ref=model_ref)
    model = load_tts_model(config)
    ensure_served(model)
    # trace the compiled streaming_step shapes serving will hit, so
    # the first request pays no compile (failure just defers the
    # trace to that request)
    started = time.perf_counter()
    try:
        shapes = prewarm_streaming(model, DEFAULT_STREAM_INTERVAL, config.streaming_initial_interval)
        print(f"tts streaming prewarm (shapes {shapes}) done in {time.perf_counter() - started:.1f}s", file=sys.stderr)
    except Exception as exc:
        print(f"warning: tts streaming prewarm failed ({exc}); first request will trace on demand", file=sys.stderr)
    return TTSService(model, config)


def _tts_synthesize(args) -> int:
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
