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
    parser.add_argument("--model", "-m", default=None, help="Hugging Face repo or local path of the chat model to serve (optional when --tts-model is set)")
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
    parser.add_argument(
        "--tts-model",
        default=None,
        help="also serve speech synthesis on /v1/audio/* (Qwen3-TTS CustomVoice via the [tts] extra); the only model when --model is omitted",
    )
    parser.add_argument("--log-level", default="info", help="uvicorn log level (default: info)")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "tts":
        return _tts(argv[1:])
    args = build_parser().parse_args(argv)
    if not args.model and not args.tts_model:
        build_parser().error("--model or --tts-model is required")

    import uvicorn

    from .server import create_app

    backend = None
    if args.model:
        from .backends import load_backend

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

    tts_service = None
    if args.tts_model:
        import time as _time

        from .tts.config import TTSConfig, load_tts_model
        from .tts.service import DEFAULT_STREAM_INTERVAL, TTSService
        from .tts.stream_loop import prewarm_streaming

        try:
            tts_config = TTSConfig(model_ref=args.tts_model)
            model = load_tts_model(tts_config)
            # trace the compiled streaming_step shapes serving will hit, so
            # the first request pays no compile (failure just defers the
            # trace to that request)
            started = _time.perf_counter()
            try:
                shapes = prewarm_streaming(model, DEFAULT_STREAM_INTERVAL, tts_config.streaming_initial_interval)
                print(f"tts streaming prewarm (shapes {shapes}) done in {_time.perf_counter() - started:.1f}s", file=sys.stderr)
            except Exception as exc:
                print(f"warning: tts streaming prewarm failed ({exc}); first request will trace on demand", file=sys.stderr)
            tts_service = TTSService(model, tts_config)
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"error: failed to load TTS model '{args.tts_model}': {exc}", file=sys.stderr)
            return 1

    names = [n for n in (getattr(backend, "name", None), getattr(tts_service, "name", None)) if n]
    print(f"serving {', '.join(names)} on http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(create_app(backend, api_key=args.api_key, tts_service=tts_service), host=args.host, port=args.port, log_level=args.log_level)
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
