"""CLI, shaped like upstream vllm-omni's `vllm serve <model> --omni`.

    vllm-omni-mlx serve mlx-community/Qwen2.5-0.5B-Instruct-4bit
    vllm-omni-mlx serve mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit --omni
"""

from __future__ import annotations

import argparse
import sys


def build_serve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-omni-mlx serve",
        description="Serve a model on the OpenAI- and Anthropic-compatible API.",
        add_help=False,
    )
    parser.add_argument("model", nargs="?", default=None, help="HF repo or local path to serve (optional when --tts-model is set)")
    parser.add_argument(
        "--omni",
        action="store_true",
        help="serve the model omni-modally: a Qwen3-TTS, VoxCPM2 or MOSS Nano checkpoint serves /v1/audio/*, anything else forces the mlx-vlm backend",
    )
    parser.add_argument(
        "--tts-model",
        default=None,
        help="also serve speech synthesis on /v1/audio/* (Qwen3-TTS, VoxCPM2 or MOSS Nano via the [tts] extra); the only model when <model> is omitted",
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        help="also serve speech recognition on /v1/audio/transcriptions (Qwen3-ASR via the [asr] extra); combine with --tts-model for a one-process voice agent",
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
    """`vllm-omni-mlx tts --voice vivian --text "..." --out out.wav` (#15)."""
    parser = argparse.ArgumentParser(prog="vllm-omni-mlx tts", description="Synthesize speech to a WAV file.", add_help=False)
    parser.add_argument("--model", default=None, help="TTS model repo or path (default: the [tts] default)")
    parser.add_argument("--voice", default=None, help="preset CustomVoice speaker (e.g. vivian, ryan); VoxCPM2 accepts only 'default'; MOSS Nano requires --ref-audio instead")
    parser.add_argument("--language", default=None, help="spoken language hint (default: auto; not supported on VoxCPM2; MOSS Nano accepts only auto)")
    parser.add_argument("--instruct", default=None, help="emotion/style instruction, or a VoxCPM2 voice description (not supported on MOSS Nano)")
    parser.add_argument(
        "--ref-audio",
        default=None,
        help="path to a reference clip for VoxCPM2 or MOSS Nano voice cloning (0.5–30 s); required for MOSS Nano",
    )
    parser.add_argument("--text", required=True, help="text to synthesize")
    parser.add_argument("--out", required=True, help="output WAV path")
    parser.add_argument("--temperature", type=float, default=None, help="sampling temperature (MOSS Nano: audio sampler only, must be positive)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-tokens", default=None, help="Qwen3-TTS tokens, VoxCPM2 audio patches (~20 ms each, default 2000), or MOSS Nano audio frames per text chunk (80 ms each, default 375)")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-omni-mlx",
        description="Lightweight OpenAI- and Anthropic-compatible omni-modality server for Apple Silicon.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{serve,tts}")
    sub.add_parser("serve", parents=[build_serve_parser()], help="serve a model")
    sub.add_parser("tts", parents=[build_tts_parser()], help="one-shot speech synthesis to a WAV file")
    return parser


def _looks_like_tts(config: dict) -> bool:
    """TTS checkpoints announce themselves in ``config.json``: Qwen3-TTS via
    ``tts_model_type`` (the same key tts/variants.py dispatches on after
    load) or ``model_type``, VoxCPM2 via ``architecture`` (#71),
    MOSS Nano via ``model_type`` (#73)."""
    return (
        "tts_model_type" in config
        or config.get("model_type") in ("qwen3_tts", "moss_tts_nano")
        or config.get("architecture") == "voxcpm2"
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    return _tts_synthesize(args)


def _serve(args) -> int:
    if not args.model and not args.tts_model and not args.asr_model:
        build_serve_parser().error("a model is required (or --tts-model / --asr-model to serve audio alone)")
    if args.omni and not args.model:
        build_serve_parser().error("--omni applies to the served model")
    if args.omni and args.tts_model:
        build_serve_parser().error("--omni and --tts-model are mutually exclusive: --omni already serves the model as TTS when it is a TTS checkpoint")

    import uvicorn

    from .server import create_app

    backend = None
    tts_service = None
    asr_service = None
    if args.model:
        from .backends import _peek_config, load_backend

        try:
            if args.omni:
                if _looks_like_tts(_peek_config(args.model)):
                    tts_service = _load_tts(args.model)
                else:
                    backend = load_backend(args.model, preferred="omni", kv_bits=args.kv_bits, kv_group_size=args.kv_group_size)
            elif _looks_like_tts(_peek_config(args.model)):
                print(f"error: '{args.model}' is a TTS checkpoint; pass --omni to serve speech synthesis", file=sys.stderr)
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

    if args.asr_model:
        try:
            asr_service = _load_asr(args.asr_model)
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"error: failed to load ASR model '{args.asr_model}': {exc}", file=sys.stderr)
            return 1

    names = [n for n in (getattr(backend, "name", None), getattr(tts_service, "name", None), getattr(asr_service, "name", None)) if n]
    print(f"serving {', '.join(names)} on http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(create_app(backend, api_key=args.api_key, tts_service=tts_service, asr_service=asr_service), host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _load_asr(model_ref: str):
    """Load an ASR checkpoint for serving (#68). The capability matrix is
    consulted on config.json *before* any weights load, so an unsupported
    family fails at boot with guidance instead of a per-request error."""
    import time

    from .asr.capabilities import family_of, require_served
    from .asr.config import ASRConfig, load_asr_model
    from .asr.service import ASRService
    from .backends import _peek_config

    try:
        import multipart  # noqa: F401  (starlette's form parsing)
    except ImportError as exc:
        raise RuntimeError("ASR uploads need python-multipart; pip install 'vllm-omni-mlx[asr]'") from exc
    require_served(family_of(_peek_config(model_ref)))
    started = time.perf_counter()
    config = ASRConfig(model_ref=model_ref)
    service = ASRService(load_asr_model(config), config)
    print(f"asr loaded in {time.perf_counter() - started:.1f}s ({service.model_type})", file=sys.stderr)
    return service


def _load_tts(model_ref: str):
    """Load a TTS checkpoint for serving — shared by the --omni and
    --tts-model paths. Qwen3-TTS: load, reject variants this build can't
    synthesize at startup (a 400 per request is the late signal otherwise),
    prewarm the compiled streaming shapes, wrap. VoxCPM2 (#71): eager load
    of the mlx-audio model — nothing to prewarm (its generate is
    single-yield; compiled/incremental decode is follow-up loop work).
    MOSS Nano (#73): load the model and codec, then wrap its cloning service."""
    import time

    from .backends import _peek_config

    model_config = _peek_config(model_ref)
    if model_config.get("model_type") == "moss_tts_nano":
        from .tts.moss_nano import MossNanoConfig, MossNanoService, load_moss_nano_model

        started = time.perf_counter()
        config = MossNanoConfig(model_ref=model_ref)
        service = MossNanoService(load_moss_nano_model(config), config)
        print(
            f"moss nano loaded in {time.perf_counter() - started:.1f}s "
            f"({service.sample_rate} Hz, voice cloning)",
            file=sys.stderr,
        )
        return service

    if model_config.get("architecture") == "voxcpm2":
        from .tts.voxcpm2 import VoxCPM2Config, VoxCPM2Service, load_voxcpm2_model

        started = time.perf_counter()
        config = VoxCPM2Config(model_ref=model_ref)
        service = VoxCPM2Service(load_voxcpm2_model(config), config)
        print(
            f"voxcpm2 loaded in {time.perf_counter() - started:.1f}s "
            f"({service.sample_rate} Hz, voices {service.voices})",
            file=sys.stderr,
        )
        return service

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
    from .backends import _peek_config

    if args.model:
        try:
            model_config = _peek_config(args.model)
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"error: failed to read TTS model config '{args.model}': {exc}", file=sys.stderr)
            return 1
        if model_config.get("model_type") == "moss_tts_nano":
            return _tts_synthesize_moss_nano(args)
        if model_config.get("architecture") == "voxcpm2":
            return _tts_synthesize_voxcpm2(args)

    import time

    from .tts.config import TTSConfig, load_tts_model
    from .tts.generate import synthesize, wav_bytes

    if args.ref_audio:
        print("error: --ref-audio needs a VoxCPM2 or MOSS Nano model; Qwen3-TTS clones via the serving API's voice object", file=sys.stderr)
        return 1
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


def _tts_synthesize_moss_nano(args) -> int:
    """Clone a reference voice through the same service used by /v1/audio/speech."""
    import base64
    import io
    import math
    import time
    import wave

    if not args.ref_audio:
        print("error: MOSS Nano voice cloning requires --ref-audio (0.5–30 s)", file=sys.stderr)
        return 1
    if args.voice is not None:
        print("error: MOSS Nano has no preset voices; use --ref-audio without --voice", file=sys.stderr)
        return 1
    if args.instruct:
        print("error: --instruct is not supported on MOSS Nano", file=sys.stderr)
        return 1
    if args.language not in (None, "", "auto"):
        print("error: MOSS Nano selects language from the text; omit --language or use auto", file=sys.stderr)
        return 1

    from .tts.moss_nano import MossNanoConfig, MossNanoService, load_moss_nano_model

    try:
        overrides = {}
        if args.max_tokens is not None:
            try:
                overrides["max_new_frames"] = int(args.max_tokens)
            except ValueError:
                raise ValueError("--max-tokens must be a positive integer for MOSS Nano") from None
            if overrides["max_new_frames"] < 1:
                raise ValueError("--max-tokens must be a positive integer for MOSS Nano")
        if args.temperature is not None:
            if not math.isfinite(args.temperature) or args.temperature <= 0:
                raise ValueError("--temperature must be finite and positive for MOSS Nano")
            overrides["audio_temperature"] = args.temperature
        if args.seed is not None:
            overrides["seed"] = args.seed
        config = MossNanoConfig(model_ref=args.model, **overrides)
        with open(args.ref_audio, "rb") as f:
            voice = {"ref_audio": base64.b64encode(f.read()).decode("ascii")}
        service = MossNanoService(load_moss_nano_model(config), config)
        start = time.perf_counter()
        data, _ = service.speech_bytes(args.text, voice=voice, response_format="wav")
        elapsed = time.perf_counter() - start
        with wave.open(io.BytesIO(data), "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
        if duration <= 0:
            raise RuntimeError("model produced no audio")
        with open(args.out, "wb") as f:
            f.write(data)
    except (RuntimeError, ValueError, OSError, wave.Error) as exc:
        print(f"error: MOSS Nano synthesis failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.out}: {duration:.2f}s of audio in {elapsed:.2f}s (RTF {elapsed / duration:.2f})")
    return 0


def _tts_synthesize_voxcpm2(args) -> int:
    """`vllm-omni-mlx tts --model mlx-community/VoxCPM2-4bit ...` — zero-shot
    (default voice), `--instruct` voice design, or `--ref-audio` cloning."""
    import time

    from .tts.voxcpm2 import (
        VoxCPM2Config,
        decode_ref_audio,
        load_voxcpm2_model,
        synthesize,
        wav_bytes,
    )

    if args.language:
        print("error: --language is not supported on VoxCPM2 (multilingual by default)", file=sys.stderr)
        return 1
    if args.voice and args.voice.lower() != "default":
        print(f"error: voice '{args.voice}' is not one of the preset voices; VoxCPM2 speaks zero-shot as 'default', clones via --ref-audio, designs via --instruct", file=sys.stderr)
        return 1
    if args.temperature is not None or args.seed is not None:
        print("warning: --temperature/--seed do not apply to VoxCPM2 (no categorical sampling); ignoring", file=sys.stderr)

    config = VoxCPM2Config(model_ref=args.model, instruct=args.instruct).with_overrides(max_tokens=args.max_tokens)
    try:
        model = load_voxcpm2_model(config)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: failed to load TTS model '{config.model_ref}': {exc}", file=sys.stderr)
        return 1

    ref_audio = None
    if args.ref_audio:
        try:
            with open(args.ref_audio, "rb") as f:
                ref_audio = decode_ref_audio(f.read(), int(model.sample_rate))
        except (OSError, ValueError) as exc:
            print(f"error: --ref-audio: {exc}", file=sys.stderr)
            return 1

    start = time.perf_counter()
    data = wav_bytes(synthesize(model, config, args.text, ref_audio=ref_audio), int(model.sample_rate))
    elapsed = time.perf_counter() - start
    if not data:
        print("error: model produced no audio", file=sys.stderr)
        return 1
    with open(args.out, "wb") as f:
        f.write(data)
    rate = int(model.sample_rate)
    frames = (len(data) - 44) // 2
    print(f"wrote {args.out}: {frames / rate:.2f}s of audio in {elapsed:.2f}s (RTF {elapsed / (frames / rate):.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
