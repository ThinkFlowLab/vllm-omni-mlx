"""CLI, shaped like upstream vllm-omni's `vllm serve <model> --omni`.

    vllm-omni-mlx serve mlx-community/Qwen2.5-0.5B-Instruct-4bit
    vllm-omni-mlx serve mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit --omni
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace


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
        help="serve the model omni-modally: a Qwen3-TTS checkpoint serves /v1/audio/*, anything else forces the mlx-vlm backend",
    )
    parser.add_argument(
        "--tts-model",
        default=None,
        help="also serve speech synthesis on /v1/audio/* (Qwen3-TTS via the [tts] extra); the only model when <model> is omitted",
    )
    parser.add_argument(
        "--image-model",
        default=None,
        help="also serve image generation on /v1/images/generations (mflux via the [image] extra); the only model when <model> is omitted",
    )
    parser.add_argument(
        "--image-lora",
        action="append",
        default=None,
        help="LoRA adapter for the image model (repeatable, HF repo or path; e.g. the Viggle turbo distilled LoRA)",
    )
    parser.add_argument(
        "--image-scheduler",
        default=None,
        help="image sampler override (e.g. viggle_turbo for the Viggle turbo LoRA; default: the model's linear scheduler)",
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
    parser.add_argument("--voice", default=None, help="preset CustomVoice speaker (e.g. vivian, ryan); VoxCPM2 accepts only 'default'")
    parser.add_argument("--language", default=None, help="spoken language hint (default: auto; not supported on VoxCPM2)")
    parser.add_argument("--instruct", default=None, help="emotion/style instruction, or a VoxCPM2 voice description")
    parser.add_argument(
        "--ref-audio",
        default=None,
        help="path to a reference clip for VoxCPM2 voice cloning (0.5–30 s)",
    )
    parser.add_argument("--text", required=True, help="text to synthesize")
    parser.add_argument("--out", required=True, help="output WAV path")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-tokens", default=None, help="Qwen3-TTS tokens or VoxCPM2 audio patches (~20 ms each, default 2000)")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-omni-mlx",
        description="Lightweight OpenAI- and Anthropic-compatible omni-modality server for Apple Silicon.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{serve,tts,image}")
    sub.add_parser("serve", parents=[build_serve_parser()], help="serve a model")
    sub.add_parser("tts", parents=[build_tts_parser()], help="one-shot speech synthesis to a WAV file")
    sub.add_parser("image", parents=[build_image_parser()], help="one-shot image generation to a PNG file")
    return parser


def build_image_parser() -> argparse.ArgumentParser:
    """`vllm-omni-mlx image --model <repo> --prompt "..." --out out.png`."""
    parser = argparse.ArgumentParser(prog="vllm-omni-mlx image", description="Generate an image to a PNG file.", add_help=False)
    parser.add_argument("--model", default=None, help="image model repo or path (default: the [image] default)")
    parser.add_argument("--prompt", required=True, help="text prompt")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--size", default=None, help="WIDTHxHEIGHT, sides multiples of 16 (default: 1024x1024)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", required=True, help="output PNG path")
    parser.add_argument("--lora", action="append", default=None, help="LoRA adapter HF repo or path (repeatable)")
    parser.add_argument("--lora-scale", type=float, action="append", default=None, help="scale per --lora (repeatable)")
    parser.add_argument("--scheduler", default=None, help="sampler override (e.g. viggle_turbo)")
    parser.add_argument("--quantize", type=int, default=None, help="on-the-fly quantization for dense repos (e.g. 4)")
    return parser


def _looks_like_tts(config: dict) -> bool:
    """TTS checkpoints announce themselves in ``config.json``: Qwen3-TTS via
    ``tts_model_type`` (the same key tts/variants.py dispatches on after
    load) or ``model_type``, VoxCPM2 via ``architecture`` (#71)."""
    return (
        "tts_model_type" in config
        or config.get("model_type") == "qwen3_tts"
        or config.get("architecture") == "voxcpm2"
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    if args.command == "image":
        return _image_synthesize(args)
    return _tts_synthesize(args)


def _looks_like_image(model_ref: str) -> bool:
    """Diffusion checkpoints announce themselves by their diffusers/mflux
    layout rather than a root config.json: the official Qwen-Image-2.1 repo
    has model_index.json, mflux-format conversions keep the transformer/
    subdir with an index (the q4 repo has no config.json at all)."""
    import os

    markers = ("model_index.json", "transformer/config.json", "transformer/model.safetensors.index.json")
    for marker in markers:
        try:
            if os.path.isdir(model_ref):
                if not os.path.exists(os.path.join(model_ref, marker)):
                    continue
                with open(os.path.join(model_ref, marker)) as f:
                    content = f.read()
            else:
                from huggingface_hub import hf_hub_download

                with open(hf_hub_download(model_ref, marker)) as f:
                    content = f.read()
            return "_class_name" not in content or "QwenImage" in content or "DiffusionPipeline" in content
        except Exception:
            continue
    return False


def _serve(args) -> int:
    if not args.model and not args.tts_model and not args.image_model:
        build_serve_parser().error("a model is required (or --tts-model / --image-model to serve speech or images alone)")
    if args.omni and not args.model:
        build_serve_parser().error("--omni applies to the served model")
    if args.omni and args.tts_model:
        build_serve_parser().error("--omni and --tts-model are mutually exclusive: --omni already serves the model as TTS when it is a Qwen3-TTS checkpoint")
    if args.image_model and args.model and _looks_like_image(args.model):
        build_serve_parser().error("--image-model is redundant: <model> is itself an image checkpoint and is served as one")

    import uvicorn

    from .server import create_app

    backend = None
    tts_service = None
    image_service = None
    if args.model:
        from .backends import _peek_config, load_backend

        try:
            if _looks_like_image(args.model):
                image_service = _load_image_service(
                    args.model, lora_paths=args.image_lora, scheduler=args.image_scheduler
                )
            elif args.omni:
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

    if args.image_model:
        try:
            image_service = _load_image_service(
                args.image_model, lora_paths=args.image_lora, scheduler=args.image_scheduler
            )
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"error: failed to load image model '{args.image_model}': {exc}", file=sys.stderr)
            return 1

    names = [
        n
        for n in (
            getattr(backend, "name", None),
            getattr(tts_service, "name", None),
            getattr(image_service, "name", None),
        )
        if n
    ]
    print(f"serving {', '.join(names)} on http://{args.host}:{args.port}", file=sys.stderr)
    if image_service is not None:
        print(f"image model license: {image_service.license}", file=sys.stderr)
    uvicorn.run(
        create_app(backend, api_key=args.api_key, tts_service=tts_service, image_service=image_service),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


def _load_image_service(model_ref: str, *, lora_paths=None, scheduler=None):
    import time

    from .diffusion.config import ImageConfig, load_image_model
    from .diffusion.service import ImageService

    config = ImageConfig(model_ref=model_ref)
    if lora_paths:
        config = replace(config, lora_paths=tuple(lora_paths))
    if scheduler:
        config = config.with_overrides(scheduler=scheduler)
    started = time.perf_counter()
    model = load_image_model(config)
    service = ImageService(model, config)
    print(f"image model loaded in {time.perf_counter() - started:.1f}s ({service.license})", file=sys.stderr)
    return service


def _load_tts(model_ref: str):
    """Load a TTS checkpoint for serving — shared by the --omni and
    --tts-model paths. Qwen3-TTS: load, reject variants this build can't
    synthesize at startup (a 400 per request is the late signal otherwise),
    prewarm the compiled streaming shapes, wrap. VoxCPM2 (#71): eager load
    of the mlx-audio model — nothing to prewarm (its generate is
    single-yield; compiled/incremental decode is follow-up loop work)."""
    import time

    from .backends import _peek_config

    if _peek_config(model_ref).get("architecture") == "voxcpm2":
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


def _image_synthesize(args) -> int:
    """`vllm-omni-mlx image --model <repo> --prompt "..." --out out.png` —
    zero-shot t2i, `--lora` (e.g. the Viggle turbo distilled adapter) +
    `--scheduler viggle_turbo` for the 6-step fast lane."""

    from .diffusion.config import ImageConfig, load_image_model
    from .diffusion.service import ImageService, parse_size

    config = ImageConfig()
    if args.model:
        config = replace(config, model_ref=args.model)
    if args.lora:
        config = replace(config, lora_paths=tuple(args.lora), lora_scales=tuple(args.lora_scale or ()))
    config = config.with_overrides(
        size=args.size,
        steps=args.steps,
        guidance=args.guidance,
        negative_prompt=args.negative_prompt,
        scheduler=args.scheduler,
        quantize=args.quantize,
    )
    width, height = parse_size(config.size)
    try:
        model = load_image_model(config)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: failed to load image model '{config.model_ref}': {exc}", file=sys.stderr)
        return 1
    print(f"image model license: {config.license}", file=sys.stderr)

    service = ImageService(model, config)
    try:
        results = service.generate(args.prompt, seed=args.seed, n=1)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    result = results[0]
    with open(args.out, "wb") as f:
        f.write(result.png)
    print(
        f"wrote {args.out}: {width}x{height} in {result.elapsed:.2f}s"
        f" ({result.steps} steps, seed {result.seed})"
    )
    return 0


def _tts_synthesize(args) -> int:
    from .backends import _peek_config

    if args.model and _peek_config(args.model).get("architecture") == "voxcpm2":
        return _tts_synthesize_voxcpm2(args)

    import time

    from .tts.config import TTSConfig, load_tts_model
    from .tts.generate import synthesize, wav_bytes

    if args.ref_audio:
        print("error: --ref-audio needs a VoxCPM2 model (--model mlx-community/VoxCPM2-4bit); Qwen3-TTS clones via the serving API's voice object", file=sys.stderr)
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
