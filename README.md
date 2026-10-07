# vllm-omni-mlx

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/vllm-omni-mlx-logo-dark.svg">
    <img alt="vLLM-Omni-MLX" src="assets/vllm-omni-mlx-logo-light.svg" width="620">
  </picture>
</p>

<h3 align="center">
Easy, fast, and lightweight omni-modality model serving for Apple Silicon
</h3>

<p align="center">
| <a href="docs/architecture.md"><b>Architecture</b></a> | <a href="docs/speech.md"><b>Speech Guide</b></a> | <a href="docs/profiling.md"><b>Profiling Guide</b></a> | <a href="examples/"><b>Examples</b></a> | <a href="CONTRIBUTING.md"><b>Contributing</b></a> |
</p>

---

*Latest News* 🔥
- [2026/10] **[v0.1.0 released](https://github.com/ThinkFlowLab/vllm-omni-mlx/releases/tag/v0.1.0)** — the Qwen3-TTS family is fully served on Apple Silicon: preset/instructed voices, zero-shot voice cloning, and text-described voices (VoiceDesign), buffered and streaming, across all five 4-bit checkpoints (0.6B/1.7B). First audio in **~0.1 s** (88 ms on M1 Max), sustained RTF **0.29–0.45**, bit-reproducible streams — numbers reproduced on two machines ([#84](https://github.com/ThinkFlowLab/vllm-omni-mlx/issues/84)).
- [2026/10] Batch-1 latency stack: compiled per-frame decode, per-voice prefix cache, single-codec-frame first chunk for TTS; cross-turn prompt cache (8.3× faster TTFT), `--draft-model` and `--kv-bits` for chat.

---

## About

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) serves omni-modality models on large GPU clusters.
vllm-omni-mlx is its lightweight Apple Silicon counterpart: the same serving surface —
OpenAI- and Anthropic-compatible chat APIs plus OpenAI speech synthesis — on [MLX](https://github.com/ml-explore/mlx),
in a single process built for batch-1 low latency.

- **Omni-modality, in and out**: text, image, and audio in; text and speech out — chat, ASR (speech → text), and TTS
- **API compatibility**: OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`, both with SSE streaming
- **Lightweight by design**: one model per process, no scheduler, no worker pool, no FastAPI/pydantic
  in the core — just Starlette plus `mlx-lm` and `mlx-vlm`

vllm-omni-mlx is fast with:

- Streaming TTS: first audio in ~0.1 s (per-voice prefix cache + single-codec-frame first chunk), sustained RTF 0.29–0.45 across machines
- Compiled per-frame decode (fused predictor, shapeless talker step) and bitwise-reproducible streams per voice
- Cross-turn prompt cache: continuing a conversation prefills only the new suffix (measured 8.3× faster time-to-first-token, text engine)
- `--draft-model` speculative decoding and `--kv-bits` quantized KV cache for long contexts (text engine)

vllm-omni-mlx is flexible and easy to use with:

- Seamless loading of popular Hugging Face models through `mlx-vlm` (vision/audio) and `mlx-audio` (speech), on the MLX engine stack
- Small optional-dependency footprint: ~440 MB core install, no torch
- Streaming outputs, preset and instructed TTS voices, one-shot CLI synthesis

## Supported Models

| Modality | Examples | Status |
| --- | --- | --- |
| **TTS** — text → speech | Qwen3-TTS (CustomVoice · Base · VoiceDesign, 0.6B/1.7B) | ✅ verified end-to-end, buffered + streaming — [speech guide](docs/speech.md) |
| **Omni** — any-to-any chat | Qwen3-Omni 30B-A3B | 🚧 chat works via mlx-vlm; speech-out chat in progress |
| **ASR** — speech → text | Whisper, Qwen3-ASR, Voxtral, … | 🚧 planned — `/v1/audio/transcriptions` (#68) |
| **Diffusion** — text/image → image | Z-Image-Turbo (6B, 9-step distilled, 4-bit) | ✅ `/v1/images/generations` (mflux seam) — SD tier next (#95) |

| Modality | Examples | Status |
| --- | --- | --- |
| **TTS** — text → speech | Qwen3-TTS (CustomVoice · Base · VoiceDesign, 0.6B/1.7B); VoxCPM2 (zero-shot · cloned · described voice, 30+ languages, 48 kHz) | ✅ verified end-to-end, buffered + streaming — [speech guide](docs/speech.md) |
| **Omni** — any-to-any chat | Qwen3-Omni 30B-A3B | 🚧 chat works via mlx-vlm; speech-out chat in progress |
| **ASR** — speech → text | Whisper, Qwen3-ASR, Voxtral, … | 🚧 planned — `/v1/audio/transcriptions` (#68) |
| **Diffusion** — text/image → image | Z-Image-Turbo (6B, 9-step distilled, 4-bit) | ✅ `/v1/images/generations` (mflux seam) — SD tier next (#95) |

Text-only LLMs and image-in/text-out VLMs load through their engines but are
not this server's target categories.

## Getting Started

Requires Python 3.10+ on an Apple Silicon Mac (MLX ships arm64-only wheels).

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e .            # server core (MLX engine stack)
pip install -e '.[omni]'    # + vision/audio models (mlx-vlm)
pip install -e '.[tts]'     # + speech synthesis (mlx-audio)
pip install -e '.[image]'  # + image generation, Z-Image-Turbo (mflux)
```

### Dependency footprint

| Install | Direct deps | Resolved packages | Disk |
| --- | --- | --- | --- |
| core | `mlx-lm`, `starlette`, `uvicorn` | 38 | ~440 MB |
| + `[tts]` | + `mlx-audio` | 44 | ~560 MB |
| + `[omni]` | + `mlx-vlm` | 59 | ~750 MB |
| + `[image]` | + `mflux` | ~55 | ~870 MB |

The core install pulls in the MLX stack (`mlx` + `mlx-metal` kernels, `transformers`,
`tokenizers`, `huggingface_hub`) plus starlette/uvicorn and almost nothing else —
**no FastAPI, no pydantic, no torch**. The `[tts]` extra adds ~125 MB through
`mlx-audio` (miniaudio, sounddevice — still no torch). The `[omni]` extra adds
~315 MB through `mlx-vlm` (opencv, pillow, scipy — and mlx-audio, so `[omni]`
implies `[tts]`; that path does drag in fastapi/pydantic, contained to the
optional extra). The `[image]` extra adds ~870 MB through `mflux` — the one
extra that does pull `torch` (583 MB of it, for mflux's utility stack; MLX
still runs every model). Measured on macOS arm64 / Python 3.13 with mlx-lm 0.32,
mlx-vlm 0.7, mlx-audio 0.5.7, and mflux 0.21.

### Run

```sh
# omni-modality server: Qwen3-Omni chat + speech synthesis in one process
vllm-omni-mlx serve mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit \
    --tts-model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit      # needs [omni] + [tts]

# speech-only server
vllm-omni-mlx serve mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit --omni

# image-only server (Z-Image-Turbo, 4-bit ~5.5 GiB of weights)
vllm-omni-mlx serve --image-model z-image-turbo                      # needs [image]
```

Qwen3-Omni serves text-out chat today (speech-out chat is in progress); the 30B-A3B
4-bit checkpoint is ~22 GB, so pick a Mac with the memory for it.

Options: `--host` (default `127.0.0.1`), `--port` (default `8000`), `--backend auto|text|omni`
(auto sniffs `config.json` for vision/audio sections), `--omni` (serve the model omni-modally:
a Qwen3-TTS checkpoint serves `/v1/audio/*`, anything else forces the omni backend),
`--api-key` to require `Authorization: Bearer …` or `x-api-key`.

Performance flags:

- `--draft-model <repo>` — speculative decoding for the text engine: pass a smaller
  model that shares the main model's tokenizer.
  While a draft model is set, the cross-turn prompt cache is bypassed (each turn re-prefills).
- `--kv-bits <n>` (`--kv-group-size`, default 64) — quantize the KV cache to `n` bits to cut
  memory on long contexts (mlx-lm quantizes entries beyond its first-5000-token window;
  same kwargs are honored by mlx-vlm for the omni backend).

Conversations continuing a previous turn reuse its KV cache: only the new suffix is
prefilled (text backend; the divergence or edit of resent history falls back to a full
re-prefill, so correctness never depends on the cache).

## API

| Endpoint | Format |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI (SSE streaming, multimodal content parts) |
| `POST /v1/messages` | Anthropic (SSE streaming, image blocks) |
| `POST /v1/audio/speech` | OpenAI audio (`wav` / chunked `pcm` with `stream: true`) |
| `GET /v1/audio/voices` | preset speakers of the loaded TTS model |
| `GET /v1/models`, `GET /health` | model list, liveness |

Bring up a speech server and talk to it:

```sh
pip install 'vllm-omni-mlx[tts]'
vllm-omni-mlx serve mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit --omni --api-key demo
curl -H 'Authorization: Bearer demo' -H 'Content-Type: application/json' \
    -d '{"input": "Hello from vllm omni mlx.", "voice": "vivian"}' \
    http://127.0.0.1:8000/v1/audio/speech -o speech.wav
```

Chat works the same way on the same server (`/v1/chat/completions`,
`/v1/messages`, both with `"stream": true`). Voices, instructions, voice
cloning, streaming knobs, and per-checkpoint differences: the
**[speech guide](docs/speech.md)**; runnable scripts in [`examples/`](examples/);
one-shot synthesis without a server: `vllm-omni-mlx tts --voice ryan --text "..." --out out.wav`.

## Design Notes & Limits

See [docs/architecture.md](docs/architecture.md) for the architecture diagram and rationale.

- **Single model, serialized generation.** One model instance per process; a lock serializes
  generation. Concurrent requests queue instead of racing the GPU. This is the intended
  lightweight trade-off, not an oversight.
- **Sampling**: `temperature`, `top_p`, `top_k`, `max_tokens`, stop sequences are mapped onto
  both APIs. Tools/function calling are not supported yet.
- **Media**: the chat path targets Qwen3-Omni — text, image, and audio in, text out today;
  speech-out chat via its talker and video input are future work. Speech out today is the TTS
  endpoint; speech in (ASR via mlx-audio stt) is planned. Vision-language and text-only
  checkpoints load through their engines but are not supported categories.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — setup, running the (weight-gated) tests,
the A/B rule for performance PRs, and the repository layout.
