# vllm-omni-mlx

High-performance OpenAI and Anthropic compatible omni-modality model inference server for Apple Silicon.

Built on [MLX](https://github.com/ml-explore/mlx) and deliberately lightweight: one model per process,
no scheduler, no worker pool, no FastAPI/pydantic — just Starlette plus `mlx-lm` (text) and `mlx-vlm`
(image/audio/video, optional).

## Install

Requires Python 3.10+ on an Apple Silicon Mac (MLX ships arm64-only wheels).

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e .            # text models (mlx-lm)
pip install -e '.[omni]'    # + vision/audio models (mlx-vlm)
```

### Dependency footprint

| Install | Direct deps | Resolved packages | Disk |
| --- | --- | --- | --- |
| core | `mlx-lm`, `starlette`, `uvicorn` | 38 | ~440 MB |
| + `[omni]` | + `mlx-vlm` | 59 | ~750 MB |

The core install pulls in the MLX stack (`mlx` + `mlx-metal` kernels, `transformers`,
`tokenizers`, `huggingface_hub`) plus starlette/uvicorn and almost nothing else —
**no FastAPI, no pydantic, no torch**. The `[omni]` extra adds ~315 MB through
`mlx-vlm` (opencv, pillow, scipy, mlx-audio — which does drag in fastapi/pydantic,
contained to the optional path). Measured on macOS arm64 / Python 3.13 with
mlx-lm 0.32 and mlx-vlm 0.7.

## Run

```sh
vllm-mlx serve mlx-community/Qwen2.5-7B-Instruct-4bit            # text LLM
vllm-mlx serve mlx-community/Qwen2.5-VL-7B-Instruct-4bit --omni  # omni-modality (needs [omni])
```

Options: `--host` (default `127.0.0.1`), `--port` (default `8000`), `--backend auto|text|omni`
(auto sniffs `config.json` for vision/audio sections), `--omni` (serve the model omni-modally:
a Qwen3-TTS checkpoint serves `/v1/audio/*`, anything else forces the omni backend),
`--api-key` to require `Authorization: Bearer …` or `x-api-key`.

Performance flags:

- `--draft-model <repo>` — speculative decoding for the text backend: pass a smaller
  model that shares the main model's tokenizer (e.g. serve
  `Qwen2.5-7B-Instruct-4bit` with `--draft-model mlx-community/Qwen2.5-0.5B-Instruct-4bit`).
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
| `POST /v1/chat/completions` | OpenAI (streaming via SSE, `stop`, multimodal `image_url` / `input_audio` content parts) |
| `POST /v1/messages` | Anthropic (streaming via SSE, `stop_sequences`, base64/URL image blocks) |
| `POST /v1/audio/speech` | OpenAI audio (`wav` 24 kHz mono / raw `pcm`; needs a TTS model via `--omni` or `--tts-model`, `[tts]` extra) |
| `GET /v1/audio/voices` | preset CustomVoice speakers for the loaded TTS model |
| `GET /v1/models` | OpenAI model list |
| `GET /health` | liveness |

Speech synthesis quickstart:

```sh
pip install 'vllm-omni-mlx[tts]'
vllm-mlx serve mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit --omni --api-key demo
curl -H 'Authorization: Bearer demo' -H 'Content-Type: application/json' \
    -d '{"input": "Hello from vllm omni em el ex.", "voice": "vivian"}' \
    http://127.0.0.1:8000/v1/audio/speech -o speech.wav
```

`voice` picks a preset speaker (`GET /v1/audio/voices` lists them), `instructions`
adds an emotion/style prompt, `language` forces a language (default auto);
`speed` must be 1.0 for now. Streaming: pass `"stream": true` for chunked raw
PCM (24 kHz 16-bit mono, `X-Audio-*` response headers) instead of a buffered
WAV — first audio typically lands in under 0.5 s instead of after the full
generation; `streaming_interval` (default 0.5 s) trades first-audio latency
for chunk cadence, and `streaming_initial_interval` (default 0.2 s) emits the
first chunk as soon as that much audio exists (quantized to a power-of-two
frame bucket, so compiled decode shapes stay bounded) — measured time to
first audio: 195–290 ms at the default, 132 ms at 0.08 s. One-shot synthesis without a server:
`vllm-mlx tts --voice ryan --text "..." --out out.wav`. See `examples/`.

OpenAI-style request:

```sh
curl http://127.0.0.1:8000/v1/chat/completions -d '{
  "model": "any",
  "messages": [{"role": "user", "content": "Say hi in three words."}],
  "max_tokens": 32
}'
```

Anthropic-style request:

```sh
curl http://127.0.0.1:8000/v1/messages -d '{
  "model": "any",
  "max_tokens": 32,
  "messages": [{"role": "user", "content": "Say hi in three words."}]
}'
```

Both accept `"stream": true`; media arrives as data URLs (OpenAI `image_url`) or
base64/URL sources (Anthropic `image` blocks). Either way it reaches the model the same way.

## Design notes & limits

See [docs/architecture.md](docs/architecture.md) for the architecture diagram and rationale.

- **Single model, serialized generation.** One model instance per process; a lock serializes
  generation. Concurrent requests queue instead of racing the GPU. This is the intended
  lightweight trade-off, not an oversight.
- **Sampling**: `temperature`, `top_p`, `top_k`, `max_tokens`, stop sequences are mapped onto
  both APIs. Tools/function calling are not supported yet.
- **Media**: images and audio in, text out. mlx-vlm also supports video — plumbing it through the
  OpenAI/Anthropic request shape is future work, as are audio-out models.

## Development

```sh
python -m unittest discover -s tests   # stdlib unittest, no extra deps
```

Profiling and benchmarking on Apple Silicon: [docs/profiling.md](docs/profiling.md) —
timing harness (time the `mx.eval` bracket, mind GPU clock ramp), Metal GPU capture
for per-kernel truth, powermetrics/xctrace for SoC counters.

Layout: `schemas.py` (OpenAI/Anthropic → one internal request), `backends.py` (mlx-lm text backend,
mlx-vlm omni backend, stop-sequence filtering), `server.py` (routes, SSE), `__main__.py` (CLI).
