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
vllm-omni-mlx --model mlx-community/Qwen2.5-7B-Instruct-4bit            # text LLM
vllm-omni-mlx --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit         # omni-modality (needs [omni])
```

Options: `--host` (default `127.0.0.1`), `--port` (default `8000`), `--backend auto|text|omni`
(auto sniffs `config.json` for vision/audio sections), `--api-key` to require `Authorization: Bearer …`
or `x-api-key`.

## API

| Endpoint | Format |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI (streaming via SSE, `stop`, multimodal `image_url` / `input_audio` content parts) |
| `POST /v1/messages` | Anthropic (streaming via SSE, `stop_sequences`, base64/URL image blocks) |
| `GET /v1/models` | OpenAI model list |
| `GET /health` | liveness |

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

Layout: `schemas.py` (OpenAI/Anthropic → one internal request), `backends.py` (mlx-lm text backend,
mlx-vlm omni backend, stop-sequence filtering), `server.py` (routes, SSE), `__main__.py` (CLI).
