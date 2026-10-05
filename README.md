# vllm-omni-mlx

<h3 align="center">
Easy, fast, and lightweight omni-modality model serving for Apple Silicon
</h3>

<p align="center">
| <a href="docs/architecture.md"><b>Architecture</b></a> | <a href="docs/profiling.md"><b>Profiling Guide</b></a> | <a href="examples/"><b>Examples</b></a> |
</p>

---

*Latest News* 🔥
- [2026/10] Streaming speech: `"stream": true` chunked PCM on `/v1/audio/speech`, with a first-chunk fast path — time to first audio 132–290 ms and sustained RTF ~0.9 on a quiet M4.
- [2026/10] TTS voice-prefix cache: per-voice prompt state (prompt pieces + static-prefix KV) is reused across requests — first audio another ~20–40 ms sooner on a warm voice, sustained RTF unchanged, reproducible streams.
- [2026/10] M1 speech milestone: Qwen3-TTS CustomVoice synthesis on MLX — `/v1/audio/speech` + `/v1/audio/voices`, one-shot `tts` CLI, 64 weight-gated tests.
- [2026/10] v0.1 core: OpenAI + Anthropic compatible APIs on Starlette, cross-turn prompt cache (8.3× faster TTFT on continued conversations), `--draft-model` and `--kv-bits` performance flags.

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

- Streaming TTS first-chunk fast path: first audio in 132–290 ms instead of after full generation
- TTS voice-prefix cache: the per-voice static prompt (instruct/role/codec prefix) prefills once per voice and is spliced into every request — no re-prefill, bitwise-reproducible streams per voice
- Cross-turn prompt cache: continuing a conversation prefills only the new suffix (measured 8.3× faster time-to-first-token, text engine)
- `--draft-model` speculative decoding and `--kv-bits` quantized KV cache for long contexts (text engine)

vllm-omni-mlx is flexible and easy to use with:

- Seamless loading of popular Hugging Face models through `mlx-vlm` (vision/audio) and `mlx-audio` (speech), on the MLX engine stack
- Small optional-dependency footprint: ~440 MB core install, no torch
- Streaming outputs, preset and instructed TTS voices, one-shot CLI synthesis

## Supported Models

Like vLLM-Omni, vllm-omni-mlx targets omni-modality serving across the speech stack — ASR (speech in), TTS (speech out), and any-to-any chat. Plain text-LLM serving (vLLM proper and mlx-lm's own server territory) and vision-language (image-in, text-out) models are not supported categories. Supported:

- **Omni-modality models** (Qwen3-Omni — text, image, and audio in; text and speech out)
- **ASR models** (e.g. Whisper, Parakeet, Qwen3-ASR, Voxtral, SenseVoice)
- **TTS models** (Qwen3-TTS CustomVoice, VoiceDesign)

| Modality | Models | Example HF models | Engine | Status |
| --- | --- | --- | --- | --- |
| Text / image / audio in → text + speech out | Qwen3-Omni | `mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit` | `mlx-vlm` (`[omni]` extra) | ⚠️ via mlx-vlm — speech-out chat not plumbed yet, not verified here <sup>1</sup> |
| Speech in → text out (ASR) | Whisper, Parakeet, Qwen3-ASR, Qwen2-Audio, Voxtral, SenseVoice, Moonshine, … <sup>2</sup> | `mlx-community/whisper-large-v3-turbo` | `mlx-audio` (`[tts]` extra) | 🚧 planned — engine support via mlx-audio stt, transcription endpoint not built yet |
| Text → speech out | Qwen3-TTS-12Hz-1.7B-CustomVoice | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit` | `mlx-audio` (`[tts]` extra) | ✅ verified end-to-end (4-bit) <sup>3</sup> |
| Text → speech out (described voice) | Qwen3-TTS-12Hz-1.7B-VoiceDesign | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit` | `mlx-audio` (`[tts]` extra) | ✅ verified end-to-end buffered + streaming (4-bit, weight-gated tests) <sup>3</sup> |

<sup>1</sup> mlx-vlm 0.7 ships the full Qwen3-Omni thinker/talker implementation; this server currently
consumes its text output only — speech-out chat and video input are future work. The 30B-A3B MoE
weighs ~22 GB at 4-bit, so it needs a large-memory Mac. Vision-language and text-only checkpoints
still load through their engines but are not supported categories.
<sup>2</sup> mlx-audio 0.5.7's stt package implements these families (Whisper is its default); serving
them on an OpenAI-style `/v1/audio/transcriptions` endpoint is planned. Families listed are present
in the installed engine, not verified through this server.
<sup>3</sup> Verified on the 4-bit quantization; the bf16 variant loads but is untested. `speed` must be 1.0 for now.

## Getting Started

Requires Python 3.10+ on an Apple Silicon Mac (MLX ships arm64-only wheels).

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e .            # server core (MLX engine stack)
pip install -e '.[omni]'    # + vision/audio models (mlx-vlm)
pip install -e '.[tts]'     # + speech synthesis (mlx-audio)
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

### Run

```sh
# omni-modality server: Qwen3-Omni chat + speech synthesis in one process
vllm-omni-mlx serve mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit \
    --tts-model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit      # needs [omni] + [tts]

# speech-only server
vllm-omni-mlx serve mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit --omni
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
| `POST /v1/chat/completions` | OpenAI (streaming via SSE, `stop`, multimodal `image_url` / `input_audio` content parts) |
| `POST /v1/messages` | Anthropic (streaming via SSE, `stop_sequences`, base64/URL image blocks) |
| `POST /v1/audio/speech` | OpenAI audio (`wav` 24 kHz mono / raw `pcm`; needs a TTS model via `--omni` or `--tts-model`, `[tts]` extra) |
| `GET /v1/audio/voices` | preset CustomVoice speakers for the loaded TTS model (empty on Base/VoiceDesign checkpoints) |
| `GET /v1/models` | OpenAI model list |
| `GET /health` | liveness |

Speech synthesis quickstart:

```sh
pip install 'vllm-omni-mlx[tts]'
vllm-omni-mlx serve mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit --omni --api-key demo
curl -H 'Authorization: Bearer demo' -H 'Content-Type: application/json' \
    -d '{"input": "Hello from vllm omni em el ex.", "voice": "vivian"}' \
    http://127.0.0.1:8000/v1/audio/speech -o speech.wav
```

`voice` picks a preset speaker (`GET /v1/audio/voices` lists them), `instructions`
add an emotion/style prompt, `language` forces a language (default auto);
`speed` must be 1.0 for now. On a VoiceDesign checkpoint
(`mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit`) the same `instructions`
field **is the voice** — a description like "A cheerful young female voice with
high pitch and energetic tone" (required; `voice` is rejected — presets don't
exist there). The field's meaning is set by the loaded checkpoint, mirroring
mlx-audio's own mapping; VoiceDesign works buffered and streaming, on the same
first-chunk fast path as CustomVoice (compiled decode, #65). Streaming: pass `"stream": true` for chunked raw
PCM (24 kHz 16-bit mono, `X-Audio-*` response headers) instead of a buffered
WAV — first audio typically lands in under 0.5 s instead of after the full
generation; `streaming_interval` (default 0.5 s) trades first-audio latency
for chunk cadence, and `streaming_initial_interval` (default 0.2 s) emits the
first chunk as soon as that much audio exists (quantized to a power-of-two
frame bucket, so compiled decode shapes stay bounded) — measured time to
first audio: 195–290 ms at the default, 132 ms at 0.08 s. One-shot synthesis without a server:
`vllm-omni-mlx tts --voice ryan --text "..." --out out.wav`. See `examples/`.

**Voice cloning** (Base checkpoints, e.g.
`mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit`): `voice` is instead an object
carrying a short reference clip and its transcript —

```sh
vllm-omni-mlx serve mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit --omni --api-key demo
REF=$(base64 -i reference.wav)
curl -H 'Authorization: Bearer demo' -H 'Content-Type: application/json' \
    -d "{\"input\": \"Any text in the cloned voice.\", \"voice\": {\"ref_audio\": \"$REF\", \"ref_text\": \"transcript of the reference clip\"}}" \
    http://127.0.0.1:8000/v1/audio/speech -o cloned.wav
```

The clip is decoded and resampled to 24 kHz mono server-side (any format
miniaudio/ffmpeg reads); keep it 0.5–30 s of clean speech. Base checkpoints
have no preset voices (`GET /v1/audio/voices` returns `[]`) and CustomVoice
checkpoints ignore cloning — send the form matching your checkpoint.
Cloning streams too: `stream: true` + the voice object emits chunked PCM on
the same first-chunk fast path as presets (the vendored loop is
token-exact against mlx-audio's ICL loop; chunked audio differs from the
buffered WAV at waveform level by nature — the vocoder is stateful).

The 0.6B CustomVoice variant
(`mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit`) serves the same
preset voices and endpoint; `instructions` are rejected with a 400 there —
emotion/style prompts are a 1.7B capability.

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

```sh
python -m unittest discover -s tests   # stdlib unittest, no extra deps
```

Weight-gated tests (TTS, prompt cache) run against locally cached checkpoints and skip
where the weights are absent — a green CI run does not by itself mean the weight-dependent
paths were exercised.

Profiling and benchmarking on Apple Silicon: [docs/profiling.md](docs/profiling.md) —
timing harness (time the `mx.eval` bracket, mind GPU clock ramp), Metal GPU capture
for per-kernel truth, powermetrics/xctrace for SoC counters.

Layout: `schemas.py` (OpenAI/Anthropic → one internal request), `backends.py` (mlx-lm text backend,
mlx-vlm omni backend, stop-sequence filtering), `server.py` (routes, SSE), `__main__.py` (CLI),
`tts/` (Qwen3-TTS pipeline: config, loader, talker, code2wav, streaming loop).
