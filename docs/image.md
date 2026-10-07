# Image Guide — Qwen-Image-2.1 on Apple Silicon

Text → image through the OpenAI images API, served by the same lightweight
server as chat and speech. The model math lives entirely in
[mflux](https://github.com/filipstrand/mflux) (pinned by the `[image]` extra,
`>=0.21,<0.22`) — this repo owns config, loading, validation, and serving only.

## License — read before serving

**Qwen-Image-2.1 and its fast variants ship under the
[Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE):
non-commercial, research/evaluation purposes only.** Commercial use requires a
separate license from Qwen (`model-business@notice.qwencloud.com`). This
applies to the base checkpoint and equally to the Viggle turbo LoRA and Pruna
distillations (both are derivatives). The server prints the license at load;
if you serve this model, your operators and users inherit the terms.

The Apache-2.0 fast-lane tier (Z-Image-Turbo #99, FLUX.2-klein #100) is the
commercial-safe alternative on the same seam; this model's value is prompt
understanding/world knowledge + upstream vllm-omni parity, not latency.

## Install & run

```sh
pip install 'vllm-omni-mlx[image]'        # +mflux (brings torch, ~750 MB)

# image-only server (solo — see memory below)
vllm-omni-mlx serve mlx-community/Qwen-Image-2.1-mflux-q4

# one-shot, no server
vllm-omni-mlx image --prompt "a cozy harbor town at dawn" --out out.png --seed 42
```

Weights: `mlx-community/Qwen-Image-2.1-mflux-q4` (~9.6 GB on disk, 4-bit
DiT + 4-bit text encoder in mflux layout). First call downloads via
`huggingface_hub` — set `HF_ENDPOINT=https://hf-mirror.com` if the Hub is
unreachable from your network. Dense repos also work with `--quantize 4|8`
(on-the-fly quantization at load).

## API

`POST /v1/images/generations` — OpenAI images shape, `b64_json` responses:

```sh
curl -d '{"prompt": "a cozy harbor town at dawn", "size": "1024x1024",
          "steps": 40, "seed": 42}' http://127.0.0.1:8000/v1/images/generations
```

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | required | non-empty string |
| `size` | `1024x1024` | `WIDTHxHEIGHT`, sides multiples of 16, 256–2048 (rejected, not rounded) |
| `steps` | `40` | 1–100; the card's default is 40-step guidance-free |
| `guidance` | `1.0` | 1.0 = guidance-free; >1 with a `negative_prompt` enables true CFG |
| `seed` | random | with `n>1` images run seed, seed+1, … (deterministic per request) |
| `negative_prompt` | none | |
| `n` | `1` | 1–4, generated sequentially (batch-1 doctrine) |
| `response_format` | `b64_json` | only `b64_json` is served (no URL hosting) |

Generation is serialized by the single-user lock, same as TTS: concurrent
requests queue. Same seed + same parameters reproduce byte-identical output
(weight-gated test: `tests/test_image_e2e.py`).

## The Viggle turbo fast lane (6 steps)

The fast-variant decision recorded on #101: **Viggle turbo** — a DMD-distilled
LoRA over the base DiT, sampled on its 6 trained sigma nodes through mflux's
built-in `viggle_turbo` scheduler (6 steps fixed, no CFG). It won on
mflux-first-class support (the scheduler ships in mflux itself); Pruna ships
merged dense transformers with no mflux loader path, and the base model at 40
steps is the quality reference.

```sh
vllm-omni-mlx serve mlx-community/Qwen-Image-2.1-mflux-q4 \
    --image-lora Viggle/Qwen-Image-2.1-viggle-turbo:Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors \
    --image-scheduler viggle_turbo
```

`org/repo:filename` LoRA refs download on first use. The v0.3 LoRA edges out
v0.2.1 on ghosting; expect visible distillation artifacts at 6 steps (ghosted
structures) against the 40-step base — that is the latency/quality trade,
priced below. One-shot: `vllm-omni-mlx image --lora … --scheduler viggle_turbo`.
Steps must be 6 on this scheduler (validated; the sampler is the LoRA's trained
sigma table).

## Measured on Apple Silicon (M4, 24 GB)

Solo residency, `mlx-community/Qwen-Image-2.1-mflux-q4` (4-bit), guidance-free,
fixed seed, cool-down sleeps between runs; per-step cadence from mflux's
in-loop callback; MLX peak = `mx.get_peak_memory()` after
`mx.reset_peak_memory()`; RSS = process high-water. Repro:
`scripts/bench_image.py --preset base|turbo --out bench.json`.

### Base path — 40 steps, guidance-free (the quality reference)

| Size | Runs | Time-to-image (s) | Per-step (s) | Peak MLX (GiB) | RSS (GiB) | PNG (KiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 512x512 | 3 | 190 / 192 / 192 | 4.6±0.7 | 13.6 | 8.4 | 388 |
| 768x768 | 1 | 449 | 10.9±1.7 | 18.1 | 8.8 | 980 |
| 1280x720 | 1 | 739 | 17.9±2.8 | 21.6 | 8.8 | 1312 |
| 1024x1024 | 2 | 852 / 852 | 20.6±3.3 | 23.3 | 8.8 | 1564 |

### Fast lane — Viggle turbo v0.3 LoRA, 6 steps

| Size | Runs | Time-to-image (s) | Per-step (s) | Peak MLX (GiB) | RSS (GiB) | PNG (KiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 512x512 | 1 | 31 | 4.1±1.4 | 16.9 | 9.1 | 354 |
| 1024x1024 | 1 | 134 | 17.6±8.5 | 24.6 | 9.1 | 1420 |

Weights resident at load: 8.93 GiB (4-bit), 12.2 GiB with the baked r128 LoRA
(227 adapter-targeted layers re-quantized q8); load 7–11 s. 1328×1328 was not
measured: the 1024² fast lane already peaks at 24.6 GiB MLX-active on this 24 GB
machine — the base path at ≥1024² and any path at 1328² want more than 24 GB.

## Design notes

- **Solo-only residency.** Weights are ~8.9 GiB resident and generation peaks
  ~14–17 GiB MLX-active; on a 16 GB Mac serve this model alone and keep chat
  out of the same process. The 24 GB class runs it comfortably.
- **The step loop is the decode loop** (same doctrine as the TTS per-frame
  work): latency scales with latent token count ((H/16)·(W/16)) — a 4×
  pixel-count step costs ~4× (compute-bound at 4-bit), so resolution is the
  primary latency knob, then steps.
- **Prompt encoding is per-request** (~2 s at load-warm state): the text
  encoder runs on every generate. A prompt-embedding cache keyed by prompt
  hash is the obvious follow-up (the TTS prefix-cache analog).
- **Zero native model code** — the architecture rule holds: mflux carries the
  model; `diffusion/` is config + validation + serving.
- **Follow-ups**: `/v1/images/edits` + Qwen-Image-2.1 reference editing
  (prefix-KV, mflux-native — second edit-native model after #100), Z-Image-Turbo
  (#99) and FLUX.2-klein (#100) as Apache-2.0 entries on the same seam,
  step-cache (`step_cache_ratio`) as a latency lever.
