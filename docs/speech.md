# Speech (TTS) Usage

Everything about `/v1/audio/speech` beyond the quickstart in the README: voices,
instructions, streaming knobs, cloning, and the per-checkpoint differences.
Setup and server commands live in the [README](../README.md); runnable scripts
in [`examples/`](../examples/).

## Voices by checkpoint

The loaded checkpoint decides what `voice` and `instructions` mean — mirroring
mlx-audio's own mapping:

| Checkpoint | `voice` | `instructions` |
| --- | --- | --- |
| CustomVoice (0.6B / 1.7B) | preset speaker name — `GET /v1/audio/voices` lists them (9 presets: vivian, ryan, aiden, …) | optional emotion/style prompt (1.7B only — rejected with 400 on 0.6B) |
| Base (1.7B) | **cloning object** — `{"ref_audio": <base64>, "ref_text": "…"}` | — |
| VoiceDesign (1.7B) | rejected (no presets) | **required** — the voice description itself, e.g. "A cheerful young female voice with high pitch and energetic tone" |
| VoxCPM2 (2.5B-class, 48 kHz) | `"default"` (zero-shot) **or** a cloning object `{"ref_audio": <base64>}` — `ref_text` accepted but unused (VoxCPM2 conditions on the clip alone) | **voice design** — a description of the speaker, e.g. "A young woman with a warm and gentle voice" |

`language` forces a language (default auto; not supported on VoxCPM2, which is
multilingual natively); `speed` must be 1.0 for now. VoxCPM2 output is 48 kHz
mono (Qwen3-TTS is 24 kHz); the streaming response headers carry the rate.

## Streaming

`"stream": true` switches from a buffered WAV to chunked raw PCM (24 kHz
16-bit mono, `X-Audio-*` response headers):

- first audio typically lands in **~0.1 s** (single-codec-frame first chunk,
  warm voice — per-voice prefix cache);
- `streaming_interval` (default 0.5 s) sets the steady chunk size;
- `streaming_initial_interval` (default 0.08 s) sets the first-chunk size —
  raise to 0.2 s for a chunkier first beat.

```sh
curl -N -H 'Authorization: Bearer demo' -H 'Content-Type: application/json' \
    -d '{"input": "Hello.", "voice": "vivian", "stream": true}' \
    http://127.0.0.1:8000/v1/audio/speech -o speech.pcm
```

## Voice cloning (Base checkpoints)

`voice` carries a short reference clip and its transcript:

```sh
vllm-omni-mlx serve mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit --omni --api-key demo
REF=$(base64 -i reference.wav)
curl -H 'Authorization: Bearer demo' -H 'Content-Type: application/json' \
    -d "{\"input\": \"Any text in the cloned voice.\", \"voice\": {\"ref_audio\": \"$REF\", \"ref_text\": \"transcript of the reference clip\"}}" \
    http://127.0.0.1:8000/v1/audio/speech -o cloned.wav
```

The clip is decoded and resampled to 24 kHz mono server-side (any format
miniaudio/ffmpeg reads); keep it 0.5–30 s of clean speech. Cloning streams too
(`stream: true` + the voice object) on the same fast path as presets — the
loop is token-exact against mlx-audio's ICL path; chunked audio differs from
the buffered WAV at waveform level by nature (the vocoder is stateful).
Cloning pays per-request setup (reference decode + the ICL prompt prefill;
the reference *re-encode* itself is cached per clip), so its
time-to-first-audio is ~0.28 s on 1.7B and ~0.15 s on 0.6B vs ~0.1 s for
presets (quiet M4, 2026-10-08). The prefill dominates and is compute-bound
at the 4-bit GEMM floor — measured and closed as no kernel headroom on the
1.7B ([#112](https://github.com/ThinkFlowLab/vllm-omni-mlx/issues/112));
the **0.6B pair is the low-latency, small-memory clone pick** (clone-path
peak 3.7 vs 5.9 GiB).

## VoxCPM2

```sh
vllm-omni-mlx serve mlx-community/VoxCPM2-4bit --omni        # needs [tts]
vllm-omni-mlx tts --model mlx-community/VoxCPM2-4bit \
    --text "Hello from VoxCPM2." --out out.wav                # or --instruct / --ref-audio clip.wav
```

A tokenizer-free AR + diffusion model (MiniCPM4 backbone → CFM solver →
48 kHz AudioVAE), 30+ languages, no speaker presets: `voice: "default"`
speaks zero-shot, `instructions` designs a voice, and cloning rides the same
`voice` object as Base (`ref_audio` required, `ref_text` unused). `language`
is rejected — the model is multilingual natively.

Generation runs through a vendored compiled loop (the whole CFM solver in one
fixed-shape trace, KV-as-arrays LM steps — bitwise-reproducible against the
library under a fixed seed); `VLLM_OMNI_VOXCPM2_EAGER=1` serves the plain
mlx-audio path. mlx-audio's generate is single-yield, so `stream: true`
delivers interval-sized chunks of the **finished** buffer — first audio lands
when synthesis completes; incremental per-patch decode is follow-up work
(#88).

#88's serving defaults, each gated on quality: `inference_timesteps` 8 and
load-time quantization of the blocks the checkpoint ships in bf16 (**8-bit
DiT + 4-bit encoder**; `VLLM_OMNI_VOXCPM2_QUANT` = `off`/`4bit`/`8bit`).
The compound matters: t=6 was equivalent to t=10 on bf16 blocks, but under
the quantized weights it degraded the ASR round-trip measurably (−0.18 mean
over 6 paired seeds) — t=8 is the knee that survives both knobs together
(found in review on #98 when the gate test was fixed to actually run).
Quiet-M4 steady RTF: 2.0 baseline (bf16, t=10, library eager) → **~0.8**
(default) — 2.5×, with the HNR floors and the ASR round-trip battery passing
(`tests/asr_oracle.py`, oracle built by `scripts/build_asr_oracle.py`).

## One-shot, no server

```sh
vllm-omni-mlx tts --voice ryan --text "Hello from the CLI." --out out.wav
```

## Performance & verification

Per-checkpoint numbers, one quiet session (M4 base · 16 GB, 2026-10-08,
direct serving path without the HTTP layer; texts/seeds/interval identical
to the bench scripts — provenance in
[#2](https://github.com/ThinkFlowLab/vllm-omni-mlx/issues/2)):

| Checkpoint | Path | First audio (p50) | Sustained RTF (p50) | Peak |
| --- | --- | --- | --- | --- |
| CustomVoice 1.7B | preset | 106 ms | 0.40 | — |
| CustomVoice 0.6B | preset | 81 ms | 0.32 | 3.7 GiB |
| VoiceDesign 1.7B | described voice | 85 ms | 0.41 | 4.3 GiB |
| Base 1.7B | clone | 276 ms | 0.42 | 5.9 GiB |
| Base 0.6B | clone | 148 ms | 0.33 | 3.7 GiB |

- With a 0.5B chat model resident: speech RTF 0.47 (worst chunk-gap ratio
  0.66), first audio 104 ms, chat TTFT 40→65 ms
  ([#39](https://github.com/ThinkFlowLab/vllm-omni-mlx/issues/39)).
- Clone first audio is floored by the ICL prefill at the 4-bit compute
  floor ([#112](https://github.com/ThinkFlowLab/vllm-omni-mlx/issues/112)) —
  16 GB machines should prefer the 0.6B pair.
- Longer sustained runs still throttle the M4 base ~3×; the thermal
  envelope, cross-machine numbers (M1 Max RTF 0.286), and the public
  reproduction protocol live in
  [issue #84](https://github.com/ThinkFlowLab/vllm-omni-mlx/issues/84); the
  benchmark scripts are `scripts/bench_all_checkpoints.py`,
  `scripts/bench_prefix_cache.py`, `scripts/bench_stream_rtf.py`.
  Correctness gates: HNR floors (harmonics-to-noise, the
  catastrophic-decode detector) plus token-exactness harnesses in `tests/`.
