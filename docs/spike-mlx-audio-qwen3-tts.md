# M1.0 spike: mlx-audio Qwen3-TTS — adapt vs hand-port

Closes #9 (evidence for the go/no-go; the same verdict was first posted on
the closed research record #1 and is restated on the roadmap #2).

**Question.** Qwen3-TTS-12Hz-1.7B-CustomVoice is already implemented in
[mlx-audio](https://github.com/Blaizzy/mlx-audio) (MIT). Do we adapt/wrap it,
or hand-port ~2.8–3.2k lines of load-bearing torch math from the vllm-omni
reference (`~/code/vllm-omni`, `vllm_omni/model_executor/models/qwen3_tts/`)?

**Verdict: ADAPT mlx-audio.** Functional coverage is complete on every spec
point, the license is clean, it already sits in our `[omni]` dependency tree,
and the measured gap to the RTF ≤ 0.3 target is decode-side performance —
exactly what M1.8 targets — not a porting problem. Hand-porting would spend
the milestone budget to land at the same RTF with zero differentiation; our
value-add is the streaming, voice-managed server around the model.

## Coverage audit (mlx-audio 0.5.7 vs torch reference)

| Spec point | Status | Evidence (mlx-audio 0.5.7) | Reference counterpart |
|---|---|---|---|
| CustomVoice dual-track prompt | ✓ | text track (2048-d text table + resize MLP) summed position-wise with codec track; `spk_id[speaker]` → single backbone-embedding row as the speaker slot; think/no-think control prefix; non-streaming prefill default | `prompt_embeds_builder.py:884-1412` (branch :1309) |
| MTP code predictor | ✓ | 5-layer Qwen3-style transformer, hidden 1024, GQA 16/8; 15 per-codebook embeddings + 15 lm_heads; conditions on last talker hidden + sampled layer-0 code; KV reset per frame | `common/qwen3_code_predictor.py` (re-prefills, no KV — same math, different strategy) |
| 12 Hz tokenizer decode | ✓ | RVQ (1 semantic + 15 acoustic) → causal conv → 8-layer transformer (sliding window 72) → ConvNeXt/SnakeBeta decoder, total upsample 1920 → 24 kHz; weights ship in the same snapshot under `speech_tokenizer/`; codebooks rebuilt from `cluster_usage`/`embedding_sum` | `tokenizer_12hz/` + `chunked_decode` (300/25) |
| Chunked / streaming decode | ✓ | generator API; incremental `streaming_step` keeps causal-conv state + transformer KV; `streaming_interval` (s) controls chunk size; non-streaming `chunked_decode` 300/25 | connector-level 25-frame chunks, 1-frame initial chunk for TTFA |
| Sampling defaults | ✓ | temperature 0.9, top_k 50, top_p 1.0, repetition_penalty 1.05, EOS 2150, max_tokens 4096 | identical (`deploy/qwen3_tts.yaml`) |

Extras beyond our spec: ICL voice cloning on Base variants (ECAPA x-vector +
ref-audio codec conditioning), a continuous-batching session API, 4-bit
quantization with embeddings/vocoder kept full-precision, `mx.compile` on the
decoder/rope/swiglu hot paths.

**Known gaps vs the official service** (acceptable for M1, track as
follow-ups): no watermarking; `speed` accepted but unimplemented; `pitch` is
batch-gating only; streamed ICL chunks skip ref-code acoustic context;
batch ICL limited to one shared reference.

**Parity-test nuance found:** the torch reference masks talker logits to
ids `[1, 2048) ∪ {EOS}` (code 0 excluded); mlx-audio instead suppresses
special ids `[vocab−1024, vocab)` per step. A parity test should pin whether
code 0 can be sampled and what it decodes to.

## License

- mlx-audio 0.5.7: `License-Expression: MIT` (package dist-info) — adapt path
  needs only license + notice attribution. No fastapi/pydantic/server imports
  inside `tts/models/qwen3_tts/`; the module is cleanly vendorable (~6.2k
  lines + ~1.8k shared) if we ever need the escape hatch.
- Model: apache-2.0 per the cached model card; no extra CustomVoice
  restriction text found in the card, but re-check the online card before any
  product surface uses preset voices.

## Measurements (independent re-run, harness in `scripts/`)

Machine: M4 16 GB, mlx 0.32.3, mlx-audio 0.5.7, python 3.13.5. Model:
`mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit` (2.15 GiB cached; bf16
original does not fit this disk — bf16-vs-4-bit stays an open empirical
question deferred until disk allows). Load 8.6 s. GPU clocks warmed by one
short generation; the warmup call itself absorbed the first-call
`mx.compile` traces (minutes of one-time compile — see #1's 6.2 s warmup TTFA
note; per-shape retraces during the first real cases are the likely cause of
the slow first-case tail below).

`scripts/spike_mlxaudio_qwen3tts.py`, streaming_interval 0.5 s:

| case | TTFA s | RTF (stream) | RTF (full) | audio s | peak GB |
|---|---|---|---|---|---|
| en/Vivian, 27 words | 0.92 | 4.60 | 4.86 | 25.9 / 14.8 | 3.1 / 6.8 |
| zh/Dylan, 25 chars | 52.7 ⚠ | 8.94 | 6.35 | 7.0 / 7.3 | 3.0 / 5.6 |
| en/Ryan + instruct | 9.4 | 2.72 | 1.36 | 6.4 / 4.1 | 3.0 / 4.7 |

(RTF = wall / audio duration; stream/full columns, audio s stream / full,
peak = peak wired GPU memory incl. MLX cache.) All six WAVs carry
speech-level energy (RMS 1.1k–3.5k, no clipping); en + zh + instruct paths
all produce audio.

Cross-run comparison with the #1 comment (interval 2.0 s, quieter machine
state): TTFA 2.7 s, RTF 2.06, peak 3.83 GB. The variance between runs and
cases (RTF 1.4–8.9) is larger than either number alone suggests; known
contributors here: 4× finer streaming interval (more incremental decode
syncs), first-shape compile retraces mid-run, MLX cache growth in full-utterance
mode (peaks 4.7–6.8 GB), heavy memory pressure while only 2.2 GiB disk
remained free, and the unexplained zh TTFA outlier (first chunk held ~53 s,
subsequent chunks 0.66 s p50 — worth one look when the streaming wrapper
lands, not a verdict-changer). None of these are porting questions; they are
M1.8 material.

## Approach for M1.2–M1.5

Wrap, don't reimplement: `vllm_omni_mlx/tts/` modules delegate to pinned
mlx-audio components behind our own interfaces; parity tests against the
torch reference where it is runnable, golden-output tests where it is not
(the logit-mask nuance above is the first parity-test candidate). mlx-audio
is declared directly as the `[tts]` extra (#27, `mlx-audio>=0.5.7`), no
longer only transitive via `[omni]`.
Vendoring the MIT module in-tree stays open as an escape hatch if its decode
loop becomes the ceiling after M1.8's `mx.compile` pass.

## Reproducing

```sh
python scripts/spike_mlxaudio_qwen3tts.py            # default cases, /tmp out
python scripts/spike_mlxaudio_qwen3tts.py --streaming-interval 2.0
```

WAV artifacts + `report.json` land in a temp dir (listening artifacts, not
repo content). Requires the `[tts]` extra (mlx-audio, #27); first run pays
one-time compile.
