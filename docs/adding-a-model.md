# Adding a new model

Two very different jobs hide behind "add a model":

- **Chat models** (text LLMs, vision/audio-in VLMs) are generic — `--model`
  hands the repo to `mlx-lm` / `mlx-vlm` and `backends.py` auto-selects by
  sniffing `config.json`. No per-model code; if one fails it's a bug, not a
  wiring task.
- **Speech-out models** (TTS today; ASR planned) *generate* media, so they
  need explicit plumbing: loader, generation paths, endpoints, tests — and
  they are latency-sensitive enough to deserve real optimization work.

This guide covers the second kind: the seams a speech model hangs off, the
steps to wire a new one, and the optimization playbook distilled from the
Qwen3-TTS work (#39–#43) so the next model starts where that one ended.

## The seams

| Seam | File | Owns |
| --- | --- | --- |
| Config / loader | `vllm_omni_mlx/tts/config.py` | `TTSConfig` serving defaults, `load_tts_model`, `local_snapshot` (the weight-gate for tests) |
| Dispatch | `vllm_omni_mlx/tts/variants.py` | which checkpoint type serves which generation *path* (`SERVED_BY_PATH`), request-shaped errors instead of deep generation failures |
| Inputs | `vllm_omni_mlx/tts/prompt_embeds.py` | text + speaker + language + instruct → the model's input layout |
| Pipeline stages | `talker.py` · `code_predictor.py` · `code2wav.py` | the two-stage pipeline (text → codec tokens → audio), the MTP seam, `streaming_step` |
| Generation | `generate.py` · `stream_loop.py` | buffered synthesis and the vendored streaming loop |
| Serving | `service.py` · `server.py` · `__main__.py` | batch-1 lock, request validation, `/v1/audio/*` endpoints, `tts` CLI |
| Tests | `tests/` | stub-model seam tests (run in CI) + weight-gated e2e |

The pattern that makes these seams work: **validation happens outside the
generation lock, generation inside it** (see `service.py`'s `_clone_inputs`
vs `speech_bytes`), and **the dispatch table answers "is this served?" at
the boundary** — a novel checkpoint type fails at boot with its raw value
in the message, an unsupported type × path fails as a 400 that names the
tracking issue.

## Adding a variant of a family we already serve

The cheap case — another checkpoint of a family whose pipeline is plumbed
(what #47/#53 did for Qwen3-TTS dispatch and #49/#54 for cloning):

1. **Dispatch row.** Add the type to `variants.py`: `KNOWN` if it is new,
   `SERVED_BY_PATH` for the paths you actually synthesize on, and a
   `_TRACKING` message for the ones you don't (yet). Served-ness is a
   path × type question — `base` checkpoints clone voices but have no
   presets; `custom_voice` is the reverse.
2. **Defaults.** Only touch `TTSConfig` if the variant changes serving
   defaults (temperature, frame rate, `streaming_interval`).
3. **Weight-gated test.** Gate on `local_snapshot(model_ref)`; the test
   runs where the checkpoint is cached and skips in CI — so it must not be
   the *only* test. Pair it with stub-model tests of the dispatch rows
   (`tests/test_variants.py` needs no weights).
4. **A/B numbers** before merging (below) and a README supported-models
   row.

## Adding a new family

1. **Spike adapt-vs-port first.** Try adapting the upstream MLX
   implementation (for TTS: mlx-audio) before hand-porting — loading the
   checkpoint is itself the weight-mapping validation (conv transposes,
   RVQ codebooks, quantized variants), and the spike doc
   ([spike-mlx-audio-qwen3-tts.md](spike-mlx-audio-qwen3-tts.md)) is the
   template. One early decision matters more than the rest: if the
   library's decode loop hardcodes something you need control of (for us,
   a single fixed streaming interval), plan to **vendor the loop** from
   day one rather than fight it from outside.
2. **Build bottom-up.** Config/loader → input builder → buffered
   generation → service → endpoint → CLI. Buffered first: join chunks,
   assert exact parity with the library's reference output, *then* add
   streaming. Match the OpenAI audio API shapes (`/v1/audio/speech`,
   voice list endpoint) so clients work unchanged.
3. **Streaming second, vendored.** Copy the reference decode loop
   step-for-step — same sampler call, same caches, same single
   `mx.eval` sync per frame — and drive the library's own components
   through your seams (`stream_loop.py` is the worked example). Parity is
   *token-exact draws*, not bitwise audio: the vocoder is chunk-boundary
   sensitive (identical tokens still shift the output envelope by
   ~2–3e-3 mean across chunkings, ~8× that if context handling breaks),
   so assert on tokens plus an envelope-mean tripwire (~8e-3).
4. **Tests at each layer.** Stub-model tests for every seam (these run in
   CI), weight-gated tests for mapping and e2e (these skip in CI — green
   CI never means "verified"; local runs and human audition close that
   gap).

## Optimization playbook (from Qwen3-TTS)

Ordered by leverage; each number is from the 1.7B 4-bit model on a quiet
M4 unless noted.

**1. Measure the right things before optimizing.** The metrics that
matter for a speech model here: time-to-first-audio (p50/p95), sustained
RTF as a *chunk distribution* (not a mean — it hides tail stalls), chunk
jitter p95/max once streaming, an audio-correctness gate, and peak memory
as the capacity constraint. Throughput is explicitly not a target
(batch-1 by design — see [architecture.md](architecture.md)).
Methodology in [profiling.md](profiling.md): time the `mx.eval` bracket,
prewarm the GPU clock first, `mx.get_peak_memory` for peaks; harnesses in
`scripts/` (`latency_probe.py`, `coexistence_bench.py`).

**2. TTFA is a streaming-policy problem, not only a compute problem.**
Waiting for a full steady chunk floors first audio at
`streaming_interval` (the 2 s default once meant 2 s TTFA). Fix it with
an *initial* chunk boundary smaller than the steady chunk — upstream's
`initial_codec_chunk_frames` trick — emitted as soon as that much audio
exists. This took TTFA from 345–473 ms (0.5 s interval) to 195–290 ms at
defaults and 132 ms at a 0.08 s initial interval. Total RTF is
interval-independent (~0.85–0.9 here), so you are only trading chunk
cadence for first audio.

**3. Keep the compiled-shape set closed.** Every distinct shape an
`mx.compile`'d step traces costs a seconds-long retrace mid-request.
Quantize the initial boundary to a power-of-two bucket ≤ the steady chunk
and pad-then-trim the final remainder, so a configuration traces at most
two shapes (`initial_frames_bucket` / `_pad_target` in `stream_loop.py`);
prewarm them at boot (`prewarm_streaming`). A new model should never let
request parameters widen the shape set.

**4. Compile step closures, not attribute paths.** Wrapping the whole
per-frame step in a closure (the `_predict_step_logits` pattern upstream
uses, and the remaining RTF lever for our talker) works; compiling via
attribute patching of library objects failed outright (#39.6). Inside a
vendored loop you own the step boundary — use it.

**5. Cache what is request-independent.** Per-voice prompt embeddings are
the next TTS lever (our `PromptEmbeds` builds per request; upstream keeps
a per-voice cache). On the chat side the cross-turn prompt cache bought
8.3× TTFT, which is the template: find the prefix that repeats and reuse
it.

**6. Pick quantization empirically.** 4-bit is the serving default and
held audio quality here (HNR-gated), but bf16-vs-4-bit must be A/B'd per
family, not assumed. Chat-side reference points: KV 8-bit cut memory
~47%; draft-model speculative decoding was a net loss below ~2B — don't
reach for it for small codecs.

**7. Benchmark coexistence, not solo.** The deployment reality is TTS
with a chat model resident: solo RTF 0.877 became 1.118 mixed, peak
memory 3.21 GB. A model that only hits its numbers alone isn't done —
`scripts/coexistence_bench.py` measures the mixed case.

**8. Gate correctness while you optimize.** The HNR check (autocorrelation
80–400 Hz, `tests/audio_metrics.py`) is the catastrophic-decode detector:
calibrate a floor per voice — healthy output sits several dB above it,
garbage decode falls through. Greedy decode drifts ~1 dB run-to-run from
kernel nondeterminism, so leave margin. No perf claim lands without the
gate green and a human audition of the actual WAV.

**9. A/B in every PR.** Perf-relevant changes carry before→after numbers
measured the same session, same conditions, baseline taken fresh from
`main` (never quoted from an old comment), plus the repro command. For
context on remaining headroom: upstream vllm-omni's fused single-stage
Qwen3-TTS reaches TTFP 64 ms / RTF 0.16 on an H200 — our numbers are the
Apple-Silicon budget, not the ceiling.

## Definition of done

- [ ] Dispatch table answers every type × path with serve, or a message
      naming the tracking issue
- [ ] Stub-model seam tests run in CI; weight-gated tests run (and pass)
      on a machine with the checkpoint
- [ ] Streaming parity: token-exact vs the reference loop, envelope
      tripwire green
- [ ] Numbers in the PR: TTFA p50/p95, RTF distribution, peak memory —
      solo *and* with a chat model resident
- [ ] HNR gate calibrated per voice, human audition done

## Image models (`diffusion/`, #91)

The seam is adapter-shaped, not loop-shaped — no streaming, one blocking
call per request:

1. **Family registry** (`diffusion/config.py`): add a `Family` (aliases,
   canonical repo, default steps, `supports_guidance`) and route refs in
   `resolve_model()` → `ResolvedModel(family, weights_ref, quantize)`.
   Weight routing matters: pre-quantized mirror repos honor their stored
   level (`quantize=None`); canonical fp16 repos quantize on load
   (`quantize=4`). Keep `config.py` importable without the `[image]`
   extra (CI installs none).
2. **Service** (`diffusion/mflux_service.py`): `MFluxService` maps the
   request onto the mflux backend call, validates family capabilities
   (guidance on a guidance-distilled model → `ValueError`, the route
   maps it to a 400), runs load+generate on one dedicated worker thread
   (MLX stream affinity — serve smoke is mandatory, see #71), and
   returns `ImageResult` (PNG bytes + seed + timings + peak
   memory) — the metrics the doctrine reports.
3. **Tests** (`tests/diffusion/`, mirroring src): CI-safe resolution +
   endpoint tests with a fake service; one weight-gated file per model
   (per-file subprocess, offline, donor release in `tearDownClass`).
   Gates: valid PNG above a size floor, dimensions/steps honored,
   **same seed → byte-identical PNG** (characterize, don't assume, if
   fusion flips appear), family-capability 400s.
4. **Docs**: README models-table row + the `[image]` footprint row.
