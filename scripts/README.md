# scripts

Operational and development scripts.

- `latency_probe.py` (#6): TTFT / inter-token latency probe against a running
  server — cold vs cached prefix, per model — so "extreme low latency" has
  numbers. Stdlib only:

  ```sh
  python scripts/latency_probe.py --url http://127.0.0.1:8000 \
      --system-words 500 --turns 3 --max-tokens 128
  ```

  Turn 1 measures the cold prefix; later turns extend the same conversation and
  measure the server's cached-prefix path. With a key: `--api-key` or
  `VLLM_OMNI_MLX_KEY`.

- `spike_mlxaudio_qwen3tts.py` (#9): one-shot harness for the M1.0 spike —
  loads Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit through mlx-audio, runs
  streaming + full-utterance generation per voice/language case, and reports
  load time, TTFA, inter-chunk latency, RTF, and peak GPU memory. WAVs go to
  a temp dir. Needs the `[tts]` extra (mlx-audio); the first run pays
  one-time `mx.compile` cost:

  ```sh
  python scripts/spike_mlxaudio_qwen3tts.py --streaming-interval 0.5
  ```

- `profile_stream_loop.py` (#65): per-frame phase split of the vendored
  stream loop — talker forward / predictor forwards / sampling / vocoder
  decode / Python residue, plus decile-binned frame wall across the
  generation (sustained-RTF drift). Eval brackets add sync overhead, so
  read the phase *shares*, not absolute RTF. Weight-gated (local snapshot):

  ```sh
  python scripts/profile_stream_loop.py --max-tokens 800
  ```

- `bench_stream_rtf.py` (#65): sustained-RTF A/B bench for the streaming
  speech path — prewarm + clock ramp, then N full generations of a fixed
  long text at serving defaults; reports per-turn wall/audio/RTF, p50/min/
  max, first-chunk latency, peak memory. Runs unchanged across branches;
  `VLLM_OMNI_TTS_EAGER_STREAM=1` forces the uncompiled loop on the
  compiled branch (isolation ablation):

  ```sh
  python scripts/bench_stream_rtf.py --label run --turns 4
  ```

- `profile_prefix_cost.py` (#66): per-request TTFA cost split — prompt
  build (tokenizer / text projection / voice-static pieces / assembly) vs
  the multi-row prefill forward vs the single-row splice decode, plus the
  splice-vs-fresh logits delta (the kernel-batching numerics that motivate
  #66's draw-level, not bitwise, parity claim). Weight-gated:

  ```sh
  python scripts/profile_prefix_cost.py --iters 5
  ```

- `bench_prefix_cache.py` (#66): interleaved cool-machine A/B for the
  voice-prefix cache — `VLLM_OMNI_TTS_PREFIX_CACHE` off vs warm-cache hits,
  alternating order per pair with cool-down sleeps, reporting first-chunk
  p50/min and RTF per side plus one cold-cache miss turn. Weight-gated:

  ```sh
  python scripts/bench_prefix_cache.py --pairs 4 --cooldown 8
  ```

- `profile_compiled_loop.py` (#77): phase split of the COMPILED stream
  loop — eval-bracketed wrappers around the compiled closures (talker
  decode / predictor frame / sampler / input prep) plus loop residue
  (vocoder), and the serving weight dtypes. The #65 profiler measures
  the eager loop; this one measures what serving runs. Weight-gated:

  ```sh
  python scripts/profile_compiled_loop.py --max-tokens 400
  ```

- `asr_roundtrip.py` (#68): TTS → ASR round-trip gate. Synthesizes fixed
  sentences with the serving TTS checkpoint, transcribes them with the serving
  ASR checkpoint, and fails if mean word-error-rate exceeds `--max-wer`
  (default 0.15). No external fixtures; catches catastrophic breakage on
  either side, not fine WER differences. Both models stay resident (~5.4 GiB
  peak on the 4-bit defaults), so run it on its own:

  ```sh
  python scripts/asr_roundtrip.py --voices vivian ryan
  ```
