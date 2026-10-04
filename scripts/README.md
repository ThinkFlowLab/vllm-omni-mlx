# scripts

Operational and development scripts.

- `run_tests.sh`: run the suite **one file per process** — each file's
  checkpoints and Metal buffers return to the OS when its process exits,
  instead of stacking every weight-gated class's model in one `pytest tests/`
  process (which can swamp a 16 GB Mac). The heavy Base+donor clone batteries
  (`BaseCloneE2ETest`, `ICLStreamE2ETest`, ~8 GB peak) are skipped by default;
  pass `--heavy` (or set `VOMX_HEAVY_TESTS=1`) to include them:

  ```sh
  scripts/run_tests.sh                 # light suite, per-file isolation
  scripts/run_tests.sh --heavy         # include the heavy batteries
  scripts/run_tests.sh tests/test_tts_generate.py   # specific files
  ```

  `PYTHON` overrides the interpreter (default `.venv/bin/python`); pytest is
  used when importable, else unittest; `HF_HUB_OFFLINE` defaults to 1.

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
