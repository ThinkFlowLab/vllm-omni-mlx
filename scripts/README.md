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
