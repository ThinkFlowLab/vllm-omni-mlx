# Profiling on Apple Silicon

There is no `ncu` on macOS. GPU truth comes in three tiers — escalate only when
the previous one can't answer the question. All verified on mlx 0.32 / arm64.

| Tier | Answers | Cost |
| --- | --- | --- |
| 1. Timing + memory harness | where does time go, what's the footprint, did a change help | none — a script |
| 2. Metal GPU capture | which kernel, per-kernel time, gaps in the stream | Xcode to read |
| 3. SoC counters | throttling, frequency, bandwidth ceilings | sudo / Xcode |

## Tier 1 — time the eval bracket

MLX is lazy: graph-building calls only queue work; `mx.eval()` executes it.
Time the eval bracket, never the construction.

```python
import time
import mlx.core as mx

def step():
    out = backend.generate_one_token(...)  # build graph (cheap to "call")
    mx.eval(out)                           # execute — the thing you time

# warmup: shader compile + buffer caches, then ~1s sustained load (see below)
for _ in range(10):
    step()
t0 = time.perf_counter()
while time.perf_counter() - t0 < 1.0:
    step()

mx.reset_peak_memory()
times = []
for _ in range(30):
    t = time.perf_counter()
    step()
    times.append((time.perf_counter() - t) * 1000)
```

Memory accounting: `mx.get_active_memory()` (live arrays), `mx.get_cache_memory()`
(reusable buffers — growing is not a leak), `mx.get_peak_memory()` after
`mx.reset_peak_memory()`. The `mx.metal.get_*` spellings are deprecated.

**Clock ramp.** Apple GPUs raise clocks only under sustained load. The same
1024³ matmul measured 4.2 ms on an idle machine and 1.1 ms warm — a ~4x gap
that no count-based warmup closes for millisecond-scale kernels. Pre-warm with
~1 s of sustained work before timing, and treat per-iteration times that trend
downward as ramp, not as the workload.

Reading the numbers: report p50 and p90, not the mean alone (windowserver and
other GPU clients cause spikes). For generation, measure one decode step at
fixed context and end-to-end generation separately — TTFT and inter-token
latency have different bottlenecks. A/B comparisons must pre-warm identically.

## Tier 2 — Metal GPU capture

Per-kernel truth. `mx.metal.start_capture()` fails with
`Capture layer is not inserted` unless the process is **launched** with
`METAL_CAPTURE_ENABLED=1` — it cannot be set after start.

```python
mx.metal.start_capture("/tmp/step.gputrace")
step()                      # exactly one hot iteration; traces grow fast
mx.metal.stop_capture()
```

```sh
METAL_CAPTURE_ENABLED=1 .venv/bin/python bench.py
open /tmp/step.gputrace     # opens Xcode's capture view
```

Sort the kernel list by duration. MLX names its kernels — `gemm`/matmul
variants, `rms_norm`, `rope`, `quantized_*`, attention, copies. Big gaps
between kernels with an empty encoder mean the GPU is waiting for the CPU to
encode work (Python overhead), not that a kernel is slow. Captured runs are
for attribution only — the capture layer adds overhead; never report timings
from them.

## Tier 3 — SoC counters

```sh
sudo powermetrics --samplers gpu_power -i 500 -n 30   # alongside the bench
xcrun xctrace record --template 'Game Performance' --attach <pid> --time-limit 10s
```

`GPU HW active residency` near 100% with slow wall-clock → the kernel or
memory bandwidth is the wall. Residency low → gaps (CPU-bound encoding, sync
stalls). GPU frequency sagging across a long run → thermal throttling.
powermetrics output is plain text and script-parseable; xctrace gives real
hardware counters via the Game Performance template but is Xcode-bound.
