# Contributing

Thanks for helping build vllm-omni-mlx. The short version: PRs against
`main`, tests green (locally with weights where the path needs them), and
any performance claim carries its own before/after numbers.

## Setup

Requires Python 3.10+ on an Apple Silicon Mac (MLX ships arm64-only wheels).

```sh
git clone https://github.com/ThinkFlowLab/vllm-omni-mlx && cd vllm-omni-mlx
python -m venv .venv && source .venv/bin/activate
pip install -e '.[tts]'        # + speech (mlx-audio); '.[omni]' for vision/audio chat
```

## Running the tests

```sh
python -m unittest discover -s tests   # stdlib unittest, no extra deps
```

Two classes of tests:

- **Plain tests** run everywhere (CI runs these).
- **Weight-gated tests** (TTS, prompt cache) load locally cached Hugging Face
  checkpoints and **skip** where the weights are absent — set
  `HF_HUB_OFFLINE=1` so loads resolve from the cache without network stalls.
  A green CI run does *not* mean the weight-dependent paths were exercised;
  run them locally on a machine with the checkpoints.

The full TTS battery is heavy on a 16 GB machine — run test files one per
process rather than one big `unittest discover` when checkpoints are cached
(checkpoints stack inside a single process):

```sh
for f in tests/test_*.py; do python -m unittest "${f%.py}".replace('/', '.') ; done   # illustrative
python -m unittest tests.test_stream_loop            # one file at a time in practice
```

## Performance changes

- Every perf-relevant PR carries a **before/after A/B measured on both
  sides of the change, same conditions, with the repro command** — fresh
  baselines, never numbers quoted from old comments.
- Measure on a cool, idle machine (GPU clock ramp and thermal throttling
  swing Apple Silicon numbers ~3×; interleave A/B runs with cool-down
  sleeps and alternate order). Methodology:
  [docs/profiling.md](docs/profiling.md) — time the `mx.eval` bracket, mind
  the clock ramp, Metal GPU capture for per-kernel truth.
- Public reproduction protocol for the headline streaming numbers:
  [issue #84](https://github.com/ThinkFlowLab/vllm-omni-mlx/issues/84);
  bench scripts live in [`scripts/`](scripts/) (`bench_all_checkpoints.py`,
  `bench_prefix_cache.py`, `bench_stream_rtf.py`, `profile_*`).

## Correctness bars for speech

- HNR (harmonics-to-noise) floors are the catastrophic-decode detector —
  new audio paths need them.
- Token-exactness harnesses compare our vendored loops against mlx-audio's
  own generation; where bit-parity is impossible (kernel batching under
  compile), the test carries the documented tolerance and the claim is
  envelope-level, not bitwise.
- We vendor control flow over mlx-audio (MIT) internals rather than
  reimplementing model math; the pin is `mlx-audio>=0.5.7,<0.6` — bumping
  past it means re-verifying every vendored function against the new
  source (an adaptation PR, not a version bump).

## Contributor self-review

Before requesting review, inspect the complete diff against the target branch,
check scope and correctness, and record commands, results, and checks not run.
Keep documentation and PR claims aligned with implemented behavior. Contributor
self-review helps maintainers; it does not replace maintainer review.

Agent-assisted review entry points live in
[`.agents/skills/`](.agents/skills/):
[precheck-pr](.agents/skills/precheck-pr/SKILL.md) for a readiness pass,
[self-review](.agents/skills/self-review/SKILL.md) for a full contributor review,
and [vllm-omni-mlx-review](.agents/skills/vllm-omni-mlx-review/SKILL.md) for
reviewing a PR. All three paths apply the artifact and size rules below.

### Committed artifact hygiene

Review artifacts by their maintained purpose and actual consumer, not by file
extension. Keep necessary configuration, request examples, benchmark inputs,
and small deterministic fixtures or reference oracles. Identify the test, tool,
or maintained workflow that consumes each retained artifact.

One-off run summaries, response dumps, cache statistics, profiler output, agent
process notes, and duplicate historical results belong in durable artifact
storage or PR/CI evidence, not the source tree. A documentation or PR link alone
does not justify committing generated output. Preserve raw measurements,
failures, and provenance with links to the exact revision or run; do not discard
evidence to shrink a diff or hide it in a committed archive.

Before removing redundant output, check callers, links, and reproduction
commands. Keep replay inputs and expected responses intact, verify their hashes,
and rerun affected replay or documentation checks. See the
[self-review checklist](.agents/skills/self-review/SKILL.md#committed-artifact-hygiene).

### Large code changes

PRs with **more than 3,000 changed lines of authored code** need extra contributor
attention before requesting review. Count additions plus deletions against the
PR's merge base in source files, tests, and build or validation scripts. Report
this count separately from the total diff size; exclude documentation, generated
output, lockfiles, and static fixtures from the code count, while still reviewing
those files for relevance and correctness.

- Complete a full self-review of every affected component and its integration
  boundaries. A quick precheck alone is insufficient; keep the PR in draft until
  the contributor self-review is complete.
- Consider splitting independent features, refactors, and cleanup into focused
  PRs. If the change needs to stay together, explain why in the PR description
  and provide a component map and suggested review order.
- Include the code-line count and a validation summary for each affected area in
  the PR description: commands, results, and unverified behavior with reasons.
  Cover changed interfaces between components as well as individual components.

Size signals the need for closer review; it is not itself a correctness finding.
Choose checks based on the changed behavior and risk. Crossing this threshold
alone does not require GPU benchmarks or other expensive experiments.

## Repository layout

| Path | Responsibility |
| --- | --- |
| `schemas.py` | OpenAI/Anthropic → one internal request |
| `backends.py` | mlx-lm text backend, mlx-vlm omni backend, stop filtering, cross-turn prompt cache |
| `server.py` | routes, SSE + chunked streaming |
| `__main__.py` | CLI (`serve`, `tts`) |
| `tts/` | Qwen3-TTS pipeline: `config`/`variants` (checkpoint typing), `service` (validation + lock), `generate` (buffered) + `stream_loop` (streaming fast path), `prompt_embeds` + `prefix_cache` (per-voice prompt state), `compiled_steps` (mx.compile'd decode closures), `code2wav`/`talker`/`code_predictor` seams |
| `docs/` | [architecture](docs/architecture.md) · [speech guide](docs/speech.md) · [profiling](docs/profiling.md) |

Architecture rationale (why no scheduler, why prefix caching has none of
upstream's paged-KV conflicts): [docs/architecture.md](docs/architecture.md).
