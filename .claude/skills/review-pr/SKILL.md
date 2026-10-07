---
name: review-pr
description: Delegated-reviewer protocol for vllm-omni-mlx PRs — freeze the head SHA, verify library/API claims against the repo .venv, check the A/B and audio-quality evidence bars, post inline findings, keep verdicts local (own PRs can't approve; use --comment). Use when the user says "review PR <n>", "delegated review", or "verdict on <n>".
---

# Delegated Review (vllm-omni-mlx)

Reviewing PRs in a repo where the reviewer may also be the author, sessions
run in parallel, and most correctness evidence is weight-gated (local-only).
The precheck-pr skill is the author-side must-check; this is the reviewer-side
protocol for someone else's PR — or your own, with the verdict-posting caveat
below.

## Step 1: Freeze and census

```bash
gh pr view <n> --json headRefOmitted 2>/dev/null || true
gh pr diff <n> --name-only
HEAD_SHA=$(gh pr view <n> --json headRefOid --jq .headRefOid)
```

Everything is judged against `HEAD_SHA`. If it moves mid-review: stop posting,
mark the review stale in your notes, re-freeze before continuing.

Checkout the head for verification in a THROWAWAY worktree — never review
from a shared worktree another session may be using.

## Step 2: Verify library claims against the venv

The PR says "mlx-audio does X" — check it:

```bash
.venv/bin/python - <<'EOF'
import inspect, mlx_audio.tts.<module> as m
print(inspect.getsource(m.<function>))
EOF
```

Version reality: the pin is `mlx-audio>=0.5.7,<0.6` (deliberate — vendored
internals). Any claim about behavior beyond the pinned version is a finding.
Do the same for mlx/mlx-lm (`mlx.core` semantics, `mx.compile` constraints).

## Step 3: Evidence bars by PR type

**Perf PRs** (anything touching generation paths):
- Fresh before→after A/B with a repro command? Numbers quoted from old
  comments = finding.
- Probe value recorded with each number; thermal order-bias addressed
  (fast variant measured first or standalone)?
- Draw count and seeds for stochastic models?

**Audio-quality PRs**:
- Fleet battery (`scripts/acc_all_checkpoints.py`) cited, or the specific
  per-family battery named?
- New defaults affecting output: paired-seed equivalence evidence + audition
  wavs + verdict? Numeric-only evidence for a default change = finding.
- New floors/envelopes carry their calibration?

**Loop/compile PRs** (vendored loops, closures):
- Parity discipline: bitwise vs the library where achievable, else a drift
  envelope on the pre-divergence prefix + bounded stop-point tolerance.
- MLX traps to grep for: captured array constants under shapeless (Scatter
  fold), nested compiles, `.item()` inside traced fns, thread-binding of
  closures, KVCache objects in traces, closure-cache keys without thread id.

**Serving PRs**:
- Worker-thread coverage — main-thread tests structurally miss
  thread-binding and CPU-pinned-mmap bugs; a serve smoke or a
  worker-thread test is required for anything load-adjacent.

**Cross-cutting**: weight-gated tests follow the taxonomy (offline env,
snapshot gate, ReleaseAfterClass, one heavy model per process); CI-safe
tests don't import mlx_audio ungated.

## Step 4: Post findings inline, one at a time

Post each finding as it's proved (partial work survives context loss):

```bash
gh api repos/ThinkFlowLab/vllm-omni-mlx/pulls/<n>/comments \
  -F body="<finding>" -F commit_id="<HEAD_SHA>" \
  -F path="<file>" -F line=<line>   # -F, not -f, for integers
```

Line numbers verified against the frozen SHA. Praise sparingly, cite the
measurement, never speculate — if you can't prove it in the venv or the diff,
it's a question, not a finding.

## Step 5: Verdict — local, or comment on own PRs

`gh pr review --approve` fails when the token == the author (own PRs).
Present the verdict in the session; for own PRs post it as a `--comment`
review with the evidence list. Verdict vocabulary: approve / approve-with-
nits (list them) / request-changes (blockers must each have a finding posted).

Recommended per-PR reading of the evidence: run the PR's own battery
commands if weights are cached; otherwise verify the harness logic and mark
weight-gated claims "not executed — evidence reviewed only".
