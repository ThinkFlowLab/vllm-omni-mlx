---
name: precheck-pr
description: Self-check a vllm-omni-mlx branch before opening or updating a PR — CI gates, weight-gated test taxonomy, the A/B + probe-gated perf rule, audio-quality gates (HNR floors, ASR round-trip, paired-seed equivalence), parity discipline for loop/compile work, and parallel-session safety. Use when the user says "precheck", "must check", "self review", or "check my PR". Never posts to GitHub.
---

# PR Pre-Check (vllm-omni-mlx)

Author-side must-check before a PR is declared ready. The report stays in the
terminal; findings are fixed on the branch, not posted. Run it on every PR —
including "just tests" and "just docs" PRs; three of the five findings in
#102's pass were in a test-only diff, one of them introduced by the pass
itself and caught only by the re-run gate (step 6).

## Step 1: Freeze the diff

```bash
BASE_SHA=$(git merge-base HEAD origin/main)
git diff --name-only ${BASE_SHA}...HEAD
```

Everything below runs against this snapshot. If anything commits mid-check,
re-freeze and redo the affected steps.

## Step 2: Machine gates (always)

```bash
python -m ruff check --select E9,F .        # exactly CI's subset
python -m unittest discover -s tests        # CI-safe suites must pass;
                                            # weight-gated ones skip cleanly
python -m py_compile <changed .py files>    # cheap, catches what ruff misses
```

A green CI is NOT completion evidence for anything weight-gated — CI has no
weights. "Local battery green" claims must name which battery ran.

## Step 3: Test-taxonomy audit (per changed/added test)

- **CI-safe** (no checkpoint): must run in discover without weights, and must
  not import `mlx_audio` at module level unless gated by
  `requires_mlx_audio` (the #56 rule).
- **Weight-gated** (loads a checkpoint): `os.environ.setdefault("HF_HUB_OFFLINE", "1")`
  at the top; skip via `local_snapshot(MODEL) is None`; inherit
  `ReleaseAfterClass`; one heavy model per process — a multi-checkpoint
  battery goes through per-model subprocesses (`scripts/acc_all_checkpoints.py`
  is the pattern).
- New floors or envelopes: the calibration evidence belongs in the PR body or
  a comment — a number without its measurement is a review finding.

## Step 4: Perf-claim audit (any PR touching generation paths)

- **A/B rule**: before→after numbers measured fresh, same conditions,
  reproducible command in the PR. Never quote a number from an old comment.
- **Probe gate**: bracket every measurement with the matmul probe
  (`mlx-perf` recipe; ~8 ms/iter = quiet on this M4). Probe > 15 ms ⇒ defer
  or disclose. Record the probe value WITH the number.
- **Thermal order-bias**: in-process A/Bs that run the faster variant second
  lose 20–40% to heat. Measure the fast variant first, or standalone.
- Stochastic models: report draw counts and seeds; RTF varies with stop-point
  randomness (VoxCPM2 ±0.1) — medians, not single draws.

## Step 5: Audio-quality audit (any PR that can change output audio)

Run the fleet battery and cite it:

```bash
python scripts/acc_all_checkpoints.py            # all checkpoints, exit 1 on breach
python scripts/acc_all_checkpoints.py --model voxcpm2   # one checkpoint
```

- HNR floors = catastrophic-decode tripwires; per-family test files hold the
  precise calibrations.
- ASR round-trip (needs the oracle: `scripts/build_asr_oracle.py`) — zh is
  report-only at whisper-base quality; do not "fix" a zh failure by tweaking
  audio without paired-seed evidence.
- **Equivalence claims** (same quality, faster): paired seeds — same noise
  draws, one knob different — paired median ≥ −0.02 per text, plus HNR.
  Absolute similarity floors don't work (systematic per-sentence ASR errors).
- Quality-affecting defaults need audition wavs in /tmp, paths in the PR, and
  the verdict recorded before merge.

## Step 6: Re-run gate

After ANY fix made during this pass, re-run the thing the fix touched — full
battery for battery changes, the specific suite for test changes. A diff
that "looks right" is not evidence; #102 shipped a print-dedent regression
mid-pass that only the re-run caught.

## Step 7: Parallel-session safety

Other sessions work this repo concurrently. Check `git worktree list` before
relying on a worktree; commit early (worktrees have been wiped mid-session);
run benches only after probing for contention; claim issues by comment.
PYTHONPATH-shadow the worktree when using the shared .venv.

## Step 8: Docs and PR hygiene

- Conventional-commit title (`feat(tts): …`, `perf(tts): …`, `test(tts): …`);
  body carries the numbers and the gates that ran.
- README stays consolidated (status-led news, compact matrix, launch-only
  API); speech detail goes in `docs/speech.md`.
- Overlap check: if another open PR touches your new files with identical
  content, say which PR is canonical — identical files merge clean but the
  reviewer should know.

## Report format

One line per finding: `severity  file:line  finding  fix`. Verdict at the
end: `READY` or `NOT READY (n blockers)`. Anything you wouldn't defend to a
maintainer is a blocker.
