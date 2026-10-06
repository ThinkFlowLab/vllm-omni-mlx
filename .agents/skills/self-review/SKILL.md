---
name: self-review
description: Perform a full contributor self-review of vllm-omni-mlx changes, including artifacts, integration boundaries, and validation evidence.
---

# Contributor self-review

Read [CONTRIBUTING.md](../../../CONTRIBUTING.md). Review the complete diff
against the target branch for correctness, focused scope, tests, and accurate
documentation and PR claims. Record head/base commit IDs and use their merge
base; include relevant uncommitted changes and disclose any unrefreshed base.

Apply [large code changes](../../../CONTRIBUTING.md#large-code-changes):
report authored-code additions plus deletions separately from total diff size.
Above 3,000 changed code lines, require a full self-review, split
decision/rationale, component map and review order, and validation for each
affected component and interface. Keep the PR draft until contributor
self-review is complete. A quick precheck is insufficient; size alone is not a
correctness finding or a requirement for GPU benchmarks.

This skill prepares a contributor report. It does not authorize edits, commits,
pushes, external posts, review-status changes, or paid execution.

## Review and validation

- Read changed files and their callers, tests, and affected integration
  boundaries. Check API/streaming contracts and model-specific behavior when
  touched; use current code and the [architecture guide](../../../docs/architecture.md)
  rather than assuming another repository's runtime design applies.
- Select checks from current CI and the contributor guide based on changed
  behavior. Preserve speech correctness bars and fresh before/after A/B evidence
  for performance claims. Do not silently weaken tolerances.
- Distinguish personally run checks, author-reported evidence, and CI results.
  Record commands, results, skips, and unverified behavior with reasons.
  Weight-gated skips do not validate inference; use Apple Silicon and cached
  checkpoints where needed, with heavy test files isolated as documented.
- Validate documentation links and reproduction commands affected by the diff.
  Documentation-only changes do not automatically need model runs.

## Committed artifact hygiene

Apply this check in every review, including quick reviews.
Inspect added and changed artifacts in the complete diff, including JSON/JSONL,
CSV, logs, reports, source/binary hash inventories, and generated media. Classify
them by purpose and actual consumer, rather than rejecting a file extension.

- Keep necessary configuration, request examples, maintained benchmark inputs,
  and small deterministic fixtures or reference oracles in intended locations.
  Identify the test, tool, or documented workflow that needs each retained item.
- Flag one-off run summaries, response dumps, cache statistics, profiler output,
  agent process notes, and duplicate historical results with no maintained
  source-tree role. A link from PR prose or documentation alone does not justify
  committing generated run output.
- Preserve raw measurements, failures, and provenance in a durable artifact
  archive or PR/CI evidence, linking the exact revision or run from the summary.
  Do not discard evidence to reduce the diff or hide it in a committed archive.
- When removing redundant output, check callers, links, and reproduction
  commands. Keep replay inputs and expected responses intact; verify their
  hashes and rerun affected replay or documentation checks.

## Report

Provide findings with concrete paths and impact, code and total diff counts,
the split decision and review map when required, and a validation summary for
each affected area and changed interface. Include evidence for behavior/perf
claims, retained artifact consumers, and remaining readiness gaps. Contributor
self-review does not replace maintainer review.
