---
name: vllm-omni-mlx-review
description: Review vllm-omni-mlx pull requests for correctness, evidence, artifact hygiene, and contributor readiness.
---

# vllm-omni-mlx review

Read [CONTRIBUTING.md](../../../CONTRIBUTING.md), the complete merge-base
diff, and changed files with their callers and tests. Record head/base commit
IDs and whether the remote base was refreshed. Use current code, CI, and the
[architecture guide](../../../docs/architecture.md) as the repository baseline.

This skill produces review findings; it does not authorize edits, commits,
pushes, external posts, review-status changes, or paid execution.

## Review contracts

- Apply [large code changes](../../../CONTRIBUTING.md#large-code-changes).
  Report authored-code and total diff counts separately. Above 3,000 changed
  code lines, verify the contributor's full self-review, split
  decision/rationale, component map/review order, and validation across affected
  components and interfaces. Keep the PR draft until contributor self-review is
  complete. Missing preparation is a readiness gap; size alone is not a
  correctness finding or a requirement for GPU benchmarks.
- Apply [committed artifact hygiene](../self-review/SKILL.md#committed-artifact-hygiene)
  in every review, including quick reviews, to the complete changed-file
  inventory, including generated data. Distinguish
  maintained fixtures/configuration from one-off run output. Report redundant
  artifacts with concrete paths and consumers; preserve reviewer-accessible raw
  evidence and verify callers, replay inputs/expected responses, hashes, links,
  and reproduction commands before recommending cleanup.
- Check changed behavior against actual API, backend, and speech contracts.
  Verify the current batch-1/serialized-generation behavior against the target
  revision and assess any intentional changes to it; do not import
  system1-omni's Rust/CUDA assumptions.
- For speech/performance changes, apply the existing correctness and A/B bars in
  the contributor guide. Verify weight-gated coverage instead of treating skips
  as passes, and assess measured claims using the documented profiling method.
- Choose validation based on changed behavior and risk. Distinguish checks run
  personally from author reports and CI; disclose missing Apple Silicon,
  checkpoints, or other prerequisites and unsupported claims.

## Output

Lead with actionable findings, citing paths/lines and concrete impact.
Separately list readiness gaps, code/total counts, validated scope, unrun
checks, and evidence links. If there are no findings, say so without implying
that untested behavior passed.
