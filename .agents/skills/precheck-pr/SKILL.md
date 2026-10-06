---
name: precheck-pr
description: Check vllm-omni-mlx PR readiness, scope, artifact hygiene, and validation before requesting review.
---

# PR precheck

Read [CONTRIBUTING.md](../../../CONTRIBUTING.md) and current repository
configuration. This skill produces a readiness report; it does not authorize
edits, commits, pushes, external posts, or review-status changes.

1. Identify the target branch, head and base commit IDs. Review the complete
   merge-base diff and relevant staged, unstaged, and untracked changes; state
   what is not yet in the PR and whether the remote base could be refreshed.
2. Count additions plus deletions in authored code separately from total diff
   size using [large code changes](../../../CONTRIBUTING.md#large-code-changes).
   Above 3,000 changed code lines, require full contributor self-review, a split
   decision/rationale, a component map and review order, and component/interface
   validation. A quick precheck cannot establish readiness; keep the PR draft
   until contributor self-review is complete. Size alone is not a defect or a
   requirement for GPU benchmarks.
3. Read changed files, surrounding code, and callers. Check relevance to the
   stated problem and flag unused code, duplication, or unrelated cleanup.
4. Apply [committed artifact hygiene](../self-review/SKILL.md#committed-artifact-hygiene)
   in every precheck, including quick checks. Account for retained fixtures and
   redundant run output; preserve raw evidence and verify cleanup does not break
   consumers, replay commands, or documentation links.
5. Select checks for changed behavior using the contributor guide and current
   CI configuration. Distinguish personally run checks, author-reported results,
   CI results, failures, skips, and checks not run. A green CI result is not
   evidence that weight-gated paths ran.
6. Report concrete correctness findings separately from readiness gaps. Include
   code/total counts, tested scope, remaining validation, and evidence links.
