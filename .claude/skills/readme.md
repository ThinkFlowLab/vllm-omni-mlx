# vllm-omni-mlx repo skills

Author-side and reviewer-side review protocols, kept in-repo so they stay
current with the bars they enforce.

| Skill | Who runs it | What it encodes |
| --- | --- | --- |
| [`precheck-pr`](precheck-pr/SKILL.md) | the PR author (must-check before ready) | CI gates, weight-gated test taxonomy, A/B + probe-gated perf rule, audio-quality gates (HNR / ASR round-trip / paired-seed equivalence), re-run gate, parallel-session safety |
| [`review-pr`](review-pr/SKILL.md) | the delegated reviewer | head-SHA freeze, verify library claims against the pinned .venv, evidence bars by PR type, inline finding mechanics, verdict vocabulary |

Both are terminal-only by default — neither posts approvals. The bars they
reference live in [CONTRIBUTING.md](../../CONTRIBUTING.md) (setup, test
semantics, the A/B rule) and the batteries under `scripts/` and `tests/`.
