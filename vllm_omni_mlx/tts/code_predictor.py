"""MTP code predictor seam (M1.3, #12) — adapting mlx-audio (MIT).

mlx-audio's ``CodePredictorModel`` (inside ``talker.py``) implements the MTP
predictor: per-code-group embeddings and LM heads (Q−1 residual codebooks),
Qwen3-style blocks with KV cache, driven one group per step by the generation
loop with a ``generation_step`` position offset. This module is our stable
seam over it for tests and the M1.8 mx.compile work; generation (#15) drives
it through mlx-audio's loop.

Audit vs the reference numerics policy (divergences recorded, none blocking
under the 4-bit adapt path):

- per-group embeddings + heads: present (``num_code_groups − 1`` each) ✓
- per-group temperature / top-k / top-p: **shared across groups** in
  mlx-audio's ``_sample_token_batch`` — the reference samples each codebook
  group with its own params. Not exposed here; revisit if audition (#15
  listen test) flags it.
- fp32 RMSNorm variance / fp32 RoPE cos-sin: mlx-audio uses stock
  ``nn.RMSNorm`` whose accumulate dtype follows the weight dtype — with the
  4-bit checkpoint this differs from the HF-exact policy either way. Revisit
  under #17 when hot paths are compiled.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


class CodePredictor:
    """Wrap a loaded mlx-audio talker's code predictor."""

    def __init__(self, talker: Any):
        self._predictor = talker.code_predictor

    @property
    def num_code_groups(self) -> int:
        return self._predictor.model.config.num_code_groups

    @property
    def num_layers(self) -> int:
        return len(self._predictor.model.layers)

    @property
    def residual_embeddings(self) -> list:
        return self._predictor.codec_embedding

    @property
    def residual_heads(self) -> list:
        return self._predictor.lm_head

    def make_cache(self) -> list:
        return self._predictor.make_cache()

    def step(self, embeds: mx.array, cache: list, generation_step: int) -> mx.array:
        """One predictor forward: residual-group embeddings in, next-group
        logits out, KV cache advanced in place."""
        logits, _, _ = self._predictor(embeds, cache=cache, generation_step=generation_step)
        return logits
