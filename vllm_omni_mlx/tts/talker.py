"""Talker backbone seam (M1.4, #13) — adapting mlx-audio (MIT).

mlx-audio's ``Qwen3TTSTalkerForConditionalGeneration`` implements the
dual-track backbone: text track (``text_embedding`` + ``text_projection``
ResizeMLP, 2048d, Qwen vocab) and codec track (``codec_embedding``, 1024d,
3072-entry codec vocab), channel-summed into the shared Qwen3-style decoder
with MRoPE. Codec-logit masking takes the form of a suppress list built from
the checkpoint config — special ids ``[vocab_size-1024, vocab_size)`` minus
``codec_eos_token_id`` — the adapt-path equivalent of the reference's
``[1, codec_vocab) ∪ {eos}`` mask. ids come from config, never hardcoded.

Parity vs transformers ``Qwen3Model`` on random embeddings is deferred: the
local checkpoint is 4-bit (dequantization error dwarfs a 1e-4 tolerance) and
the bf16 snapshot does not fit this machine's disk; recorded for a rerun
where bf16 weights are available.
"""

from __future__ import annotations

from typing import Any, List

import mlx.core as mx


class Talker:
    """Wrap a loaded mlx-audio talker (``model.talker``)."""

    def __init__(self, talker: Any):
        self._talker = talker
        self._config = talker.config

    @property
    def text_hidden_size(self) -> int:
        return self._config.text_hidden_size

    @property
    def text_vocab_size(self) -> int:
        return self._config.text_vocab_size

    @property
    def codec_hidden_size(self) -> int:
        return self._config.hidden_size

    @property
    def codec_vocab_size(self) -> int:
        return self._config.vocab_size

    @property
    def codec_eos_token_id(self) -> int:
        return self._config.codec_eos_token_id

    def codec_embeds(self, code_tokens: List[mx.array]) -> mx.array:
        """Channel-summed codec-track embeddings: group-0 through the codec
        embedding, each residual group through its predictor embedding."""
        embed = self._talker.get_input_embeddings()(code_tokens[0])
        predictor_embeddings = self._talker.code_predictor.codec_embedding
        for index, code in enumerate(code_tokens[1:]):
            embed = embed + predictor_embeddings[index](code)
        return embed

    def suppressed_codec_ids(self) -> List[int]:
        """Special codec ids that must never be sampled, except EOS — built
        from checkpoint config (mlx-audio's generation-loop mask)."""
        return [i for i in range(self.codec_vocab_size - 1024, self.codec_vocab_size) if i != self.codec_eos_token_id]
