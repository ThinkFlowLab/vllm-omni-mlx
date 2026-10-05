"""CustomVoice prompt-embeds seam (M1.5, #14) — adapting mlx-audio (MIT).

The dual-track prompt layout (the piece where a wrong layout silently
produces garbage audio) is implemented by mlx-audio's
``_prepare_generation_inputs`` and matches the reference spec exactly:

    [instruct embeds]                          (only when instruct is set)
    role header  text_embed[:3]                <|im_start|>assistant\\n
    codec track  tts_pad×(n−1) + tts_bos, channel-summed with codec[:-1]
                 codec prefix = think tags (+language id unless Auto)
                              + spk_embed + codec_pad + codec_bos   (n rows)
    first text   text_embed[3:4] + codec[-1]   (the decode-step text row is
    trailing     text_embed[4:-5] + tts_eos     the returned tts_pad embed)

CustomVoice speakers resolve through the checkpoint's ``spk_id`` map (no
ECAPA / ref-audio — that is M2). All special ids come from config.

#66 splits the build by data dependence: the voice-static pieces (codec
prefix, tts specials, instruct projection) come from the per-voice
:mod:`prefix_cache` — computed once per ``(speaker, language, instruct)``
key — while the tokenizer + text embedding pass runs fresh per request.
The ops and their order are mlx-audio's own, so the assembled prompt is
bit-identical to a direct ``_prepare_generation_inputs`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx

from .prefix_cache import prompt_pieces


@dataclass(frozen=True)
class PromptLayout:
    input_embeds: mx.array
    trailing_text_hidden: mx.array
    decode_text_embed: mx.array
    codec_prefix_len: int

    @property
    def hidden_size(self) -> int:
        return self.input_embeds.shape[-1]

    @property
    def prefix_rows(self) -> int:
        """Rows of ``input_embeds`` that are voice-static — everything but
        the final first-text row (#66's Level-2 splice boundary)."""
        return self.input_embeds.shape[1] - 1

    @property
    def speaker_position(self) -> int:
        """Index within input_embeds carrying the speaker embedding: 3 role
        rows + the speaker's index inside the codec prefix. The speaker sits
        after the think rows — index 3 for Auto, 4 with an explicit language
        (the pad/bos base aligns index-to-index with codec[:-1])."""
        return 3 + (3 if self.codec_prefix_len == 6 else 4)

    @property
    def expected_len(self) -> int:
        # role(3) + combined(n−1) + first-text(1)
        return 3 + self.codec_prefix_len - 1 + 1


class PromptEmbeds:
    """Build the CustomVoice dual-track prompt for a loaded mlx-audio model."""

    def __init__(self, model: Any):
        self._model = model
        self._config = model.config.talker_config

    @property
    def speakers(self) -> List[str]:
        return list((self._config.spk_id or {}).keys())

    @property
    def languages(self) -> List[str]:
        return list((self._config.codec_language_id or {}).keys())

    def speaker_id(self, speaker: str) -> int:
        try:
            return self._config.spk_id[speaker.lower()]
        except (KeyError, AttributeError):
            known = ", ".join(sorted(self.speakers))
            raise ValueError(f"unknown speaker '{speaker}'; available: {known}") from None

    def build(self, text: str, speaker: str, language: str = "auto", instruct: Optional[str] = None) -> PromptLayout:
        pieces = prompt_pieces(self._model, speaker, language, instruct)

        # per-request half (the text rows): tokenize with the chat template,
        # embed, project — the only part of the prompt that depends on `text`
        chat_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = mx.array(self._model.tokenizer.encode(chat_text))[None, :]
        talker = self._model.talker
        text_embed = talker.text_projection(talker.get_text_embeddings()(input_ids))

        codec_embed = pieces.codec_embed
        # tts_pad * (codec_len - 2) + tts_bos, channel-summed with codec[:-1]
        pad_count = codec_embed.shape[1] - 2
        pad_embeds = mx.broadcast_to(pieces.tts_pad_embed, (1, pad_count, pieces.tts_pad_embed.shape[-1]))
        combined_embed = mx.concatenate([pad_embeds, pieces.tts_bos_embed], axis=1)
        combined_embed = combined_embed + codec_embed[:, :-1, :]

        # role(3) [+ instruct prepended] + combined + first text token
        head = (
            [pieces.instruct_embed, text_embed[:, :3, :]]
            if pieces.instruct_embed is not None
            else [text_embed[:, :3, :]]
        )
        input_embeds = mx.concatenate(head + [combined_embed], axis=1)
        first_text_embed = text_embed[:, 3:4, :] + codec_embed[:, -1:, :]
        input_embeds = mx.concatenate([input_embeds, first_text_embed], axis=1)

        # trailing text (tokens 4 to -5, plus EOS)
        trailing_text_hidden = mx.concatenate(
            [text_embed[:, 4:-5, :], pieces.tts_eos_embed],
            axis=1,
        )
        return PromptLayout(
            input_embeds=input_embeds,
            trailing_text_hidden=trailing_text_hidden,
            decode_text_embed=pieces.tts_pad_embed,
            codec_prefix_len=codec_embed.shape[1],
        )
