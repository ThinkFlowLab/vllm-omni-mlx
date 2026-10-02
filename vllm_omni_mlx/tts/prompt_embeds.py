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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx


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
        auto = language.lower() == "auto" or not self._config.codec_language_id
        # nothink path (Auto): 3 think rows + spk + [pad, bos]; explicit
        # language adds one row (think id + language id)
        codec_prefix_len = (3 if auto else 4) + 1 + 2
        input_embeds, trailing, tts_pad = self._model._prepare_generation_inputs(
            text=text, language=language, speaker=speaker, instruct=instruct
        )
        return PromptLayout(
            input_embeds=input_embeds,
            trailing_text_hidden=trailing,
            decode_text_embed=tts_pad,
            codec_prefix_len=codec_prefix_len,
        )
