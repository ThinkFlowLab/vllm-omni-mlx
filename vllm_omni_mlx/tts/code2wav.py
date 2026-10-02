"""Stage 1 (code2wav): the 12 Hz codec decoder, wrapped for M1.2 (#11).

mlx-audio's ``Qwen3TTSSpeechTokenizer`` owns the numerics (SplitRVQ dequant →
CausalConv → 8-layer sliding-window transformer → 2× ConvNeXt upsample →
SnakeBeta transposed-conv vocoder → clamp(-1, 1), all fp32). This module is
our seam over it: one-shot and chunked decode behind a stable interface, so
generation (#15), the endpoint (#16), and mx.compile (#17) do not touch
mlx-audio types directly.
"""

from __future__ import annotations

from typing import Iterator

import mlx.core as mx


class Code2Wav:
    """Decode 12 Hz codec codes to a 24 kHz mono waveform.

    The wrapped tokenizer comes from :func:`vllm_omni_mlx.tts.config.load_tts_model`
    (``model.speech_tokenizer``).
    """

    def __init__(self, speech_tokenizer):
        self._tok = speech_tokenizer

    @property
    def sample_rate(self) -> int:
        return 24000

    @property
    def num_quantizers(self) -> int:
        return self._tok.decoder.config.num_quantizers

    def decode(self, codes: mx.array) -> mx.array:
        """One-shot decode of ``[batch, num_quantizers, time]`` codes to
        ``[batch, samples]`` audio, clamped to (-1, 1)."""
        wav = self._tok.decoder(codes).squeeze(1)
        return wav

    def chunks(self, codes: mx.array, chunk_size: int = 300) -> Iterator[mx.array]:
        """Yield ``[samples]`` audio per code chunk — the streaming shape the
        /v1/audio/speech endpoint serves. Uses mlx-audio's stateful
        ``streaming_step`` (conv buffers + transformer KV cache carry context
        across chunks), which is both more accurate than left-context windowed
        decoding and the natural fit for codes arriving as they generate.

        Streaming approximates full-sequence decode — limited visible context
        puts transients at chunk boundaries (measured on random codes: mean
        |Δ| ~2.5e-4 vs full decode via streaming_step, vs ~1.6e-2 for
        left-context windows). Inherent to real-time chunked serving, not a
        wrapper defect. Streaming state is reset at the start of each call."""
        decoder = self._tok.decoder
        decoder.reset_streaming_state()
        for start in range(0, codes.shape[-1], chunk_size):
            yield decoder.streaming_step(codes[..., start : start + chunk_size]).squeeze(1)
