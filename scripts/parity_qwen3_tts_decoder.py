#!/usr/bin/env python3
"""Cross-framework parity check: our MLX code2wav decode vs the torch
reference in the vllm-omni checkout (tokenizer_12hz).

Runs in TWO processes because no single environment has both stacks:
the MLX side under the repo venv (mlx-audio), the torch side under a python
that has torch+transformers (e.g. the miniconda base). Usage:

    # MLX side (repo venv): decode fixed-seed random codes, save artifacts
    python scripts/parity_qwen3_tts_decoder.py mlx [--codes 60] [--seed 7]

    # torch side (torch env): load the reference decoder from the checkout,
    # decode the same codes, compare against the saved MLX waveform
    python scripts/parity_qwen3_tts_decoder.py torch --ref ~/code/vllm-omni@codex

Status 2026-10-02 (#11): the harness runs end to end, but parity is NOT yet
established — under transformers 5.2 (plus a mask-arg shim the reference
needs there) torch vs MLX agree on waveform statistics (RMS within 0.2%)
yet differ pointwise (Pearson r≈0.83, mean |Δ|≈2.5e-2). Unresolved whether
the gap is the shim's mask semantics on the torch side or mlx-audio's
causal-only decoder attention (the reference alternates sliding-window
layers). Re-run under the transformers version the reference pins before
trusting either conclusion; mlx side is empirically fine (intelligible
speech in the M1.0 e2e run).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

MLX_WAV = "/tmp/parity_mlx.wav.npy"
CODES_NPY = "/tmp/parity_codes.npy"


def _snapshot_tokenizer_dir() -> str:
    from vllm_omni_mlx.tts.config import DEFAULT_MODEL, local_snapshot

    snap = local_snapshot(DEFAULT_MODEL)
    if snap is None:
        raise SystemExit("checkpoint not cached locally")
    return os.path.join(snap, "speech_tokenizer")


def run_mlx(n_codes: int, seed: int) -> None:
    import numpy as np

    import mlx.core as mx

    from vllm_omni_mlx.tts.code2wav import Code2Wav
    from vllm_omni_mlx.tts.config import TTSConfig, load_tts_model

    model = load_tts_model(TTSConfig())
    dec = Code2Wav(model.speech_tokenizer)
    mx.random.seed(seed)
    codes = mx.random.randint(0, 2048, (1, dec.num_quantizers, n_codes))
    wav = dec.decode(codes)
    np.save(CODES_NPY, np.array(codes[0]))
    np.save(MLX_WAV, np.array(wav[0]))
    print(f"mlx: saved {n_codes} codes (seed {seed}) -> wav {wav.shape[-1]} samples")


def run_torch(ref_checkout: str) -> None:
    import inspect
    import types

    import numpy as np

    base = os.path.join(ref_checkout, "vllm_omni")

    def reg(name: str, path: str) -> None:
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod

    # register the package chain without executing __init__ files (the
    # checkout's package root imports a vllm version this env may not match)
    reg("vllm_omni", base)
    reg("vllm_omni.model_executor", base + "/model_executor")
    reg("vllm_omni.model_executor.models", base + "/model_executor/models")
    reg("vllm_omni.model_executor.models.common", base + "/model_executor/models/common")
    reg("vllm_omni.model_executor.models.qwen3_tts", base + "/model_executor/models/qwen3_tts")
    reg("vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz", base + "/model_executor/models/qwen3_tts/tokenizer_12hz")

    import torch
    import transformers.masking_utils as MU

    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2 as M

    params = inspect.signature(MU.create_causal_mask).parameters

    def make_shim(orig):
        def shim(**kwargs):
            kw = dict(kwargs)
            if "input_embeds" in kw and "inputs_embeds" in params:
                kw["inputs_embeds"] = kw.pop("input_embeds")
            if "inputs_embeds" in kw and "input_embeds" in params:
                kw["input_embeds"] = kw.pop("inputs_embeds")
            if "cache_position" not in kw and "cache_position" in params:
                emb = kw.get("inputs_embeds", kw.get("input_embeds"))
                kw["cache_position"] = torch.arange(emb.shape[1])
            return orig(**kw)

        return shim

    M.create_causal_mask = make_shim(MU.create_causal_mask)
    if hasattr(MU, "create_sliding_window_causal_mask"):
        M.create_sliding_window_causal_mask = make_shim(MU.create_sliding_window_causal_mask)

    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit/snapshots/*/speech_tokenizer"))[0]
    model = M.Qwen3TTSTokenizerV2Model.from_pretrained(snap, torch_dtype=torch.float32).eval()
    codes = torch.from_numpy(np.load(CODES_NPY)).long()[None]
    with torch.no_grad():
        wav = model.decoder(codes)
    tw = wav[0, 0].float().numpy()
    mw = np.load(MLX_WAV)
    d = np.abs(tw - mw)
    r = float(np.corrcoef(tw, mw)[0, 1])
    print(f"torch: {tw.shape[-1]} samples | mlx rms {np.sqrt((mw**2).mean()):.4f} torch rms {np.sqrt((tw**2).mean()):.4f}")
    print(f"max |Δ| {d.max():.3e}  mean |Δ| {d.mean():.3e}  pearson r {r:.4f}  -> {'MATCH' if d.max() < 1e-4 else 'DIVERGE'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("side", choices=("mlx", "torch"))
    parser.add_argument("--codes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ref", default=os.path.expanduser("~/code/vllm-omni@codex"), help="vllm-omni checkout path (torch side)")
    args = parser.parse_args()
    if args.side == "mlx":
        run_mlx(args.codes, args.seed)
    else:
        run_torch(args.ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
