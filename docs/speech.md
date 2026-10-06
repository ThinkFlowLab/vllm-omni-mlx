# Speech (TTS) Usage

Everything about `/v1/audio/speech` beyond the quickstart in the README: voices,
instructions, streaming knobs, cloning, and the per-checkpoint differences.
Setup and server commands live in the [README](../README.md); runnable scripts
in [`examples/`](../examples/).

## Voices by checkpoint

The loaded checkpoint decides what `voice` and `instructions` mean — mirroring
mlx-audio's own mapping:

| Checkpoint | `voice` | `instructions` |
| --- | --- | --- |
| CustomVoice (0.6B / 1.7B) | preset speaker name — `GET /v1/audio/voices` lists them (9 presets: vivian, ryan, aiden, …) | optional emotion/style prompt (1.7B only — rejected with 400 on 0.6B) |
| Base (1.7B) | **cloning object** — `{"ref_audio": <base64>, "ref_text": "…"}` | — |
| VoiceDesign (1.7B) | rejected (no presets) | **required** — the voice description itself, e.g. "A cheerful young female voice with high pitch and energetic tone" |

`language` forces a language (default auto); `speed` must be 1.0 for now.

## Streaming

`"stream": true` switches from a buffered WAV to chunked raw PCM (24 kHz
16-bit mono, `X-Audio-*` response headers):

- first audio typically lands in **~0.1 s** (single-codec-frame first chunk,
  warm voice — per-voice prefix cache);
- `streaming_interval` (default 0.5 s) sets the steady chunk size;
- `streaming_initial_interval` (default 0.08 s) sets the first-chunk size —
  raise to 0.2 s for a chunkier first beat.

```sh
curl -N -H 'Authorization: Bearer demo' -H 'Content-Type: application/json' \
    -d '{"input": "Hello.", "voice": "vivian", "stream": true}' \
    http://127.0.0.1:8000/v1/audio/speech -o speech.pcm
```

## Voice cloning (Base checkpoints)

`voice` carries a short reference clip and its transcript:

```sh
vllm-omni-mlx serve mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit --omni --api-key demo
REF=$(base64 -i reference.wav)
curl -H 'Authorization: Bearer demo' -H 'Content-Type: application/json' \
    -d "{\"input\": \"Any text in the cloned voice.\", \"voice\": {\"ref_audio\": \"$REF\", \"ref_text\": \"transcript of the reference clip\"}}" \
    http://127.0.0.1:8000/v1/audio/speech -o cloned.wav
```

The clip is decoded and resampled to 24 kHz mono server-side (any format
miniaudio/ffmpeg reads); keep it 0.5–30 s of clean speech. Cloning streams too
(`stream: true` + the voice object) on the same fast path as presets — the
loop is token-exact against mlx-audio's ICL path; chunked audio differs from
the buffered WAV at waveform level by nature (the vocoder is stateful).
Cloning pays per-request setup (reference re-encode + ICL prefill), so its
time-to-first-audio is ~0.35–0.7 s vs ~0.1 s for presets — a known
optimization target.

## One-shot, no server

```sh
vllm-omni-mlx tts --voice ryan --text "Hello from the CLI." --out out.wav
```

## Performance & verification

Measured numbers (first-audio, sustained RTF, per-checkpoint tables) and the
public reproduction protocol live in
[issue #84](https://github.com/ThinkFlowLab/vllm-omni-mlx/issues/84); the
benchmark scripts are `scripts/bench_all_checkpoints.py`,
`scripts/bench_prefix_cache.py`, `scripts/bench_stream_rtf.py`. Correctness
gates: HNR floors (harmonics-to-noise, the catastrophic-decode detector) plus
token-exactness harnesses in `tests/`.
