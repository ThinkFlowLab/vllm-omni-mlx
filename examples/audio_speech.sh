#!/bin/sh
# /v1/audio/speech round trip (#16): start the server with a TTS model first —
#   vllm-omni-mlx --tts-model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit --api-key demo
# then run this script.

BASE="${BASE:-http://127.0.0.1:8000}"
KEY="${KEY:-demo}"

echo "== preset voices =="
curl -s -H "Authorization: Bearer $KEY" "$BASE/v1/audio/voices"
echo

echo "== synthesize (wav) =="
curl -s -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d '{"input": "Hello from vllm omni em el ex on Apple Silicon.", "voice": "vivian"}' \
    "$BASE/v1/audio/speech" -o speech.wav
file speech.wav 2>/dev/null || ls -la speech.wav

echo "== style instruction + explicit language =="
curl -s -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d '{"input": "今天天气很好，我们一起去公园散步吧。", "voice": "ryan", "language": "chinese", "instructions": "Very happy."}' \
    "$BASE/v1/audio/speech" -o speech_zh.wav
ls -la speech_zh.wav
