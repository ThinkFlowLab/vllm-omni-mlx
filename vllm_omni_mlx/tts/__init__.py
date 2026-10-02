"""Qwen3-TTS on MLX — milestone M1 (roadmap #2), adapting mlx-audio (MIT).

M1.1 (#10) scaffolds config + weights loading; M1.2–M1.6 (#11–#15) wrap the
model stages behind this package with parity tests; M1.7 (#16) serves
OpenAI-compatible POST /v1/audio/speech on the existing Starlette app.
"""
