"""Diffusion (text → image) seam over mflux (#91, #101).

Same doctrine as tts/ over mlx-audio: this package owns config, loading and
serving; the model math lives entirely in mflux (pinned by the [image]
extra). Served through OpenAI-compatible POST /v1/images/generations.
"""
