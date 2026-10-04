"""Base ICL streaming clone (#50) — weight-gated battery: token-exactness
of the vendored `generate_icl_frames` against mlx-audio's `_generate_icl`,
streaming e2e through the service, and TTFA sanity. Runs where the Base
checkpoint is cached, skips in CI."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import unittest

from tests._teardown import ReleaseAfterClass

import mlx.core as mx

from vllm_omni_mlx.tts.config import DEFAULT_MODEL, TTSConfig, load_tts_model, local_snapshot
from vllm_omni_mlx.tts.service import TTSService
from vllm_omni_mlx.tts.stream_loop import generate_icl_frames, synthesize_clone_stream

BASE = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"
REF_TEXT = "This is the voice we are cloning today."
TEXT = "The streaming clone loop must draw the very same tokens."


class ICLStreamE2ETest(ReleaseAfterClass, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if local_snapshot(BASE) is None:
            raise unittest.SkipTest(f"{BASE} not cached locally")
        cls.model = load_tts_model(TTSConfig(model_ref=BASE))
        cls.service = TTSService(cls.model, TTSConfig(model_ref=BASE))

    @classmethod
    def reference_audio(cls) -> "mx.array":
        # donor clip synthesized once per class, donor model released
        # immediately — keeps peak memory at Base + a 3 s waveform
        if getattr(cls, "_ref", None) is None:
            if local_snapshot(DEFAULT_MODEL) is None:
                raise unittest.SkipTest("no CustomVoice snapshot cached to synthesize a reference clip")
            from vllm_omni_mlx.tts.generate import decode_ref_audio, synthesize, wav_bytes

            donor = load_tts_model(TTSConfig())
            try:
                cls._wav = wav_bytes(synthesize(donor, TTSConfig(), REF_TEXT, seed=7))
                cls._ref = decode_ref_audio(cls._wav)
                mx.eval(cls._ref)
            finally:
                del donor
                mx.clear_cache()
        return cls._ref

    def test_vendored_loop_is_token_exact_vs_mlx_audio(self):
        """#43's method: identical sampler draws on identical seeds. Records
        every _sample_token call (group-0 + all 15 residual groups) from both
        loops and compares the sequences — parity is claimed at the sampler
        level; audio differs across decode chunkings by nature."""
        ref = self.reference_audio()

        def record_tokens(model, seed, run):
            original = model._sample_token
            recorded = []

            def recorder(logits, **kwargs):
                token = original(logits, **kwargs)
                recorded.append(int(token[0, 0]))
                return token

            model._sample_token = recorder
            try:
                mx.random.seed(seed)
                for chunk in run():
                    mx.eval(chunk)
            finally:
                model._sample_token = original
            return recorded

        # token-exact parity is asserted against the eager loop (recorded
        # via _sample_token, which cannot see draws inside the compiled
        # closures); the compiled path's rare fp16 near-tie drift is covered
        # by tests.test_stream_loop.test_compiled_drift_bounded
        from vllm_omni_mlx.tts import stream_loop

        was, stream_loop.EAGER_STREAM = stream_loop.EAGER_STREAM, True
        try:
            ours = record_tokens(
                self.model, 11,
                lambda: generate_icl_frames(
                    self.model, text=TEXT, ref_audio=ref, ref_text=REF_TEXT,
                    temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.05,
                    max_tokens=256, initial_frames=2, chunk_frames=6,
                ),
            )
        finally:
            stream_loop.EAGER_STREAM = was
        theirs = record_tokens(
            self.model, 11,
            lambda: self.model.generate(
                TEXT, ref_audio=ref, ref_text=REF_TEXT, stream=True,
                streaming_interval=0.5, temperature=0.9, top_k=50, top_p=1.0,
                repetition_penalty=1.05, max_tokens=256, verbose=False,
            ),
        )
        self.assertGreater(len(ours), 32, "implausibly few draws — EOS broke immediately?")
        self.assertEqual(ours, theirs, "vendored ICL loop drew different tokens than mlx-audio's")

    @classmethod
    def reference_voice(cls) -> dict:
        # base64 voice object for the service path, from the shared clip
        import base64

        wav = getattr(cls, "_wav", None)
        if wav is None:
            cls.reference_audio()  # populates _wav alongside _ref
            wav = cls._wav
        return {"ref_audio": base64.b64encode(wav).decode(), "ref_text": REF_TEXT}

    def test_streaming_clone_e2e_is_speech(self):
        from tests.audio_metrics import CATASTROPHIC_HNR_DB, int16_pcm_hnr_db

        voice = self.reference_voice()
        t0 = _now()
        chunks = list(self.service.speech_stream(TEXT, voice=voice))
        ttfa = _now() - t0
        self.assertGreater(len(chunks), 1, "expected chunked audio, not one blob")
        pcm = b"".join(chunks)
        self.assertGreater(len(pcm), 2 * 24000)
        hnr = int16_pcm_hnr_db(pcm)
        self.assertGreater(hnr, CATASTROPHIC_HNR_DB, f"HNR {hnr:.2f} dB below catastrophic floor")
        print(f"\nicl stream: ttfa {ttfa:.2f}s, {len(chunks)} chunks, HNR {hnr:.2f} dB")

    def test_synthesize_clone_stream_rejects_custom_voice(self):
        # guard fires at the entry, mirroring the buffered clone entry
        stub = type("M", (), {})()
        stub.config = type("C", (), {"tts_model_type": "custom_voice", "tts_model_size": "1b7"})()
        with self.assertRaisesRegex(ValueError, "needs a Base checkpoint"):
            list(synthesize_clone_stream(stub, TTSConfig(), "hi", mx.zeros(24000), "ref"))


def _now():
    import time

    return time.perf_counter()


if __name__ == "__main__":
    unittest.main()
