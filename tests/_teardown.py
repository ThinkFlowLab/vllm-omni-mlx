"""Suite infrastructure: release weight-gated class fixtures.

unittest keeps class attributes alive for the whole discover run, so every
weight-gated class holding a checkpoint keeps it resident while later
classes load theirs — four checkpoints (CustomVoice + Base + 0.6B +
VoiceDesign) exceed a 16 GB machine. Classes that set ``cls.model`` /
``cls.service`` in ``setUpClass`` inherit :class:`ReleaseAfterClass` to
drop the references and clear the MLX allocator cache once their last test
finishes. Harmless on skip-gated runs (attributes absent)."""

import mlx.core as mx


class ReleaseAfterClass:
    @classmethod
    def tearDownClass(cls):
        for attr in ("model", "service", "donor", "_clip", "_ref", "_wav"):
            if getattr(cls, attr, None) is not None:
                setattr(cls, attr, None)
        mx.clear_cache()
