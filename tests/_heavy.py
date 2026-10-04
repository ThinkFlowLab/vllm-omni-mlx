"""Opt-in gate for the suite's heaviest weight-gated batteries.

The Base-checkpoint clone batteries (`test_clone_scaffold.BaseCloneE2ETest`,
`test_icl_stream.ICLStreamE2ETest`) load the 1.7B Base model plus — for their
reference clips — a donor CustomVoice model, peaking around 8 GB. Alone that
fits a 16 GB Mac; stacked on a serving process, a benchmark, or a second
suite run it can push the machine into swap. They are skipped by default and
included with ``VOMX_HEAVY_TESTS=1`` (or ``scripts/run_tests.sh --heavy``).
CI is unaffected either way — it caches no weights, so these classes skip.
"""

import os
import unittest

requires_heavy = unittest.skipUnless(
    os.environ.get("VOMX_HEAVY_TESTS", "") not in ("", "0"),
    "heavy battery: set VOMX_HEAVY_TESTS=1 (Base + donor checkpoint, ~8 GB peak)",
)
