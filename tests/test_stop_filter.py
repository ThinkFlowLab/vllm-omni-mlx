"""Stop-sequence filtering across chunk boundaries."""

import unittest

from vllm_omni_mlx.backends import _stop_filter


class StopFilterTest(unittest.TestCase):
    def test_holds_back_tail_without_match(self):
        # the last len(stop)-1 chars are held until more text arrives or the
        # stream ends (the caller flushes the held tail on clean finish)
        emit, pending, hit = _stop_filter("hello world", ("STOP",))
        self.assertEqual((emit, hit), ("hello wo", None))
        self.assertEqual(pending, "rld")

    def test_holds_back_potential_prefix(self):
        # "STO" could still become "STOP" — held back
        emit, pending, hit = _stop_filter("hello STO", ("STOP",))
        self.assertEqual(emit, "hello ")
        self.assertEqual(pending, "STO")
        self.assertIsNone(hit)

    def test_truncates_on_match(self):
        emit, pending, hit = _stop_filter("hello STOP world", ("STOP",))
        self.assertEqual(emit, "hello ")
        self.assertEqual(pending, "")
        self.assertEqual(hit, "STOP")

    def test_earliest_match_wins(self):
        emit, _, hit = _stop_filter("a BB c AA d", ("AA", "BB"))
        self.assertEqual(emit, "a ")
        self.assertEqual(hit, "BB")

    def test_chunk_boundary_assembly(self):
        stops = ("</end>",)
        pending = ""
        emitted = []
        for piece in ("answer ", "tok</e", "nd>leftover"):
            pending += piece
            emit, pending, hit = _stop_filter(pending, stops)
            emitted.append(emit)
            if hit:
                break
        self.assertEqual("".join(emitted), "answer tok")
        self.assertEqual(hit, "</end>")


if __name__ == "__main__":
    unittest.main()
