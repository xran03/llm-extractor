"""The execution core's contract, which every stage depends on.

These run against whichever backend is installed — the sequential
implementation, or the accelerated core when its wheel is present — so both are
held to the same contract.
"""
from __future__ import annotations

import unittest

from llm_extractor._exec import (BACKEND, BACKEND_DETAIL, map_completed, map_ordered,
                                 run_both)


class ContractTest(unittest.TestCase):
    def test_backend_is_reported(self):
        self.assertIn(BACKEND, ("sequential", "accelerated"))
        self.assertIn(BACKEND_DETAIL, ("sequential", "compiled", "python"))

    def test_map_ordered_preserves_order(self):
        self.assertEqual(map_ordered(lambda x: x * 2, [1, 2, 3], workers=4), [2, 4, 6])

    def test_map_ordered_empty(self):
        self.assertEqual(map_ordered(lambda x: x, []), [])

    def test_map_ordered_propagates_exceptions(self):
        def boom(x):
            raise ValueError("bad")

        with self.assertRaises(ValueError):
            map_ordered(boom, [1])

    def test_map_ordered_accepts_a_generator(self):
        self.assertEqual(map_ordered(str, (i for i in range(3))), ["0", "1", "2"])

    def test_map_completed_returns_every_result(self):
        self.assertEqual(sorted(map_completed(lambda x: x * 2, [1, 2, 3], workers=4)),
                         [2, 4, 6])

    def test_map_completed_empty(self):
        self.assertEqual(list(map_completed(lambda x: x, [])), [])

    def test_map_completed_visits_every_item(self):
        seen = []
        for value in map_completed(seen.append, [1, 2, 3]):
            self.assertIsNone(value)
        self.assertEqual(sorted(seen), [1, 2, 3])

    def test_run_both_returns_both_results(self):
        self.assertEqual(run_both(lambda: "a", lambda: "b"), ("a", "b"))

    def test_run_both_returns_exceptions_rather_than_raising(self):
        def boom():
            raise RuntimeError("stage failed")

        first, second = run_both(boom, lambda: "ok")
        self.assertIsInstance(first, RuntimeError)
        self.assertEqual(second, "ok")

    def test_run_both_handles_two_failures(self):
        def boom():
            raise RuntimeError("x")

        first, second = run_both(boom, boom)
        self.assertIsInstance(first, RuntimeError)
        self.assertIsInstance(second, RuntimeError)

    def test_worker_hint_never_changes_results(self):
        for workers in (0, 1, 8, None, "not a number"):
            with self.subTest(workers=workers):
                self.assertEqual(map_ordered(lambda x: x, [1, 2], workers=workers),
                                 [1, 2])


if __name__ == "__main__":
    unittest.main()
