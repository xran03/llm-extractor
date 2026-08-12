"""Chunking must always terminate and make progress, for any parameters."""
from __future__ import annotations

import unittest

from llm_extractor.extract import chunk_text


class ChunkTerminationTest(unittest.TestCase):
    def test_overlap_larger_than_size_still_terminates(self):
        text = "word " * 20000
        chunks = chunk_text(text, size=1000, overlap=5000)
        self.assertLess(len(chunks), 200)

    def test_overlap_equal_to_size_still_terminates(self):
        chunks = chunk_text("word " * 5000, size=2000, overlap=2000)
        self.assertLess(len(chunks), 100)

    def test_zero_overlap(self):
        text = "abcdefghij" * 100
        chunks = chunk_text(text, size=200, overlap=0)
        self.assertEqual("".join(chunks), text)

    def test_chunk_count_is_proportional_to_length(self):
        text = "x" * 10000
        self.assertLessEqual(len(chunk_text(text, size=1000, overlap=100)), 15)

    def test_size_of_one_terminates(self):
        self.assertLessEqual(len(chunk_text("abcdef", size=1, overlap=0)), 6)


if __name__ == "__main__":
    unittest.main()
