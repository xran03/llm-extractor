"""The cache is a cost control *and* an audit substrate — both must hold."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_extractor.cache import (CachedProvider, ResponseCache, VERDICT_CONFIRMED,
                                 VERDICT_UNVERIFIED, make_key)

from ._fakes import FakeProvider


class CacheKeyTest(unittest.TestCase):
    def test_same_request_same_key(self):
        args = ("llmhub", "m", [{"role": "user", "content": "hi"}], 0.0, 100)
        self.assertEqual(make_key(*args), make_key(*args))

    def test_any_input_change_changes_the_key(self):
        base = ("llmhub", "m", [{"role": "user", "content": "hi"}], 0.0, 100)
        variants = [
            ("aimodelhub", *base[1:]),
            (base[0], "other-model", *base[2:]),
            (*base[:2], [{"role": "user", "content": "different"}], *base[3:]),
            (*base[:3], 0.7, base[4]),
            (*base[:4], 200),
        ]
        keys = {make_key(*base)} | {make_key(*v) for v in variants}
        self.assertEqual(len(keys), len(variants) + 1)

    def test_json_schema_participates_in_the_key(self):
        base = ("llmhub", "m", [{"role": "user", "content": "hi"}], 0.0, 100)
        schema = {"name": "a", "schema": {"type": "object"}}
        self.assertNotEqual(make_key(*base), make_key(*base, schema))


class ResponseCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = ResponseCache(self.tmp.name)

    def tearDown(self):
        self.cache.close()
        self.tmp.cleanup()

    def test_roundtrip_and_stats(self):
        self.assertIsNone(self.cache.get("missing"))
        self.cache.put("k1", "payload", {"prompt_tokens": 10, "completion_tokens": 5},
                       meta={"stage": "extract", "doc_id": "d1"})
        entry = self.cache.get("k1")
        self.assertEqual(entry["text"], "payload")
        self.assertEqual(self.cache.stats.hits, 1)
        self.assertEqual(self.cache.stats.misses, 1)
        self.assertEqual(self.cache.stats.saved_prompt_tokens, 10)

    def test_ttl_expires_entries(self):
        cache = ResponseCache(str(Path(self.tmp.name) / "ttl"), ttl_seconds=-1)
        cache.put("k", "v")
        self.assertIsNone(cache.get("k"))
        cache.close()

    def test_index_records_metadata(self):
        self.cache.put("k1", "v", {"prompt_tokens": 3},
                       meta={"stage": "ocr", "doc_id": "d9", "model": "m"})
        rows = self.cache.query(stage="ocr")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["doc_id"], "d9")
        self.assertEqual(rows[0]["verdict"], VERDICT_UNVERIFIED)

    def test_request_is_stored_for_replay(self):
        self.cache.put("k1", "v", request={"messages": [], "model": "m"})
        self.assertIsNotNone(self.cache.load_entry("k1")["request"])

    def test_store_request_can_be_disabled(self):
        cache = ResponseCache(str(Path(self.tmp.name) / "norq"), store_request=False)
        cache.put("k1", "v", request={"messages": []})
        self.assertIsNone(cache.load_entry("k1")["request"])
        cache.close()

    def test_sample_respects_replayable_only(self):
        self.cache.put("with", "v", request={"messages": []}, meta={"stage": "extract"})
        self.cache.put("without", "v", meta={"stage": "extract"})
        keys = {row["key"] for row in self.cache.sample(10, replayable_only=True)}
        self.assertEqual(keys, {"with"})

    def test_sample_is_reproducible_with_a_seed(self):
        for i in range(20):
            self.cache.put(f"k{i}", "v", request={"messages": []},
                           meta={"stage": "extract"})
        first = [r["key"] for r in self.cache.sample(5, seed=7)]
        second = [r["key"] for r in self.cache.sample(5, seed=7)]
        self.assertEqual(first, second)

    def test_sample_oldest_strategy_orders_by_age(self):
        import time

        for i in range(3):
            self.cache.put(f"k{i}", "v", request={"messages": []},
                           meta={"stage": "extract"})
            time.sleep(0.01)
        self.assertEqual(self.cache.sample(1, strategy="oldest")[0]["key"], "k0")

    def test_mark_writes_a_verdict(self):
        self.cache.put("k1", "v", meta={"stage": "extract"})
        self.cache.mark("k1", VERDICT_CONFIRMED, 0.95, verified_by="referee")
        row = self.cache.query(verdict=VERDICT_CONFIRMED)[0]
        self.assertAlmostEqual(row["agreement"], 0.95)
        self.assertEqual(row["verified_by"], "referee")

    def test_invalidate_removes_blob_and_index_row(self):
        self.cache.put("k1", "v", meta={"stage": "extract"})
        self.assertEqual(self.cache.invalidate(["k1"]), 1)
        self.assertIsNone(self.cache.load_entry("k1"))
        self.assertEqual(self.cache.query(), [])

    def test_summary_groups_by_stage_and_verdict(self):
        self.cache.put("a", "v", meta={"stage": "extract"})
        self.cache.put("b", "v", meta={"stage": "ocr"})
        summary = self.cache.summary()
        self.assertEqual(summary["entries"], 2)
        self.assertEqual(summary["by_stage"], {"extract": 1, "ocr": 1})

    def test_clear_empties_everything(self):
        self.cache.put("a", "v", meta={"stage": "extract"})
        self.assertEqual(self.cache.clear(), 1)
        self.assertEqual(self.cache.summary()["entries"], 0)


class CachedProviderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = ResponseCache(self.tmp.name)
        self.inner = FakeProvider()
        self.provider = CachedProvider(self.inner, self.cache)
        self.messages = [{"role": "user", "content": "hello"}]

    def tearDown(self):
        self.cache.close()
        self.tmp.cleanup()

    def test_second_identical_call_does_not_hit_the_api(self):
        self.provider.complete(self.messages, model="m", meta={"stage": "extract"})
        self.provider.complete(self.messages, model="m", meta={"stage": "extract"})
        self.assertEqual(len(self.inner.calls), 1)

    def test_cached_completion_is_flagged(self):
        self.provider.complete(self.messages, model="m", meta={"stage": "extract"})
        again = self.provider.complete(self.messages, model="m", meta={"stage": "extract"})
        self.assertTrue(again.usage.cached)

    def test_different_model_is_a_different_entry(self):
        self.provider.complete(self.messages, model="m1", meta={"stage": "extract"})
        self.provider.complete(self.messages, model="m2", meta={"stage": "extract"})
        self.assertEqual(len(self.inner.calls), 2)

    def test_bypass_forces_a_live_call_and_refreshes_the_entry(self):
        self.provider.complete(self.messages, model="m", meta={"stage": "extract"})
        bypassing = CachedProvider(self.inner, self.cache, bypass=True)
        bypassing.complete(self.messages, model="m", meta={"stage": "extract"})
        self.assertEqual(len(self.inner.calls), 2)

    def test_meta_is_persisted_to_the_index(self):
        self.provider.complete(self.messages, model="m",
                               meta={"stage": "ocr", "doc_id": "doc7"})
        self.assertEqual(self.cache.query(stage="ocr")[0]["doc_id"], "doc7")


if __name__ == "__main__":
    unittest.main()
