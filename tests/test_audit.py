"""Cache revalidation: sampling, replay, scoring and verdict write-back."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_extractor.audit import (audit_cache, compare_records, record_signature,
                                 value_agreement, wilson_interval)
from llm_extractor.cache import (CachedProvider, ResponseCache, VERDICT_CONFIRMED,
                                 VERDICT_DRIFTED, VERDICT_SUSPECT, VERDICT_UNVERIFIED)

from ._fakes import DEFAULT_RECORDS, FakeProvider


class ValueAgreementTest(unittest.TestCase):
    def test_identical_values(self):
        self.assertEqual(value_agreement("a", "a"), 1.0)
        self.assertEqual(value_agreement(1.0, 1.0), 1.0)
        self.assertEqual(value_agreement(None, None), 1.0)

    def test_one_sided_null(self):
        self.assertEqual(value_agreement("a", None), 0.0)

    def test_numeric_tolerance(self):
        self.assertEqual(value_agreement(100.0, 100.00001), 1.0)
        self.assertEqual(value_agreement(100.0, 120.0), 0.0)

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(value_agreement(" Group A ", "group a"), 1.0)

    def test_partial_text_overlap_scores_between(self):
        score = value_agreement("group A high", "group A")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_lists_compare_structurally(self):
        self.assertEqual(value_agreement([1, 2], [1, 2]), 1.0)
        self.assertEqual(value_agreement([1, 2], [2, 1]), 0.0)


class CompareRecordsTest(unittest.TestCase):
    def test_identical_sets_agree_fully(self):
        agreement, detail = compare_records(DEFAULT_RECORDS, DEFAULT_RECORDS)
        self.assertEqual(agreement, 1.0)
        self.assertEqual(detail["added"], 0)
        self.assertEqual(detail["removed"], 0)

    def test_changed_field_is_reported(self):
        changed = json.loads(json.dumps(DEFAULT_RECORDS))
        changed[0]["value"] = 99.0
        agreement, detail = compare_records(DEFAULT_RECORDS, changed)
        self.assertLess(agreement, 1.0)
        self.assertIn("value", detail["changed_fields"])

    def test_added_and_removed_records_are_counted(self):
        _, detail = compare_records(DEFAULT_RECORDS, DEFAULT_RECORDS[:1])
        self.assertEqual(detail["removed"], 1)
        self.assertEqual(detail["added"], 0)

    def test_disjoint_sets_score_zero(self):
        other = [{"subject": "z", "source_span": "totally different evidence"}]
        agreement, _ = compare_records(DEFAULT_RECORDS, other)
        self.assertEqual(agreement, 0.0)

    def test_audit_flags_are_ignored_in_comparison(self):
        with_flags = [{**DEFAULT_RECORDS[0], "_grounded": True, "doc_id": "d1"}]
        without = [{**DEFAULT_RECORDS[0], "_grounded": False, "doc_id": "d2"}]
        agreement, _ = compare_records(with_flags, without)
        self.assertEqual(agreement, 1.0)

    def test_signature_prefers_the_evidence_span(self):
        a = {"subject": "x", "source_span": "the same quoted sentence"}
        b = {"subject": "y", "source_span": "the same quoted sentence"}
        self.assertEqual(record_signature(a), record_signature(b))

    def test_signature_falls_back_to_content(self):
        signature = record_signature({"subject": "x", "value": 1})
        self.assertIn("subject", signature)

    def test_two_empty_sets_agree(self):
        agreement, _ = compare_records([], [])
        self.assertEqual(agreement, 1.0)


class WilsonIntervalTest(unittest.TestCase):
    def test_zero_sample_is_zero_width(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))

    def test_perfect_small_sample_has_a_wide_interval(self):
        low, high = wilson_interval(5, 5)
        self.assertLess(low, 0.9)
        self.assertEqual(high, 1.0)

    def test_larger_sample_narrows_the_interval(self):
        small = wilson_interval(9, 10)
        large = wilson_interval(900, 1000)
        self.assertGreater(small[1] - small[0], large[1] - large[0])


class AuditCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = ResponseCache(str(Path(self.tmp.name) / "cache"))
        self.inner = FakeProvider()
        self.messages = [{"role": "user", "content": "extract this"}]
        # Seed the cache with real entries through the caching provider.
        writer = CachedProvider(self.inner, self.cache)
        for i in range(6):
            writer.complete([{"role": "user", "content": f"doc {i}"}], model="m",
                            meta={"stage": "extract", "doc_id": f"d{i}"})

    def tearDown(self):
        self.cache.close()
        self.tmp.cleanup()

    def _auditor(self, provider=None):
        return CachedProvider(provider or self.inner, self.cache, bypass=True)

    def test_stable_model_confirms_every_entry(self):
        report = audit_cache(self.cache, self._auditor(), n=3, seed=1)
        self.assertEqual(report.sampled, 3)
        self.assertEqual(report.confirmed, 3)
        self.assertEqual(report.pass_rate, 1.0)

    def test_verdicts_are_written_back_to_the_index(self):
        audit_cache(self.cache, self._auditor(), n=2, seed=1)
        self.assertEqual(len(self.cache.query(verdict=VERDICT_CONFIRMED)), 2)

    def test_a_changed_model_answer_is_detected(self):
        drifted = FakeProvider(records=[{"subject": "completely different",
                                         "source_span": "new evidence entirely"}])
        report = audit_cache(self.cache, self._auditor(drifted), n=3, seed=1)
        self.assertEqual(report.confirmed, 0)
        self.assertEqual(report.suspect, 3)

    def test_partial_drift_is_classified_as_drifted(self):
        partial = json.loads(json.dumps(DEFAULT_RECORDS))
        partial[0]["value"] = 99.0
        partial[0]["direction"] = "lower"
        partial[0]["significant"] = "no"
        report = audit_cache(self.cache, self._auditor(FakeProvider(records=partial)),
                             n=2, seed=1)
        self.assertEqual(report.confirmed + report.drifted + report.suspect, 2)
        self.assertLess(report.mean_agreement, 1.0)

    def test_referee_model_is_used_for_the_replay(self):
        audit_cache(self.cache, self._auditor(), n=1, seed=1, referee_model="stronger")
        self.assertIn("stronger", [c["model"] for c in self.inner.calls])

    def test_replay_failure_is_an_error_verdict_not_a_crash(self):
        report = audit_cache(self.cache, self._auditor(FakeProvider(fail_stages={
            "audit:extract"})), n=2, seed=1)
        self.assertEqual(report.errors, 2)
        self.assertEqual(report.pass_rate, 0.0)

    def test_invalidate_drifted_removes_bad_entries(self):
        before = self.cache.summary()["entries"]
        drifted = FakeProvider(records=[{"subject": "different",
                                         "source_span": "new evidence"}])
        audit_cache(self.cache, self._auditor(drifted), n=2, seed=1,
                    invalidate_drifted=True)
        self.assertEqual(self.cache.summary()["entries"], before - 2)

    def test_only_unverified_skips_already_confirmed_entries(self):
        audit_cache(self.cache, self._auditor(), n=6, seed=1)
        report = audit_cache(self.cache, self._auditor(), n=6, only_unverified=True)
        self.assertEqual(report.sampled, 0)

    def test_stage_filter_restricts_the_sample(self):
        writer = CachedProvider(self.inner, self.cache)
        writer.complete([{"role": "user", "content": "figure"}], model="v",
                        meta={"stage": "ocr", "doc_id": "d99"})
        report = audit_cache(self.cache, self._auditor(), n=10, stage="ocr")
        self.assertEqual(report.sampled, 1)

    def test_empty_cache_produces_an_empty_report(self):
        empty = ResponseCache(str(Path(self.tmp.name) / "empty"))
        try:
            report = audit_cache(empty, self._auditor(), n=5)
            self.assertEqual(report.sampled, 0)
            self.assertEqual(report.to_dict()["pass_rate"], 0.0)
        finally:
            empty.close()

    def test_report_serializes_with_a_confidence_interval(self):
        data = audit_cache(self.cache, self._auditor(), n=3, seed=1).to_dict()
        self.assertEqual(len(data["pass_rate_ci95"]), 2)
        self.assertEqual(len(data["entries"]), 3)

    def test_entries_without_a_stored_request_are_not_sampled(self):
        cache = ResponseCache(str(Path(self.tmp.name) / "norq"), store_request=False)
        try:
            CachedProvider(FakeProvider(), cache).complete(
                self.messages, model="m", meta={"stage": "extract"})
            self.assertEqual(audit_cache(cache, self._auditor(), n=5).sampled, 0)
        finally:
            cache.close()

    def test_unaudited_entries_start_unverified(self):
        rows = self.cache.query()
        self.assertTrue(all(r["verdict"] == VERDICT_UNVERIFIED for r in rows))


if __name__ == "__main__":
    unittest.main()
