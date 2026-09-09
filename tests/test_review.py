"""The review pass: value, unit and row-coherence verdicts, live and batched.

The alignment cases matter most. A review answer names records by the index its
request handed out, so a short, reordered or unknown-index reply must annotate
only what it legitimately covers — shifting a verdict onto the next row would
mark a good record bad and let a bad one through.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_extractor import batchapi, review
from llm_extractor.batchapi import KIND_REVIEW, STAGE_REVIEW, Manifest
from llm_extractor.serialize import record_columns, read_csv
from llm_extractor.settings import Settings
from llm_extractor.sources import SourceDocument
from llm_extractor.templates import BUILTIN_TEMPLATES

from .test_batchapi import make_provider

TEMPLATE = BUILTIN_TEMPLATES["immunogenicity"]


def records():
    return [
        {"doc_id": "d", "assay": "opa", "group_label": "PCV13", "value": 7132,
         "value_unit": "opsonic index", "serotype": "6B",
         "source_span": "post-vaccination: 7132"},
        {"doc_id": "d", "assay": "opa", "group_label": "control", "value": 164,
         "value_unit": "opsonic index", "serotype": "6B",
         "source_span": "post-challenge: 164"},
    ]


def verdict_payload(*items) -> str:
    return json.dumps({"verdicts": list(items)})


class PlanTest(unittest.TestCase):
    def test_records_are_routed_to_the_chunk_holding_their_evidence(self):
        text = "A" * 40000 + "\npost-challenge: 164 was the control arm.\n"
        text = "post-vaccination: 7132 in the vaccine arm.\n" + text
        plan = review.plan(records(), text)
        self.assertTrue(plan)
        placed = {position for _, positions in plan for position in positions}
        self.assertEqual(placed, {0, 1}, "every record must be reviewed somewhere")

    def test_a_record_with_no_span_is_still_reviewed(self):
        rows = [{"doc_id": "d", "value": 1}]
        plan = review.plan(rows, "some document text")
        self.assertEqual([p for _, positions in plan for p in positions], [0])

    def test_no_text_means_nothing_to_review_against(self):
        self.assertEqual(review.plan(records(), ""), [])

    def test_requests_are_capped(self):
        rows = [dict(doc_id="d", value=i, source_span=f"v{i}") for i in range(45)]
        plan = review.plan(rows, "a document", limit=20)
        self.assertEqual([len(p) for _, p in plan], [20, 20, 5])


class ApplyVerdictsTest(unittest.TestCase):
    def test_verdicts_land_on_the_records_they_describe(self):
        rows = records()
        review.apply_verdicts(rows, [0, 1], verdict_payload(
            {"index": 0, "value_ok": True, "unit_ok": True, "row_ok": True,
             "issue": None},
            {"index": 1, "value_ok": False, "unit_ok": True, "row_ok": False,
             "issue": "value belongs to the vaccine arm"},
        ))
        self.assertTrue(rows[0]["_review_value"])
        self.assertFalse(rows[1]["_review_value"])
        self.assertFalse(rows[1]["_review_row"])
        self.assertIn("vaccine arm", rows[1]["_review_note"])

    def test_a_reordered_reply_still_lands_correctly(self):
        rows = records()
        review.apply_verdicts(rows, [0, 1], verdict_payload(
            {"index": 1, "value_ok": False, "unit_ok": True, "row_ok": True,
             "issue": "no"},
            {"index": 0, "value_ok": True, "unit_ok": True, "row_ok": True,
             "issue": None},
        ))
        self.assertTrue(rows[0]["_review_value"])
        self.assertFalse(rows[1]["_review_value"])

    def test_a_short_reply_leaves_the_rest_unreviewed(self):
        rows = records()
        applied = review.apply_verdicts(rows, [0, 1], verdict_payload(
            {"index": 0, "value_ok": True, "unit_ok": True, "row_ok": True,
             "issue": None}))
        self.assertEqual(applied, 1)
        self.assertTrue(rows[0]["_review_value"])
        self.assertNotIn("_review_value", rows[1])

    def test_an_unknown_index_is_ignored_rather_than_shifted(self):
        rows = records()
        applied = review.apply_verdicts(rows, [0, 1], verdict_payload(
            {"index": 9, "value_ok": False, "unit_ok": False, "row_ok": False,
             "issue": "x"}))
        self.assertEqual(applied, 0)
        self.assertNotIn("_review_value", rows[0])
        self.assertNotIn("_review_value", rows[1])

    def test_positions_map_indexes_back_to_the_document(self):
        """The second request's index 0 is the document's third record."""
        rows = [dict(doc_id="d", value=i) for i in range(4)]
        review.apply_verdicts(rows, [2, 3], verdict_payload(
            {"index": 0, "value_ok": False, "unit_ok": True, "row_ok": True,
             "issue": "wrong"}))
        self.assertNotIn("_review_value", rows[0])
        self.assertFalse(rows[2]["_review_value"])

    def test_malformed_replies_annotate_nothing(self):
        for payload in ("not json", "[]", json.dumps({"verdicts": "no"}),
                        json.dumps({"verdicts": [{"index": "a"}]})):
            rows = records()
            self.assertEqual(review.apply_verdicts(rows, [0, 1], payload), 0, payload)

    def test_an_abstention_is_not_a_rejection(self):
        rows = records()
        review.apply_verdicts(rows, [0], verdict_payload(
            {"index": 0, "value_ok": None, "unit_ok": None, "row_ok": None,
             "issue": None}))
        self.assertIsNone(rows[0]["_review_value"])
        self.assertEqual(review.summary(rows)["value_rejected"], 0)


class SummaryTest(unittest.TestCase):
    def test_counts_each_kind_of_rejection(self):
        rows = records()
        review.apply_verdicts(rows, [0, 1], verdict_payload(
            {"index": 0, "value_ok": True, "unit_ok": False, "row_ok": True,
             "issue": "mg/mL not ug/mL"},
            {"index": 1, "value_ok": False, "unit_ok": True, "row_ok": False,
             "issue": "mismatched arm"},
        ))
        stats = review.summary(rows)
        self.assertEqual(stats["reviewed"], 2)
        self.assertEqual(stats["unit_rejected"], 1)
        self.assertEqual(stats["value_rejected"], 1)
        self.assertEqual(stats["row_rejected"], 1)
        self.assertEqual(stats["flagged"], 2)

    def test_unreviewed_records_are_not_counted(self):
        self.assertEqual(review.summary(records())["reviewed"], 0)


class ColumnTest(unittest.TestCase):
    def test_review_columns_appear_only_once_a_review_has_run(self):
        plain = record_columns(TEMPLATE)
        reviewed = record_columns(TEMPLATE, reviewed=True)
        self.assertNotIn("_review_row", plain)
        self.assertEqual(reviewed[:len(plain)], plain, "existing columns must not move")
        self.assertEqual(reviewed[len(plain):], list(review.REVIEW_COLUMNS))


class LiveReviewTest(unittest.TestCase):
    def test_records_are_annotated_through_the_provider(self):
        class FakeProvider:
            def complete(self, messages, model, **kwargs):
                class C:
                    text = verdict_payload(
                        {"index": 0, "value_ok": True, "unit_ok": True,
                         "row_ok": True, "issue": None},
                        {"index": 1, "value_ok": False, "unit_ok": True,
                         "row_ok": False, "issue": "arm mismatch"})
                return C()

        rows = records()
        annotated = review.review_document(
            FakeProvider(), rows, "post-vaccination: 7132 post-challenge: 164",
            TEMPLATE, "d", model="scout-gpt-5.1")
        self.assertEqual(annotated, 2)
        self.assertFalse(rows[1]["_review_row"])


class BatchReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.out = self.dir / "out"
        self.out.mkdir()
        self.doc = self.dir / "d.txt"
        self.doc.write_text("post-vaccination: 7132\npost-challenge: 164\n",
                            encoding="utf-8")
        (self.out / "documents").mkdir(parents=True, exist_ok=True)
        (self.out / "documents" / "d.records.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records()) + "\n", encoding="utf-8")
        self.settings = Settings(
            api="aimodelhub", base_url="https://gw", api_key="k",
            model="gpt-5.6-sol", review_model="scout-gpt-5.1",
            cache_dir=str(self.dir / "c"), cache_enabled=False, aggregate=False)
        self.source = SourceDocument(doc_id="d", title="d", path=str(self.doc),
                                     source_name="folder")

    def test_requests_use_the_review_model_and_carry_their_positions(self):
        provider = make_provider()
        lines, tasks = batchapi.build_review_requests(
            provider, [self.source], {"d": records()}, TEMPLATE, self.settings)
        self.assertTrue(lines)
        self.assertEqual(tasks[0].stage, STAGE_REVIEW)
        self.assertEqual(tasks[0].payload["positions"], [0, 1])
        self.assertEqual(lines[0]["body"]["model"], "scout-gpt-5.1")

    def test_a_review_batch_is_marked_as_one(self):
        provider = make_provider(**{"/v1/files": {"id": "f1"},
                                    "/v1/batches": {"id": "b1"}})
        manifest = batchapi.submit_review(
            provider, [self.source], {"d": records()}, TEMPLATE, self.settings,
            self.out)
        self.assertEqual(manifest.kind, KIND_REVIEW)
        self.assertEqual(manifest.model, "scout-gpt-5.1")

    def test_verdicts_are_merged_back_into_the_stored_records(self):
        lines, tasks = batchapi.build_review_requests(
            make_provider(), [self.source], {"d": records()}, TEMPLATE, self.settings)
        manifest = Manifest(batch_id="b1", kind=KIND_REVIEW,
                            template=TEMPLATE.name, tasks=tasks)

        answer = json.dumps({"custom_id": tasks[0].custom_id, "response": {
            "status_code": 200,
            "body": {"output": [{"type": "message", "content": [
                {"type": "output_text", "text": verdict_payload(
                    {"index": 0, "value_ok": True, "unit_ok": True, "row_ok": True,
                     "issue": None},
                    {"index": 1, "value_ok": False, "unit_ok": True, "row_ok": False,
                     "issue": "value belongs to the other arm"})}]}]}}})
        provider = make_provider(**{
            "/v1/batches": {"id": "b1", "status": "completed",
                            "output_file_id": "o1"},
            "/v1/files/o1/content": answer.encode(),
        })

        summary = batchapi.collect_review(provider, manifest, TEMPLATE,
                                          self.settings, self.out)
        self.assertEqual(summary["reviewed"], 2)
        self.assertEqual(summary["flagged"], 1)

        rows = read_csv(self.out / "records.csv")
        self.assertEqual(rows[0]["_review_row"], "true")
        self.assertEqual(rows[1]["_review_row"], "false")
        self.assertIn("other arm", rows[1]["_review_note"])

    def test_a_missing_records_file_is_reported_not_crashed(self):
        manifest = Manifest(batch_id="b1", kind=KIND_REVIEW, template=TEMPLATE.name,
                            tasks=[batchapi.Task(custom_id="v-0", doc_id="ghost",
                                                 stage=STAGE_REVIEW)])
        provider = make_provider(**{
            "/v1/batches": {"id": "b1", "status": "completed", "output_file_id": "o1"},
            "/v1/files/o1/content": b"",
        })
        summary = batchapi.collect_review(provider, manifest, TEMPLATE,
                                          self.settings, self.out)
        self.assertEqual(summary["documents"], 0)
        self.assertTrue(any("ghost" in e for e in summary["errors"]))


if __name__ == "__main__":
    unittest.main()

