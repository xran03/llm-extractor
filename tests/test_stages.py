"""Extraction, OCR and the aggregation agent as isolated stages."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_extractor.agent import aggregate, deterministic_aggregate
from llm_extractor.extract import (chunk_text, dedupe, extract_records,
                                   grounding_summary)
from llm_extractor.ingest import Document, load_document
from llm_extractor.ocr import ocr_document, ocr_figure, ocr_summary, ocr_to_text
from llm_extractor.templates import BUILTIN_TEMPLATES

from ._fakes import SAMPLE_TEXT, FakeProvider, write_png


class ChunkingTest(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("short"), ["short"])

    def test_empty_text_is_no_chunks(self):
        self.assertEqual(chunk_text("   "), [])

    def test_long_text_is_split_with_overlap(self):
        text = ("paragraph\n\n" * 4000)
        chunks = chunk_text(text, size=5000, overlap=200)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 5000 for c in chunks))

    def test_chunks_cover_the_whole_document(self):
        text = "".join(f"line {i}\n\n" for i in range(3000))
        joined = "".join(chunk_text(text, size=4000, overlap=100))
        self.assertIn("line 0", joined)
        self.assertIn("line 2999", joined)


class ExtractTest(unittest.TestCase):
    def setUp(self):
        self.template = BUILTIN_TEMPLATES["generic"]
        self.document = Document(doc_id="d1", text=SAMPLE_TEXT)
        self.provider = FakeProvider()

    def test_records_are_extracted_and_coerced(self):
        result = extract_records(self.provider, self.document, self.template, model="m")
        self.assertEqual(len(result.records), 2)
        self.assertEqual(result.records[0]["doc_id"], "d1")
        self.assertEqual(result.records[0]["unit"], "µg/mL")

    def test_grounding_flags_are_attached(self):
        result = extract_records(self.provider, self.document, self.template, model="m")
        self.assertTrue(all(r["_grounded"] for r in result.records))
        self.assertTrue(all(r["_value_grounded"] for r in result.records))

    def test_fabricated_value_is_flagged(self):
        provider = FakeProvider(records=[{
            "subject": "ghost", "value": 999,
            "source_span": "Group A reached 12.5 ug/mL, higher than group B (p<0.01).",
        }])
        result = extract_records(provider, self.document, self.template, model="m")
        self.assertFalse(result.records[0]["_value_grounded"])

    def test_json_schema_is_sent_to_the_provider(self):
        extract_records(self.provider, self.document, self.template, model="m")
        schema = self.provider.stage_calls("extract")[0]["json_schema"]
        self.assertEqual(schema["name"], "generic_records")

    def test_usage_is_accumulated(self):
        result = extract_records(self.provider, self.document, self.template, model="m")
        self.assertEqual(result.prompt_tokens, 100)
        self.assertEqual(result.completion_tokens, 50)

    def test_long_document_makes_multiple_parallel_calls(self):
        document = Document(doc_id="big", text="paragraph\n\n" * 20000)
        result = extract_records(self.provider, document, self.template, model="m",
                                 chunk_chars=20000)
        self.assertGreater(result.chunks, 1)
        self.assertEqual(len(self.provider.stage_calls("extract")), result.chunks)

    def test_duplicates_from_overlapping_chunks_are_removed(self):
        document = Document(doc_id="big", text=SAMPLE_TEXT * 500)
        result = extract_records(self.provider, document, self.template, model="m",
                                 chunk_chars=2000)
        self.assertEqual(len(result.records), 2)

    def test_unparsable_output_is_recorded_not_raised(self):
        class Broken(FakeProvider):
            def complete(self, messages, model, **kwargs):
                completion = super().complete(messages, model, **kwargs)
                completion.text = "not json"
                return completion

        result = extract_records(Broken(), self.document, self.template, model="m")
        self.assertEqual(result.records, [])
        self.assertTrue(result.errors)

    def test_empty_document_makes_no_call(self):
        result = extract_records(self.provider, Document(doc_id="d", text=""),
                                 self.template, model="m")
        self.assertEqual(result.records, [])
        self.assertEqual(self.provider.calls, [])

    def test_dedupe_keeps_distinct_records(self):
        records = [
            {"subject": "a", "attribute": "x", "source_span": "s1", "value": 1},
            {"subject": "a", "attribute": "x", "source_span": "s1", "value": 1},
            {"subject": "a", "attribute": "x", "source_span": "s2", "value": 2},
        ]
        self.assertEqual(len(dedupe(records, self.template)), 2)

    def test_grounding_summary_counts(self):
        records = [
            {"_grounded": True, "_value_grounded": True},
            {"_grounded": False, "_value_grounded": False},
            {"_grounded": True, "_value_grounded": None},
        ]
        summary = grounding_summary(records)
        self.assertEqual(summary["records"], 3)
        self.assertEqual(summary["grounded"], 2)
        self.assertEqual(summary["values_ungrounded"], 1)


class OcrTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.provider = FakeProvider()

    def tearDown(self):
        self.tmp.cleanup()

    def test_figure_is_transcribed_to_json(self):
        outcome = ocr_figure(self.provider, write_png(self.dir), model="v", doc_id="d")
        self.assertEqual(outcome["ocr"]["figure_type"], "chart")
        self.assertEqual(len(outcome["ocr"]["items"]), 2)

    def test_ocr_json_schema_is_requested(self):
        ocr_figure(self.provider, write_png(self.dir), model="v")
        self.assertEqual(self.provider.calls[0]["json_schema"]["name"], "figure_ocr")

    def test_image_is_attached_to_the_message(self):
        ocr_figure(self.provider, write_png(self.dir), model="v")
        parts = self.provider.calls[0]["messages"][0]["content"]
        self.assertTrue(any(p["type"] == "image" for p in parts))

    def test_all_figures_are_read(self):
        document = load_document(write_png(self.dir, "a.png"))
        document.figures = [write_png(self.dir, "a.png"), write_png(self.dir, "b.png")]
        result = ocr_document(self.provider, document, model="v")
        self.assertEqual(len(result.figures), 2)

    def test_one_bad_figure_does_not_fail_the_document(self):
        document = Document(doc_id="d", text="", figures=[self.dir / "missing.png"])
        result = ocr_document(self.provider, document, model="v")
        self.assertEqual(result.figures, [])
        self.assertTrue(result.errors)

    def test_no_figures_means_no_calls(self):
        result = ocr_document(self.provider, Document(doc_id="d", text=""), model="v")
        self.assertEqual(result.figures, [])
        self.assertEqual(self.provider.calls, [])

    def test_partial_payload_is_normalized(self):
        provider = FakeProvider(ocr={"figure_type": "table"})
        outcome = ocr_figure(provider, write_png(self.dir), model="v")
        self.assertEqual(outcome["ocr"]["items"], [])
        self.assertEqual(outcome["ocr"]["text_blocks"], [])

    def test_ocr_to_text_is_readable(self):
        outcome = ocr_figure(self.provider, write_png(self.dir), model="v")
        text = ocr_to_text([outcome])
        self.assertIn("group A", text)
        self.assertIn("12.5", text)

    def test_ocr_summary_counts_numeric_items(self):
        outcome = ocr_figure(self.provider, write_png(self.dir), model="v")
        self.assertEqual(ocr_summary([outcome])["numeric_items"], 2)


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.provider = FakeProvider()
        self.records = [{"subject": "A", "_grounded": True, "source_span": "span"}]
        self.figures = [{"image": "f.png", "ocr": {
            "items": [{"label": "group C", "value": 7.25, "unit": "ug/mL"}]}}]

    def test_agent_returns_the_aggregate_envelope(self):
        result = aggregate(self.provider, "d1", self.records, self.figures, model="m")
        self.assertIn("summary", result)
        self.assertEqual(result["figure_insights"][0]["value"], 7.25)

    def test_agent_uses_the_aggregate_schema(self):
        aggregate(self.provider, "d1", self.records, self.figures, model="m")
        call = self.provider.stage_calls("aggregate")[0]
        self.assertEqual(call["json_schema"]["name"], "document_aggregate")

    def test_agent_receives_both_views_as_json(self):
        aggregate(self.provider, "d1", self.records, self.figures, model="m")
        content = self.provider.stage_calls("aggregate")[0]["messages"][1]["content"]
        self.assertIn("TEXT_RECORDS", content)
        self.assertIn("FIGURE_OCR", content)

    def test_agent_failure_falls_back_to_deterministic(self):
        provider = FakeProvider(fail_stages={"aggregate"})
        result = aggregate(provider, "d1", self.records, self.figures, model="m")
        self.assertTrue(result["summary"])
        self.assertTrue(any("unavailable" in gap for gap in result["coverage_gaps"]))

    def test_no_inputs_means_no_call(self):
        result = aggregate(self.provider, "d1", [], [], model="m")
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(result["summary"], "")

    def test_deterministic_aggregate_needs_no_provider(self):
        result = deterministic_aggregate(self.records, self.figures)
        self.assertIn("1 records extracted", result["summary"])
        self.assertEqual(result["figure_insights"][0]["value"], 7.25)

    def test_deterministic_aggregate_reports_ungrounded_values(self):
        result = deterministic_aggregate(
            [{"_grounded": True, "_value_grounded": False, "value": 1}], [])
        self.assertTrue(result["coverage_gaps"])

    def test_agent_prompt_drops_internal_flags(self):
        aggregate(self.provider, "d1", self.records, self.figures, model="m")
        content = self.provider.stage_calls("aggregate")[0]["messages"][1]["content"]
        payload = json.loads(content[content.index("{"):])
        self.assertNotIn("_grounded", payload["TEXT_RECORDS"][0])


if __name__ == "__main__":
    unittest.main()
