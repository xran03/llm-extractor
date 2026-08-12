"""Grounding, unit consistency and vision triage — the anti-hallucination layer."""
from __future__ import annotations

import unittest

from llm_extractor.normalize import (annotate_grounding, canon_unit, number_unit_pairs,
                                     span_coverage, span_is_grounded,
                                     unit_agrees_with_span)
from llm_extractor.sources import build_source
from llm_extractor.sources.rest import RestSourceError

DOC = ("Group A reached 12.5 ug/mL at day 28. The GMT was 1.23 ug/mL in the "
       "per-protocol population. A total of 1,165 subjects were enrolled.")


class SpanGroundingTest(unittest.TestCase):
    """A quote counts only when the document really contains it."""

    def test_exact_quote(self):
        self.assertTrue(span_is_grounded("Group A reached 12.5 ug/mL", DOC))

    def test_whitespace_differences_are_tolerated(self):
        self.assertTrue(span_is_grounded("Group   A    reached 12.5", DOC))

    def test_case_differences_are_tolerated(self):
        self.assertTrue(span_is_grounded("group a REACHED 12.5 ug/ml", DOC))

    def test_ocr_damage_is_tolerated(self):
        """'rn' misread for 'm' must not invalidate a genuine quote."""
        self.assertTrue(span_is_grounded("Group A reached 12.5 ug/rnL", DOC))

    def test_fabricated_tail_is_rejected(self):
        """Regression: quoting a real prefix then inventing the rest."""
        self.assertFalse(span_is_grounded(
            "The GMT was 1.23 ug/mL and the hazard ratio was 0.42 with p<0.001", DOC))

    def test_fabricated_number_is_rejected(self):
        self.assertFalse(span_is_grounded("Group A reached 99 ug/mL", DOC))

    def test_substituted_label_is_rejected(self):
        self.assertFalse(span_is_grounded("Group Z reached 12.5 ug/mL", DOC))

    def test_unrelated_text_is_rejected(self):
        self.assertFalse(span_is_grounded("Mice were housed at 22 degrees", DOC))

    def test_empty_span_is_not_grounded(self):
        self.assertFalse(span_is_grounded("", DOC))
        self.assertFalse(span_is_grounded(None, DOC))

    def test_empty_document_grounds_nothing(self):
        self.assertFalse(span_is_grounded("anything", ""))

    def test_coverage_is_higher_for_a_real_quote(self):
        real = span_coverage("Group A reached 12.5 ug/mL", DOC)
        fake = span_coverage("Totally unrelated sentence here", DOC)
        self.assertGreater(real, fake)


class UnitConsistencyTest(unittest.TestCase):
    def test_micro_spellings_canonicalise_together(self):
        self.assertEqual(canon_unit("\u00b5g/mL"), canon_unit("ug / ML"))
        self.assertEqual(canon_unit("\u03bcg/mL"), canon_unit("ug/ml"))

    def test_prefix_is_preserved(self):
        self.assertNotEqual(canon_unit("mg/mL"), canon_unit("ug/mL"))

    def test_number_unit_pairs_are_extracted(self):
        self.assertIn((1.23, "ug/ml"), number_unit_pairs("GMT was 1.23 ug/mL"))

    def test_thousandfold_error_is_caught(self):
        """The failure this check exists for: mg/mL reported for ug/mL."""
        self.assertIs(unit_agrees_with_span(1.23, "mg/mL", "GMT was 1.23 ug/mL"), False)

    def test_matching_unit_agrees(self):
        self.assertIs(unit_agrees_with_span(1.23, "ug/mL", "GMT was 1.23 ug/mL"), True)

    def test_micro_sign_still_agrees(self):
        self.assertIs(unit_agrees_with_span(1.23, "\u00b5g/mL", "GMT was 1.23 ug/mL"), True)

    def test_span_without_a_unit_is_unknown_not_wrong(self):
        self.assertIsNone(unit_agrees_with_span(1.23, "ug/mL", "GMT was 1.23"))

    def test_absent_number_is_unknown(self):
        self.assertIsNone(unit_agrees_with_span(9.9, "ug/mL", "GMT was 1.23 ug/mL"))

    def test_no_declared_unit_is_unknown(self):
        self.assertIsNone(unit_agrees_with_span(1.23, None, "GMT was 1.23 ug/mL"))


class RecordGroundingTest(unittest.TestCase):
    def _annotate(self, record, numeric=("value",), units=("unit",)):
        return annotate_grounding(record, DOC, numeric_fields=numeric, unit_fields=units)

    def test_clean_record_passes_every_check(self):
        record = self._annotate({"source_span": "Group A reached 12.5 ug/mL",
                                 "value": 12.5, "unit": "ug/mL"})
        self.assertTrue(record["_grounded"])
        self.assertTrue(record["_value_grounded"])
        self.assertTrue(record["_unit_grounded"])
        self.assertEqual(record["_ungrounded"], [])

    def test_every_numeric_field_is_checked(self):
        """Regression: only the first populated number used to be validated."""
        record = self._annotate(
            {"source_span": "The GMT was 1.23 ug/mL", "value": 1.23,
             "ci_lower": 9999.0, "unit": "ug/mL"},
            numeric=("value", "ci_lower"))
        self.assertFalse(record["_value_grounded"])
        self.assertIn("ci_lower", record["_ungrounded"])

    def test_wrong_unit_is_flagged(self):
        record = self._annotate({"source_span": "The GMT was 1.23 ug/mL",
                                 "value": 1.23, "unit": "mg/mL"})
        self.assertTrue(record["_value_grounded"], "the number itself is fine")
        self.assertFalse(record["_unit_grounded"])
        self.assertIn("unit", record["_ungrounded"])

    def test_record_without_numbers_reports_not_applicable(self):
        record = self._annotate({"source_span": "Group A reached 12.5 ug/mL",
                                 "value": None, "unit": None})
        self.assertIsNone(record["_value_grounded"])
        self.assertIsNone(record["_unit_grounded"])

    def test_ungrounded_span_fails_its_numbers(self):
        record = self._annotate({"source_span": "Invented sentence entirely",
                                 "value": 12.5, "unit": "ug/mL"})
        self.assertFalse(record["_grounded"])
        self.assertFalse(record["_value_grounded"])

    def test_per_field_unit_column_is_matched(self):
        record = annotate_grounding(
            {"source_span": "The GMT was 1.23 ug/mL", "value": 1.23,
             "value_unit": "mg/mL"},
            DOC, numeric_fields=("value",), unit_fields=("value_unit",))
        self.assertFalse(record["_unit_grounded"])


class SearchParameterTest(unittest.TestCase):
    """`--param query=...` is what the docs show; it must work."""

    def test_query_as_a_string_is_the_search_term(self):
        source = build_source("openalex", query="pneumococcal conjugate")
        self.assertIn("search=pneumococcal+conjugate", source._build_url({}))

    def test_search_parameter_still_works(self):
        source = build_source("openalex", search="pneumococcal conjugate")
        self.assertIn("search=pneumococcal+conjugate", source._build_url({}))

    def test_query_as_a_dict_stays_static_parameters(self):
        source = build_source("openalex", query={"filter": "x"}, search="pcv")
        url = source._build_url({})
        self.assertIn("filter=x", url)
        self.assertIn("search=pcv", url)

    def test_unusable_query_type_is_a_clean_error(self):
        with self.assertRaises(RestSourceError):
            build_source("openalex", query=12345)


class VisionTriageTest(unittest.TestCase):
    """Only pages the text pass could not read are worth a vision call."""

    def setUp(self):
        try:
            import pymupdf
        except ImportError:  # pragma: no cover - optional dependency
            self.skipTest("PyMuPDF is not installed")
        self.pymupdf = pymupdf

    def _prose_page(self, doc):
        """A page of ordinary prose, long enough to look like real body text."""
        page = doc.new_page()
        for line in range(12):
            page.insert_text((60, 80 + line * 14),
                             "Immunogenicity was assessed at day 28 in the cohort. ",
                             fontsize=9)
        return page

    def test_prose_page_is_skipped(self):
        from llm_extractor.ingest import page_needs_vision

        doc = self.pymupdf.open()
        self.assertFalse(page_needs_vision(self._prose_page(doc)))

    def test_page_with_an_embedded_image_is_rendered(self):
        from llm_extractor.ingest import page_needs_vision

        doc = self.pymupdf.open()
        page = self._prose_page(doc)
        pix = self.pymupdf.Pixmap(self.pymupdf.csRGB,
                                  self.pymupdf.IRect(0, 0, 60, 60))
        pix.set_rect(pix.irect, (10, 20, 30))
        page.insert_image(self.pymupdf.Rect(80, 300, 300, 500), pixmap=pix)
        self.assertTrue(page_needs_vision(page))

    def test_page_with_a_vector_chart_is_rendered(self):
        from llm_extractor.ingest import page_needs_vision

        doc = self.pymupdf.open()
        page = self._prose_page(doc)
        for x in range(25):
            page.draw_line((80 + x * 15, 400), (95 + x * 15, 380 - x))
        self.assertTrue(page_needs_vision(page))

    def test_scanned_page_without_a_text_layer_is_rendered(self):
        """A scan has almost no extractable text: it must not be skipped."""
        from llm_extractor.ingest import page_needs_vision

        doc = self.pymupdf.open()
        self.assertTrue(page_needs_vision(doc.new_page()))

    def test_unreadable_page_defaults_to_rendering(self):
        from llm_extractor.ingest import page_needs_vision

        class Broken:
            def get_images(self):
                raise RuntimeError("damaged page")

        self.assertTrue(page_needs_vision(Broken()))


if __name__ == "__main__":
    unittest.main()
