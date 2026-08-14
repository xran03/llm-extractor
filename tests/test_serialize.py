"""CSV is the analysis artifact — column order, encoding and content must hold."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_extractor.serialize import (FIGURE_COLUMNS, NO_READING_NOTE, append_records_csv,
                                     figure_rows, read_csv, record_columns,
                                     write_figures_csv, write_records_csv)
from llm_extractor.templates import BUILTIN_TEMPLATES

from ._fakes import DEFAULT_OCR


def sample_records():
    return [
        {"doc_id": "d1", "subject": "group A", "attribute": "concentration",
         "value": 12.5, "unit": "µg/mL", "direction": "higher", "significant": "yes",
         "p_value": "<0.01", "source_span": "Group A reached 12.5 ug/mL",
         "_grounded": True, "_value_grounded": True},
        {"doc_id": "d1", "subject": "group B", "attribute": "concentration",
         "value": None, "unit": None, "notes": "not detected",
         "source_span": "Group B was not detected",
         "_grounded": True, "_value_grounded": None},
    ]


class ColumnLayoutTest(unittest.TestCase):
    def setUp(self):
        self.template = BUILTIN_TEMPLATES["generic"]

    def test_provenance_first_audit_flags_last(self):
        columns = record_columns(self.template)
        self.assertEqual(columns[:2], ["doc_id", "doc_title"])
        self.assertEqual(columns[-4:], ["_grounded", "_value_grounded",
                                        "_unit_grounded", "_ungrounded"])

    def test_every_template_field_has_a_column(self):
        columns = set(record_columns(self.template))
        for name in self.template.field_names:
            self.assertIn(name, columns)

    def test_no_duplicate_columns(self):
        columns = record_columns(self.template)
        self.assertEqual(len(columns), len(set(columns)))

    def test_column_order_is_stable_across_calls(self):
        self.assertEqual(record_columns(self.template), record_columns(self.template))


class WriteRecordsCsvTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "records.csv"
        self.template = BUILTIN_TEMPLATES["generic"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_row_per_record(self):
        write_records_csv(self.path, sample_records(), self.template)
        self.assertEqual(len(read_csv(self.path)), 2)

    def test_values_round_trip(self):
        write_records_csv(self.path, sample_records(), self.template)
        row = read_csv(self.path)[0]
        self.assertEqual(row["subject"], "group A")
        self.assertEqual(row["value"], "12.5")
        self.assertEqual(row["p_value"], "<0.01")

    def test_none_becomes_an_empty_cell(self):
        write_records_csv(self.path, sample_records(), self.template)
        row = read_csv(self.path)[1]
        self.assertEqual(row["value"], "")
        self.assertEqual(row["_value_grounded"], "")

    def test_booleans_are_readable(self):
        write_records_csv(self.path, sample_records(), self.template)
        self.assertEqual(read_csv(self.path)[0]["_grounded"], "true")

    def test_doc_title_is_filled_in(self):
        write_records_csv(self.path, sample_records(), self.template,
                          doc_title="My Report")
        self.assertEqual(read_csv(self.path)[0]["doc_title"], "My Report")

    def test_utf8_bom_is_written_for_excel(self):
        write_records_csv(self.path, sample_records(), self.template)
        self.assertTrue(self.path.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_micro_sign_survives_the_round_trip(self):
        write_records_csv(self.path, sample_records(), self.template)
        self.assertEqual(read_csv(self.path)[0]["unit"], "µg/mL")

    def test_embedded_commas_and_quotes_are_escaped(self):
        records = [{"doc_id": "d", "subject": 'a, b "quoted"',
                    "source_span": "line1\nline2"}]
        write_records_csv(self.path, records, self.template)
        row = read_csv(self.path)[0]
        self.assertEqual(row["subject"], 'a, b "quoted"')
        self.assertEqual(row["source_span"], "line1\nline2")

    def test_empty_records_still_writes_a_header(self):
        write_records_csv(self.path, [], self.template)
        self.assertEqual(read_csv(self.path), [])
        self.assertIn("subject", self.path.read_text(encoding="utf-8-sig"))

    def test_template_change_changes_the_columns(self):
        write_records_csv(self.path, [], BUILTIN_TEMPLATES["immunogenicity"])
        header = self.path.read_text(encoding="utf-8-sig").splitlines()[0]
        self.assertIn("assay", header)
        self.assertNotIn("subject", header)


class AppendRecordsCsvTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "combined.csv"
        self.template = BUILTIN_TEMPLATES["generic"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_appending_accumulates_rows_under_one_header(self):
        append_records_csv(self.path, sample_records(), self.template,
                           write_header=True)
        append_records_csv(self.path, sample_records(), self.template)
        rows = read_csv(self.path)
        self.assertEqual(len(rows), 4)
        self.assertEqual(self.path.read_text(encoding="utf-8-sig").count("doc_title"), 1)

    def test_write_header_truncates_a_previous_run(self):
        append_records_csv(self.path, sample_records(), self.template, write_header=True)
        append_records_csv(self.path, sample_records()[:1], self.template,
                           write_header=True)
        self.assertEqual(len(read_csv(self.path)), 1)


class FiguresCsvTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "figures.csv"
        self.figures = [{"image": "fig1.png", "ocr": DEFAULT_OCR}]

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_row_per_readable_item(self):
        rows = figure_rows(self.figures, "d1", "Report")
        self.assertEqual(len(rows), len(DEFAULT_OCR["items"]))

    def test_axes_and_caption_repeat_on_each_row(self):
        rows = figure_rows(self.figures, "d1")
        self.assertTrue(all(r["axis_y"] == DEFAULT_OCR["axis_y"] for r in rows))

    def test_a_figure_with_no_items_still_appears(self):
        rows = figure_rows([{"image": "blank.png",
                             "ocr": {"figure_type": "empty", "items": []}}], "d1")
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["value"])

    def test_a_figure_that_yielded_nothing_says_so(self):
        # Otherwise the row is indistinguishable from a reading whose values
        # went missing, which is the opposite of what happened.
        rows = figure_rows([{"image": "blank.png",
                             "ocr": {"figure_type": "chart", "items": []}}], "d1")
        self.assertEqual(rows[0]["note"], NO_READING_NOTE)

    def test_the_model_s_own_note_is_not_overwritten(self):
        rows = figure_rows([{"image": "blank.png",
                             "ocr": {"figure_type": "chart", "items": [],
                                     "notes": "axis unreadable"}}], "d1")
        self.assertEqual(rows[0]["note"], "axis unreadable")

    def test_written_columns_match_the_schema(self):
        write_figures_csv(self.path, figure_rows(self.figures, "d1", "Report"))
        rows = read_csv(self.path)
        self.assertEqual(list(rows[0]), list(FIGURE_COLUMNS))
        self.assertEqual(rows[0]["value"], "12.5")

    def test_no_figures_is_no_rows(self):
        self.assertEqual(figure_rows([], "d1"), [])


if __name__ == "__main__":
    unittest.main()
