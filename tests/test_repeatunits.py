"""Repeat-unit resolution and the derived alignment column."""
from __future__ import annotations

import unittest

from llm_extractor import repeatunits
from llm_extractor.repeatunits import (annotate_records, load_table, lookup,
                                       normalise_serotype)
from llm_extractor.serialize import record_columns
from llm_extractor.templates import BUILTIN_TEMPLATES


class NormaliseTest(unittest.TestCase):
    def test_spellings_reach_the_same_token(self):
        for written in ("19F", "19f", "19 F", "serotype 19F", "Pn19F", "type-19f"):
            with self.subTest(written=written):
                self.assertEqual(normalise_serotype(written), "19F")

    def test_plain_numbers_keep_their_form(self):
        self.assertEqual(normalise_serotype("3"), "3")
        self.assertEqual(normalise_serotype(" 3 "), "3")

    def test_multi_letter_suffix_keeps_danish_casing(self):
        self.assertEqual(normalise_serotype("6eb"), "6Eb")

    def test_non_serotypes_resolve_to_nothing(self):
        for value in (None, "", "na", "control group", "CRM197", "adjuvant"):
            with self.subTest(value=value):
                self.assertEqual(normalise_serotype(value), "")


class TableTest(unittest.TestCase):
    def test_the_shipped_table_loads(self):
        table = load_table()
        self.assertGreater(len(table), 50)

    def test_common_serotypes_resolve(self):
        for serotype in ("1", "3", "4", "6B", "14", "19F", "23F"):
            with self.subTest(serotype=serotype):
                entry = lookup(serotype)
                self.assertIsNotNone(entry, f"{serotype} missing from the table")
                self.assertTrue(entry["repeat_unit"].strip())

    def test_distinct_serotypes_have_distinct_units(self):
        """The column is only worth having if it varies by serotype."""
        units = {s: lookup(s)["repeat_unit"] for s in ("1", "3", "6B", "19F", "23F")}
        self.assertEqual(len(set(units.values())), len(units), units)

    def test_an_unknown_serotype_is_not_guessed(self):
        self.assertIsNone(lookup("999Z"))


class AnnotateTest(unittest.TestCase):
    def test_a_serotype_record_gains_its_repeat_unit(self):
        records = [{"serotype": "19F", "value": 12.5}]
        stats = annotate_records(records)
        self.assertEqual(stats["annotated"], 1)
        self.assertTrue(records[0]["repeat_unit"])
        self.assertIn("reference table", records[0]["repeat_unit_source"])

    def test_a_record_without_a_serotype_is_left_empty(self):
        records = [{"value": 1.0}]
        annotate_records(records)
        self.assertIsNone(records[0]["repeat_unit"])
        self.assertIsNone(records[0]["repeat_unit_source"])

    def test_na_is_treated_as_absent(self):
        records = [{"serotype": "na"}]
        stats = annotate_records(records)
        self.assertEqual(stats["no_serotype"], 1)

    def test_an_unknown_serotype_is_counted_not_guessed(self):
        records = [{"serotype": "999Z"}]
        stats = annotate_records(records)
        self.assertEqual(stats["unknown_serotype"], 1)
        self.assertIsNone(records[0]["repeat_unit"])

    def test_a_unit_read_from_the_document_wins(self):
        """A paper reporting a structure is more current than a static table."""
        records = [{"serotype": "19F", "repeat_unit": "novel unit from the paper"}]
        stats = annotate_records(records)
        self.assertEqual(records[0]["repeat_unit"], "novel unit from the paper")
        self.assertEqual(records[0]["repeat_unit_source"], "document")
        self.assertEqual(stats["from_document"], 1)

    def test_different_serotypes_get_different_units(self):
        records = [{"serotype": "3"}, {"serotype": "19F"}]
        annotate_records(records)
        self.assertNotEqual(records[0]["repeat_unit"], records[1]["repeat_unit"])

    def test_non_dict_rows_are_ignored(self):
        records = ["not a record", {"serotype": "3"}]
        annotate_records(records)
        self.assertTrue(records[1]["repeat_unit"])


class ColumnTest(unittest.TestCase):
    def test_repeat_unit_is_a_csv_column(self):
        columns = record_columns(BUILTIN_TEMPLATES["immunogenicity"])
        self.assertIn("repeat_unit", columns)
        self.assertIn("repeat_unit_source", columns)

    def test_derived_columns_sit_before_the_audit_flags(self):
        columns = record_columns(BUILTIN_TEMPLATES["immunogenicity"])
        self.assertLess(columns.index("repeat_unit"), columns.index("_grounded"))

    def test_no_column_is_duplicated(self):
        for template in BUILTIN_TEMPLATES.values():
            with self.subTest(template=template.name):
                columns = record_columns(template)
                self.assertEqual(len(columns), len(set(columns)))


if __name__ == "__main__":
    unittest.main()
