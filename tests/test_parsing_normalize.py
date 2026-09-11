"""JSON recovery from imperfect model output, and value normalization/grounding."""
from __future__ import annotations

import unittest

from llm_extractor.normalize import (annotate_grounding, canon_category, coerce_record,
                                     fix_units, normalize_p_value, number_unit_pairs,
                                     numbers_in_text, parse_number, span_is_grounded,
                                     token_set_similarity, value_supported_by_span)
from llm_extractor.parsing import extract_json_array, extract_json_object, strip_fences
from llm_extractor.templates import BUILTIN_TEMPLATES


class ParsingTest(unittest.TestCase):
    def test_bare_array(self):
        self.assertEqual(extract_json_array('[{"a": 1}]'), [{"a": 1}])

    def test_records_envelope_is_preferred(self):
        payload = '{"other": [1], "records": [{"a": 1}]}'
        self.assertEqual(extract_json_array(payload), [{"a": 1}])

    def test_fenced_json_block(self):
        self.assertEqual(extract_json_array('```json\n[{"a": 1}]\n```'), [{"a": 1}])

    def test_array_embedded_in_prose(self):
        self.assertEqual(extract_json_array('Here you go: [{"a": 1}] done'), [{"a": 1}])

    def test_truncated_array_is_salvaged(self):
        truncated = '[{"a": 1}, {"a": 2}, {"a": 3'
        self.assertEqual(extract_json_array(truncated), [{"a": 1}, {"a": 2}])

    def test_single_object_becomes_one_record(self):
        self.assertEqual(extract_json_array('{"a": 1}'), [{"a": 1}])

    def test_empty_output_is_empty_list(self):
        self.assertEqual(extract_json_array("   "), [])

    def test_none_raises(self):
        with self.assertRaises(ValueError):
            extract_json_array(None)

    def test_unparsable_raises(self):
        with self.assertRaises(ValueError):
            extract_json_array("no json at all here")

    def test_extract_json_object(self):
        self.assertEqual(extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json_object("nothing"), {})

    def test_strip_fences(self):
        self.assertEqual(strip_fences("```json\n{}\n```"), "{}")


class UnitAndNumberTest(unittest.TestCase):
    def test_micro_sign_repair(self):
        self.assertEqual(fix_units("12.5 \ufffdg/mL"), "12.5 µg/mL")
        self.assertEqual(fix_units("5 ug/ml"), "5 µg/mL")

    def test_greek_mu_is_normalized(self):
        self.assertEqual(fix_units("3 \u03bcg/mL"), "3 µg/mL")

    def test_none_passes_through(self):
        self.assertIsNone(fix_units(None))

    def test_parse_number_variants(self):
        self.assertEqual(parse_number("1,165"), 1165.0)
        self.assertEqual(parse_number("<0.01"), 0.01)
        self.assertEqual(parse_number("2.05e3"), 2050.0)
        self.assertEqual(parse_number(7), 7.0)
        self.assertIsNone(parse_number("no digits"))
        self.assertIsNone(parse_number(None))
        self.assertIsNone(parse_number(True))

    def test_canon_category_defaults_blanks(self):
        self.assertEqual(canon_category(None, ["a"]), "na")
        self.assertEqual(canon_category("  A ", ["a"]), "a")

    def test_unexpected_category_is_preserved_not_dropped(self):
        self.assertEqual(canon_category("weird", ["a", "b"]), "weird")


class CoercionTest(unittest.TestCase):
    def setUp(self):
        self.template = BUILTIN_TEMPLATES["generic"]

    def test_unknown_fields_are_dropped(self):
        record = coerce_record({"subject": "A", "bogus": 1}, self.template, "d1")
        self.assertNotIn("bogus", record)
        self.assertEqual(record["doc_id"], "d1")

    def test_missing_fields_become_none(self):
        record = coerce_record({"subject": "A"}, self.template, "d1")
        self.assertIsNone(record["unit"])

    def test_numbers_are_parsed(self):
        record = coerce_record({"value": "1,165"}, self.template, "d1")
        self.assertEqual(record["value"], 1165.0)

    def test_integer_fields_are_ints(self):
        template = BUILTIN_TEMPLATES["immunogenicity"]
        record = coerce_record({"valency": "20 valent"}, template, "d1")
        self.assertEqual(record["valency"], 20)
        self.assertIsInstance(record["valency"], int)

    def test_enum_fields_are_lowercased(self):
        record = coerce_record({"direction": "HIGHER"}, self.template, "d1")
        self.assertEqual(record["direction"], "higher")

    def test_units_are_repaired_in_text_fields(self):
        record = coerce_record({"unit": "\ufffdg/mL"}, self.template, "d1")
        self.assertEqual(record["unit"], "µg/mL")


class PValueNormalizationTest(unittest.TestCase):
    """The schema asks for the value; models return the sentence's phrasing.

    A consumer filtering on significance parses `<0.05`, not `P <0.05`, so the
    label has to come off before the row is written.
    """

    def test_the_p_label_and_equals_sign_are_removed(self):
        self.assertEqual(normalize_p_value("P = 0.0005"), "0.0005")
        self.assertEqual(normalize_p_value("p=0.0228"), "0.0228")

    def test_a_comparator_is_kept(self):
        self.assertEqual(normalize_p_value("P <0.05"), "<0.05")
        self.assertEqual(normalize_p_value("p < 0.0005"), "<0.0005")
        self.assertEqual(normalize_p_value(">0.99"), ">0.99")

    def test_unicode_comparators_become_ascii(self):
        self.assertEqual(normalize_p_value("p \u2264 0.01"), "<=0.01")
        self.assertEqual(normalize_p_value("P \u2265 0.2"), ">=0.2")

    def test_a_bare_value_is_unchanged(self):
        self.assertEqual(normalize_p_value("0.0039"), "0.0039")
        self.assertEqual(normalize_p_value("<0.001"), "<0.001")

    def test_an_unparseable_value_is_kept_rather_than_dropped(self):
        self.assertEqual(normalize_p_value("not significant"), "not significant")

    def test_empty_input(self):
        self.assertIsNone(normalize_p_value(None))
        self.assertIsNone(normalize_p_value("   "))

    def test_coercion_applies_it_to_the_record(self):
        template = BUILTIN_TEMPLATES["immunogenicity"]
        record = coerce_record({"p_value": "P = 0.0005"}, template, "d")
        self.assertEqual(record["p_value"], "0.0005")


class GroundingTest(unittest.TestCase):
    def setUp(self):
        self.text = "Group A reached 12.5 ug/mL, higher than group B (p<0.01)."

    def test_span_present_in_document_is_grounded(self):
        self.assertTrue(span_is_grounded("Group A reached 12.5 ug/mL", self.text))

    def test_span_absent_from_document_is_not_grounded(self):
        self.assertFalse(span_is_grounded("Group Z reached 99 ug/mL", self.text))

    def test_missing_span_is_not_grounded(self):
        self.assertFalse(span_is_grounded(None, self.text))

    def test_whitespace_differences_are_tolerated(self):
        self.assertTrue(span_is_grounded("Group   A    reached 12.5", self.text))

    def test_value_must_appear_in_its_span(self):
        self.assertTrue(value_supported_by_span(12.5, "reached 12.5 ug/mL"))
        self.assertFalse(value_supported_by_span(99.0, "reached 12.5 ug/mL"))

    def test_thousands_separators_are_handled(self):
        self.assertTrue(value_supported_by_span(1165.0, "titer of 1,165"))

    def test_comma_separated_list_is_read_as_separate_numbers(self):
        """A compact list must not be merged into one oversized number.

        Time series and tabulated readings are often written without spaces
        after the comma, and every entry in such a list is a value a record may
        legitimately cite as its evidence.
        """
        self.assertEqual(numbers_in_text("5,094,5,227,5,296,5,325"),
                         [5094.0, 5227.0, 5296.0, 5325.0])

    def test_value_quoted_from_a_comma_separated_list_is_grounded(self):
        span = "Taw = 1,365, 4,484, 3,386, 4,088"
        for value in (1365.0, 4484.0, 3386.0, 4088.0):
            self.assertTrue(value_supported_by_span(value, span))
        self.assertFalse(value_supported_by_span(1365448433864088.0, span))

    def test_thousands_groups_must_be_three_digits(self):
        self.assertEqual(numbers_in_text("1, 2, and 5 seconds"), [1.0, 2.0, 5.0])
        self.assertEqual(numbers_in_text("1,234.56"), [1234.56])
        self.assertEqual(numbers_in_text("-1,234"), [-1234.0])
        self.assertEqual(numbers_in_text("2.05e3"), [2050.0])

    def test_hyphenated_range_is_two_positive_numbers(self):
        """The hyphen in a written range is a separator, not a minus sign.

        Ranges and confidence intervals are routinely typed with a plain
        hyphen, so reading it as a sign turns the upper bound negative and the
        bound is then absent from the very span that states it.
        """
        self.assertEqual(numbers_in_text("10-15%"), [10.0, 15.0])
        self.assertEqual(numbers_in_text("95% CI 1.2-3.4"), [95.0, 1.2, 3.4])
        self.assertEqual(numbers_in_text("1.2\u20133.4"), [1.2, 3.4])

    def test_a_genuine_minus_sign_still_parses(self):
        self.assertEqual(numbers_in_text("a drop of -15%"), [-15.0])
        self.assertEqual(numbers_in_text("(-15)"), [-15.0])
        self.assertEqual(numbers_in_text("-0.5 to -0.1"), [-0.5, -0.1])
        self.assertEqual(numbers_in_text("2.05e-3"), [0.00205])

    def test_range_bounds_are_grounded_by_the_span_that_states_them(self):
        span = "lots were targeted to a degree of activation of 10-15%"
        self.assertTrue(value_supported_by_span(10.0, span))
        self.assertTrue(value_supported_by_span(15.0, span))

    def test_units_are_paired_with_numbers_in_a_list(self):
        self.assertEqual(number_unit_pairs("doses of 1,000 mg"), [(1000.0, "mg")])

    def test_annotate_flags_a_real_record(self):
        record = {"value": 12.5, "source_span": "Group A reached 12.5 ug/mL"}
        annotate_grounding(record, self.text)
        self.assertTrue(record["_grounded"])
        self.assertTrue(record["_value_grounded"])

    def test_annotate_flags_a_fabricated_value(self):
        record = {"value": 99.0, "source_span": "Group A reached 12.5 ug/mL"}
        annotate_grounding(record, self.text)
        self.assertTrue(record["_grounded"])
        self.assertFalse(record["_value_grounded"])

    def test_annotate_flags_a_fabricated_span(self):
        record = {"value": 12.5, "source_span": "text that is not in the document"}
        annotate_grounding(record, self.text)
        self.assertFalse(record["_grounded"])
        self.assertFalse(record["_value_grounded"])

    def test_non_numeric_record_has_value_grounding_none(self):
        record = {"source_span": "Group A reached 12.5 ug/mL"}
        annotate_grounding(record, self.text)
        self.assertIsNone(record["_value_grounded"])


class SimilarityTest(unittest.TestCase):
    def test_identical_labels(self):
        self.assertEqual(token_set_similarity("group A", "Group  A"), 1.0)

    def test_disjoint_labels(self):
        self.assertEqual(token_set_similarity("alpha", "beta"), 0.0)

    def test_partial_overlap(self):
        self.assertGreater(token_set_similarity("250 kDa lot", "250kDa"), 0.0)

    def test_both_empty_is_equal(self):
        self.assertEqual(token_set_similarity("", ""), 1.0)


if __name__ == "__main__":
    unittest.main()
