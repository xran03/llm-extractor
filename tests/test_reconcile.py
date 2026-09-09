"""Cross-channel reconciliation: one measurement, one record.

The passes overlap. A titer stated in the prose, plotted in a figure and
measured off that figure's geometry is one fact read three times, and
concatenating the passes inflates every count taken from the corpus — silently,
because each duplicate is individually correct.

These tests pin the two behaviours that make the difference: duplicates
collapse and record who corroborated them, while disagreements survive as
flagged conflicts rather than being quietly resolved.
"""
from __future__ import annotations

import unittest

from llm_extractor.reconcile import (CHANNEL_TRUST, identity, reconcile,
                                     values_agree)
from llm_extractor.templates import load_template


def record(**overrides) -> dict:
    base = {
        "assay": "igg",
        "assay_platform": "elisa",
        "serotype": "6B",
        "group_label": "PCV13",
        "timepoint": "post-dose 3",
        "species": "human infant",
        "value_kind": "geometric_mean",
        "value_unit": "ug/mL",
        "value": 4.5,
        "extraction_mode": "llm",
    }
    base.update(overrides)
    return base


class TemplateBackedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = load_template("templates/conjugate-titer.json")


class DuplicateTest(TemplateBackedTest):
    def test_the_same_value_from_two_channels_becomes_one_record(self):
        records, stats = reconcile(
            [record(extraction_mode="llm"), record(extraction_mode="ocr")],
            self.template)
        self.assertEqual(len(records), 1)
        self.assertEqual(stats["duplicates_merged"], 1)

    def test_the_more_reliable_channel_survives(self):
        records, _ = reconcile(
            [record(extraction_mode="ocr"), record(extraction_mode="llm")],
            self.template)
        self.assertEqual(records[0]["extraction_mode"], "llm")

    def test_agreement_between_channels_is_kept_as_evidence(self):
        records, stats = reconcile(
            [record(extraction_mode="llm"), record(extraction_mode="ocr")],
            self.template)
        self.assertEqual(records[0]["corroborated_by"], "ocr")
        self.assertEqual(stats["corroborated"], 1)

    def test_rounding_between_channels_still_counts_as_agreement(self):
        records, _ = reconcile(
            [record(value=4.5, extraction_mode="llm"),
             record(value=4.52, extraction_mode="ocr")], self.template)
        self.assertEqual(len(records), 1)

    def test_a_measurement_seen_once_is_untouched(self):
        records, stats = reconcile([record()], self.template)
        self.assertEqual(len(records), 1)
        self.assertEqual(stats["duplicates_merged"], 0)
        self.assertNotIn("corroborated_by", records[0])


class NotDuplicatesTest(TemplateBackedTest):
    """Things that look alike and are not the same measurement."""

    def test_different_serotypes_are_different_measurements(self):
        records, _ = reconcile(
            [record(serotype="6B"), record(serotype="19F")], self.template)
        self.assertEqual(len(records), 2)

    def test_different_timepoints_are_different_measurements(self):
        records, _ = reconcile(
            [record(timepoint="pre"), record(timepoint="post-dose 3")], self.template)
        self.assertEqual(len(records), 2)

    def test_different_platforms_are_different_measurements(self):
        """An ELISA GMC and a dLIA GMC are not the same quantity."""
        records, _ = reconcile(
            [record(assay_platform="elisa", extraction_mode="llm"),
             record(assay_platform="luminex", extraction_mode="ocr")], self.template)
        self.assertEqual(len(records), 2)

    def test_a_mean_and_an_individual_are_different_facts(self):
        records, _ = reconcile(
            [record(value_kind="geometric_mean"),
             record(value_kind="individual")], self.template)
        self.assertEqual(len(records), 2)

    def test_different_units_are_not_merged(self):
        records, _ = reconcile(
            [record(value_unit="ug/mL"), record(value_unit="OPA titer")],
            self.template)
        self.assertEqual(len(records), 2)


class DistributionTest(TemplateBackedTest):
    """A group of subjects shares every key field. That is the data, not a bug."""

    def test_individual_points_are_never_collapsed_into_one(self):
        points = [record(value_kind="individual", value=v,
                         extraction_mode="vector_geometry")
                  for v in (0.4, 1.2, 3.5, 8.0, 12.5, 40.0)]
        records, _ = reconcile(points, self.template)
        self.assertEqual(len(records), 6)

    def test_a_distribution_is_not_reported_as_a_conflict(self):
        points = [record(value_kind="individual", value=v,
                         extraction_mode="vector_geometry")
                  for v in (0.4, 1.2, 3.5)]
        points.append(record(value_kind="individual", value=9.9,
                             extraction_mode="ocr"))
        _, stats = reconcile(points, self.template)
        self.assertEqual(stats["conflicts_flagged"], 0)

    def test_two_subjects_sharing_a_value_collapse_only_across_channels(self):
        """Geometry measured it; the vision pass guessed the same number."""
        records, stats = reconcile(
            [record(value_kind="individual", value=3.5,
                    extraction_mode="vector_geometry"),
             record(value_kind="individual", value=3.5, extraction_mode="ocr")],
            self.template)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["extraction_mode"], "vector_geometry")


class ConflictTest(TemplateBackedTest):
    """A disagreement is a finding, not something to resolve quietly."""

    def setUp(self):
        self.records, self.stats = reconcile(
            [record(value=4.5, extraction_mode="llm"),
             record(value=9.1, extraction_mode="ocr")], self.template)

    def test_both_readings_are_kept(self):
        self.assertEqual(len(self.records), 2)

    def test_the_disagreement_is_flagged(self):
        self.assertTrue(all(r["modality_conflict"] == "value_mismatch"
                            for r in self.records))
        self.assertEqual(self.stats["conflicts_flagged"], 2)

    def test_the_note_names_the_other_reading(self):
        text = self.records[0]["notes"]
        self.assertIn("read", text)
        self.assertIn("9.1", text + self.records[1]["notes"])

    def test_agreement_is_not_flagged(self):
        _, stats = reconcile(
            [record(value=4.5, extraction_mode="llm"),
             record(value=4.5, extraction_mode="ocr")], self.template)
        self.assertEqual(stats["conflicts_flagged"], 0)

    def test_one_channel_reporting_twice_is_not_a_conflict(self):
        """Repeated measurements within a pass are data, not contradiction."""
        _, stats = reconcile(
            [record(value=4.5, extraction_mode="llm"),
             record(value=9.1, extraction_mode="llm")], self.template)
        self.assertEqual(stats["conflicts_flagged"], 0)


class TrustOrderTest(unittest.TestCase):
    def test_measurement_outranks_quotation_outranks_estimate(self):
        self.assertGreater(CHANNEL_TRUST["vector_geometry"], CHANNEL_TRUST["llm"])
        self.assertGreater(CHANNEL_TRUST["llm"], CHANNEL_TRUST["ocr"])

    def test_an_unknown_channel_never_wins(self):
        self.assertEqual(CHANNEL_TRUST["na"], 0)


class IdentityTest(unittest.TestCase):
    def test_evidence_is_not_part_of_identity(self):
        """The old chunk dedupe keyed on the span, so channels never matched."""
        keys = ["assay", "serotype"]
        quoted = record(source_span="GMC was 4.5 ug/mL")
        measured = record(source_span="log axis calibrated on ticks [...]")
        self.assertEqual(identity(quoted, keys), identity(measured, keys))

    def test_case_and_padding_do_not_split_a_measurement(self):
        keys = ["serotype"]
        self.assertEqual(identity(record(serotype=" 6B "), keys),
                         identity(record(serotype="6b"), keys))


class ToleranceTest(unittest.TestCase):
    def test_close_values_agree(self):
        self.assertTrue(values_agree(100.0, 101.0))

    def test_distant_values_do_not(self):
        self.assertFalse(values_agree(100.0, 140.0))

    def test_missing_values_agree_only_with_each_other(self):
        self.assertTrue(values_agree(None, None))
        self.assertFalse(values_agree(None, 1.0))

    def test_booleans_are_not_numbers(self):
        self.assertFalse(values_agree(True, 1.0))


if __name__ == "__main__":                      # pragma: no cover
    unittest.main()
