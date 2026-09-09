"""The shipped templates and demo output are part of the contract — verify them."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from llm_extractor.serialize import read_csv
from llm_extractor.templates import load_template, validate_template_dict

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "templates"
DEMO = REPO / "demo"
RESULTS = DEMO / "results"


class ShippedTemplateTest(unittest.TestCase):
    def test_example_templates_exist(self):
        self.assertTrue(sorted(TEMPLATES.glob("*.json")), "no example templates shipped")

    def test_every_example_template_is_valid(self):
        for path in sorted(TEMPLATES.glob("*.json")):
            with self.subTest(template=path.name):
                data = json.loads(path.read_text(encoding="utf-8"))
                validate_template_dict(data)

    def test_every_example_template_builds_a_strict_schema(self):
        for path in sorted(TEMPLATES.glob("*.json")):
            with self.subTest(template=path.name):
                schema = load_template(str(path)).json_schema()
                self.assertTrue(schema["strict"])
                item = schema["schema"]["properties"]["records"]["items"]
                self.assertFalse(item["additionalProperties"])
                self.assertEqual(sorted(item["required"]), sorted(item["properties"]))

    def test_every_example_template_requires_evidence(self):
        for path in sorted(TEMPLATES.glob("*.json")):
            with self.subTest(template=path.name):
                self.assertIn("source_span", load_template(str(path)).field_names)


class DemoAssetTest(unittest.TestCase):
    def test_demo_inputs_are_present(self):
        self.assertTrue((DEMO / "naca-report-1372-excerpt.pdf").is_file())
        self.assertTrue((DEMO / "naca-figure-2.png").is_file())

    def test_demo_documents_the_source_and_its_licence(self):
        readme = (DEMO / "README.md").read_text(encoding="utf-8")
        self.assertIn("public domain", readme.lower())
        self.assertIn("ntrs.nasa.gov", readme)

    def test_demo_inputs_stay_small(self):
        for name in ("naca-report-1372-excerpt.pdf", "naca-figure-2.png"):
            size_kb = (DEMO / name).stat().st_size / 1024
            self.assertLess(size_kb, 600, f"{name} is {size_kb:.0f} KB")


class DemoResultsTest(unittest.TestCase):
    """Reference output must stay consistent with the code that writes it."""

    def test_headline_table_exists_and_parses(self):
        rows = read_csv(RESULTS / "records.csv")
        self.assertTrue(rows)

    def test_every_reference_record_is_grounded(self):
        for row in read_csv(RESULTS / "records.csv"):
            self.assertEqual(row["_grounded"], "true", row.get("attribute"))

    def test_no_reference_value_is_ungrounded(self):
        for row in read_csv(RESULTS / "records.csv"):
            self.assertNotEqual(row["_value_grounded"], "false", row.get("attribute"))

    def test_figures_table_holds_values_absent_from_the_text(self):
        rows = read_csv(RESULTS / "figures.csv")
        self.assertTrue(rows)
        self.assertTrue(any(r["value"] for r in rows))

    def test_document_json_carries_all_views(self):
        payload = json.loads(
            (RESULTS / "naca-report-1372-excerpt.document.json").read_text(encoding="utf-8"))
        for key in ("records", "figures", "aggregate", "stats"):
            self.assertIn(key, payload)

    def test_reference_stats_report_honest_grounding(self):
        payload = json.loads(
            (RESULTS / "naca-report-1372-excerpt.document.json").read_text(encoding="utf-8"))
        stats = payload["stats"]
        self.assertEqual(stats["ungrounded"], 0)
        self.assertEqual(stats["values_ungrounded"], 0)

    def test_reference_output_is_machine_independent(self):
        """Committed artifacts must not carry the author's paths or timings."""
        for path in sorted(RESULTS.glob("*.json")):
            with self.subTest(artifact=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn(":\\", text, "absolute Windows path committed")
                self.assertNotIn("file:///", text, "file URI committed")
                self.assertNotIn("/home/", text, "absolute POSIX path committed")

    def test_reference_output_has_no_wall_clock_values(self):
        payload = json.loads(
            (RESULTS / "naca-report-1372-excerpt.document.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["generated_at"], 0)
        self.assertEqual(payload["stats"]["duration_s"], 0.0)

    def test_csv_columns_match_the_current_generic_template(self):
        from llm_extractor.serialize import record_columns

        header = (RESULTS / "records.csv").read_text(
            encoding="utf-8-sig").splitlines()[0].strip().split(",")
        self.assertEqual(header, record_columns(load_template("generic")))


if __name__ == "__main__":
    unittest.main()
