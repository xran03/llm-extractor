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
DOCUMENTS = RESULTS / "documents"


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
    INPUTS = ("pcv13-opa-colonisation.pdf", "pcv13-opa-figure2.png",
              "fda-pcv15-approval-letter.pdf", "opa-scatter.png",
              "h5-titre-histogram-scatter.jpg")

    def test_demo_inputs_are_present(self):
        for name in self.INPUTS:
            self.assertTrue((DEMO / name).is_file(), name)

    def test_every_redistributed_document_is_attributed(self):
        """CC BY permits redistribution only with attribution; say where each came from."""
        readme = (DEMO / "README.md").read_text(encoding="utf-8")
        for marker in ("creativecommons.org/licenses/by/4.0/",
                       "10.1016/j.vaccine.2022.09.069", "fda.gov", "public domain"):
            self.assertIn(marker, readme, f"{marker} missing from demo/README.md")

    def test_no_demo_input_bloats_the_repository(self):
        for path in sorted(DEMO.rglob("*")):
            if path.is_file() and path.suffix.lower() in (".pdf", ".png", ".jpg"):
                size_mb = path.stat().st_size / (1024 * 1024)
                self.assertLess(size_mb, 2.0, f"{path.name} is {size_mb:.1f} MB")



class DemoResultsTest(unittest.TestCase):
    """Reference output must stay consistent with the code that writes it."""

    def setUp(self):
        self.rows = read_csv(RESULTS / "records.csv")

    def test_the_headline_table_sits_at_the_top_level(self):
        """What a run is opened for should not be buried among per-document files."""
        self.assertTrue(self.rows)
        self.assertTrue((RESULTS / "figures.csv").is_file())
        self.assertTrue((RESULTS / "summary.json").is_file())
        self.assertFalse(sorted(RESULTS.glob("*.records.jsonl")),
                         "per-document artifacts belong under documents/")

    def test_intermediates_live_in_one_subdirectory(self):
        self.assertTrue(sorted(DOCUMENTS.glob("*.records.jsonl")))
        self.assertTrue(sorted(DOCUMENTS.glob("*.document.json")))

    def test_every_reference_record_is_grounded(self):
        for row in self.rows:
            self.assertEqual(row["_grounded"], "true", row.get("group_label"))
            self.assertNotEqual(row["_value_grounded"], "false", row.get("group_label"))

    def test_every_serotype_resolves_to_a_repeat_unit(self):
        named = [r for r in self.rows if r.get("serotype")]
        self.assertTrue(named, "no serotype-bearing records in the demo")
        for row in named:
            self.assertTrue(row["repeat_unit"], f"serotype {row['serotype']} unresolved")

    def test_the_control_arm_survived_alongside_the_responders(self):
        """A run that keeps only the impressive numbers has misread the study."""
        values = [float(r["value"]) for r in self.rows if r.get("value")]
        self.assertTrue(any(v < 500 for v in values), "no low/control values kept")
        self.assertTrue(any(v > 5000 for v in values), "no post-vaccination values kept")

    def test_the_figure_yields_numbers_the_text_layer_lacks(self):
        rows = [r for r in read_csv(RESULTS / "figures.csv")
                if r["doc_id"].startswith("pcv13-opa-figure2")]
        self.assertTrue(rows, "the real scatter plot produced nothing")
        recovered = {r["value_text"] for r in rows}
        for printed in ("R2= 0.351", "p= 0.012"):
            self.assertIn(printed, recovered, f"{printed} is drawn but was not read")

    def test_reference_output_is_machine_independent(self):
        """Committed artifacts must not carry the author's paths or timings."""
        for path in sorted(RESULTS.rglob("*.json")):
            with self.subTest(artifact=path.relative_to(RESULTS).as_posix()):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn(":\\", text, "absolute Windows path committed")
                self.assertNotIn("file:///", text, "file URI committed")
                self.assertNotIn("/home/", text, "absolute POSIX path committed")

    def test_reference_output_has_no_wall_clock_values(self):
        payload = json.loads((DOCUMENTS / "pcv13-opa-colonisation.document.json")
                             .read_text(encoding="utf-8"))
        self.assertEqual(payload["generated_at"], 0)
        self.assertEqual(payload["stats"]["duration_s"], 0.0)
        for key in ("records", "figures", "aggregate", "stats"):
            self.assertIn(key, payload)

    def test_csv_columns_match_the_immunogenicity_template(self):
        from llm_extractor.serialize import record_columns

        header = (RESULTS / "records.csv").read_text(
            encoding="utf-8-sig").splitlines()[0].strip().split(",")
        self.assertEqual(header, record_columns(load_template("immunogenicity")))


if __name__ == "__main__":
    unittest.main()
