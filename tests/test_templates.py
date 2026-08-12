"""Templates define the JSON contract; they must stay strict and loadable."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_extractor.templates import (AGGREGATE_JSON_SCHEMA, BUILTIN_TEMPLATES,
                                     ExtractionTemplate, Field, OCR_JSON_SCHEMA,
                                     STARTER_TEMPLATE, TemplateError,
                                     empty_aggregate, empty_ocr_payload, load_template,
                                     validate_template_dict)


def assert_strict(testcase, node, path="root"):
    """Strict structured output: every object lists all properties as required."""
    if not isinstance(node, dict):
        return
    if node.get("type") == "object" or "properties" in node:
        properties = node.get("properties", {})
        testcase.assertFalse(node.get("additionalProperties", True),
                             f"{path}: additionalProperties must be false")
        testcase.assertEqual(sorted(node.get("required", [])), sorted(properties),
                             f"{path}: required must list every property")
        for name, child in properties.items():
            assert_strict(testcase, child, f"{path}.{name}")
    if "items" in node:
        assert_strict(testcase, node["items"], f"{path}[]")


class BuiltinTemplateTest(unittest.TestCase):
    def test_both_templates_available(self):
        self.assertEqual(sorted(BUILTIN_TEMPLATES), ["generic", "immunogenicity"])

    def test_every_template_requires_evidence(self):
        for template in BUILTIN_TEMPLATES.values():
            self.assertIn("source_span", template.field_names, template.name)

    def test_schemas_are_strict(self):
        for template in BUILTIN_TEMPLATES.values():
            schema = template.json_schema()
            self.assertTrue(schema["strict"])
            assert_strict(self, schema["schema"], template.name)

    def test_records_envelope(self):
        schema = BUILTIN_TEMPLATES["generic"].json_schema()["schema"]
        self.assertEqual(list(schema["properties"]), ["records"])
        self.assertEqual(schema["properties"]["records"]["type"], "array")

    def test_fields_are_nullable_so_unknown_is_expressible(self):
        schema = BUILTIN_TEMPLATES["generic"].json_schema()["schema"]
        item = schema["properties"]["records"]["items"]
        for name, node in item["properties"].items():
            self.assertIn("null", node["type"], name)

    def test_enums_include_null(self):
        schema = BUILTIN_TEMPLATES["generic"].json_schema()["schema"]
        direction = schema["properties"]["records"]["items"]["properties"]["direction"]
        self.assertIn(None, direction["enum"])

    def test_prompt_lists_every_field_and_the_grounding_rule(self):
        template = BUILTIN_TEMPLATES["immunogenicity"]
        prompt = template.prompt()
        for name in template.field_names:
            self.assertIn(name, prompt)
        self.assertIn("source_span", prompt)
        self.assertIn("NEVER output a numeric value", prompt)

    def test_immunogenicity_keeps_the_original_record_format(self):
        fields = set(BUILTIN_TEMPLATES["immunogenicity"].field_names)
        original = {"assay", "endpoint", "factor_type", "comparison_a", "comparison_b",
                    "dose_ug", "p_value", "effect_direction", "significant",
                    "group_label", "value", "value_unit", "value_kind", "censoring",
                    "value_source", "serotype", "carrier", "conjugation_chemistry",
                    "valency", "double_dose", "figure", "notes", "source_span"}
        self.assertTrue(original.issubset(fields), original - fields)

    def test_empty_record_covers_all_fields(self):
        template = BUILTIN_TEMPLATES["generic"]
        self.assertEqual(sorted(template.empty_record()), sorted(template.field_names))


class OcrAndAggregateSchemaTest(unittest.TestCase):
    def test_ocr_schema_is_strict(self):
        assert_strict(self, OCR_JSON_SCHEMA["schema"], "ocr")

    def test_ocr_empty_payload_matches_schema_keys(self):
        self.assertEqual(sorted(empty_ocr_payload()),
                         sorted(OCR_JSON_SCHEMA["schema"]["properties"]))

    def test_aggregate_schema_is_strict(self):
        assert_strict(self, AGGREGATE_JSON_SCHEMA["schema"], "aggregate")

    def test_aggregate_empty_payload_matches_schema_keys(self):
        self.assertEqual(sorted(empty_aggregate()),
                         sorted(AGGREGATE_JSON_SCHEMA["schema"]["properties"]))


class LoadTemplateTest(unittest.TestCase):
    def test_default_is_generic(self):
        self.assertEqual(load_template(None).name, "generic")

    def test_builtin_by_name(self):
        self.assertEqual(load_template("immunogenicity").name, "immunogenicity")

    def test_unknown_name_is_rejected(self):
        with self.assertRaises(TemplateError):
            load_template("does-not-exist")

    def test_inline_dict_is_accepted(self):
        template = load_template({
            "name": "inline",
            "fields": [
                {"name": "a", "type": "string", "description": "d"},
                {"name": "source_span", "type": "string", "description": "e"},
            ],
        })
        self.assertEqual(template.name, "inline")

    def test_template_instance_passes_through(self):
        original = BUILTIN_TEMPLATES["generic"]
        self.assertIs(load_template(original), original)

    def test_custom_template_from_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.json"
            path.write_text(json.dumps({
                "name": "patent_claims",
                "description": "claims",
                "instructions": "extract claims",
                "key_fields": ["claim_number"],
                "fields": [
                    {"name": "claim_number", "type": "integer", "description": "n"},
                    {"name": "claim_text", "type": "string", "description": "text"},
                    {"name": "source_span", "type": "string", "description": "evidence"},
                ],
            }), encoding="utf-8")
            template = load_template(str(path))
            self.assertEqual(template.name, "patent_claims")
            self.assertEqual(template.key_fields, ["claim_number"])
            assert_strict(self, template.json_schema()["schema"], "custom")

    def test_malformed_json_file_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(TemplateError) as ctx:
                load_template(str(path))
            self.assertIn("broken.json", str(ctx.exception))

    def test_starter_template_is_valid_and_strict(self):
        template = load_template(STARTER_TEMPLATE)
        assert_strict(self, template.json_schema()["schema"], "starter")
        self.assertIn("source_span", template.field_names)

    def test_roundtrip_to_dict_and_back(self):
        original = BUILTIN_TEMPLATES["generic"]
        clone = ExtractionTemplate.from_dict(original.to_dict())
        self.assertEqual(clone.field_names, original.field_names)
        self.assertEqual(clone.json_schema(), original.json_schema())

    def test_field_type_and_enum_lookup(self):
        template = ExtractionTemplate(name="t", fields=[
            Field("n", "number", "num"), Field("c", "string", "cat", ["a", "b"])
        ])
        self.assertEqual(template.type_for("n"), "number")
        self.assertEqual(template.enum_for("c"), ["a", "b"])
        self.assertIsNone(template.enum_for("n"))


class TemplateValidationTest(unittest.TestCase):
    """A hand-written schema must fail with a message that says how to fix it."""

    def _bad(self, data):
        with self.assertRaises(TemplateError) as ctx:
            validate_template_dict(data)
        return str(ctx.exception)

    def _fields(self, *extra):
        return [*extra, {"name": "source_span", "type": "string", "description": "e"}]

    def test_not_an_object(self):
        self.assertIn("JSON object", self._bad(["a"]))

    def test_missing_fields(self):
        self.assertIn("non-empty 'fields'", self._bad({"name": "t"}))

    def test_empty_fields(self):
        self.assertIn("non-empty 'fields'", self._bad({"name": "t", "fields": []}))

    def test_field_must_be_an_object(self):
        self.assertIn("must be an object", self._bad({"fields": ["x"]}))

    def test_missing_field_name(self):
        self.assertIn("'name' is required", self._bad({"fields": [{"type": "string"}]}))

    def test_duplicate_field_name(self):
        message = self._bad({"fields": self._fields(
            {"name": "a", "type": "string", "description": "d"},
            {"name": "a", "type": "string", "description": "d"})})
        self.assertIn("duplicate field name 'a'", message)

    def test_reserved_field_name(self):
        message = self._bad({"fields": self._fields(
            {"name": "doc_id", "type": "string", "description": "d"})})
        self.assertIn("reserved", message)

    def test_unsupported_type_lists_the_valid_ones(self):
        message = self._bad({"fields": self._fields(
            {"name": "a", "type": "float", "description": "d"})})
        self.assertIn("not supported", message)
        self.assertIn("number", message)

    def test_unknown_key_is_caught(self):
        message = self._bad({"fields": self._fields(
            {"name": "a", "type": "string", "description": "d", "reqired": True})})
        self.assertIn("unknown key", message)

    def test_enum_must_be_a_non_empty_list(self):
        message = self._bad({"fields": self._fields(
            {"name": "a", "type": "string", "description": "d", "enum": []})})
        self.assertIn("non-empty list", message)

    def test_enum_only_on_string_fields(self):
        message = self._bad({"fields": self._fields(
            {"name": "a", "type": "number", "description": "d", "enum": ["x"]})})
        self.assertIn("only supported on string", message)

    def test_key_fields_must_exist(self):
        message = self._bad({"key_fields": ["nope"], "fields": self._fields(
            {"name": "a", "type": "string", "description": "d"})})
        self.assertIn("do not exist", message)

    def test_key_fields_must_be_a_list(self):
        self.assertIn("must be a list", self._bad(
            {"key_fields": "a", "fields": self._fields()}))

    def test_source_span_is_mandatory(self):
        message = self._bad({"fields": [
            {"name": "a", "type": "string", "description": "d"}]})
        self.assertIn("source_span", message)

    def test_a_minimal_valid_template_passes(self):
        validate_template_dict({"name": "ok", "fields": self._fields(
            {"name": "a", "type": "string", "description": "d"})})


if __name__ == "__main__":
    unittest.main()
