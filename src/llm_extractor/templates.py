"""Extraction templates: the JSON contract between you and the model.

A template declares *what* to extract and *in what shape*. It produces:

* the prompt instructions sent to the model,
* a strict JSON Schema used for structured output (``json_schema`` response
  format on chat-completions, ``text.format`` on the Responses API),
* the field list used to coerce and validate every returned record.

Two templates ship with the package — ``generic`` (works on any document) and
``immunogenicity`` (the original vaccine-study record format, kept verbatim so
existing downstream consumers keep working). A custom template is just a JSON
file passed to ``--template``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "You are a meticulous document data-extraction agent. You extract ONLY facts "
    "explicitly stated in the provided material. You never guess, never infer "
    "unstated values, and you always quote your evidence. If a field is unknown, "
    "use null."
)

GROUNDING_RULE = (
    "Every record MUST include source_span: a SHORT verbatim quote (<=200 chars) "
    "copied from the document that supports the record. NEVER output a numeric "
    "value whose digits do not appear verbatim in that record's source_span. "
    "Return JSON only - no prose, no markdown fences."
)


@dataclass
class Field:
    name: str
    type: str = "string"          # string | number | integer | boolean
    description: str = ""
    enum: list | None = None

    def json_type(self) -> dict:
        # Nullable everywhere: strict structured output requires each property to
        # be present, so "unknown" must be expressible as null rather than absent.
        node: dict = {"type": [self.type, "null"]}
        if self.enum:
            node = {"type": [self.type, "null"], "enum": [*self.enum, None]}
        if self.description:
            node["description"] = self.description
        return node


@dataclass
class ExtractionTemplate:
    name: str
    description: str = ""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    instructions: str = ""
    fields: list = field(default_factory=list)
    key_fields: list = field(default_factory=list)

    @property
    def field_names(self) -> list:
        return [f.name for f in self.fields]

    def empty_record(self) -> dict:
        return {f.name: None for f in self.fields}

    def enum_for(self, name: str):
        for f in self.fields:
            if f.name == name:
                return f.enum
        return None

    def type_for(self, name: str) -> str:
        for f in self.fields:
            if f.name == name:
                return f.type
        return "string"

    def prompt(self) -> str:
        lines = [self.instructions.strip(), "", "Fields for each record:"]
        for f in self.fields:
            enum = f" (one of: {', '.join(str(e) for e in f.enum)})" if f.enum else ""
            lines.append(f"- {f.name} [{f.type}]{enum}: {f.description}")
        lines += ["", GROUNDING_RULE]
        return "\n".join(lines)

    def json_schema(self) -> dict:
        """Strict JSON Schema for a ``{"records": [...]}`` envelope."""
        properties = {f.name: f.json_type() for f in self.fields}
        record_schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        return {
            "name": f"{self.name}_records",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"records": {"type": "array", "items": record_schema}},
                "required": ["records"],
                "additionalProperties": False,
            },
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "instructions": self.instructions,
            "key_fields": self.key_fields,
            "fields": [
                {k: v for k, v in
                 {"name": f.name, "type": f.type, "description": f.description,
                  "enum": f.enum}.items() if v is not None}
                for f in self.fields
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionTemplate":
        validate_template_dict(data)
        return cls(
            name=data.get("name", "custom"),
            description=data.get("description", ""),
            system_prompt=data.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
            instructions=data.get("instructions", ""),
            key_fields=data.get("key_fields", []),
            fields=[Field(**f) for f in data["fields"]],
        )


#: Field names the pipeline sets itself; a template must not redefine them.
RESERVED_FIELD_NAMES = {"doc_id", "doc_title", "_grounded", "_value_grounded"}
ALLOWED_FIELD_TYPES = {"string", "number", "integer", "boolean"}
ALLOWED_FIELD_KEYS = {"name", "type", "description", "enum"}


class TemplateError(ValueError):
    """Raised when a user-supplied template is not usable.

    Messages name the offending field and say how to fix it, because a template
    is written by hand and a vague error costs far more than a precise one.
    """


def validate_template_dict(data) -> dict:
    """Validate a template definition; raises :class:`TemplateError` if invalid."""
    if not isinstance(data, dict):
        raise TemplateError("template must be a JSON object")

    fields = data.get("fields")
    if not isinstance(fields, list) or not fields:
        raise TemplateError("template needs a non-empty 'fields' list")

    if data.get("key_fields") is not None and not isinstance(data["key_fields"], list):
        raise TemplateError("'key_fields' must be a list of field names")

    seen: set = set()
    for index, field_def in enumerate(fields):
        where = f"fields[{index}]"
        if not isinstance(field_def, dict):
            raise TemplateError(f"{where}: each field must be an object")

        unknown = set(field_def) - ALLOWED_FIELD_KEYS
        if unknown:
            raise TemplateError(
                f"{where}: unknown key(s) {sorted(unknown)}; "
                f"allowed: {sorted(ALLOWED_FIELD_KEYS)}"
            )

        name = field_def.get("name")
        if not name or not isinstance(name, str):
            raise TemplateError(f"{where}: 'name' is required and must be a string")
        if name in seen:
            raise TemplateError(f"{where}: duplicate field name '{name}'")
        if name in RESERVED_FIELD_NAMES:
            raise TemplateError(
                f"{where}: '{name}' is reserved and set automatically; "
                f"reserved names are {sorted(RESERVED_FIELD_NAMES)}"
            )
        seen.add(name)

        field_type = field_def.get("type", "string")
        if field_type not in ALLOWED_FIELD_TYPES:
            raise TemplateError(
                f"{where} ('{name}'): type '{field_type}' is not supported; "
                f"use one of {sorted(ALLOWED_FIELD_TYPES)}"
            )

        enum = field_def.get("enum")
        if enum is not None:
            if not isinstance(enum, list) or not enum:
                raise TemplateError(f"{where} ('{name}'): 'enum' must be a non-empty list")
            if field_type != "string":
                raise TemplateError(
                    f"{where} ('{name}'): 'enum' is only supported on string fields"
                )

    missing_keys = [k for k in (data.get("key_fields") or []) if k not in seen]
    if missing_keys:
        raise TemplateError(
            f"'key_fields' names fields that do not exist: {missing_keys}; "
            f"defined fields are {sorted(seen)}"
        )

    if "source_span" not in seen:
        raise TemplateError(
            "template must define a 'source_span' string field: it carries the "
            "verbatim evidence that makes a record checkable"
        )
    return data


# --------------------------------------------------------------------------
# Built-in template 1: generic — subject / attribute / value on any document.
# --------------------------------------------------------------------------
GENERIC = ExtractionTemplate(
    name="generic",
    description="Domain-agnostic fact records: subject, attribute, value, unit, evidence.",
    instructions=(
        "Extract every explicitly stated quantitative or factual claim from the "
        "document as a JSON array of records. Capture negative and qualitative "
        "findings too - they are data. Do not summarise and do not merge distinct "
        "facts into one record."
    ),
    key_fields=["subject", "attribute"],
    fields=[
        Field("subject", "string", "The entity the fact is about (e.g. a group, product, cohort)."),
        Field("attribute", "string", "The measured or stated property."),
        Field("value", "number", "Numeric value exactly as written, else null."),
        Field("value_text", "string", "Verbatim value when it is not numeric (e.g. 'not detected')."),
        Field("unit", "string", "Unit of the value as written, else null."),
        Field("qualifier", "string", "Condition/context that scopes the fact (timepoint, dose, subgroup)."),
        Field("comparator", "string", "Baseline the subject is compared against, else null."),
        Field("direction", "string", "Direction of the effect vs the comparator.",
              ["higher", "lower", "no_change", "mixed", "na"]),
        Field("significant", "string", "Statistical significance if stated.", ["yes", "no", "na"]),
        Field("p_value", "string", "P-value as written (e.g. '<0.01'), else null."),
        Field("section", "string", "Section/figure/table the fact came from, else null."),
        Field("value_source", "string", "Where the value was read from.",
              ["text", "table", "figure", "na"]),
        Field("notes", "string", "Short context in the source's own words."),
        Field("source_span", "string", "REQUIRED verbatim quote (<=200 chars) supporting this record."),
    ],
)

# --------------------------------------------------------------------------
# Built-in template 2: immunogenicity — original record format, preserved.
# --------------------------------------------------------------------------
IMMUNOGENICITY = ExtractionTemplate(
    name="immunogenicity",
    description="Vaccine immunogenicity comparison + measurement records (original format).",
    instructions=(
        "Extract immunogenicity records and emit ONE JSON array. Capture "
        "quantitative comparisons, numeric measurements, AND qualitative / "
        "negative findings - negatives and causal context ARE data.\n\n"
        "(A) COMPARISON records - any comparison or directional finding, with or "
        "without a statistical test. Fill comparison_a/comparison_b; leave "
        "value/group_label null. Use significant='no' when the text explicitly "
        "reports no significant difference.\n"
        "(B) MEASUREMENT records - one object per explicitly reported numeric data "
        "point. Fill group_label/value/value_unit; leave comparison_a/comparison_b "
        "null.\n"
        "(C) NEGATIVE findings - values below LOD/LLOQ become measurement records "
        "with censoring='left_censored'; assays that could not be run become "
        "records with value=null and an explanation in notes.\n"
        "(D) DESIGN COVARIATES - one atomic fact per field. Never pack serotype, "
        "carrier, chemistry, valency or dose into group_label."
    ),
    key_fields=["assay", "endpoint", "factor_type", "dose_ug"],
    fields=[
        Field("assay", "string", "Assay type.", ["opa", "igg", "na"]),
        Field("endpoint", "string", "Endpoint, e.g. 'GMT', 'GMC', 'Concentration'."),
        Field("factor_type", "string", "Design factor being compared.",
              ["chemistry", "carrier", "polysaccharide_size", "degree_of_activation",
               "chemistry_plus_carrier", "na"]),
        Field("comparison_a", "string", "Test group (comparison records only)."),
        Field("comparison_b", "string", "Comparator/baseline group (comparison records only)."),
        Field("dose_ug", "number", "Dose in micrograms if stated."),
        Field("p_value", "string", "P-value as written (e.g. '<0.01'), else null."),
        Field("effect_direction", "string", "Direction of a vs b.",
              ["higher", "lower", "mixed", "na"]),
        Field("significant", "string", "Significance at p<0.05 if stated.", ["yes", "no", "na"]),
        Field("group_label", "string", "Single group/arm a measurement describes."),
        Field("value", "number", "Numeric value exactly as reported."),
        Field("value_unit", "string", "Unit, e.g. 'ug/mL' or 'GMT titer'."),
        Field("value_kind", "string", "Statistic reported.",
              ["geometric_mean", "mean", "median", "individual", "na"]),
        Field("censoring", "string", "Censoring status of the value.",
              ["observed", "left_censored", "right_censored", "na"]),
        Field("value_source", "string", "Where the value was read from.",
              ["text", "table", "figure", "na"]),
        Field("serotype", "string", "Serotype token as written, e.g. '3', '6B'."),
        Field("carrier", "string", "Carrier protein.", ["crm197", "scp", "tt", "dt", "na"]),
        Field("conjugation_chemistry", "string", "Conjugation chemistry.",
              ["click", "rac_dmso", "cdi", "reductive_amination", "na"]),
        Field("valency", "integer", "Number of serotypes in the formulation."),
        Field("double_dose", "string", "Double-dose regimen?", ["yes", "no", "na"]),
        Field("figure", "string", "Figure/table number the record came from."),
        Field("notes", "string", "Causal/context statement in the source's own words."),
        Field("source_span", "string", "REQUIRED verbatim quote (<=200 chars) supporting this record."),
    ],
)

BUILTIN_TEMPLATES = {t.name: t for t in (GENERIC, IMMUNOGENICITY)}

#: Starting point for ``llm-extract templates --init``: a complete, valid
#: template that already follows the rules (evidence field, atomic fields).
STARTER_TEMPLATE = {
    "name": "my_template",
    "description": "One line describing what this template extracts.",
    "instructions": (
        "Extract every explicitly stated <thing> from the document as records. "
        "Capture negative and qualitative findings too. Do not summarise, and do "
        "not merge distinct facts into one record."
    ),
    "key_fields": ["subject"],
    "fields": [
        {"name": "subject", "type": "string",
         "description": "The entity this record is about."},
        {"name": "measure", "type": "string",
         "description": "What was measured or stated."},
        {"name": "value", "type": "number",
         "description": "Numeric value exactly as written, else null."},
        {"name": "unit", "type": "string",
         "description": "Unit of the value as written, else null."},
        {"name": "status", "type": "string",
         "description": "Outcome category if stated.",
         "enum": ["confirmed", "unconfirmed", "na"]},
        {"name": "notes", "type": "string",
         "description": "Short context in the source's own words."},
        {"name": "source_span", "type": "string",
         "description": "REQUIRED verbatim quote (<=200 chars) supporting this record."},
    ],
}


def load_template(name_or_path) -> ExtractionTemplate:
    """Return a built-in template by name, a template from JSON, or from a dict.

    Accepting a dict lets the HTTP API take a template inline in the request
    body, so a frontend can define a schema without shipping a file.
    """
    if not name_or_path:
        return GENERIC
    if isinstance(name_or_path, dict):
        return ExtractionTemplate.from_dict(name_or_path)
    if isinstance(name_or_path, ExtractionTemplate):
        return name_or_path
    if name_or_path in BUILTIN_TEMPLATES:
        return BUILTIN_TEMPLATES[name_or_path]

    path = Path(name_or_path)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TemplateError(f"{path}: not valid JSON ({exc})") from exc
        try:
            return ExtractionTemplate.from_dict(data)
        except TemplateError as exc:
            raise TemplateError(f"{path}: {exc}") from exc
    raise TemplateError(
        f"unknown template '{name_or_path}'; use one of {sorted(BUILTIN_TEMPLATES)} "
        f"or a path to a template JSON file (see 'llm-extract templates --init')"
    )


# --------------------------------------------------------------------------
# OCR template — the vision model must also answer in JSON (requirement 1).
# --------------------------------------------------------------------------
OCR_INSTRUCTIONS = (
    "You are reading one page or figure image from a document. Transcribe ONLY "
    "what is visibly present: axis labels and units, plotted or tabulated values, "
    "p-values, group/series labels, captions and headings. Repair obvious unit "
    "glyph damage (a replacement character before 'g/mL' is the micro sign). "
    "Never infer, extrapolate or invent a value you cannot read. If the image "
    "contains no data, return empty arrays. Return JSON only."
)

OCR_JSON_SCHEMA = {
    "name": "figure_ocr",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "figure_type": {
                "type": ["string", "null"],
                "enum": ["chart", "table", "text", "diagram", "photo", "mixed", "empty", None],
                "description": "What kind of content the image holds.",
            },
            "caption": {"type": ["string", "null"], "description": "Verbatim caption/title if visible."},
            "axis_x": {"type": ["string", "null"], "description": "X axis label incl. unit."},
            "axis_y": {"type": ["string", "null"], "description": "Y axis label incl. unit."},
            "items": {
                "type": "array",
                "description": "One entry per readable data point or labelled value.",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": ["string", "null"]},
                        "series": {"type": ["string", "null"]},
                        "value": {"type": ["number", "null"]},
                        "value_text": {"type": ["string", "null"]},
                        "unit": {"type": ["string", "null"]},
                        "note": {"type": ["string", "null"]},
                    },
                    "required": ["label", "series", "value", "value_text", "unit", "note"],
                    "additionalProperties": False,
                },
            },
            "tables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": ["string", "null"]},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "rows": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": ["string", "null"]}},
                        },
                    },
                    "required": ["title", "columns", "rows"],
                    "additionalProperties": False,
                },
            },
            "text_blocks": {
                "type": "array",
                "description": "Verbatim text visible in the image, in reading order.",
                "items": {"type": "string"},
            },
            "notes": {"type": ["string", "null"]},
        },
        "required": ["figure_type", "caption", "axis_x", "axis_y", "items", "tables",
                     "text_blocks", "notes"],
        "additionalProperties": False,
    },
}


def empty_ocr_payload() -> dict:
    return {
        "figure_type": "empty",
        "caption": None,
        "axis_x": None,
        "axis_y": None,
        "items": [],
        "tables": [],
        "text_blocks": [],
        "notes": None,
    }


# --------------------------------------------------------------------------
# Aggregation agent output template (text records + OCR JSON -> one document).
# --------------------------------------------------------------------------
AGGREGATE_JSON_SCHEMA = {
    "name": "document_aggregate",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "2-4 sentence factual summary of what the document reports.",
            },
            "key_findings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short verbatim-grounded findings, most important first.",
            },
            "figure_insights": {
                "type": "array",
                "description": "Values that only the figure/OCR pass revealed.",
                "items": {
                    "type": "object",
                    "properties": {
                        "image": {"type": ["string", "null"]},
                        "finding": {"type": "string"},
                        "value": {"type": ["number", "null"]},
                        "unit": {"type": ["string", "null"]},
                    },
                    "required": ["image", "finding", "value", "unit"],
                    "additionalProperties": False,
                },
            },
            "conflicts": {
                "type": "array",
                "description": "Disagreements between the text pass and the OCR pass.",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": ["string", "null"]},
                        "text_says": {"type": ["string", "null"]},
                        "figure_says": {"type": ["string", "null"]},
                        "resolution": {"type": ["string", "null"]},
                    },
                    "required": ["field", "text_says", "figure_says", "resolution"],
                    "additionalProperties": False,
                },
            },
            "coverage_gaps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Data the document references but that could not be extracted.",
            },
        },
        "required": ["summary", "key_findings", "figure_insights", "conflicts", "coverage_gaps"],
        "additionalProperties": False,
    },
}


def empty_aggregate() -> dict:
    return {
        "summary": "",
        "key_findings": [],
        "figure_insights": [],
        "conflicts": [],
        "coverage_gaps": [],
    }
