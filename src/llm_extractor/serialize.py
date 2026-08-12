"""Tabular serialization — CSV is the format most people actually analyse.

Extracted records are flat and homogeneous, so a spreadsheet is the natural
final artifact: one row per fact, columns fixed by the template. JSONL stays
the lossless machine format (it keeps nested figure payloads and audit flags),
and CSV is the analysis format.

Two tables are produced:

``records.csv``  one row per extracted record, columns in template order;
``figures.csv``  one row per value read out of a figure by the vision pass.

Files are written with a UTF-8 BOM so Excel renders ``µg/mL`` correctly instead
of mojibake — the single most common complaint about CSV exports.
"""
from __future__ import annotations

import csv
from pathlib import Path

#: Provenance columns placed before the template's own fields.
LEADING_COLUMNS = ("doc_id", "doc_title")
#: Audit columns placed after them; these are the anti-hallucination flags.
#: ``_ungrounded`` comes last because it is the one a reviewer reads: it names
#: the fields that failed, so a flagged row can be checked without re-reading
#: the whole record.
TRAILING_COLUMNS = ("_grounded", "_value_grounded", "_unit_grounded", "_ungrounded")

FIGURE_COLUMNS = (
    "doc_id", "doc_title", "image", "figure_type", "caption",
    "axis_x", "axis_y", "series", "label", "value", "value_text", "unit", "note",
)


def record_columns(template) -> list:
    """Stable column order: provenance, template fields, audit flags."""
    fields = [f for f in template.field_names if f not in LEADING_COLUMNS]
    return [*LEADING_COLUMNS, *fields, *TRAILING_COLUMNS]


def _cell(value):
    """Render one value for a spreadsheet cell."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return "; ".join(str(v) for v in value) if isinstance(value, list) else str(value)
    return str(value)


def write_records_csv(path, records, template, doc_title: str = "") -> Path:
    """Write records as CSV; returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = record_columns(template)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for record in records:
            row = dict(record)
            row.setdefault("doc_title", doc_title)
            writer.writerow([_cell(row.get(column)) for column in columns])
    return path


def figure_rows(figures, doc_id: str = "", doc_title: str = "") -> list:
    """Flatten the OCR payload into one row per readable value."""
    rows = []
    for figure in figures or []:
        payload = figure.get("ocr") or {}
        base = {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "image": figure.get("image"),
            "figure_type": payload.get("figure_type"),
            "caption": payload.get("caption"),
            "axis_x": payload.get("axis_x"),
            "axis_y": payload.get("axis_y"),
        }
        items = payload.get("items") or []
        if not items:
            rows.append({**base, "label": None, "series": None, "value": None,
                         "value_text": None, "unit": None,
                         "note": payload.get("notes")})
            continue
        for item in items:
            rows.append({
                **base,
                "series": item.get("series"),
                "label": item.get("label"),
                "value": item.get("value"),
                "value_text": item.get("value_text"),
                "unit": item.get("unit"),
                "note": item.get("note"),
            })
    return rows


def write_figures_csv(path, rows) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(FIGURE_COLUMNS)
        for row in rows:
            writer.writerow([_cell(row.get(column)) for column in FIGURE_COLUMNS])
    return path


def append_records_csv(path, records, template, doc_title: str = "",
                       write_header: bool = False) -> Path:
    """Append rows to the run-level combined CSV.

    Documents finish one at a time, so the combined table is appended to as the
    run progresses rather than held in memory until the end.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = record_columns(template)
    mode = "w" if write_header or not path.exists() else "a"
    with path.open(mode, encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        if mode == "w":
            writer.writerow(columns)
        for record in records:
            row = dict(record)
            row.setdefault("doc_title", doc_title)
            writer.writerow([_cell(row.get(column)) for column in columns])
    return path


def append_figures_csv(path, rows, write_header: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header or not path.exists() else "a"
    with path.open(mode, encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        if mode == "w":
            writer.writerow(FIGURE_COLUMNS)
        for row in rows:
            writer.writerow([_cell(row.get(column)) for column in FIGURE_COLUMNS])
    return path


def read_csv(path) -> list:
    """Read a CSV written by this module (used by tests and consumers)."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))
