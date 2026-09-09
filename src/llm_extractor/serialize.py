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

from .normalize import parse_number

#: Provenance columns placed before the template's own fields.
LEADING_COLUMNS = ("doc_id", "doc_title")
#: Derived columns appended after the template's own fields: resolved from a
#: reference table rather than read out of the document, so they sit apart from
#: what the model extracted.
DERIVED_COLUMNS = ("repeat_unit", "repeat_unit_source")
#: Audit columns placed after them; these are the anti-hallucination flags.
#: ``_ungrounded`` comes last because it is the one a reviewer reads: it names
#: the fields that failed, so a flagged row can be checked without re-reading
#: the whole record.
TRAILING_COLUMNS = ("_grounded", "_value_grounded", "_unit_grounded", "_ungrounded")
#: Second-opinion columns, written only once a review pass has run. They are
#: kept out of TRAILING_COLUMNS so an un-reviewed run does not ship four empty
#: columns implying a check nobody performed.
REVIEW_COLUMNS = ("_review_value", "_review_unit", "_review_row", "_review_note")


def record_columns(template, reviewed: bool = False) -> list:
    """Stable column order: provenance, template fields, derived, audit flags."""
    skip = set(LEADING_COLUMNS) | set(DERIVED_COLUMNS)
    fields = [f for f in template.field_names if f not in skip]
    columns = [*LEADING_COLUMNS, *fields, *DERIVED_COLUMNS, *TRAILING_COLUMNS]
    return [*columns, *REVIEW_COLUMNS] if reviewed else columns

FIGURE_COLUMNS = (
    "doc_id", "doc_title", "image", "figure_type", "caption",
    "axis_x", "axis_y", "series", "label", "value", "value_text", "unit", "note",
)


def _cell(value):
    """Render one value for a spreadsheet cell."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return "; ".join(str(v) for v in value) if isinstance(value, list) else str(value)
    return str(value)


def write_records_csv(path, records, template, doc_title: str = "",
                      reviewed: bool = False) -> Path:
    """Write records as CSV; returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = record_columns(template, reviewed=reviewed)
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
        table_rows = _table_rows(base, payload.get("tables") or [])
        if not items and not table_rows:
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
        rows.extend(table_rows)
    return rows


def _table_rows(base: dict, tables: list) -> list:
    """One row per table cell — a table printed inside a figure is data too.

    Papers routinely print a statistics table beside the plot it belongs to,
    and the vision pass reads it into ``tables``. Only ``items`` used to be
    flattened, so those readings reached ``.ocr.json`` and were counted in the
    run summary, but never appeared in the table people actually analyse.

    A cell means nothing without its coordinates, so the column header is kept
    as the label, and the row's own leading cell, when the table has one, is
    kept in front of it.
    """
    rows = []
    for index, table in enumerate(tables, start=1):
        columns = list(table.get("columns") or [])
        # An unnamed table still has to be told apart from the next one, and a
        # figure often prints several side by side.
        series = table.get("title") or f"table {index}"
        for cells in table.get("rows") or []:
            cells = list(cells or [])
            row_label = None
            # A labelled table carries one cell more than it has columns: the
            # leading cell names the row rather than answering a column.
            if columns and len(cells) == len(columns) + 1:
                row_label, cells = cells[0], cells[1:]
            for position, cell in enumerate(cells):
                if cell is None or not str(cell).strip():
                    continue
                column = columns[position] if position < len(columns) else None
                label = " / ".join(str(p) for p in (row_label, column) if p)
                rows.append({
                    **base,
                    "series": series,
                    "label": label or None,
                    "value": parse_number(cell),
                    "value_text": str(cell),
                    "unit": None,
                    "note": None,
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


def header_matches(path, columns) -> bool:
    """True when an existing CSV already carries exactly these columns.

    Resuming into a directory should extend the combined table, not replace it.
    But only when the columns still line up: a run with a different template
    writes different columns, and appending those under the old header would
    produce a file whose rows no longer mean what the header says.
    """
    path = Path(path)
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        first = fh.readline()
    if not first.strip():
        return False
    return next(csv.reader([first]), []) == list(columns)


def append_records_csv(path, records, template, doc_title: str = "",
                       write_header: bool = False, reviewed: bool = False) -> Path:
    """Append rows to the run-level combined CSV.

    Documents finish one at a time, so the combined table is appended to as the
    run progresses rather than held in memory until the end.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = record_columns(template, reviewed=reviewed)
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
