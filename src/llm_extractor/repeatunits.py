"""Repeat units per pneumococcal serotype.

A capsular polysaccharide is a repeating unit, and which unit it is depends on
the serotype — so a table of measurements is only half-interpretable without
it. Two records reading "12.5 µg/mL" mean different things for serotype 3 and
serotype 19F, and the difference is the structure.

The shipped table is the harmonised Danish-type reference; a record naming a
serotype is annotated from it. When a document states a structure the table
does not have, the extracted one is kept and marked as coming from the
document, because a paper reporting a novel or corrected structure is more
current than a static reference.

Resolution is deliberately narrow: the serotype token is normalised (``19f`` →
``19F``) and looked up. No fuzzy matching — silently attaching serotype 6B's
structure to a record about 6C would be worse than attaching nothing.
"""
from __future__ import annotations

import csv
import functools
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent / "data" / "pn_serotype_repeat_units.csv"

#: Structure statuses that carry an actual repeat unit worth attaching.
#: ``constituents_only`` and ``prose_partial`` describe a structure without
#: giving one, so attaching them would put prose in a structure column.
USABLE_STATUS = ("full_structure", "composite")

#: Column added to alignment output.
REPEAT_UNIT_FIELD = "repeat_unit"
SOURCE_FIELD = "repeat_unit_source"


def normalise_serotype(value) -> str:
    """Canonical Danish type token, or "" when the value is not one.

    Serotypes are written every way a keyboard allows — ``19f``, ``19 F``,
    ``Pn19F``, ``serotype 19F`` — and they all have to reach the same row.
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    for prefix in ("serotype", "sero", "type", "pn", "st"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip(" -_:")
    text = text.replace(" ", "").replace("-", "").replace("_", "")
    if not text or not text[0].isdigit():
        return ""
    digits = "".join(c for c in text if c.isdigit())
    letters = "".join(c for c in text if c.isalpha())
    if not digits:
        return ""
    # Danish types capitalise the first letter only: 6A, 15B, 6Eb.
    suffix = (letters[:1].upper() + letters[1:].lower()) if letters else ""
    return f"{digits}{suffix}"


@functools.lru_cache(maxsize=1)
def load_table(path=None) -> dict:
    """Map a normalised serotype token to its reference repeat unit."""
    source = Path(path) if path else DATA_FILE
    table: dict = {}
    if not source.is_file():
        return table
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            token = normalise_serotype(row.get("Danish_Type"))
            if not token or token in table:
                continue
            unit = (row.get("Repeat_Unit_Machine")
                    or row.get("Backbone_Normalized")
                    or row.get("IUPAC_Condensed_Repeat_Unit_Verbatim") or "").strip()
            if not unit:
                continue
            table[token] = {
                "repeat_unit": unit,
                "status": (row.get("Structure_Status") or "").strip(),
                "reference": (row.get("Reference") or "").strip(),
            }
    return table


def lookup(serotype) -> dict | None:
    """Reference entry for one serotype, or ``None`` when it is not in the table."""
    token = normalise_serotype(serotype)
    if not token:
        return None
    return load_table().get(token)


def annotate_records(records, serotype_field: str = "serotype") -> dict:
    """Attach ``repeat_unit`` to every record that names a serotype.

    Returns counts for the run summary. A record that already carries a repeat
    unit read out of the document keeps it: the table is a fallback, not an
    override.
    """
    stats = {"annotated": 0, "from_document": 0, "unknown_serotype": 0, "no_serotype": 0}
    for record in records:
        if not isinstance(record, dict):
            continue
        existing = str(record.get(REPEAT_UNIT_FIELD) or "").strip()
        if existing:
            record.setdefault(SOURCE_FIELD, "document")
            stats["from_document"] += 1
            continue

        serotype = record.get(serotype_field)
        if not serotype or str(serotype).strip().lower() in ("", "na", "none"):
            record[REPEAT_UNIT_FIELD] = None
            record[SOURCE_FIELD] = None
            stats["no_serotype"] += 1
            continue

        entry = lookup(serotype)
        if entry is None or entry["status"] not in USABLE_STATUS:
            record[REPEAT_UNIT_FIELD] = None
            record[SOURCE_FIELD] = None
            stats["unknown_serotype"] += 1
            continue

        record[REPEAT_UNIT_FIELD] = entry["repeat_unit"]
        record[SOURCE_FIELD] = f"reference table ({entry['status']})"
        stats["annotated"] += 1
    return stats


def summary(stats: dict) -> dict:
    return {
        "repeat_units_annotated": stats.get("annotated", 0),
        "repeat_units_from_document": stats.get("from_document", 0),
        "repeat_units_unknown_serotype": stats.get("unknown_serotype", 0),
    }
