"""Value normalization, coercion and grounding checks.

Two responsibilities:

* **Coercion** — force every returned record into the template's field set and
  types, so downstream consumers see one stable shape regardless of model or
  backend.
* **Grounding** — the anti-hallucination guarantee. A record is only trusted
  when its ``source_span`` is locatable in the document, and a numeric value is
  only trusted when its digits appear inside that span.
"""
from __future__ import annotations

import re

# Matches integers/decimals with thousands separators and scientific notation.
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")

_MICRO_FIXES = (
    (re.compile(r"[\ufffd\u00b5\u03bc]\s*g\s*/\s*m\s*[lL]"), "µg/mL"),
    (re.compile(r"[\ufffd\u00b5\u03bc]\s*g\b"), "µg"),
    (re.compile(r"[\ufffd\u00b5\u03bc]\s*[lL]\b"), "µL"),
    (re.compile(r"\bug/ml\b", re.IGNORECASE), "µg/mL"),
)


def fix_units(text):
    """Repair unit glyphs mangled by text extraction (micro sign in particular)."""
    if text is None:
        return None
    out = str(text)
    for pattern, replacement in _MICRO_FIXES:
        out = pattern.sub(replacement, out)
    return out


def parse_number(value):
    """Parse a reported number ('1,165', '<0.01', '2.05e3') to float, else None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMBER_RE.search(str(value))
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def numbers_in_text(text) -> list:
    out = []
    for token in _NUMBER_RE.findall(str(text or "")):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


def canon_text(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def canon_category(value, allowed, default: str = "na") -> str:
    """Lowercase a categorical value; blanks become ``default``.

    Values outside ``allowed`` are preserved rather than dropped, so an
    unexpected label surfaces in review instead of silently vanishing.
    """
    text = canon_text(value).lower()
    if text in ("", "none", "nan", "null"):
        return default
    return text


def coerce_record(raw: dict, template, doc_id: str) -> dict:
    """Project a raw model object onto the template's schema."""
    record = template.empty_record()
    for key, value in (raw or {}).items():
        if key in record:
            record[key] = value

    for name in template.field_names:
        value = record.get(name)
        field_type = template.type_for(name)
        enum = template.enum_for(name)
        if value is None:
            continue
        if field_type == "number":
            record[name] = parse_number(value)
        elif field_type == "integer":
            number = parse_number(value)
            record[name] = int(number) if number is not None else None
        elif field_type == "boolean":
            record[name] = bool(value) if not isinstance(value, str) else \
                value.strip().lower() in ("true", "yes", "1")
        elif enum:
            record[name] = canon_category(value, enum)
        else:
            record[name] = fix_units(canon_text(value)) or None

    record["doc_id"] = doc_id
    return record


def span_is_grounded(span, doc_text: str) -> bool:
    """True when the quoted evidence can be located in the document."""
    if not span:
        return False
    needle = " ".join(str(span).split()).lower()
    haystack = " ".join((doc_text or "").split()).lower()
    probe = needle[:20]
    return bool(probe) and probe in haystack


def value_supported_by_span(value, span, rel_tol: float = 1e-6) -> bool:
    """True when the numeric value's digits actually appear in the span."""
    if value is None or span is None:
        return False
    try:
        target = float(value)
    except (TypeError, ValueError):
        return False
    for candidate in numbers_in_text(span):
        if candidate == target or (
            target != 0 and abs(candidate - target) <= rel_tol * abs(target)
        ):
            return True
    return False


def annotate_grounding(record: dict, doc_text: str, numeric_fields=("value",)) -> dict:
    """Attach ``_grounded`` / ``_value_grounded`` audit flags to a record.

    ``_value_grounded`` is ``None`` when the record carries no number (the check
    does not apply), ``False`` when the number is unsupported by its evidence
    (a fabricated value), ``True`` when it traces back to the source.
    """
    span = record.get("source_span")
    record["_grounded"] = span_is_grounded(span, doc_text)

    numeric_value = None
    for name in numeric_fields:
        if record.get(name) is not None:
            numeric_value = record[name]
            break

    if numeric_value is None:
        record["_value_grounded"] = None
    elif not record["_grounded"]:
        record["_value_grounded"] = False
    else:
        record["_value_grounded"] = value_supported_by_span(numeric_value, span)
    return record


def token_set(value) -> set:
    """Tokenize a free-text label for order-insensitive comparison."""
    text = canon_text(value).lower()
    text = re.sub(r"(?<=\d)(?=[a-z])", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return {t for t in text.split() if t}


def token_set_similarity(a, b) -> float:
    """Jaccard similarity between two label token sets, in [0, 1]."""
    sa, sb = token_set(a), token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
