"""Value normalization, coercion and grounding checks.

Two responsibilities:

* **Coercion** — force every returned record into the template's field set and
  types, so downstream consumers see one stable shape regardless of model or
  backend.
* **Grounding** — the anti-hallucination guarantee. A record is only trusted
  when its ``source_span`` is locatable in the document, when *every* number it
  reports appears inside that span, and when the units it declares agree with
  the units the span actually states.

The checks are deliberately deterministic and local: they cost no tokens and no
API calls, so they run on every record of every document rather than on a
sample.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

# Matches integers/decimals with thousands separators and scientific notation.
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")

#: Fraction of a quoted span that must be found in the document for the span to
#: count as grounded. Below 1.0 so that whitespace, hyphenation and OCR noise do
#: not reject a genuine quote; high enough that a fabricated clause cannot pass.
SPAN_COVERAGE_THRESHOLD = 0.85

#: Fraction of a quoted span's *words* that must also appear. Character
#: coverage alone is dominated by boilerplate, so a swapped label or group
#: name would otherwise survive; this is what rejects it.
WORD_COVERAGE_THRESHOLD = 0.85

#: How alike two words must be before one is accepted as a damaged spelling of
#: the other rather than a different word. Calibrated against the two cases
#: that matter: "ug/rnl" for "ug/ml" scores 0.73, while a substituted group
#: label ("z" for "a") scores 0.0 — the two are far apart, so the exact cut-off
#: is not delicate.
NEAR_WORD_RATIO = 0.7

#: Bound on how many candidate positions the fuzzy search inspects, so a long
#: document with a common anchor word cannot make grounding quadratic.
MAX_ANCHOR_POSITIONS = 40

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


def span_is_grounded(span, doc_text: str, min_coverage: float = SPAN_COVERAGE_THRESHOLD) -> bool:
    """True when the quoted evidence can be located in the document.

    An exact match settles it. Otherwise the quote is compared against the
    best-matching stretch of the document on three counts, because character
    similarity alone is not enough: in "Group A reached 12.5 ug/mL" the words
    that carry the meaning are short, so swapping the group and the number
    still leaves most characters intact.

    A span therefore has to survive all three:

    * every number it quotes must appear in that stretch — a fabricated figure
      is the failure this whole module exists to catch;
    * enough of its words must appear, so a substituted label is rejected;
    * enough of its characters must match, which is what tolerates whitespace,
      hyphenation and OCR damage in an otherwise genuine quote.
    """
    needle = canon_text(span).lower()
    haystack = canon_text(doc_text).lower()
    if not needle or not haystack:
        return False
    if needle in haystack:
        return True

    ratio, window = _best_window(needle, haystack)
    if ratio < min_coverage:
        return False

    window_numbers = numbers_in_text(window)
    for number in numbers_in_text(needle):
        if not any(number == seen for seen in window_numbers):
            return False

    needle_words = needle.split()
    if not needle_words:
        return False
    return _word_coverage(needle_words, window.split()) >= WORD_COVERAGE_THRESHOLD


def _word_coverage(needle_words, window_words) -> float:
    """Fraction of the quote's words present in the window, verbatim or near.

    A word absent from the window still counts when a close variant is there,
    which is what separates OCR damage from substitution: ``ug/rnl`` has an
    obvious counterpart in ``ug/ml``, whereas the ``z`` of a swapped "Group Z"
    has none, and so is charged as missing.
    """
    present = set(window_words)
    found = 0
    for word in needle_words:
        if word in present:
            found += 1
            continue
        if any(SequenceMatcher(None, word, candidate).ratio() >= NEAR_WORD_RATIO
               for candidate in present):
            found += 1
    return found / len(needle_words)


def span_coverage(needle: str, haystack: str) -> float:
    """Fraction of ``needle`` found in the best-matching window of ``haystack``."""
    return _best_window(canon_text(needle).lower(), canon_text(haystack).lower())[0]


def _best_window(needle: str, haystack: str):
    """Return ``(coverage, window)`` for the best-matching stretch of haystack.

    The search is anchored on the needle's longest (most distinctive) word so
    that a long document does not turn every record into a full scan; the
    number of candidate positions is capped for the same reason. Coverage is
    asymmetric on purpose — a model that quotes a real sentence and then
    appends an invented clause only gets credit for the part that exists.
    """
    needle_words = needle.split()
    hay_words = haystack.split()
    if not needle_words or not hay_words:
        return 0.0, ""

    width = len(needle_words)
    anchor = max(needle_words, key=len)
    positions = [i for i, word in enumerate(hay_words) if word == anchor]
    if not positions:
        # Nothing distinctive in common: probe a few evenly spaced windows so
        # an unrelated span scores low instead of being skipped entirely.
        step = max(1, width)
        positions = list(range(0, max(1, len(hay_words) - width + 1), step))

    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq2(needle)
    best, best_window = 0.0, ""
    for position in positions[:MAX_ANCHOR_POSITIONS]:
        start = max(0, position - width)
        window = " ".join(hay_words[start:position + width])
        if not window:
            continue
        matcher.set_seq1(window)
        matched = sum(block.size for block in matcher.get_matching_blocks())
        ratio = matched / len(needle)
        if ratio > best:
            best, best_window = ratio, window
        if best >= 1.0:
            break
    return best, best_window


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


def annotate_grounding(record: dict, doc_text: str, numeric_fields=("value",),
                       unit_fields=None) -> dict:
    """Attach ``_grounded`` / ``_value_grounded`` / ``_unit_grounded`` flags.

    Every numeric field is checked, not just the first one that happens to be
    populated: confidence bounds, denominators and secondary readings are
    exactly where an invented number hides behind a correct headline value.
    ``_ungrounded`` names the fields that failed, so review can go straight to
    them instead of re-reading the record.

    ``_value_grounded`` is ``None`` when the record carries no number (the check
    does not apply), ``False`` when any number is unsupported by its evidence,
    ``True`` when every number traces back to the source. ``_unit_grounded``
    follows the same convention for declared units.
    """
    span = record.get("source_span")
    record["_grounded"] = span_is_grounded(span, doc_text)

    unit_map = _unit_map(numeric_fields, unit_fields or ())
    failures = []
    checked_number = False
    unit_verdicts = []

    for name in numeric_fields:
        value = record.get(name)
        if value is None:
            continue
        checked_number = True
        if not record["_grounded"] or not value_supported_by_span(value, span):
            failures.append(name)
            continue
        unit_field = unit_map.get(name)
        if unit_field:
            agrees = unit_agrees_with_span(value, record.get(unit_field), span)
            if agrees is not None:
                unit_verdicts.append(agrees)
                if not agrees:
                    failures.append(unit_field)

    record["_value_grounded"] = None if not checked_number else not any(
        f in numeric_fields for f in failures
    )
    record["_unit_grounded"] = None if not unit_verdicts else all(unit_verdicts)
    record["_ungrounded"] = failures
    return record


def _unit_map(numeric_fields, unit_fields) -> dict:
    """Pair each numeric field with the field that carries its unit.

    Templates spell this differently — ``value``/``unit`` in one, ``value``/
    ``value_unit`` in another — so match ``<field>_unit`` first and fall back to
    a single shared ``unit`` column.
    """
    available = set(unit_fields)
    shared = "unit" if "unit" in available else ""
    mapping = {}
    for name in numeric_fields:
        specific = f"{name}_unit"
        mapping[name] = specific if specific in available else shared
    return mapping


#: Unit spellings that mean the same thing. Applied after case folding and
#: whitespace removal, so only genuinely different units survive comparison.
_UNIT_ALIASES = {
    "\u00b5": "u", "\u03bc": "u", "\ufffd": "u",   # micro sign variants
    "litre": "l", "liter": "l", "litres": "l", "liters": "l",
    "gram": "g", "grams": "g", "gm": "g",
    "percent": "%", "pct": "%",
    "sec": "s", "second": "s", "seconds": "s",
    "minute": "min", "minutes": "min",
    "hour": "h", "hours": "h", "hr": "h", "hrs": "h",
    "day": "d", "days": "d",
}

#: A number immediately followed by its unit, e.g. "1.23 µg/mL", "45 %".
_NUMBER_UNIT_RE = re.compile(
    r"(?P<number>[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"\s*"
    r"(?P<unit>%|[A-Za-z\u00b5\u03bc\ufffd]+(?:\s*/\s*[A-Za-z\u00b5\u03bc\ufffd0-9]+)*)?"
)


def canon_unit(value) -> str:
    """Canonical form of a unit string, for comparison only (not conversion).

    Case, spacing and the many spellings of "micro" are noise; the SI prefix is
    not. ``µg/mL``, ``ug / ml`` and ``UG/ML`` all canonicalise together, while
    ``mg/mL`` stays distinct — which is the whole point, since confusing those
    two is a thousand-fold error.
    """
    text = canon_text(value).lower()
    if not text:
        return ""
    for source, replacement in _UNIT_ALIASES.items():
        text = text.replace(source, replacement)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"[\s\u00b7*]+", "", text)
    return text


def number_unit_pairs(text) -> list:
    """Extract ``(number, unit)`` pairs as written in a piece of text."""
    pairs = []
    for match in _NUMBER_UNIT_RE.finditer(str(text or "")):
        try:
            number = float(match.group("number").replace(",", ""))
        except ValueError:
            continue
        pairs.append((number, canon_unit(match.group("unit"))))
    return pairs


def unit_agrees_with_span(value, unit, span, rel_tol: float = 1e-6):
    """Compare a declared unit against the unit written beside the same number.

    Returns ``True`` when they agree, ``False`` when the span clearly states a
    different unit, and ``None`` when the question does not arise — no unit was
    declared, or the span never writes a unit next to that number. Silence is
    reported as "unknown" rather than "wrong", because a missing unit in the
    evidence is not evidence of a wrong unit.
    """
    declared = canon_unit(unit)
    if not declared or value is None:
        return None
    try:
        target = float(value)
    except (TypeError, ValueError):
        return None

    seen_with_unit = False
    for number, written in number_unit_pairs(span):
        if number != target and not (
            target != 0 and abs(number - target) <= rel_tol * abs(target)
        ):
            continue
        if not written:
            continue
        seen_with_unit = True
        if written == declared or written.startswith(declared) or declared.startswith(written):
            return True
    return False if seen_with_unit else None


def token_set(value) -> set:
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
