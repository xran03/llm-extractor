"""Second-opinion review of extracted records against their source document.

Grounding already proves that a record's quoted span exists in the document and
that its digits appear in that span. That catches invention, but not
misattribution: a value can be real, its span real, and the row still wrong
because the number was collected under a different assay, group or timepoint
than the one the row names.

So this pass asks three questions per record, and only these three:

``value_ok``  the number is what the document reports for *this* row;
``unit_ok``   the unit is the one the document states, not a plausible guess;
``row_ok``    the fields describe one measurement rather than parts of several.

``row_ok`` is the one the deterministic checks cannot reach, and it is why the
whole row is shown to the reviewer rather than the value alone.

Records are reviewed against the chunk their evidence came from, not against a
truncated head of the document, so a record from page 40 is judged on page 40.
The reviewer is asked to abstain rather than guess when the chunk does not
settle the question — an unsupported "wrong" is as expensive as the error it
claims to find.
"""
from __future__ import annotations

import json

from .extract import chunk_text
from .normalize import span_is_grounded

#: Columns this pass adds. ``None`` means the record was never reviewed, which
#: is deliberately distinct from "reviewed and found fine".
REVIEW_COLUMNS = ("_review_value", "_review_unit", "_review_row", "_review_note")

#: How many records go into one request. Small enough that the reviewer keeps
#: the chunk and every row in view at once.
RECORDS_PER_REQUEST = 20

REVIEW_INSTRUCTIONS = (
    "You are checking records another system extracted from the document text "
    "below. You are not extracting anything new.\n\n"
    "For each record, answer exactly three questions:\n"
    "1. value_ok - does the document report this value for THIS record's "
    "subject, assay, group and timepoint? A number that appears in the "
    "document but belongs to a different row is value_ok=false.\n"
    "2. unit_ok - is the unit the one the document states for that value? "
    "A converted, assumed or omitted-but-guessed unit is unit_ok=false. Use "
    "null when the record states no unit and none is required.\n"
    "3. row_ok - do this record's fields describe ONE measurement? Set false "
    "when fields have been stitched together from different measurements - for "
    "example the group label from one arm with the value from another, or an "
    "assay that does not match the endpoint or unit.\n\n"
    "Judge only from the text provided. If the text does not settle a question, "
    "answer null for it rather than guessing. Put a short reason in 'issue' "
    "whenever you answer false, and leave it null otherwise. Return one verdict "
    "per record, using the record's 'index' exactly as given."
)

REVIEW_JSON_SCHEMA = {
    "name": "record_review",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdicts"],
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["index", "value_ok", "unit_ok", "row_ok", "issue"],
                    "properties": {
                        "index": {"type": "integer"},
                        "value_ok": {"type": ["boolean", "null"]},
                        "unit_ok": {"type": ["boolean", "null"]},
                        "row_ok": {"type": ["boolean", "null"]},
                        "issue": {"type": ["string", "null"]},
                    },
                },
            },
        },
    },
}


def reviewable_fields(template) -> tuple:
    """Fields worth showing the reviewer: the template's own, minus bookkeeping."""
    return tuple(name for name in template.field_names if not name.startswith("_"))


def _record_view(record: dict, index: int, fields) -> dict:
    """One record as the reviewer sees it: its own fields, plus its index."""
    view = {"index": index}
    for name in fields:
        value = record.get(name)
        if value not in (None, ""):
            view[name] = value
    return view


def assign_to_chunks(records: list, doc_text: str) -> list:
    """Group records by the chunk their evidence span belongs to.

    A record is reviewed against the text it was taken from. Anything whose span
    cannot be placed — including records with no span at all — goes to the first
    chunk, so it is still seen rather than silently skipped.
    """
    chunks = chunk_text(doc_text or "")
    if not chunks:
        return []
    groups: list = [[] for _ in chunks]
    for position, record in enumerate(records):
        span = record.get("source_span")
        target = 0
        if span:
            for index, chunk in enumerate(chunks):
                if span_is_grounded(span, chunk):
                    target = index
                    break
        groups[target].append(position)
    return [(chunks[i], members) for i, members in enumerate(groups) if members]


def build_messages(records: list, positions: list, chunk: str, template,
                   doc_id: str, label: str = "") -> list:
    """Render one review request for a slice of a document's records."""
    fields = reviewable_fields(template)
    payload = [_record_view(records[p], i, fields) for i, p in enumerate(positions)]
    header = f"=== DOCUMENT {doc_id}{label} START ==="
    return [
        {"role": "system",
         "content": "You are a meticulous reviewer. You verify claims against a "
                    "source text and never invent new ones."},
        {"role": "user",
         "content": (f"{REVIEW_INSTRUCTIONS}\n\n{header}\n{chunk}\n"
                     f"=== DOCUMENT END ===\n\nRECORDS TO CHECK:\n"
                     f"{json.dumps(payload, ensure_ascii=False, indent=1)}")},
    ]


def plan(records: list, doc_text: str, limit: int = RECORDS_PER_REQUEST) -> list:
    """Split a document's records into review requests.

    Returns ``(chunk, positions)`` pairs, where positions index into ``records``.
    """
    requests: list = []
    for chunk, members in assign_to_chunks(records, doc_text):
        for start in range(0, len(members), limit):
            requests.append((chunk, members[start:start + limit]))
    return requests


def parse_verdicts(payload) -> dict:
    """Read a reviewer reply into ``{index: verdict}``, ignoring malformed rows."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return {}
    if not isinstance(payload, dict):
        return {}
    verdicts: dict = {}
    for item in payload.get("verdicts") or []:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        verdicts[index] = item
    return verdicts


def apply_verdicts(records: list, positions: list, payload) -> int:
    """Attach one request's verdicts to the records it covered.

    The reply is addressed by the index handed out in the request, so a reply
    that comes back short, reordered or with an unknown index annotates only
    what it legitimately covers instead of shifting verdicts onto other rows.
    """
    verdicts = parse_verdicts(payload)
    applied = 0
    for offset, position in enumerate(positions):
        verdict = verdicts.get(offset)
        if verdict is None:
            continue
        record = records[position]
        record["_review_value"] = verdict.get("value_ok")
        record["_review_unit"] = verdict.get("unit_ok")
        record["_review_row"] = verdict.get("row_ok")
        record["_review_note"] = (verdict.get("issue") or None)
        applied += 1
    return applied


def summary(records: list) -> dict:
    """Run-level counts, so a review can be read without opening the CSV."""
    reviewed = [r for r in records if r.get("_review_row") is not None
                or r.get("_review_value") is not None
                or r.get("_review_unit") is not None]
    def _failed(field):
        return sum(1 for r in reviewed if r.get(field) is False)
    return {
        "reviewed": len(reviewed),
        "value_rejected": _failed("_review_value"),
        "unit_rejected": _failed("_review_unit"),
        "row_rejected": _failed("_review_row"),
        "flagged": sum(1 for r in reviewed
                       if any(r.get(f) is False for f in
                              ("_review_value", "_review_unit", "_review_row"))),
    }


class ReviewUnanswered(RuntimeError):
    """The reviewer returned no verdict for a whole request.

    A review that annotates nothing must not be mistaken for a review that
    found nothing wrong: the first means the check never ran. Some models
    answer this schema with an empty object, and swallowing that would let a
    run report clean records it never actually examined.
    """


def review_document(provider, records: list, doc_text: str, template, doc_id: str,
                    model: str, max_tokens: int = 4000, strict: bool = True) -> int:
    """Review one document's records live. Returns how many were annotated.

    Raises :class:`ReviewUnanswered` when a request comes back with no usable
    verdicts at all, unless ``strict`` is disabled.
    """
    requests = plan(records, doc_text)
    annotated = 0
    for index, (chunk, positions) in enumerate(requests):
        label = f" part {index + 1}/{len(requests)}" if len(requests) > 1 else ""
        messages = build_messages(records, positions, chunk, template, doc_id, label)
        completion = provider.complete(
            messages, model=model, temperature=0.0, max_tokens=max_tokens,
            json_schema=REVIEW_JSON_SCHEMA,
            meta={"stage": "review", "doc_id": doc_id},
        )
        applied = apply_verdicts(records, positions, completion.text)
        if applied == 0 and positions and strict:
            raise ReviewUnanswered(
                f"{model} returned no verdicts for {doc_id}{label} "
                f"({len(positions)} records); the review did not run"
            )
        annotated += applied
    return annotated
