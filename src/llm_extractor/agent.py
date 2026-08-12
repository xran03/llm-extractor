"""Aggregation agent — one document, one JSON answer.

The text pass and the OCR pass produce two independent views of a document.
This stage hands both, already structured, to a small model and asks it to
reconcile them: summarise what was found, surface values that only the figures
revealed, and flag disagreements instead of silently picking a winner.

Design notes:

* the agent receives **JSON, not prose**, and answers under a strict schema, so
  its output is machine-consumable by a frontend;
* it never invents records — it annotates the ones the deterministic stages
  produced, which keeps the anti-hallucination guarantee intact;
* it is optional. :func:`deterministic_aggregate` produces the same envelope
  without an API call, so ``--no-aggregate`` (or a failed call) still yields a
  complete document artifact.
"""
from __future__ import annotations

import json

from .extract import grounding_summary
from .ocr import ocr_summary
from .parsing import extract_json_object
from .templates import AGGREGATE_JSON_SCHEMA, empty_aggregate

AGENT_SYSTEM_PROMPT = (
    "You are a data-reconciliation agent. You are given structured records "
    "extracted from a document's text and structured OCR readings of its "
    "figures. You reconcile the two views. You never invent facts, never add "
    "records, and only report what is present in the inputs."
)

AGENT_INSTRUCTIONS = (
    "Reconcile the TEXT_RECORDS and FIGURE_OCR inputs for this document.\n"
    "- summary: 2-4 factual sentences on what the document reports.\n"
    "- key_findings: the most important findings, each traceable to an input record.\n"
    "- figure_insights: values present in FIGURE_OCR but absent from TEXT_RECORDS.\n"
    "- conflicts: fields where the text and the figures disagree; state both sides "
    "and which is better evidenced, or 'unresolved'.\n"
    "- coverage_gaps: data the document clearly references but that neither pass "
    "captured.\n"
    "Use empty arrays when a section has nothing. Return JSON only."
)

#: Cap the payload handed to the agent: it reconciles, it does not re-extract.
MAX_RECORDS_IN_PROMPT = 200
MAX_FIGURES_IN_PROMPT = 30


def build_agent_messages(doc_id: str, records: list, figures: list,
                         title: str = "") -> list:
    payload = {
        "doc_id": doc_id,
        "title": title,
        "TEXT_RECORDS": [_slim_record(r) for r in records[:MAX_RECORDS_IN_PROMPT]],
        "FIGURE_OCR": [
            {"image": f.get("image"), **(f.get("ocr") or {})}
            for f in figures[:MAX_FIGURES_IN_PROMPT]
        ],
    }
    return [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user",
         "content": f"{AGENT_INSTRUCTIONS}\n\n{json.dumps(payload, ensure_ascii=False)}"},
    ]


def aggregate(provider, doc_id: str, records: list, figures: list, model: str,
              title: str = "", max_tokens: int = 4000) -> dict:
    """Run the reconciliation agent; fall back to a deterministic envelope."""
    if not records and not figures:
        return empty_aggregate()
    try:
        completion = provider.complete(
            build_agent_messages(doc_id, records, figures, title),
            model=model, temperature=0.0, max_tokens=max_tokens,
            json_schema=AGGREGATE_JSON_SCHEMA,
            meta={"stage": "aggregate", "doc_id": doc_id},
        )
        parsed = extract_json_object(completion.text)
    except Exception as exc:
        result = deterministic_aggregate(records, figures)
        result["summary"] = result["summary"] or ""
        result["coverage_gaps"].append(f"aggregation agent unavailable: {exc}")
        return result

    result = empty_aggregate()
    for key in result:
        value = parsed.get(key)
        if isinstance(result[key], list) and isinstance(value, list):
            result[key] = value
        elif isinstance(result[key], str) and isinstance(value, str):
            result[key] = value
    if not result["summary"]:
        result["summary"] = deterministic_aggregate(records, figures)["summary"]
    return result


def deterministic_aggregate(records: list, figures: list) -> dict:
    """Build the aggregate envelope with no model call (offline / fallback)."""
    result = empty_aggregate()
    grounding = grounding_summary(records)
    figure_stats = ocr_summary(figures)
    result["summary"] = (
        f"{grounding['records']} records extracted "
        f"({grounding['grounded']} grounded); "
        f"{figure_stats['figures']} figures read yielding "
        f"{figure_stats['numeric_items']} numeric readings."
    )
    for record in records:
        if record.get("_grounded") and record.get("source_span"):
            result["key_findings"].append(str(record["source_span"])[:200])
        if len(result["key_findings"]) >= 10:
            break
    for figure in figures:
        for item in (figure.get("ocr") or {}).get("items") or []:
            if item.get("value") is None:
                continue
            result["figure_insights"].append({
                "image": figure.get("image"),
                "finding": str(item.get("label") or ""),
                "value": item.get("value"),
                "unit": item.get("unit"),
            })
            if len(result["figure_insights"]) >= 25:
                break
    if grounding["values_ungrounded"]:
        result["coverage_gaps"].append(
            f"{grounding['values_ungrounded']} numeric values could not be traced "
            f"to their quoted evidence"
        )
    return result


def _slim_record(record: dict) -> dict:
    """Drop nulls and long spans so the agent prompt stays cheap."""
    slim = {k: v for k, v in record.items()
            if v is not None and not k.startswith("_") and k != "doc_id"}
    if isinstance(slim.get("source_span"), str):
        slim["source_span"] = slim["source_span"][:160]
    return slim
