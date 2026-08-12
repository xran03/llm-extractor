"""Vision OCR stage — figures and scanned pages, answered in JSON.

Numbers frequently live only in charts and rasterised tables, so text
extraction alone under-reports. This stage sends each figure to a vision model
constrained by :data:`~llm_extractor.templates.OCR_JSON_SCHEMA`, so the OCR
result is structured data (items, tables, axes, text blocks) rather than prose —
which is what makes it mergeable with text records by the aggregation agent.

Figures are read one by one, and one unreadable figure never fails a document.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ._exec import map_ordered
from .ingest import mime_for
from .parsing import extract_json_object
from .providers.base import image_part, user_message
from .templates import OCR_INSTRUCTIONS, OCR_JSON_SCHEMA, empty_ocr_payload

#: Skip absurdly large images rather than paying to upload them.
MAX_IMAGE_BYTES = 12 * 1024 * 1024


@dataclass
class OCRResult:
    figures: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_calls: int = 0
    errors: list = field(default_factory=list)

    def usage(self) -> dict:
        return {
            "figures": len(self.figures),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_calls": self.cached_calls,
        }


def ocr_figure(provider, image_path, model: str, doc_id: str = "",
               context: str = "", max_tokens: int = 4000) -> dict:
    """Transcribe one image into the OCR JSON schema."""
    path = Path(image_path)
    data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        payload = empty_ocr_payload()
        payload["notes"] = f"skipped: image too large ({len(data)} bytes)"
        return {"image": path.name, "ocr": payload, "usage": {}, "skipped": True}

    prompt = OCR_INSTRUCTIONS
    if context:
        prompt += f"\n\nDocument context (do not copy from it, only use it to disambiguate labels):\n{context[:1500]}"

    completion = provider.complete(
        [user_message(prompt, image_part(data, mime=mime_for(path)))],
        model=model, temperature=0.0, max_tokens=max_tokens,
        json_schema=OCR_JSON_SCHEMA,
        meta={"stage": "ocr", "doc_id": doc_id},
    )
    payload = extract_json_object(completion.text) or empty_ocr_payload()
    return {
        "image": path.name,
        "path": str(path),
        "ocr": _normalize_payload(payload),
        "usage": completion.usage.to_dict(),
    }


def ocr_document(provider, document, model: str, max_workers: int = 4,
                 max_figures: int = 20, max_tokens: int = 4000) -> OCRResult:
    """OCR every discovered figure of a document."""
    result = OCRResult()
    figures = list(document.figures)[:max_figures]
    if not figures:
        return result

    context = (document.text or "")[:1500]

    def _one(path):
        try:
            return ocr_figure(provider, path, model, doc_id=document.doc_id,
                              context=context, max_tokens=max_tokens)
        except Exception as exc:
            result.errors.append(f"{Path(path).name}: {type(exc).__name__}: {exc}")
            return None

    for outcome in map_ordered(_one, figures, workers=max_workers):
        if outcome is None:
            continue
        usage = outcome.get("usage") or {}
        result.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        result.completion_tokens += int(usage.get("completion_tokens") or 0)
        result.cached_calls += 1 if usage.get("cached") else 0
        result.figures.append(outcome)
    return result


def _normalize_payload(payload: dict) -> dict:
    """Guarantee the OCR payload shape even when a model returns a partial object."""
    base = empty_ocr_payload()
    for key in base:
        if key in payload and payload[key] is not None:
            base[key] = payload[key]
    for list_key in ("items", "tables", "text_blocks"):
        if not isinstance(base[list_key], list):
            base[list_key] = []
    return base


def ocr_to_text(figures: list) -> str:
    """Flatten OCR JSON into readable text for prompts and grounding checks."""
    lines: list = []
    for figure in figures:
        payload = figure.get("ocr") or {}
        lines.append(f"[figure: {figure.get('image', '?')}]")
        if payload.get("caption"):
            lines.append(f"caption: {payload['caption']}")
        for axis in ("axis_x", "axis_y"):
            if payload.get(axis):
                lines.append(f"{axis}: {payload[axis]}")
        for item in payload.get("items") or []:
            value = item.get("value")
            value = item.get("value_text") if value is None else value
            parts = [str(item.get("label") or "")]
            if item.get("series"):
                parts.append(f"({item['series']})")
            parts.append(f"= {value}")
            if item.get("unit"):
                parts.append(str(item["unit"]))
            if item.get("note"):
                parts.append(f"- {item['note']}")
            lines.append(" ".join(p for p in parts if p))
        for table in payload.get("tables") or []:
            if table.get("title"):
                lines.append(f"table: {table['title']}")
            if table.get("columns"):
                lines.append(" | ".join(str(c) for c in table["columns"]))
            for row in table.get("rows") or []:
                lines.append(" | ".join("" if c is None else str(c) for c in row))
        for block in payload.get("text_blocks") or []:
            lines.append(str(block))
        lines.append("")
    return "\n".join(lines).strip()


def ocr_summary(figures: list) -> dict:
    return {
        "figures": len(figures),
        "items": sum(len((f.get("ocr") or {}).get("items") or []) for f in figures),
        "tables": sum(len((f.get("ocr") or {}).get("tables") or []) for f in figures),
        "numeric_items": sum(
            1 for f in figures
            for item in (f.get("ocr") or {}).get("items") or []
            if item.get("value") is not None
        ),
    }
