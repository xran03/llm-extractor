"""Text extraction stage: document text in, schema-constrained records out.

Long documents are split into overlapping chunks and extracted chunk by chunk,
then merged and de-duplicated. Chunking matters for two reasons: a 200-page PDF
does not fit in a context window, and per-chunk calls cache independently, so
adding one page to a document does not invalidate the whole extraction.

Every record is coerced to the template schema and annotated with grounding
flags before it leaves this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ._exec import map_ordered
from .normalize import annotate_grounding, coerce_record
from .parsing import extract_json_array

#: Characters per chunk. ~4 chars/token, so 60k chars ≈ 15k tokens of input,
#: which is comfortable for every current long-context model.
DEFAULT_CHUNK_CHARS = 60_000
DEFAULT_OVERLAP_CHARS = 2_000


@dataclass
class ExtractionResult:
    records: list = field(default_factory=list)
    chunks: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_calls: int = 0
    errors: list = field(default_factory=list)

    def usage(self) -> dict:
        return {
            "chunks": self.chunks,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_calls": self.cached_calls,
        }


def chunk_text(text: str, size: int = DEFAULT_CHUNK_CHARS,
               overlap: int = DEFAULT_OVERLAP_CHARS) -> list:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    text = text or ""
    size = max(int(size), 1)
    # Overlap must stay well below the chunk size, otherwise the window barely
    # advances and a large document explodes into near-duplicate chunks.
    overlap = max(0, min(int(overlap), size // 4))
    if len(text) <= size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Back off to the last paragraph break in the final 20% of the chunk.
            window = text.rfind("\n\n", start + int(size * 0.8), end)
            if window != -1:
                end = window
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_messages(template, doc_text: str, doc_id: str, chunk_info: str = "") -> list:
    header = f"=== DOCUMENT {doc_id}{chunk_info} START ==="
    return [
        {"role": "system", "content": template.system_prompt},
        {"role": "user",
         "content": f"{template.prompt()}\n\n{header}\n{doc_text}\n=== DOCUMENT END ==="},
    ]


def extract_records(provider, document, template, model: str,
                    max_tokens: int = 16000, temperature: float = 0.0,
                    chunk_chars: int = DEFAULT_CHUNK_CHARS,
                    max_workers: int = 4, use_json_schema: bool = True) -> ExtractionResult:
    """Run the extraction agent over one document."""
    result = ExtractionResult()
    chunks = chunk_text(document.text, size=chunk_chars)
    result.chunks = len(chunks)
    if not chunks:
        return result

    schema = template.json_schema() if use_json_schema else None

    def _one(indexed):
        index, chunk = indexed
        label = f" part {index + 1}/{len(chunks)}" if len(chunks) > 1 else ""
        messages = build_messages(template, chunk, document.doc_id, label)
        return index, provider.complete(
            messages, model=model, temperature=temperature, max_tokens=max_tokens,
            json_schema=schema,
            meta={"stage": "extract", "doc_id": document.doc_id, "template": template.name},
        )

    completions = [
        outcome for outcome in map_ordered(_safe(_one, result), list(enumerate(chunks)),
                                           workers=max_workers)
        if outcome is not None
    ]

    for _, completion in sorted(completions, key=lambda pair: pair[0]):
        result.prompt_tokens += completion.usage.prompt_tokens
        result.completion_tokens += completion.usage.completion_tokens
        result.cached_calls += 1 if completion.usage.cached else 0
        try:
            raw_records = extract_json_array(completion.text)
        except ValueError as exc:
            result.errors.append(f"unparsable model output: {exc}")
            continue
        for raw in raw_records:
            if not isinstance(raw, dict):
                continue
            record = coerce_record(raw, template, document.doc_id)
            annotate_grounding(record, document.text,
                               numeric_fields=_numeric_fields(template),
                               unit_fields=_unit_fields(template))
            result.records.append(record)

    result.records = dedupe(result.records, template)
    return result


def _numeric_fields(template) -> tuple:
    return tuple(f.name for f in template.fields if f.type in ("number", "integer"))


def _unit_fields(template) -> tuple:
    """Fields that carry a unit, by convention ``unit`` or ``<field>_unit``."""
    return tuple(f.name for f in template.fields
                 if f.type == "string" and (f.name == "unit" or f.name.endswith("_unit")))


def _safe(fn, result: ExtractionResult):
    """Wrap a chunk worker so one failed chunk does not lose the others."""
    def wrapper(item):
        try:
            return fn(item)
        except Exception as exc:
            result.errors.append(f"{type(exc).__name__}: {exc}")
            return None

    return wrapper


def dedupe(records: list, template) -> list:
    """Drop duplicates produced by overlapping chunks.

    Identity is the template's key fields plus the evidence span, so the same
    fact quoted from the same sentence collapses to one record while genuinely
    repeated measurements are preserved.
    """
    key_fields = template.key_fields or template.field_names[:2]
    seen = set()
    unique = []
    for record in records:
        signature = tuple(str(record.get(f) or "").lower() for f in key_fields)
        signature += (
            str(record.get("source_span") or "")[:120].lower(),
            str(record.get("value") if "value" in record else ""),
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(record)
    return unique


def grounding_summary(records: list) -> dict:
    """Counts used for quality gates and the document artifact."""
    numeric = [r for r in records if r.get("_value_grounded") is not None]
    return {
        "records": len(records),
        "grounded": sum(1 for r in records if r.get("_grounded")),
        "ungrounded": sum(1 for r in records if r.get("_grounded") is False),
        "numeric_records": len(numeric),
        "values_grounded": sum(1 for r in numeric if r.get("_value_grounded") is True),
        "values_ungrounded": sum(1 for r in numeric if r.get("_value_grounded") is False),
    }
