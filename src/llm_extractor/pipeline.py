"""Per-document pipeline: ingest → text extraction → figure OCR → aggregate.

Text extraction and figure OCR are independent stages; the aggregation agent
joins their results afterwards. Three OCR policies trade cost against recall:

``never``   text only — cheapest;
``auto``    OCR when the document is image-only, has no usable text layer, or
            the text pass produced nothing/ungrounded records (default);
``always``  both passes every time — highest recall, highest cost.

Artifacts written per document:

``<doc_id>.records.jsonl``   one extracted record per line (stable format)
``<doc_id>.ocr.json``        structured OCR readings per figure
``<doc_id>.document.json``   the aggregated document envelope
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ._exec import run_both
from .agent import aggregate, deterministic_aggregate
from .bus import STAGE_COMPLETED, STAGE_STARTED, Event
from .extract import ExtractionResult, extract_records, grounding_summary
from .ingest import Document, load_document
from .ocr import OCRResult, ocr_document, ocr_summary
from .serialize import figure_rows, write_figures_csv, write_records_csv

OCR_POLICIES = ("auto", "always", "never")
#: Below this many characters a "text" document is treated as having no usable
#: text layer (typical of scanned PDFs), which triggers OCR under ``auto``.
MIN_TEXT_CHARS = 200
#: Artifact formats. ``jsonl`` is lossless; ``csv`` is what people analyse.
OUTPUT_FORMATS = ("jsonl", "csv", "both")


@dataclass
class DocumentResult:
    doc_id: str
    source_path: str = ""
    records: list = field(default_factory=list)
    figures: list = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    source_doc: object = None

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "stats": self.stats,
            "aggregate": self.aggregate,
            "records": self.records,
            "figures": self.figures,
            "errors": self.errors,
            "artifacts": self.artifacts,
        }


def load_source_document(source_doc, cache_dir=None, with_figures: bool = False,
                         max_figures: int = 20) -> Document:
    """Materialise a :class:`SourceDocument` into an ingested :class:`Document`.

    File-backed documents go through the format readers; API-backed documents
    already carry text; byte blobs are spooled to the cache directory so the
    same readers apply.
    """
    if source_doc.path:
        return load_document(source_doc.path, doc_id=source_doc.doc_id,
                             with_figures=with_figures, cache_dir=cache_dir,
                             max_figures=max_figures)
    if source_doc.blob:
        suffix = f".{source_doc.media_type.lstrip('.')}" if source_doc.media_type else ".bin"
        spool_dir = Path(cache_dir or ".llm_cache") / "spool"
        spool_dir.mkdir(parents=True, exist_ok=True)
        spooled = spool_dir / f"{source_doc.doc_id}{suffix}"
        if not spooled.exists():
            spooled.write_bytes(source_doc.blob)
        return load_document(spooled, doc_id=source_doc.doc_id,
                             with_figures=with_figures, cache_dir=cache_dir,
                             max_figures=max_figures)
    return Document(doc_id=source_doc.doc_id, text=source_doc.text,
                    source_path=source_doc.uri)


def should_run_ocr(document: Document, policy: str, extraction=None) -> bool:
    """Decide whether the vision pass runs for this document.

    The format declares whether it normally has a text layer, so a scanned PDF
    and a PNG take the same path without the pipeline knowing either extension.
    """
    if policy == "never":
        return False
    if not document.figures:
        return False
    if policy == "always":
        return True
    # auto: the format has no text layer, or none was recovered in practice.
    if not document.has_text_layer:
        return True
    if len((document.text or "").strip()) < MIN_TEXT_CHARS:
        return True
    if extraction is None:
        return False
    return (not extraction.records) or any(
        r.get("_grounded") is False for r in extraction.records
    )


def run_document(provider, source_doc, settings, template, out_dir,
                 bus=None, job_id: str = "") -> DocumentResult:
    """Run the full pipeline for one document and write its artifacts."""
    started = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = DocumentResult(doc_id=source_doc.doc_id, source_doc=source_doc)

    policy = settings.ocr if settings.ocr in OCR_POLICIES else "auto"
    document = load_source_document(
        source_doc, cache_dir=settings.cache_dir,
        with_figures=policy != "never", max_figures=settings.max_figures,
    )
    result.source_path = document.source_path or source_doc.uri
    if document.read_error:
        # The text layer was unreadable; the vision pass is carrying this one.
        result.errors.append(f"ingest: {document.read_error}")

    def _emit(event_type: str, stage: str, **payload) -> None:
        if bus is not None:
            bus.publish(Event(type=event_type, job_id=job_id, doc_id=source_doc.doc_id,
                              stage=stage, payload=payload))

    # --- text ‖ OCR -------------------------------------------------------
    extraction = ExtractionResult()
    ocr_result = OCRResult()
    run_text = bool((document.text or "").strip())
    run_ocr_now = policy == "always" or (
        policy == "auto" and (not document.has_text_layer or not run_text)
    )
    run_ocr_now = run_ocr_now and bool(document.figures)

    def _text_stage():
        _emit(STAGE_STARTED, "extract")
        outcome = extract_records(
            provider, document, template, model=settings.model,
            max_tokens=settings.max_output_tokens, temperature=settings.temperature,
        )
        _emit(STAGE_COMPLETED, "extract", records=len(outcome.records), **outcome.usage())
        return outcome

    def _ocr_stage():
        _emit(STAGE_STARTED, "ocr", figures=len(document.figures))
        outcome = ocr_document(
            provider, document, model=settings.ocr_model,
            max_figures=settings.max_figures,
        )
        _emit(STAGE_COMPLETED, "ocr", **outcome.usage())
        return outcome

    if run_text and run_ocr_now:
        text_outcome, ocr_outcome = run_both(_text_stage, _ocr_stage)
        extraction = _unwrap(text_outcome, result, "extract") or ExtractionResult()
        ocr_result = _unwrap(ocr_outcome, result, "ocr") or OCRResult()
    else:
        if run_text:
            extraction = _call(_text_stage, result, "extract") or ExtractionResult()
        if run_ocr_now:
            ocr_result = _call(_ocr_stage, result, "ocr") or OCRResult()
        elif should_run_ocr(document, policy, extraction):
            # Fallback: the text pass came back empty or ungrounded.
            ocr_result = _call(_ocr_stage, result, "ocr") or OCRResult()

    result.records = extraction.records
    result.figures = ocr_result.figures
    result.errors.extend(extraction.errors + ocr_result.errors)

    # --- aggregation ------------------------------------------------------
    if settings.aggregate:
        _emit(STAGE_STARTED, "aggregate")
        result.aggregate = aggregate(
            provider, source_doc.doc_id, result.records, result.figures,
            model=settings.agent_model, title=source_doc.title,
        )
        _emit(STAGE_COMPLETED, "aggregate")
    else:
        result.aggregate = deterministic_aggregate(result.records, result.figures)

    # --- stats + artifacts ------------------------------------------------
    result.stats = {
        "template": template.name,
        "api": settings.api,
        "model": settings.model,
        "format": document.format_name,
        "media_type": document.media_type,
        "ocr_model": settings.ocr_model if result.figures else None,
        "ocr_policy": policy,
        "ocr_executed": bool(result.figures),
        "text_chars": len(document.text or ""),
        "duration_s": round(time.monotonic() - started, 3),
        "prompt_tokens": extraction.prompt_tokens + ocr_result.prompt_tokens,
        "completion_tokens": extraction.completion_tokens + ocr_result.completion_tokens,
        "cached_calls": extraction.cached_calls + ocr_result.cached_calls,
        **grounding_summary(result.records),
        **{f"ocr_{k}": v for k, v in ocr_summary(result.figures).items()},
    }
    result.artifacts = write_artifacts(result, source_doc, out_dir, template,
                                       output_format=settings.output_format)
    return result


def write_artifacts(result: DocumentResult, source_doc, out_dir: Path, template,
                    output_format: str = "both") -> dict:
    """Write the per-document artifacts and return their paths.

    JSONL keeps everything (nested figures, audit flags); CSV is the flat table
    for analysis. ``document.json`` is always written because it is the only
    artifact that carries the aggregate.
    """
    out_dir = Path(out_dir)
    artifacts: dict = {}
    wants_jsonl = output_format in ("jsonl", "both")
    wants_csv = output_format in ("csv", "both")

    if wants_jsonl:
        records_path = out_dir / f"{result.doc_id}.records.jsonl"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("w", encoding="utf-8") as fh:
            for record in result.records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        artifacts["records"] = str(records_path)

    if wants_csv:
        artifacts["records_csv"] = str(write_records_csv(
            out_dir / f"{result.doc_id}.records.csv", result.records, template,
            doc_title=source_doc.title,
        ))

    if result.figures:
        ocr_path = out_dir / f"{result.doc_id}.ocr.json"
        _write_json(ocr_path, result.figures)
        artifacts["ocr"] = str(ocr_path)
        if wants_csv:
            artifacts["figures_csv"] = str(write_figures_csv(
                out_dir / f"{result.doc_id}.figures.csv",
                figure_rows(result.figures, result.doc_id, source_doc.title),
            ))

    document_path = out_dir / f"{result.doc_id}.document.json"
    _write_json(document_path, {
        "doc_id": result.doc_id,
        "title": source_doc.title,
        "uri": source_doc.uri,
        "source": source_doc.source_name,
        "source_path": result.source_path,
        "metadata": source_doc.metadata,
        "stats": result.stats,
        "aggregate": result.aggregate,
        "records": result.records,
        "figures": result.figures,
        "errors": result.errors,
        "generated_at": time.time(),
    })
    artifacts["document"] = str(document_path)
    return artifacts


def load_records(path) -> list:
    """Read records from one ``.records.jsonl`` file, or a directory of them."""
    path = Path(path)
    files = sorted(path.glob("*.records.jsonl")) if path.is_dir() else [path]
    records = []
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _call(fn, result: DocumentResult, stage: str):
    try:
        return fn()
    except Exception as exc:
        result.errors.append(f"{stage}: {type(exc).__name__}: {exc}")
        return None


def _unwrap(outcome, result: DocumentResult, stage: str):
    """``run_both`` returns exceptions rather than raising them."""
    if isinstance(outcome, Exception):
        result.errors.append(f"{stage}: {type(outcome).__name__}: {outcome}")
        return None
    return outcome
