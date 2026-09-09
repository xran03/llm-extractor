"""Batch API — submit a whole corpus as one queued job.

The live path spends one HTTP request per chunk and waits for each. For tens of
thousands of documents that is the wrong shape: it burns rate limit, holds a
process open for hours, and loses everything if the machine sleeps. The OpenAI
batch protocol inverts it — upload every request as JSONL, let the gateway work
through them within a completion window, then collect the answers.

Three commands, because the job outlives the process that starts it:

``submit``  ingest the corpus, upload the requests, record a manifest;
``status``  ask the gateway how far along the batch is;
``fetch``   download the answers and write the ordinary artifacts.

The request bodies come from the same ``build_payload`` the live path uses, so a
batched extraction and a live one cannot drift apart. Results are reassembled
through the manifest rather than by parsing ids, so a document that was split
into chunks comes back as one record set in the original order.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .extract import build_messages, chunk_text, dedupe
from .ingest import load_document, mime_for
from .normalize import annotate_grounding, coerce_record
from .parsing import extract_json_array, extract_json_object
from .pipeline import load_source_document
from .providers.base import ProviderError, image_part, user_message
from .repeatunits import annotate_records as annotate_repeat_units
from .repeatunits import summary as repeat_summary
from .templates import OCR_INSTRUCTIONS, OCR_JSON_SCHEMA, empty_ocr_payload

MANIFEST_NAME = "batch-manifest.json"
#: Terminal states in the OpenAI batch lifecycle.
FINISHED = ("completed", "failed", "expired", "cancelled")
#: What a queued request was for. The stage decides how its answer is decoded,
#: which is the whole of the "async alignment" problem: an answer arrives hours
#: later with nothing but a custom_id, so the manifest has to carry the rest.
STAGE_EXTRACT = "extract"
STAGE_OCR = "ocr"


@dataclass
class Task:
    """One queued request, and where its answer belongs."""

    custom_id: str
    doc_id: str
    stage: str = STAGE_EXTRACT
    chunk: int = 0
    n_chunks: int = 1
    source_path: str = ""
    title: str = ""
    uri: str = ""
    source_name: str = "folder"
    figure: str = ""          # image file name, for OCR tasks
    payload: dict = field(default_factory=dict)   # stage-specific context


@dataclass
class Manifest:
    """Everything needed to turn a finished batch back into artifacts."""

    batch_id: str = ""
    input_file_id: str = ""
    api: str = ""
    model: str = ""
    template: str = ""
    endpoint: str = ""
    created_at: float = field(default_factory=time.time)
    tasks: list = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tasks"] = [asdict(t) if not isinstance(t, dict) else t for t in self.tasks]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        manifest = cls(**{k: v for k, v in data.items() if k != "tasks"})
        manifest.tasks = [Task(**t) for t in data.get("tasks", [])]
        return manifest

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "Manifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def by_custom_id(self) -> dict:
        return {t.custom_id: t for t in self.tasks}

    def documents(self) -> list:
        """Document ids in the order they were submitted."""
        seen: dict = {}
        for task in self.tasks:
            seen.setdefault(task.doc_id, None)
        return list(seen)

    def tasks_for(self, doc_id: str, stage: str = "") -> list:
        return [t for t in self.tasks
                if t.doc_id == doc_id and (not stage or t.stage == stage)]


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
def preflight(provider) -> dict:
    """Report whether this gateway can actually run a batch.

    A gateway may expose the batch routes yet reject uploads because file
    storage is unconfigured, which is a deployment gap rather than a bug here.
    Probing both tells the user which of the two they are looking at.
    """
    report = {"list_batches": None, "upload_files": None, "ready": False, "detail": ""}

    try:
        provider.list_batches(limit=1)
        report["list_batches"] = "ok"
    except ProviderError as exc:
        report["list_batches"] = "failed"
        report["detail"] = str(exc)[:300]
        return report

    probe = json.dumps({
        "custom_id": "preflight-1",
        "method": "POST",
        "url": provider.INFERENCE_PATH,
        "body": provider.build_payload(
            [{"role": "user", "content": "ping"}], model="preflight", max_tokens=1),
    }).encode("utf-8")
    try:
        uploaded = provider.upload_file(probe, filename="preflight.jsonl")
        report["upload_files"] = "ok"
        report["ready"] = True
        file_id = uploaded.get("id")
        if file_id:
            try:
                provider.request("DELETE", f"{provider.FILES_PATH}/{file_id}")
            except ProviderError:
                report["detail"] = f"probe file {file_id} left on the gateway"
    except ProviderError as exc:
        report["upload_files"] = "failed"
        report["detail"] = str(exc)[:300]
    return report


# --------------------------------------------------------------------------
# Submit
# --------------------------------------------------------------------------
def build_requests(provider, documents, template, settings, with_ocr: bool = False) -> tuple:
    """Render a corpus into batch request lines plus the manifest tasks.

    Both stages are queued in the same batch: figures cost the same round trip
    as text, and splitting them into two jobs would double the waiting.
    """
    schema = template.json_schema()
    lines: list = []
    tasks: list = []

    def add(task: Task, body: dict) -> None:
        lines.append({"custom_id": task.custom_id, "method": "POST",
                      "url": provider.INFERENCE_PATH, "body": body})
        tasks.append(task)

    for source_doc in documents:
        document = load_source_document(source_doc, cache_dir=settings.cache_dir,
                                        with_figures=with_ocr,
                                        max_figures=settings.max_figures)
        common = dict(doc_id=document.doc_id, source_path=document.source_path,
                      title=source_doc.title, uri=source_doc.uri,
                      source_name=source_doc.source_name or "folder")

        chunks = chunk_text(document.text)
        for index, chunk in enumerate(chunks):
            label = f" part {index + 1}/{len(chunks)}" if len(chunks) > 1 else ""
            add(
                Task(custom_id=f"r-{len(tasks)}", stage=STAGE_EXTRACT, chunk=index,
                     n_chunks=len(chunks), **common),
                provider.build_payload(
                    build_messages(template, chunk, document.doc_id, label),
                    model=settings.model, temperature=settings.temperature,
                    max_tokens=settings.max_output_tokens, json_schema=schema),
            )

        if not with_ocr:
            continue
        context = (document.text or "")[:1500]
        for figure in document.figures:
            figure = Path(figure)
            try:
                data = figure.read_bytes()
            except OSError:
                continue
            prompt = OCR_INSTRUCTIONS
            if context:
                prompt += ("\n\nDocument context (do not copy from it, only use it "
                           f"to disambiguate labels):\n{context}")
            add(
                Task(custom_id=f"r-{len(tasks)}", stage=STAGE_OCR,
                     figure=figure.name, **common),
                provider.build_payload(
                    [user_message(prompt, image_part(data, mime=mime_for(figure)))],
                    model=settings.ocr_model, temperature=0.0,
                    max_tokens=4000, json_schema=OCR_JSON_SCHEMA),
            )

    return lines, tasks


def submit(provider, documents, template, settings, out_dir,
           completion_window: str = "24h", with_ocr: bool = False) -> Manifest:
    """Upload the corpus as one batch and record the manifest."""
    lines, tasks = build_requests(provider, documents, template, settings,
                                  with_ocr=with_ocr)
    if not lines:
        raise ValueError("no readable text or figures in the selected documents")

    payload = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines)
    uploaded = provider.upload_file(payload.encode("utf-8"), filename="requests.jsonl")
    file_id = uploaded.get("id")
    if not file_id:
        raise ProviderError(f"upload returned no file id: {str(uploaded)[:200]}")

    batch = provider.create_batch(file_id, endpoint=provider.INFERENCE_PATH,
                                  completion_window=completion_window)
    batch_id = batch.get("id")
    if not batch_id:
        raise ProviderError(f"batch creation returned no id: {str(batch)[:200]}")

    manifest = Manifest(
        batch_id=batch_id, input_file_id=file_id, api=settings.api,
        model=settings.model, template=template.name,
        endpoint=provider.INFERENCE_PATH, tasks=tasks,
    )
    manifest.save(Path(out_dir) / MANIFEST_NAME)
    return manifest


# --------------------------------------------------------------------------
# Collect
# --------------------------------------------------------------------------
def parse_output(provider, raw_jsonl: bytes) -> dict:
    """Map ``custom_id`` to the assistant text (or an error) for each answer."""
    answers: dict = {}
    for line in raw_jsonl.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        custom_id = entry.get("custom_id")
        if not custom_id:
            continue
        if entry.get("error"):
            answers[custom_id] = {"error": str(entry["error"])[:300], "text": ""}
            continue
        response = entry.get("response") or {}
        status = response.get("status_code")
        body = response.get("body") or {}
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                body = {}
        if status and status >= 400:
            answers[custom_id] = {"error": f"HTTP {status}: {str(body)[:200]}", "text": ""}
            continue
        try:
            text = provider.parse_completion(body).text
        except Exception as exc:
            answers[custom_id] = {"error": f"unreadable response: {exc}", "text": ""}
            continue
        answers[custom_id] = {"error": "", "text": text}
    return answers


def records_for_document(answers, tasks, template, doc_text: str) -> tuple:
    """Rebuild one document's records from its chunk answers, in order."""
    from .extract import _numeric_fields, _unit_fields

    records: list = []
    errors: list = []
    for task in sorted(tasks, key=lambda t: t.chunk):
        answer = answers.get(task.custom_id)
        if answer is None:
            errors.append(f"chunk {task.chunk + 1}: no answer returned")
            continue
        if answer["error"]:
            errors.append(f"chunk {task.chunk + 1}: {answer['error']}")
            continue
        try:
            raw_records = extract_json_array(answer["text"])
        except ValueError as exc:
            errors.append(f"chunk {task.chunk + 1}: unparsable output: {exc}")
            continue
        for raw in raw_records:
            if not isinstance(raw, dict):
                continue
            record = coerce_record(raw, template, task.doc_id)
            annotate_grounding(record, doc_text,
                               numeric_fields=_numeric_fields(template),
                               unit_fields=_unit_fields(template))
            records.append(record)
    return dedupe(records, template), errors


def figures_for_document(answers, tasks) -> tuple:
    """Rebuild one document's OCR payloads from its figure answers."""
    figures: list = []
    errors: list = []
    for task in sorted(tasks, key=lambda t: t.figure):
        answer = answers.get(task.custom_id)
        if answer is None:
            errors.append(f"figure {task.figure}: no answer returned")
            continue
        if answer["error"]:
            errors.append(f"figure {task.figure}: {answer['error']}")
            continue
        payload = extract_json_object(answer["text"]) or empty_ocr_payload()
        base = empty_ocr_payload()
        for key in base:
            if key in payload and payload[key] is not None:
                base[key] = payload[key]
        for list_key in ("items", "tables", "text_blocks"):
            if not isinstance(base[list_key], list):
                base[list_key] = []
        figures.append({"image": task.figure, "ocr": base})
    return figures, errors


def collect(provider, manifest: Manifest, template, settings, out_dir) -> dict:
    """Download a finished batch and write the ordinary per-document artifacts.

    This is where the asynchronous shape is paid for: the answers arrive as one
    flat stream, in no particular order, identified only by ``custom_id``. The
    manifest maps each id back to its document, stage and position, so records
    and figures land where they belong and a document split into ten chunks
    still produces one ordered record set.
    """
    from .agent import aggregate, deterministic_aggregate
    from .extract import grounding_summary
    from .ocr import ocr_summary
    from .pipeline import DocumentResult, write_artifacts
    from .serialize import append_figures_csv, append_records_csv, figure_rows
    from .sources import SourceDocument

    batch = provider.get_batch(manifest.batch_id)
    state = batch.get("status")
    if state not in FINISHED:
        raise ProviderError(f"batch {manifest.batch_id} is still {state}")

    output_id = batch.get("output_file_id")
    if not output_id:
        raise ProviderError(
            f"batch {manifest.batch_id} finished as {state} with no output file; "
            f"errors: {batch.get('error_file_id') or 'none reported'}")

    answers = parse_output(provider, provider.download_file(output_id))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    wants_csv = settings.output_format in ("csv", "both")
    combined_records = out_path / "records.csv"
    combined_figures = out_path / "figures.csv"
    started = {"records": False, "figures": False}

    summary = {"batch_id": manifest.batch_id, "status": state, "documents": 0,
               "records": 0, "figures": 0, "errors": []}

    for doc_id in manifest.documents():
        doc_tasks = manifest.tasks_for(doc_id)
        first = doc_tasks[0]
        source_doc = SourceDocument(
            doc_id=doc_id, title=first.title, uri=first.uri,
            path=first.source_path, source_name=first.source_name)

        # Grounding needs the document's own text. Re-reading is deterministic
        # and keeps the manifest small; a moved source is reported, not guessed.
        doc_text = ""
        try:
            if first.source_path:
                doc_text = load_document(first.source_path, doc_id=doc_id).text
        except Exception as exc:
            summary["errors"].append(f"{doc_id}: source unreadable at collect time: {exc}")

        records, record_errors = records_for_document(
            answers, [t for t in doc_tasks if t.stage == STAGE_EXTRACT], template, doc_text)
        figures, figure_errors = figures_for_document(
            answers, [t for t in doc_tasks if t.stage == STAGE_OCR])

        # Same enrichment as the live path, so a batched run and a live run
        # produce the same columns.
        repeat_stats = annotate_repeat_units(records)

        result = DocumentResult(doc_id=doc_id, source_doc=source_doc,
                                source_path=first.source_path,
                                records=records, figures=figures,
                                errors=record_errors + figure_errors)
        result.aggregate = (
            aggregate(provider, doc_id, records, figures, model=settings.agent_model,
                      title=first.title)
            if settings.aggregate else deterministic_aggregate(records, figures)
        )
        result.stats = {
            "template": template.name, "api": settings.api, "model": settings.model,
            "mode": "batch", "batch_id": manifest.batch_id,
            "text_chars": len(doc_text),
            **grounding_summary(records),
            **repeat_summary(repeat_stats),
            **{f"ocr_{k}": v for k, v in ocr_summary(figures).items()},
        }
        result.artifacts = write_artifacts(result, source_doc, out_path, template,
                                           output_format=settings.output_format)

        if wants_csv and records:
            append_records_csv(combined_records, records, template,
                               doc_title=first.title,
                               write_header=not started["records"])
            started["records"] = True
        rows = figure_rows(figures, doc_id, first.title)
        if wants_csv and rows:
            append_figures_csv(combined_figures, rows,
                               write_header=not started["figures"])
            started["figures"] = True

        summary["documents"] += 1
        summary["records"] += len(records)
        summary["figures"] += len(figures)
        summary["errors"].extend(f"{doc_id}: {e}" for e in result.errors)

    (out_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
