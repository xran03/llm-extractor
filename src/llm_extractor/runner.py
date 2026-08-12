"""Job runner: source → scheduler → pipeline → artifacts.

This is the orchestration entry point shared by the CLI and the HTTP API. It
owns the run-level concerns the per-document pipeline deliberately does not:

* pulling documents from any registered :class:`~llm_extractor.sources.Source`;
* skipping work already completed for the same content hash (resume);
* fanning out across workers under a rate limit;
* persisting job/task state and emitting events for live progress;
* writing the run-level ``summary.json``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .bus import JOB_COMPLETED, JOB_FAILED, JOB_STARTED, DEFAULT_BUS, Event, EventBus
from .jobstore import (JobStore, STATUS_ERROR, STATUS_OK, STATUS_RUNNING,
                       STATUS_SKIPPED)
from .pipeline import run_document
from .providers import build_provider
from .scheduler import RateLimiter, Scheduler, Skip
from .serialize import append_figures_csv, append_records_csv, figure_rows
from .sources import build_source
from .templates import load_template


@dataclass
class RunSummary:
    job_id: str
    total: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    records: int = 0
    figures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_calls: int = 0
    duration_s: float = 0.0
    output_dir: str = ""
    tables: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        data["duration_s"] = round(self.duration_s, 2)
        return data


def run_job(settings, source_name: str = "folder", source_params: dict | None = None,
            out_dir: str = "out", bus: EventBus | None = None,
            store: JobStore | None = None, job_id: str | None = None,
            resume: bool = True, rate_limit: int = 0, scheduler=None,
            cache=None) -> RunSummary:
    """Execute one extraction job end to end.

    Resources this function creates (job store, response cache) are closed
    before returning; resources passed in by the caller are left open, because
    the caller — the CLI or the long-lived HTTP service — owns their lifetime.
    """
    bus = bus or DEFAULT_BUS
    owned_store = store is None
    store = store or JobStore(str(Path(settings.cache_dir) / "jobs.sqlite3"))
    owned_cache = None
    if cache is None and settings.cache_enabled:
        from .cache import ResponseCache

        cache = owned_cache = ResponseCache(settings.cache_dir)
    try:
        return _run_job(settings, source_name, source_params, out_dir, bus, store,
                        job_id, resume, rate_limit, scheduler, cache)
    finally:
        if owned_cache is not None:
            owned_cache.close()
        if owned_store:
            store.close()


def _run_job(settings, source_name, source_params, out_dir, bus, store, job_id,
             resume, rate_limit, scheduler, cache) -> RunSummary:
    template = load_template(settings.template)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    params = {"source": source_name, **(source_params or {}),
              "api": settings.api, "model": settings.model, "template": template.name,
              "out_dir": str(out_path)}
    job_id = store.create_job(source_name, params, job_id=job_id)
    summary = RunSummary(job_id=job_id, output_dir=str(out_path))

    def _fail(message: str) -> RunSummary:
        store.update_job(job_id, status=STATUS_ERROR, error=message,
                         finished_at=time.time())
        bus.publish(Event(type=JOB_FAILED, job_id=job_id, message=message))
        summary.errors.append(message)
        return summary

    try:
        source = build_source(source_name, **(source_params or {}))
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")

    provider = build_provider(settings, cache=cache, bus=bus)
    done_hashes = store.completed_hashes() if resume else {}

    # Enumerating a source can fail too (missing folder, unreachable API), and
    # that must surface as a failed job rather than an exception at the caller.
    try:
        with source:
            documents = list(source.iter_documents())
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")

    summary.total = len(documents)
    store.update_job(job_id, status=STATUS_RUNNING, total=summary.total)
    bus.publish(Event(type=JOB_STARTED, job_id=job_id,
                      payload={"total": summary.total, "source": source_name,
                               "api": settings.api, "model": settings.model,
                               "template": template.name, "out_dir": str(out_path)}))

    def _work(source_doc):
        digest = source_doc.content_hash()
        if resume and digest in done_hashes:
            artifact = done_hashes[digest]
            if artifact and Path(artifact).exists():
                raise Skip("unchanged since a previous successful run",
                           result={"artifact": artifact, "hash": digest})
        store.upsert_task(job_id, source_doc.doc_id, status=STATUS_RUNNING,
                          content_hash=digest)
        outcome = run_document(provider, source_doc, settings, template, out_path,
                               bus=bus, job_id=job_id)
        outcome.stats["content_hash"] = digest
        return outcome

    # One combined table per run is what most people open first; it is appended
    # to as documents finish rather than held in memory until the end.
    wants_csv = settings.output_format in ("csv", "both")
    combined_records = out_path / "records.csv"
    combined_figures = out_path / "figures.csv"
    tables_started = {"records": False, "figures": False}

    def _append_tables(outcome) -> None:
        if not wants_csv:
            return
        source_doc = getattr(outcome, "source_doc", None)
        title = getattr(source_doc, "title", "") or ""
        if outcome.records:
            append_records_csv(combined_records, outcome.records, template,
                               doc_title=title,
                               write_header=not tables_started["records"])
            tables_started["records"] = True
            summary.tables["records_csv"] = str(combined_records)
        rows = figure_rows(outcome.figures, outcome.doc_id, title)
        if rows:
            append_figures_csv(combined_figures, rows,
                               write_header=not tables_started["figures"])
            tables_started["figures"] = True
            summary.tables["figures_csv"] = str(combined_figures)

    def _collect(task_result):
        doc_id = task_result.item_id
        if task_result.status == "ok":
            outcome = task_result.result
            summary.ok += 1
            summary.records += len(outcome.records)
            summary.figures += len(outcome.figures)
            summary.prompt_tokens += outcome.stats.get("prompt_tokens", 0)
            summary.completion_tokens += outcome.stats.get("completion_tokens", 0)
            summary.cached_calls += outcome.stats.get("cached_calls", 0)
            _append_tables(outcome)
            store.upsert_task(job_id, doc_id, status=STATUS_OK,
                              content_hash=outcome.stats.get("content_hash"),
                              n_records=len(outcome.records),
                              artifact=outcome.artifacts.get("document"),
                              attempts=task_result.attempts,
                              duration=task_result.duration)
            store.bump_job(job_id, "done")
        elif task_result.status == "skipped":
            summary.skipped += 1
            store.upsert_task(job_id, doc_id, status=STATUS_SKIPPED,
                              error=task_result.error,
                              artifact=(task_result.result or {}).get("artifact"),
                              content_hash=(task_result.result or {}).get("hash"))
            store.bump_job(job_id, "skipped")
        else:
            summary.failed += 1
            summary.errors.append(f"{doc_id}: {task_result.error}")
            store.upsert_task(job_id, doc_id, status=STATUS_ERROR,
                              error=task_result.error, attempts=task_result.attempts,
                              duration=task_result.duration)
            store.bump_job(job_id, "failed")

    scheduler = scheduler or Scheduler(
        max_workers=settings.max_workers,
        rate_limiter=RateLimiter(rate_limit) if rate_limit else None,
        bus=bus,
    )
    scheduler.run(_work, documents, job_id=job_id,
                  id_of=lambda d: d.doc_id, on_result=_collect)

    summary.duration_s = time.monotonic() - started
    provider_cache = getattr(provider, "cache", None)
    summary.cache = provider_cache.summary() if provider_cache is not None else {}

    store.update_job(job_id, status=STATUS_OK, finished_at=time.time())
    bus.publish(Event(type=JOB_COMPLETED, job_id=job_id, payload=summary.to_dict()))

    summary_path = out_path / "summary.json"
    summary_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
