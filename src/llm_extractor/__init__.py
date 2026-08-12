"""llm-extractor — folder-in / JSON-out document extraction over an LLM API.

Architecture at a glance::

    sources/      where documents come from  (folder, REST, patents, literature)
        |
    ingest        format readers             (pdf xml docx pptx png jpeg txt md html)
        |
    pipeline      per document: text extraction -> figure OCR -> aggregation agent
        |         all model calls go through providers/ + cache/
    runner        scheduler + job store + event bus over many documents
        |
    cli / service CLI and HTTP API consume the same runner

Every axis is a registry, so new API backends, document sources and templates
plug in without changing core code — see :mod:`llm_extractor.registry`.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "audit",
    "bus",
    "cache",
    "cli",
    "extract",
    "ingest",
    "jobstore",
    "normalize",
    "ocr",
    "parsing",
    "pipeline",
    "providers",
    "registry",
    "runner",
    "scheduler",
    "serialize",
    "settings",
    "sources",
    "templates",
]
