"""Document sources — where documents come from.

A *source* decouples "which documents" from "how they are extracted". The
folder source reads local files today; a patent-office or literature-database
connector is the same interface over an HTTP API tomorrow, and can live in a
separate pip package discovered through the ``llm_extractor.sources`` entry
point group.

A source yields :class:`SourceDocument` items lazily, so a query returning
100k patents streams through the scheduler instead of materialising in memory.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from ..registry import Registry

SOURCES = Registry(kind="source", entry_point_group="llm_extractor.sources")


@dataclass
class SourceDocument:
    """One retrievable document, independent of where it came from.

    Exactly one of ``path``, ``text`` or ``blob`` carries the payload:

    * ``path``  — a local file the ingest readers will parse (folder source);
    * ``text``  — already-plain text supplied by an API (literature abstracts);
    * ``blob``  — raw bytes plus ``media_type`` (a PDF downloaded from an API).
    """

    doc_id: str
    title: str = ""
    uri: str = ""
    path: str = ""
    text: str = ""
    blob: bytes = b""
    media_type: str = ""
    metadata: dict = field(default_factory=dict)
    source_name: str = ""

    def content_hash(self) -> str:
        """Stable digest used for resume and cache keys."""
        if self.path and Path(self.path).is_file():
            digest = hashlib.sha256()
            with open(self.path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()[:32]
        payload = self.blob or self.text.encode("utf-8", "replace")
        return hashlib.sha256(payload).hexdigest()[:32]

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "uri": self.uri,
            "path": self.path,
            "media_type": self.media_type,
            "source": self.source_name,
            "metadata": self.metadata,
            "has_text": bool(self.text),
            "has_blob": bool(self.blob),
        }


class Source:
    """Base class for every document source.

    Subclasses implement :meth:`iter_documents`. Optional hooks let a connector
    advertise its capabilities to the CLI and the HTTP API without bespoke code.
    """

    name = "base"
    description = ""
    #: Parameter description surfaced by the API so a frontend can render a
    #: query form for any installed connector without hard-coding it.
    parameters: dict = {}

    def __init__(self, **params):
        self.params = params

    def iter_documents(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def count(self):
        """Optional total for progress reporting; ``None`` when unknown."""
        return None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def close(self) -> None:
        """Release connections/handles. Sources are usable as context managers."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def build_source(name: str, **params) -> Source:
    """Instantiate a registered source by name."""
    return SOURCES.get(name)(**params)


def available_sources() -> dict:
    """Map of source name -> description, for CLI help and the API."""
    return {
        name: getattr(cls, "description", "") for name, cls in SOURCES.items().items()
    }
