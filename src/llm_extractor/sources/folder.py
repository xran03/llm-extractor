"""Local folder source — the default input for the CLI."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..ingest import discover, supported_extensions
from .base import SOURCES, Source, SourceDocument


@SOURCES.register("folder")
class FolderSource(Source):
    name = "folder"
    description = "Recursively read supported documents from a local directory."
    parameters = {
        "input_dir": {"type": "string", "required": True,
                      "description": "Directory (or single file) to read."},
        "extensions": {"type": "array", "items": {"type": "string"},
                       "description": "Subset of the supported extensions."},
        "exclude": {"type": "array", "items": {"type": "string"},
                    "description": "Subdirectories to skip, by relative path or name."},
        "limit": {"type": "integer", "description": "Stop after N documents."},
    }

    def __init__(self, input_dir, extensions=None, limit: int = 0, exclude=None, **params):
        super().__init__(input_dir=str(input_dir), extensions=extensions,
                         limit=limit, exclude=exclude, **params)
        # Absolute from the start: `Path.as_uri()` rejects relative paths, and
        # `-i docs` (or `-i ~/docs`) is what people actually type.
        self.input_dir = Path(input_dir).expanduser().resolve()
        self.extensions = extensions
        self.exclude = list(exclude or [])
        self.limit = int(limit or 0)

    def _paths(self) -> list:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"input path does not exist: {self.input_dir}")
        paths = discover(self.input_dir, self.extensions or supported_extensions(),
                         exclude=self.exclude)
        return paths[: self.limit] if self.limit else paths

    def count(self):
        return len(self._paths())

    def iter_documents(self):
        root = self.input_dir if self.input_dir.is_dir() else self.input_dir.parent
        paths = self._paths()
        for path, doc_id in zip(paths, _unique_doc_ids(paths, root)):
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:  # pragma: no cover - path outside root
                relative = path.name
            yield SourceDocument(
                doc_id=doc_id,
                title=path.stem,
                uri=path.as_uri(),
                path=str(path),
                media_type=path.suffix.lower().lstrip("."),
                source_name=self.name,
                metadata={"relative_path": relative, "bytes": path.stat().st_size},
            )


def _doc_id(path: Path, root: Path) -> str:
    """Folder-unique id: nested files keep their subdirectory context."""
    try:
        relative = path.relative_to(root)
    except ValueError:  # pragma: no cover
        return path.stem
    parts = [*relative.parts[:-1], relative.stem]
    return "__".join(p.replace(" ", "_") for p in parts)


def _unique_doc_ids(paths: list, root: Path) -> list:
    """Give every path its own id, disambiguating only where ids collide.

    Ids drop the extension, so ``figure.pdf`` and ``figure.png`` in one folder
    both reduce to ``figure`` — and since every per-document artifact is named
    after the id, the second document silently overwrote the first one's
    records, OCR and document JSON. Extensions are appended only to the ids
    that actually clash, so ids already in use elsewhere keep their spelling.
    """
    base = [_doc_id(path, root) for path in paths]
    clashing = {doc_id for doc_id, count in Counter(base).items() if count > 1}
    unique, taken = [], set()
    for path, doc_id in zip(paths, base):
        if doc_id in clashing and path.suffix:
            doc_id = f"{doc_id}__{path.suffix.lower().lstrip('.')}"
        candidate, ordinal = doc_id, 2
        while candidate in taken:  # same stem and same extension, different dirs
            candidate, ordinal = f"{doc_id}__{ordinal}", ordinal + 1
        taken.add(candidate)
        unique.append(candidate)
    return unique
