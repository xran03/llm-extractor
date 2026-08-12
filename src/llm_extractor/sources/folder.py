"""Local folder source — the default input for the CLI."""
from __future__ import annotations

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
        "limit": {"type": "integer", "description": "Stop after N documents."},
    }

    def __init__(self, input_dir, extensions=None, limit: int = 0, **params):
        super().__init__(input_dir=str(input_dir), extensions=extensions,
                         limit=limit, **params)
        self.input_dir = Path(input_dir)
        self.extensions = extensions
        self.limit = int(limit or 0)

    def _paths(self) -> list:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"input path does not exist: {self.input_dir}")
        paths = discover(self.input_dir, self.extensions or supported_extensions())
        return paths[: self.limit] if self.limit else paths

    def count(self):
        return len(self._paths())

    def iter_documents(self):
        root = self.input_dir if self.input_dir.is_dir() else self.input_dir.parent
        for path in self._paths():
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:  # pragma: no cover - path outside root
                relative = path.name
            yield SourceDocument(
                doc_id=_doc_id(path, root),
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
