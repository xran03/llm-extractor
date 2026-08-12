"""Document sources: local folders today, external databases tomorrow.

Importing this package registers the built-in connectors. Third-party
connectors are discovered lazily through the ``llm_extractor.sources`` entry
point group, so installing a package is all it takes to add one.
"""
from __future__ import annotations

from .base import SOURCES, Source, SourceDocument, available_sources, build_source
from .folder import FolderSource
from .literature import EuropePMCSource, OpenAlexSource
from .patents import PatentSearchSource
from .rest import RestSource, RestSourceError

__all__ = [
    "EuropePMCSource",
    "FolderSource",
    "OpenAlexSource",
    "PatentSearchSource",
    "RestSource",
    "RestSourceError",
    "SOURCES",
    "Source",
    "SourceDocument",
    "available_sources",
    "build_source",
]
