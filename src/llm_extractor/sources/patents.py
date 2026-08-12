"""Patent-database connector preset.

Same pattern as the literature connectors: a :class:`RestSource` subclass that
pins field mapping and paging. Patent bodies are long, so ``to_document``
concatenates the sections that actually carry extractable facts (abstract,
claims, description) and records the bibliographic metadata separately.
"""
from __future__ import annotations

from .base import SOURCES
from .rest import RestSource, dig


@SOURCES.register("patents")
class PatentSearchSource(RestSource):
    """Generic patent search API connector.

    Defaults follow the common ``{"results": [...], "total": N}`` shape used by
    most patent search services. Override ``base_url``/``records_path``/field
    names for a specific provider, or subclass to add provider-specific joins.
    """

    name = "patents"
    description = "Query a patent search API and ingest abstract + claims + description."
    defaults = {
        "path": "/patents/query",
        "records_path": "results",
        "total_path": "total",
        "id_field": "patent_id",
        "title_field": "patent_title",
        "text_fields": ["patent_abstract", "claims", "description"],
        "uri_field": "patent_url",
        "paging": "page",
        "page_size": 50,
        "auth": "bearer",
        "auth_env": "PATENT_API_KEY",
    }

    def to_document(self, raw: dict):
        document = super().to_document(raw)
        if document is not None:
            document.metadata.update({
                "assignee": dig(raw, "assignee") or dig(raw, "assignees.0.name"),
                "filing_date": raw.get("filing_date") or raw.get("patent_date"),
                "cpc": raw.get("cpc_codes") or raw.get("cpc"),
                "inventors": raw.get("inventors"),
            })
        return document
