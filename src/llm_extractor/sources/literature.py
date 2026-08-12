"""Literature-database connector presets.

These are thin :class:`~llm_extractor.sources.rest.RestSource` subclasses: they
only pin the field mapping and paging style of a public bibliographic API. Point
them at an internal mirror by passing ``base_url``. Copy either class to add a
new database — no core code changes required.
"""
from __future__ import annotations

from .base import SOURCES
from .rest import RestSource, dig


@SOURCES.register("europepmc")
class EuropePMCSource(RestSource):
    """Europe PMC REST search (open access; no credential required)."""

    name = "europepmc"
    description = "Search Europe PMC for articles and ingest title + abstract."
    defaults = {
        "base_url": "https://www.ebi.ac.uk/europepmc/webservices/rest",
        "path": "/search",
        "query": {"format": "json", "resultType": "core"},
        "query_param": "query",
        "records_path": "resultList.result",
        "total_path": "hitCount",
        "id_field": "id",
        "title_field": "title",
        "text_fields": ["abstractText"],
        "uri_field": "doi",
        "paging": "page",
        "page_param": "page",
        "size_param": "pageSize",
        "page_size": 25,
        "auth": "none",
    }

    def to_document(self, raw: dict):
        document = super().to_document(raw)
        if document is not None:
            document.metadata.update({
                "journal": dig(raw, "journalInfo.journal.title"),
                "year": raw.get("pubYear"),
                "doi": raw.get("doi"),
                "pmid": raw.get("pmid"),
                "is_open_access": raw.get("isOpenAccess"),
            })
            if raw.get("doi"):
                document.uri = f"https://doi.org/{raw['doi']}"
        return document


@SOURCES.register("openalex")
class OpenAlexSource(RestSource):
    """OpenAlex works search (cursor paging)."""

    name = "openalex"
    description = "Search OpenAlex works and ingest title + reconstructed abstract."
    defaults = {
        "base_url": "https://api.openalex.org",
        "path": "/works",
        "query_param": "search",
        "records_path": "results",
        "total_path": "meta.count",
        "id_field": "id",
        "title_field": "display_name",
        "text_fields": [],
        "uri_field": "doi",
        "paging": "cursor",
        "cursor_param": "cursor",
        "cursor_path": "meta.next_cursor",
        "size_param": "per-page",
        "page_size": 50,
        "auth": "none",
    }

    def to_document(self, raw: dict):
        document = super().to_document(raw)
        if document is None:
            return None
        # OpenAlex ships abstracts as an inverted index; rebuild reading order.
        inverted = raw.get("abstract_inverted_index") or {}
        if inverted:
            positions = {}
            for word, spots in inverted.items():
                for spot in spots:
                    positions[spot] = word
            abstract = " ".join(positions[i] for i in sorted(positions))
            document.text = f"## title\n{document.title}\n\n## abstract\n{abstract}"
        document.metadata.update({
            "year": raw.get("publication_year"),
            "type": raw.get("type"),
            "cited_by_count": raw.get("cited_by_count"),
        })
        return document
