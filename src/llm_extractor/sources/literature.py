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
        # Europe PMC accepts pageSize up to 1000; 100 keeps `resultType=core`
        # responses (full abstracts) a reasonable size while cutting the
        # request count 4x versus the old default of 25.
        "page_size": 100,
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

    def fulltext_target(self, raw: dict):
        """The JATS full text, but only for articles Europe PMC hosts openly.

        Both flags are required. ``isOpenAccess`` says we may read it;
        ``inEPMC`` says Europe PMC actually holds it, and without that the
        endpoint answers with an error page rather than an article.

        JATS XML is preferred over the PDF on purpose: the pipeline parses it
        natively, so sections and tables survive as structure instead of being
        recovered from a rendering of themselves.
        """
        pmcid = str(raw.get("pmcid") or "").strip()
        if not pmcid:
            return None
        if raw.get("isOpenAccess") != "Y" or raw.get("inEPMC") != "Y":
            return None
        return f"{self.base_url}/{pmcid}/fullTextXML", "xml"


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
        # 200 is the OpenAlex maximum (per-page=201 is a pagination error).
        "page_size": 200,
        "auth": "none",
    }

    def to_document(self, raw: dict):
        document = super().to_document(raw)
        if document is None:
            return None
        # OpenAlex ships abstracts as an inverted index; rebuild reading order.
        # It also omits the field entirely for a growing number of works, so the
        # title is always included: a record with neither would arrive as an
        # empty document and be extracted into nothing at full price.
        inverted = raw.get("abstract_inverted_index") or {}
        parts = [f"## title\n{document.title}"] if document.title else []
        if inverted:
            positions = {}
            for word, spots in inverted.items():
                for spot in spots:
                    positions[spot] = word
            parts.append("## abstract\n" + " ".join(positions[i]
                                                    for i in sorted(positions)))
        if parts:
            document.text = "\n\n".join(parts)
        document.metadata.update({
            "year": raw.get("publication_year"),
            "type": raw.get("type"),
            "cited_by_count": raw.get("cited_by_count"),
            "oa_status": dig(raw, "open_access.oa_status"),
            "license": dig(raw, "best_oa_location.license"),
        })
        return document

    def fulltext_target(self, raw: dict):
        """The best open-access PDF OpenAlex knows about, if the work is OA.

        OpenAlex indexes locations rather than hosting them, so the URL points
        at a publisher or repository. Many publishers refuse automated fetches
        even for their own open-access articles — that refusal is expected and
        handled softly, leaving the abstract in place. Europe PMC is the more
        reliable route to full text when an article is in PMC.
        """
        if not dig(raw, "open_access.is_oa"):
            return None
        url = dig(raw, "best_oa_location.pdf_url") or ""
        if not url:
            # `oa_url` is often a landing page rather than the article, so it is
            # only trusted when it names a PDF outright.
            candidate = str(dig(raw, "open_access.oa_url") or "")
            url = candidate if candidate.lower().endswith(".pdf") else ""
        return (str(url), "pdf") if url else None
