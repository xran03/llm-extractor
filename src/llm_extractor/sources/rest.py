"""Declarative REST source — the extension path for external databases.

Patent and literature APIs differ only in URL, auth, paging and field names, so
those are *configuration*, not code. :class:`RestSource` turns a JSON REST
endpoint into a stream of :class:`SourceDocument` objects, and a new connector
usually needs nothing more than a subclass that sets defaults (see
``literature.py`` and ``patents.py``).

Supported paging styles:

``page``    ``?page=1&per_page=100``  — increment until a short/empty page
``offset``  ``?offset=0&limit=100``   — increment by page size
``cursor``  ``?cursor=<token>``       — follow a cursor field in the response
``none``    single request

Auth styles: ``none``, ``bearer`` (``Authorization: Bearer ...``), ``header``
(custom header name), ``query`` (key added to the query string).
"""
from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..credentials import get_env
from .base import SOURCES, Source, SourceDocument


class RestSourceError(RuntimeError):
    """Raised when an upstream database API cannot be read."""


#: Identifies us to publishers when fetching full text. A contactable agent is
#: what keeps a polite harvester distinguishable from a scraper.
FULLTEXT_USER_AGENT = "llm-extractor/1.0 (+https://github.com/xran03/llm-extractor)"


def sniff_media(body: bytes) -> str:
    """Identify a downloaded body from its first bytes, or ``""`` if unsure.

    Open-access links routinely resolve to a landing page rather than the
    article: the URL says PDF, the server returns HTML, and nothing complains.
    Believing the URL would feed a navigation page to the extractor and bill for
    it, so what arrived is checked against what was promised.
    """
    head = body[:2048].lstrip()
    if head.startswith(b"%PDF"):
        return "pdf"
    lowered = head[:600].lower()
    if lowered.startswith(b"<!doctype html") or lowered.startswith(b"<html"):
        return "html"
    if head.startswith(b"<?xml") or head.startswith(b"<article") \
            or b"<!doctype article" in lowered:
        return "xml"
    return ""


def dig(payload, path: str, default=None):
    """Read a dotted path out of nested JSON (``"data.items"``, ``"hits.0.id"``)."""
    if not path:
        return payload
    current = payload
    for part in path.split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
                continue
            except (ValueError, IndexError):
                return default
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def read_search_terms(source) -> list:
    """Read search terms from a list file, one query per line.

    A literature question is rarely one phrase. Pneumococcal, E. coli and GBS
    conjugate work needs a dozen wordings between them, and asking someone to
    re-run the tool once per phrase — then merge the results by hand — is how
    duplicates and gaps get in. Accepting the list instead keeps the union and
    the de-duplication inside the tool, where the ids are known.

    Plain text, CSV and TSV are all accepted because that is what people
    actually have: a text file is one term per line, while CSV/TSV take the
    first column, skipping a header if it looks like one. Blank lines and
    lines starting with ``#`` are ignored, so a list can carry comments.
    """
    path = Path(source)
    if not path.is_file():
        raise RestSourceError(f"search list not found: {path}")

    lines = [line.strip() for line in
             path.read_text(encoding="utf-8", errors="replace").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    if not lines:
        return []

    delimiter = "\t" if path.suffix.lower() in (".tsv", ".tab") else \
        ("," if path.suffix.lower() == ".csv" else "")
    if delimiter:
        rows = list(csv.reader(lines, delimiter=delimiter))
        terms = [row[0].strip() for row in rows if row and row[0].strip()]
        # Drop an obvious header rather than searching for the word "query".
        if terms and terms[0].lower() in ("query", "term", "search",
                                          "keyword", "keywords"):
            terms = terms[1:]
    else:
        terms = lines

    seen, unique = set(), []
    for term in terms:
        key = " ".join(term.lower().split())
        if key and key not in seen:
            seen.add(key)
            unique.append(term)
    return unique


#: A connector definition to edit, written by ``llm-extract sources --init``.
#: The values point at Crossref, which is public and needs no key, so the file
#: works as written and can be changed one field at a time rather than debugged
#: all at once.
STARTER_CONNECTOR = {
    "source": "rest",
    "base_url": "https://api.crossref.org",
    "path": "/works",
    "query_param": "query",
    "search": "pneumococcal conjugate vaccine",
    "query": {"filter": "type:journal-article,has-abstract:true"},
    "records_path": "message.items",
    "id_field": "DOI",
    "title_field": "title.0",
    "text_fields": ["abstract"],
    "uri_field": "URL",
    "paging": "offset",
    "offset_param": "offset",
    "size_param": "rows",
    "page_size": 20,
    "max_records": 100,
    "auth": "none",
    "auth_env": "",
}


@SOURCES.register("rest")
class RestSource(Source):
    """Generic paginated JSON API source."""

    name = "rest"
    description = "Query any paginated JSON REST API (patents, literature, internal stores)."
    parameters = {
        "base_url": {"type": "string", "required": True},
        "path": {"type": "string", "description": "Endpoint path, e.g. /v1/search."},
        "query": {"type": "object", "description": "Static query parameters."},
        "search": {"type": "string", "description": "Value for the query term parameter."},
        "search_file": {"type": "string",
                        "description": "Path to a .txt/.csv/.tsv list of search "
                                       "terms, one per line; results are unioned "
                                       "and de-duplicated."},
        "query_param": {"type": "string", "description": "Name of the search parameter."},
        "records_path": {"type": "string", "description": "Dotted path to the result list."},
        "id_field": {"type": "string"},
        "title_field": {"type": "string"},
        "text_fields": {"type": "array", "items": {"type": "string"}},
        "uri_field": {"type": "string"},
        "paging": {"type": "string", "enum": ["page", "offset", "cursor", "none"]},
        "page_size": {"type": "integer"},
        "max_records": {"type": "integer"},
        "auth": {"type": "string", "enum": ["none", "bearer", "header", "query"]},
        "auth_env": {"type": "string", "description": "Env var holding the credential."},
        "fulltext": {"type": "boolean",
                     "description": "Fetch open-access full text instead of the abstract."},
        "fulltext_field": {"type": "string",
                           "description": "Dotted path to a full-text URL in the record."},
        "fulltext_media": {"type": "string",
                           "description": "Media type of the fetched body (pdf, xml, html)."},
    }

    # Subclasses override these to become a named connector.
    defaults: dict = {}

    def __init__(self, **params):
        merged = {**self.defaults, **{k: v for k, v in params.items() if v is not None}}
        super().__init__(**merged)

        self.base_url = str(merged.get("base_url", "")).rstrip("/")
        if not self.base_url:
            raise RestSourceError(f"{self.name}: base_url is required")
        self.path = merged.get("path", "")
        # `query` is a dict of static parameters, but from the command line
        # (`--param query=vaccine`) it almost always arrives as a search term.
        # Accepting both is what makes the documented invocation work instead
        # of failing deep inside dict().
        raw_query = merged.get("query")
        search = merged.get("search")
        if isinstance(raw_query, str):
            search = search or raw_query
            raw_query = None
        elif raw_query is not None and not isinstance(raw_query, dict):
            raise RestSourceError(
                f"{self.name}: 'query' must be a search term or an object of "
                f"query parameters, not {type(raw_query).__name__}"
            )
        self.query = dict(raw_query or {})
        self.query_param = merged.get("query_param", "q")

        # One search, or many. Many is kept as a list rather than joined into a
        # single OR query, because every API spells boolean syntax differently
        # and a wrong join silently returns the wrong set.
        self.search_terms = []
        if merged.get("search_file"):
            self.search_terms = read_search_terms(merged["search_file"])
        if search:
            self.search_terms.insert(0, search)
        if self.search_terms:
            self.query[self.query_param] = self.search_terms[0]

        self.records_path = merged.get("records_path", "")
        self.id_field = merged.get("id_field", "id")
        self.title_field = merged.get("title_field", "title")
        self.text_fields = merged.get("text_fields") or ["abstract", "text", "description"]
        self.uri_field = merged.get("uri_field", "url")

        self.paging = merged.get("paging", "page")
        self.page_size = int(merged.get("page_size", 100))
        self.page_param = merged.get("page_param", "page")
        self.size_param = merged.get("size_param", "per_page")
        self.offset_param = merged.get("offset_param", "offset")
        self.cursor_param = merged.get("cursor_param", "cursor")
        self.cursor_path = merged.get("cursor_path", "next_cursor")
        self.start_page = int(merged.get("start_page", 1))
        self.max_records = int(merged.get("max_records", 0))
        # Hard stop: a misbehaving API that keeps returning the same page must
        # never spin forever inside a batch job.
        self.max_pages = int(merged.get("max_pages", 1000))
        self.total_path = merged.get("total_path", "")

        self.auth = merged.get("auth", "none")
        self.auth_env = merged.get("auth_env", "")
        self.auth_header = merged.get("auth_header", "Authorization")
        self.auth_query_param = merged.get("auth_query_param", "api_key")
        self.timeout = float(merged.get("timeout", 60.0))
        self.max_retries = int(merged.get("max_retries", 3))
        self.delay = float(merged.get("delay", 0.0))
        self.headers = dict(merged.get("headers") or {})
        self.fulltext = bool(merged.get("fulltext", False))
        self.fulltext_field = merged.get("fulltext_field", "")
        self.fulltext_media = merged.get("fulltext_media", "pdf")
        # A body large enough to be a whole issue is a mistake, not an article.
        self.fulltext_max_bytes = int(merged.get("fulltext_max_bytes", 40 << 20))
        self._fulltext_fetcher = merged.get("fulltext_fetcher")  # injected in tests
        self._fetcher = merged.get("fetcher")  # injected in tests
        self._total = None

    # ------------------------------- transport ------------------------------
    def _credential(self) -> str:
        return get_env(self.auth_env) if self.auth_env else ""

    def _build_url(self, extra: dict) -> str:
        params = {**self.query, **extra}
        if self.auth == "query":
            credential = self._credential()
            if credential:
                params[self.auth_query_param] = credential
        # Lead with the search term. Europe PMC answers hitCount=0 when `query`
        # is not the first parameter — the same request, reordered, returns the
        # article — and a search that silently finds nothing is far worse than
        # one that fails. Leading with it is harmless for every other API.
        ordered = {}
        if self.query_param in params:
            ordered[self.query_param] = params.pop(self.query_param)
        ordered.update(params)
        query = urllib.parse.urlencode(
            {k: v for k, v in ordered.items() if v is not None}, doseq=True
        )
        return f"{self.base_url}{self.path}" + (f"?{query}" if query else "")

    def fetch(self, extra: dict) -> dict:
        url = self._build_url(extra)
        if self._fetcher is not None:
            return self._fetcher(url)

        headers = {"Accept": "application/json", **self.headers}
        credential = self._credential()
        if credential and self.auth == "bearer":
            headers["Authorization"] = f"Bearer {credential}"
        elif credential and self.auth == "header":
            headers[self.auth_header] = credential

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:300]
                if exc.code != 429 and 400 <= exc.code < 500:
                    raise RestSourceError(f"{self.name} HTTP {exc.code}: {body}") from exc
                last_error = RestSourceError(f"{self.name} HTTP {exc.code}: {body}")
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
                last_error = RestSourceError(f"{self.name} request failed: {exc}")
            time.sleep(1.5 ** attempt)
        raise last_error or RestSourceError(f"{self.name}: request failed")

    # -------------------------------- paging --------------------------------
    def _pages(self):
        """Yield one decoded response body per page.

        Terminates on a short page, an exhausted cursor, ``max_records``,
        ``max_pages``, or a page that repeats content already seen — the last
        guard protects against APIs that ignore paging parameters.
        """
        if self.paging == "none":
            yield self.fetch({})
            return

        cursor = None
        page = self.start_page
        emitted = 0
        pages_read = 0
        seen_ids: set = set()
        while pages_read < self.max_pages:
            if self.paging == "page":
                extra = {self.page_param: page, self.size_param: self.page_size}
            elif self.paging == "offset":
                extra = {self.offset_param: emitted, self.size_param: self.page_size}
            else:
                extra = {self.size_param: self.page_size}
                if cursor:
                    extra[self.cursor_param] = cursor

            body = self.fetch(extra)
            pages_read += 1
            yield body

            records = dig(body, self.records_path, []) or []
            page_ids = {str(dig(r, self.id_field)) for r in records
                        if isinstance(r, dict)}
            if page_ids and page_ids <= seen_ids:
                return  # the API returned nothing new; stop rather than loop
            seen_ids |= page_ids

            emitted += len(records)
            if self.max_records and emitted >= self.max_records:
                return
            if len(records) < self.page_size or not records:
                return
            if self.paging == "cursor":
                cursor = dig(body, self.cursor_path)
                if not cursor:
                    return
            page += 1
            if self.delay:
                time.sleep(self.delay)

    def count(self):
        return self._total

    def iter_documents(self):
        """Yield documents for every search term, de-duplicated across terms.

        Terms overlap heavily by design — that is what makes a list better than
        one phrase — so the same paper will be returned by several of them. It
        is emitted once, by id, which is why the union has to happen here rather
        than in whatever merges the CSVs afterwards.
        """
        emitted = 0
        seen: set = set()
        for term in (self.search_terms or [None]):
            if term is not None:
                self.query[self.query_param] = term
                self._total = None
            for body in self._pages():
                if self.total_path and self._total is None:
                    self._total = dig(body, self.total_path)
                for raw in dig(body, self.records_path, []) or []:
                    document = self.to_document(raw)
                    if document is None or document.doc_id in seen:
                        continue
                    seen.add(document.doc_id)
                    if term is not None:
                        document.metadata.setdefault("search_term", term)
                    yield document
                    emitted += 1
                    if self.max_records and emitted >= self.max_records:
                        return

    # ------------------------------- mapping --------------------------------
    def fulltext_target(self, raw: dict):
        """Where this record's **open-access** full text lives.

        Returns ``(url, media_type)`` or ``None``. The default reads a URL out
        of the record at ``fulltext_field``; connectors override it to apply the
        access rules their API publishes.

        The rule this hook exists to enforce: only ever return a location the
        API itself marks as open access. An abstract is a poor substitute for
        full text, but it is a far better outcome than fetching something we
        were not licensed to fetch.
        """
        if not self.fulltext_field:
            return None
        url = str(dig(raw, self.fulltext_field) or "").strip()
        return (url, self.fulltext_media) if url else None

    def fetch_fulltext(self, url: str):
        """Download one full-text body. Returns bytes, or ``None`` on any failure.

        Failure is soft on purpose: a corpus of 500 papers will always contain a
        few whose full text has moved, gone behind a login or died. Losing the
        run over one of them would be worse than extracting that one from its
        abstract.
        """
        if self._fulltext_fetcher is not None:
            return self._fulltext_fetcher(url)
        request = urllib.request.Request(
            url, method="GET",
            headers={"Accept": "*/*", "User-Agent": FULLTEXT_USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                # A single read() may return short on a chunked response, so
                # read until EOF rather than trusting one call to deliver the
                # whole body. Getting this wrong truncates papers silently.
                chunks, total = [], 0
                while True:
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self.fulltext_max_bytes:
                        return None
                    chunks.append(chunk)
                return b"".join(chunks) or None
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                ConnectionError, OSError):
            return None

    def to_document(self, raw: dict):
        """Map one API record to a :class:`SourceDocument`.

        Override this in a connector subclass when the payload needs more than
        field renaming (e.g. joining claims + description for a patent).
        """
        if not isinstance(raw, dict):
            return None
        doc_id = dig(raw, self.id_field)
        if doc_id is None:
            return None
        parts = []
        for field_path in self.text_fields:
            value = dig(raw, field_path)
            if isinstance(value, list):
                value = "\n".join(str(v) for v in value if v)
            if value:
                parts.append(f"## {field_path}\n{value}")
        document = SourceDocument(
            doc_id=str(doc_id).replace("/", "_"),
            title=str(dig(raw, self.title_field) or ""),
            uri=str(dig(raw, self.uri_field) or ""),
            text="\n\n".join(parts),
            media_type="text",
            source_name=self.name,
            metadata={"raw": raw} if len(parts) == 0 else {"api_record_keys": sorted(raw)},
        )
        if self.fulltext:
            self.attach_fulltext(document, raw)
        return document

    def attach_fulltext(self, document, raw: dict) -> bool:
        """Replace the abstract with the open-access full text, when there is one.

        The abstract stays on the document either way, so a record whose full
        text could not be fetched is still extractable — just from less. What
        actually happened is recorded in ``fulltext`` metadata rather than left
        for the reader to infer from the record's length.
        """
        target = self.fulltext_target(raw)
        if not target:
            document.metadata["fulltext"] = "none advertised"
            return False
        url, media = target
        body = self.fetch_fulltext(url)
        if not body:
            document.metadata["fulltext"] = f"unavailable: {url}"
            return False
        actual = sniff_media(body)
        if actual and actual != media:
            document.metadata["fulltext"] = f"unusable: {url} returned {actual}"
            return False
        document.blob = body
        document.media_type = actual or media
        document.metadata.update({"fulltext": url, "fulltext_bytes": len(body)})
        return True
