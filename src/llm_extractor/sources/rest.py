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

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from ..credentials import get_env
from .base import SOURCES, Source, SourceDocument


class RestSourceError(RuntimeError):
    """Raised when an upstream database API cannot be read."""


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
        if search:
            self.query[self.query_param] = search

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
        query = urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None}, doseq=True
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
        emitted = 0
        seen: set = set()
        for body in self._pages():
            if self.total_path and self._total is None:
                self._total = dig(body, self.total_path)
            for raw in dig(body, self.records_path, []) or []:
                document = self.to_document(raw)
                if document is None or document.doc_id in seen:
                    continue
                seen.add(document.doc_id)
                yield document
                emitted += 1
                if self.max_records and emitted >= self.max_records:
                    return

    # ------------------------------- mapping --------------------------------
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
        return SourceDocument(
            doc_id=str(doc_id).replace("/", "_"),
            title=str(dig(raw, self.title_field) or ""),
            uri=str(dig(raw, self.uri_field) or ""),
            text="\n\n".join(parts),
            media_type="text",
            source_name=self.name,
            metadata={"raw": raw} if len(parts) == 0 else {"api_record_keys": sorted(raw)},
        )
