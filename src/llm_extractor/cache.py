"""Response cache — cost control *and* the substrate for quality audits.

Two jobs in one component:

1. **Never pay twice.** Every model call is keyed by a SHA-256 digest of
   everything that can change the answer (backend, model, full message content
   including base64 images, temperature, token budget, requested JSON schema).
   Re-running a folder after adding a few files, or iterating on downstream
   code, costs nothing for the unchanged work.

2. **Stay auditable.** Payloads are JSON files on disk, but an SQLite index
   records what each entry *is* (stage, doc id, model, template, tokens, age)
   and what we later learned about it (verdict, agreement score). Because the
   original request is stored alongside the response, any entry can be replayed
   later — that is what :mod:`llm_extractor.audit` uses to re-validate a random
   sample and detect drift or bad extractions.

Deleting the cache directory is always safe.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .providers.base import Completion, Usage

CACHE_VERSION = "1"

INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    key           TEXT PRIMARY KEY,
    api           TEXT,
    model         TEXT,
    stage         TEXT,
    doc_id        TEXT,
    template      TEXT,
    created_at    REAL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    bytes         INTEGER DEFAULT 0,
    verdict       TEXT,
    agreement     REAL,
    verified_at   REAL,
    verified_by   TEXT,
    detail        TEXT
);
CREATE INDEX IF NOT EXISTS idx_entries_stage ON entries(stage);
CREATE INDEX IF NOT EXISTS idx_entries_doc ON entries(doc_id);
CREATE INDEX IF NOT EXISTS idx_entries_verdict ON entries(verdict);
CREATE INDEX IF NOT EXISTS idx_entries_created ON entries(created_at);
"""

VERDICT_UNVERIFIED = "unverified"
VERDICT_CONFIRMED = "confirmed"
VERDICT_DRIFTED = "drifted"
VERDICT_SUSPECT = "suspect"
VERDICT_ERROR = "error"


def make_key(api: str, model: str, messages, temperature, max_tokens,
             json_schema=None, extra=None) -> str:
    """Deterministic cache key for one model call."""
    payload = {
        "v": CACHE_VERSION,
        "api": api,
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "json_schema": json_schema,
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    saved_prompt_tokens: int = 0
    saved_completion_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": self.hit_rate,
            "saved_prompt_tokens": self.saved_prompt_tokens,
            "saved_completion_tokens": self.saved_completion_tokens,
        }


@dataclass
class ResponseCache:
    """JSON blob store + SQLite index."""

    directory: str = ".llm_cache"
    ttl_seconds: float = 0.0        # 0 = entries never expire
    store_request: bool = True      # keep the request so entries can be replayed
    stats: CacheStats = field(default_factory=CacheStats)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _conn: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.directory)
        (self.root / "blobs").mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.root / "index.sqlite3"), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(INDEX_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # -------------------------------- blobs --------------------------------
    def path_for(self, key: str) -> Path:
        # Shard by the first two hex chars to keep directories small at scale.
        return self.root / "blobs" / key[:2] / f"{key}.json"

    def get(self, key: str):
        path = self.path_for(key)
        if not path.is_file():
            self._count_miss()
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._count_miss()
            return None
        if self.ttl_seconds and time.time() - entry.get("stored_at", 0) > self.ttl_seconds:
            self._count_miss()
            return None
        usage = entry.get("usage") or {}
        with self._lock:
            self.stats.hits += 1
            self.stats.saved_prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.stats.saved_completion_tokens += int(usage.get("completion_tokens") or 0)
        return entry

    def put(self, key: str, text: str, usage: dict | None = None,
            meta: dict | None = None, request: dict | None = None) -> None:
        meta = meta or {}
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "key": key,
            "stored_at": time.time(),
            "text": text,
            "usage": usage or {},
            "meta": meta,
            "request": request if self.store_request else None,
        }
        payload = json.dumps(entry, ensure_ascii=False)
        # Atomic write so an interrupted run never leaves a half-written entry.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._index(key, entry, len(payload))
        with self._lock:
            self.stats.writes += 1

    def load_entry(self, key: str):
        """Read an entry without touching hit/miss statistics."""
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _count_miss(self) -> None:
        with self._lock:
            self.stats.misses += 1

    # -------------------------------- index --------------------------------
    def _index(self, key: str, entry: dict, size: int) -> None:
        meta = entry.get("meta") or {}
        usage = entry.get("usage") or {}
        with self._lock:
            self._conn.execute(
                "INSERT INTO entries (key, api, model, stage, doc_id, template, created_at,"
                " prompt_tokens, completion_tokens, bytes, verdict)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(key) DO UPDATE SET created_at=excluded.created_at,"
                " prompt_tokens=excluded.prompt_tokens,"
                " completion_tokens=excluded.completion_tokens, bytes=excluded.bytes",
                (
                    key, meta.get("api"), meta.get("model"), meta.get("stage"),
                    meta.get("doc_id"), meta.get("template"), entry.get("stored_at"),
                    int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0),
                    size, VERDICT_UNVERIFIED,
                ),
            )
            self._conn.commit()

    def query(self, stage: str = "", doc_id: str = "", model: str = "",
              verdict: str = "", replayable_only: bool = False, limit: int = 1000) -> list:
        """List index rows matching the given filters."""
        sql = "SELECT * FROM entries WHERE 1=1"
        args: list = []
        for column, value in (("stage", stage), ("doc_id", doc_id),
                              ("model", model), ("verdict", verdict)):
            if value:
                sql += f" AND {column}=?"
                args.append(value)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(sql, args).fetchall()]
        if replayable_only:
            rows = [r for r in rows if (self.load_entry(r["key"]) or {}).get("request")]
        return rows

    def sample(self, n: int, stage: str = "", strategy: str = "random",
               seed=None, verdict: str = "", replayable_only: bool = True) -> list:
        """Pick ``n`` index rows to re-validate.

        Strategies: ``random`` (uniform), ``oldest`` (drift is time-correlated),
        ``newest``, ``unverified`` (never audited), ``largest`` (most tokens, so
        the costliest calls get checked first).
        """
        import random as _random

        rows = self.query(stage=stage, verdict=verdict, replayable_only=replayable_only,
                          limit=100000)
        if not rows:
            return []
        if strategy == "oldest":
            rows.sort(key=lambda r: r.get("created_at") or 0)
        elif strategy == "newest":
            rows.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
        elif strategy == "largest":
            rows.sort(key=lambda r: (r.get("prompt_tokens") or 0) +
                      (r.get("completion_tokens") or 0), reverse=True)
        elif strategy == "unverified":
            rows = [r for r in rows
                    if (r.get("verdict") or VERDICT_UNVERIFIED) == VERDICT_UNVERIFIED]
            _random.Random(seed).shuffle(rows)
        else:
            _random.Random(seed).shuffle(rows)
        return rows[:n]

    def mark(self, key: str, verdict: str, agreement=None,
             verified_by: str = "", detail=None) -> None:
        """Record an audit outcome against a cache entry."""
        with self._lock:
            self._conn.execute(
                "UPDATE entries SET verdict=?, agreement=?, verified_at=?, verified_by=?,"
                " detail=? WHERE key=?",
                (verdict, agreement, time.time(), verified_by,
                 json.dumps(detail, ensure_ascii=False, default=str) if detail else None,
                 key),
            )
            self._conn.commit()

    def invalidate(self, keys) -> int:
        """Drop specific entries (blob + index row)."""
        removed = 0
        for key in keys:
            path = self.path_for(key)
            if path.exists():
                path.unlink()
                removed += 1
            with self._lock:
                self._conn.execute("DELETE FROM entries WHERE key=?", (key,))
        with self._lock:
            self._conn.commit()
        return removed

    def clear(self) -> int:
        removed = 0
        for path in (self.root / "blobs").rglob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        with self._lock:
            self._conn.execute("DELETE FROM entries")
            self._conn.commit()
        return removed

    def summary(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(bytes),0) b,"
                " COALESCE(SUM(prompt_tokens),0) pt, COALESCE(SUM(completion_tokens),0) ct"
                " FROM entries"
            ).fetchone()
            by_stage = {
                r["stage"] or "unknown": r["n"] for r in self._conn.execute(
                    "SELECT stage, COUNT(*) n FROM entries GROUP BY stage")
            }
            by_verdict = {
                r["verdict"] or VERDICT_UNVERIFIED: r["n"] for r in self._conn.execute(
                    "SELECT verdict, COUNT(*) n FROM entries GROUP BY verdict")
            }
        return {
            "directory": str(self.root),
            "entries": row["n"],
            "bytes": row["b"],
            "cached_prompt_tokens": row["pt"],
            "cached_completion_tokens": row["ct"],
            "by_stage": by_stage,
            "by_verdict": by_verdict,
            **self.stats.to_dict(),
        }


class CachedProvider:
    """Provider decorator that serves repeat calls from :class:`ResponseCache`.

    ``bypass`` forces a live call while still refreshing the stored entry —
    exactly what the auditor needs to compare "what we cached" against "what the
    model says now".
    """

    def __init__(self, provider, cache: ResponseCache, bus=None, bypass: bool = False):
        self.provider = provider
        self.cache = cache
        self.bus = bus
        self.bypass = bypass
        self.name = getattr(provider, "name", "provider")

    @property
    def api_style(self) -> str:
        return getattr(self.provider, "API_STYLE", "unknown")

    def list_models(self):
        return self.provider.list_models()

    def complete(self, messages, model, temperature=0.0, max_tokens=None,
                 json_schema=None, meta=None, **kwargs) -> Completion:
        extra = kwargs.get("extra")
        key = make_key(self.name, model, messages, temperature, max_tokens, json_schema, extra)

        if not self.bypass:
            entry = self.cache.get(key)
            if entry is not None:
                usage = Usage(**{k: v for k, v in (entry.get("usage") or {}).items()
                                 if k in Usage.__dataclass_fields__})
                usage.cached = True
                if self.bus is not None:
                    from .bus import CACHE_HIT, Event

                    self.bus.publish(Event(
                        type=CACHE_HIT, doc_id=(meta or {}).get("doc_id", ""),
                        stage=(meta or {}).get("stage", ""), payload={"key": key},
                    ))
                return Completion(text=entry["text"], usage=usage, raw={"cached": True})

        completion = self.provider.complete(
            messages, model, temperature=temperature, max_tokens=max_tokens,
            json_schema=json_schema, meta=meta, **kwargs
        )
        self.cache.put(
            key, completion.text, completion.usage.to_dict(),
            meta={"api": self.name, "model": model, **(meta or {})},
            request={
                "messages": messages, "model": model, "temperature": temperature,
                "max_tokens": max_tokens, "json_schema": json_schema, "extra": extra,
            },
        )
        return completion

    def complete_text(self, messages, model, **kwargs) -> str:
        return self.complete(messages, model, **kwargs).text
