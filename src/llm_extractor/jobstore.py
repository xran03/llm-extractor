"""SQLite-backed job and task state.

Extraction runs are long and resumable, and a frontend needs to poll them after
the CLI process is gone. Persisting job/task rows (rather than keeping them in
memory) gives all three properties at once:

* **resume** — a re-run skips tasks already recorded ``ok`` for the same
  content hash;
* **observability** — ``GET /v1/jobs/{id}`` reads straight from this store;
* **audit trail** — every attempt, error and artifact path is retained.

SQLite is used in WAL mode with a single shared connection guarded by a lock,
which is more than fast enough for I/O-bound extraction workloads.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    source       TEXT,
    params       TEXT,
    created_at   REAL,
    updated_at   REAL,
    finished_at  REAL,
    total        INTEGER DEFAULT 0,
    done         INTEGER DEFAULT 0,
    failed       INTEGER DEFAULT 0,
    skipped      INTEGER DEFAULT 0,
    error        TEXT
);
CREATE TABLE IF NOT EXISTS tasks (
    job_id       TEXT NOT NULL,
    doc_id       TEXT NOT NULL,
    status       TEXT NOT NULL,
    content_hash TEXT,
    attempts     INTEGER DEFAULT 0,
    n_records    INTEGER DEFAULT 0,
    artifact     TEXT,
    error        TEXT,
    duration     REAL,
    updated_at   REAL,
    PRIMARY KEY (job_id, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_hash ON tasks(content_hash);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"
STATUS_CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    status: str
    source: str = ""
    params: dict = None
    total: int = 0
    done: int = 0
    failed: int = 0
    skipped: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        data["params"] = self.params or {}
        data["progress"] = (
            round((self.done + self.failed + self.skipped) / self.total, 4)
            if self.total else 0.0
        )
        return data


class JobStore:
    def __init__(self, path: str = ".llm_cache/jobs.sqlite3"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --------------------------------- jobs --------------------------------
    def create_job(self, source: str, params: dict | None = None, job_id: str | None = None) -> str:
        """Create a job, or adopt an id that was reserved earlier.

        The HTTP API reserves an id synchronously so the client can poll right
        away, then the runner starts the job in the background with that same
        id — so this must be idempotent.
        """
        job_id = job_id or uuid.uuid4().hex[:16]
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, status, source, params, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
                "source=excluded.source, params=excluded.params, "
                "updated_at=excluded.updated_at",
                (job_id, STATUS_PENDING, source,
                 json.dumps(params or {}, default=str), now, now),
            )
            self._conn.commit()
        return job_id

    def update_job(self, job_id: str, **fields) -> None:
        if not fields:
            return
        if "params" in fields:
            fields["params"] = json.dumps(fields["params"], default=str)
        fields["updated_at"] = time.time()
        assignments = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET {assignments} WHERE id=?", (*fields.values(), job_id)
            )
            self._conn.commit()

    def bump_job(self, job_id: str, field: str, amount: int = 1) -> None:
        """Atomically increment a counter column (done/failed/skipped)."""
        if field not in ("done", "failed", "skipped", "total"):
            raise ValueError(f"not a counter column: {field}")
        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET {field}={field}+?, updated_at=? WHERE id=?",
                (amount, time.time(), job_id),
            )
            self._conn.commit()

    def get_job(self, job_id: str):
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def list_jobs(self, status: str = "", limit: int = 50) -> list:
        sql = "SELECT * FROM jobs"
        args: tuple = ()
        if status:
            sql += " WHERE status=?"
            args = (status,)
        sql += " ORDER BY created_at DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (*args, limit)).fetchall()
        return [_row_to_job(r) for r in rows]

    # --------------------------------- tasks -------------------------------
    def upsert_task(self, job_id: str, doc_id: str, **fields) -> None:
        fields.setdefault("status", STATUS_PENDING)
        fields["updated_at"] = time.time()
        columns = ["job_id", "doc_id", *fields.keys()]
        placeholders = ",".join("?" * len(columns))
        updates = ", ".join(f"{k}=excluded.{k}" for k in fields)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO tasks ({','.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(job_id, doc_id) DO UPDATE SET {updates}",
                (job_id, doc_id, *fields.values()),
            )
            self._conn.commit()

    def get_task(self, job_id: str, doc_id: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE job_id=? AND doc_id=?", (job_id, doc_id)
            ).fetchone()
        return dict(row) if row else None

    def list_tasks(self, job_id: str, status: str = "", limit: int = 1000) -> list:
        sql = "SELECT * FROM tasks WHERE job_id=?"
        args: list = [job_id]
        if status:
            sql += " AND status=?"
            args.append(status)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def completed_hashes(self, source: str = "") -> dict:
        """Map ``content_hash -> artifact`` for every successful task.

        This is what makes re-running a folder cheap: unchanged documents are
        recognised across jobs, not just within one run.
        """
        sql = ("SELECT t.content_hash, t.artifact FROM tasks t "
               "JOIN jobs j ON j.id = t.job_id WHERE t.status=? AND t.content_hash IS NOT NULL")
        args: list = [STATUS_OK]
        if source:
            sql += " AND j.source=?"
            args.append(source)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return {r["content_hash"]: r["artifact"] for r in rows}


def _row_to_job(row) -> Job:
    data = dict(row)
    try:
        data["params"] = json.loads(data.get("params") or "{}")
    except json.JSONDecodeError:
        data["params"] = {}
    data["error"] = data.get("error") or ""
    data["finished_at"] = data.get("finished_at") or 0.0
    return Job(**data)
