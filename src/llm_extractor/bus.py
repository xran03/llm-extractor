"""In-process event bus: one progress channel for CLI, logs and the HTTP API.

Every long-running stage publishes structured events instead of printing. That
single decision is what lets the same pipeline drive a terminal progress line,
a persisted job record, and a Server-Sent-Events stream to a web frontend
without any of those consumers knowing about each other.

Subscribers are either callbacks (:meth:`EventBus.subscribe`) or queue-backed
iterators (:meth:`EventBus.stream`), the latter being what the SSE endpoint
consumes. A slow subscriber can never block a worker: queues are bounded and
drop-oldest.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

# Event types are plain strings so plugins can add their own without touching
# this module. These are the ones the core emits.
JOB_STARTED = "job.started"
JOB_PROGRESS = "job.progress"
JOB_COMPLETED = "job.completed"
JOB_FAILED = "job.failed"
DOC_STARTED = "doc.started"
DOC_COMPLETED = "doc.completed"
DOC_FAILED = "doc.failed"
DOC_SKIPPED = "doc.skipped"
STAGE_STARTED = "stage.started"
STAGE_COMPLETED = "stage.completed"
CACHE_HIT = "cache.hit"
AUDIT_PROGRESS = "audit.progress"


@dataclass
class Event:
    type: str
    job_id: str = ""
    doc_id: str = ""
    stage: str = ""
    message: str = ""
    payload: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    seq: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class EventBus:
    """Thread-safe fan-out bus."""

    def __init__(self, history: int = 500):
        self._lock = threading.Lock()
        self._subscribers: dict = {}
        self._queues: dict = {}
        self._seq = 0
        self._history_limit = history
        self._history: list = []

    # ------------------------------ publishing ------------------------------
    def publish(self, event: Event) -> Event:
        with self._lock:
            self._seq += 1
            event.seq = self._seq
            self._history.append(event)
            if len(self._history) > self._history_limit:
                del self._history[: len(self._history) - self._history_limit]
            subscribers = list(self._subscribers.values())
            queues = list(self._queues.values())

        for handler, types in subscribers:
            if types and event.type not in types:
                continue
            try:
                handler(event)
            except Exception:  # a bad subscriber must never break the pipeline
                continue

        for q, types in queues:
            if types and event.type not in types:
                continue
            _offer(q, event)
        return event

    def emit(self, type: str, **kwargs) -> Event:
        """Convenience: build and publish an event in one call."""
        return self.publish(Event(type=type, **kwargs))

    # ----------------------------- subscribing ------------------------------
    def subscribe(self, handler, types=None):
        """Register a callback; returns an unsubscribe function."""
        token = uuid.uuid4().hex
        with self._lock:
            self._subscribers[token] = (handler, set(types) if types else None)

        def unsubscribe() -> None:
            with self._lock:
                self._subscribers.pop(token, None)

        return unsubscribe

    def stream(self, types=None, maxsize: int = 1000, replay: bool = False,
               timeout=None):
        """Yield events as they arrive; used by the SSE endpoint.

        Set ``replay`` to first drain the retained history, so a frontend that
        connects mid-job still renders everything that already happened. With a
        ``timeout`` the iterator yields ``None`` when idle, letting the caller
        send a keep-alive instead of blocking forever.
        """
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        token = uuid.uuid4().hex
        with self._lock:
            wanted = set(types) if types else None
            if replay:
                for past in self._history:
                    if not wanted or past.type in wanted:
                        _offer(q, past)
            self._queues[token] = (q, wanted)

        def close() -> None:
            with self._lock:
                self._queues.pop(token, None)
            _offer(q, _SENTINEL)

        def iterator():
            try:
                while True:
                    if timeout is None:
                        item = q.get()
                    else:
                        try:
                            item = q.get(timeout=timeout)
                        except queue.Empty:
                            yield None  # idle heartbeat
                            continue
                    if item is _SENTINEL:
                        return
                    yield item
            finally:
                with self._lock:
                    self._queues.pop(token, None)

        return iterator(), close

    def history(self, job_id: str = "", after_seq: int = 0) -> list:
        with self._lock:
            return [
                e for e in self._history
                if e.seq > after_seq and (not job_id or e.job_id == job_id)
            ]


_SENTINEL = object()


def _offer(q: queue.Queue, item) -> None:
    """Non-blocking put with drop-oldest backpressure."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
            q.put_nowait(item)
        except (queue.Empty, queue.Full):  # pragma: no cover - racing consumers
            pass


#: Process-wide default bus. Components accept an explicit bus; this is only the
#: fallback so simple scripts and tests need no wiring.
DEFAULT_BUS = EventBus()
