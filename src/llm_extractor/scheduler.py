"""Rate limiting, retry and error isolation — the scheduling layer.

Extraction is a fan-out over many documents, each costing one or more API
calls. This module owns *how* that work is driven and how failures are handled,
so pipeline stages stay pure functions of a document.

Responsibilities:

* a token-bucket rate limiter, because API gateways throttle per account;
* retry with exponential backoff and jitter for transient failures only;
* per-item error isolation — one bad document never aborts a run;
* cooperative cancellation;
* event emission so progress is observable everywhere.

Execution itself is delegated to :mod:`llm_extractor._exec`.
"""
from __future__ import annotations

import random
import threading
import time
import traceback
from dataclasses import dataclass, field

from ._exec import map_completed
from .bus import DOC_COMPLETED, DOC_FAILED, DOC_SKIPPED, DOC_STARTED, JOB_PROGRESS, Event, EventBus


class Cancelled(RuntimeError):
    """Raised inside a task when the job has been cancelled."""


class RateLimiter:
    """Token bucket guarding the API call rate.

    ``max_calls`` tokens refill smoothly over ``per_seconds``. A limiter of
    ``None`` (or ``max_calls<=0``) is a no-op, so rate limiting is opt-in.
    """

    def __init__(self, max_calls: int, per_seconds: float = 60.0):
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self._tokens = float(max_calls)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> float:
        if self.max_calls <= 0:
            return 0.0
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                refill = (now - self._updated) * (self.max_calls / self.per_seconds)
                self._tokens = min(self.max_calls, self._tokens + refill)
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                deficit = tokens - self._tokens
                delay = deficit * (self.per_seconds / self.max_calls)
            time.sleep(min(delay, 1.0))
            waited += min(delay, 1.0)


@dataclass
class TaskResult:
    item_id: str
    status: str            # ok | error | skipped | cancelled
    result: object = None
    error: str = ""
    trace: str = ""
    attempts: int = 0
    duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "status": self.status,
            "error": self.error,
            "attempts": self.attempts,
            "duration": round(self.duration, 3),
        }


class CancelToken:
    """Cooperative cancellation shared by the scheduler and its tasks."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise Cancelled("job cancelled")


class Skip(Exception):
    """Raise inside a task to record the item as skipped rather than failed."""

    def __init__(self, reason: str = "", result=None):
        super().__init__(reason)
        self.reason = reason
        self.result = result


@dataclass
class Scheduler:
    max_workers: int = 8
    max_retries: int = 2
    backoff_base: float = 1.5
    backoff_jitter: float = 0.3
    rate_limiter: RateLimiter | None = None
    bus: EventBus | None = None
    cancel_token: CancelToken = field(default_factory=CancelToken)
    retry_on: tuple = (Exception,)
    dont_retry_on: tuple = ()

    def cancel(self) -> None:
        self.cancel_token.cancel()

    def run(self, func, items, job_id: str = "", id_of=None, on_result=None) -> list:
        """Run ``func(item)`` over ``items`` and return one TaskResult each.

        ``on_result`` is invoked as each item finishes, so callers can persist
        state incrementally rather than waiting for the whole run.
        """
        id_of = id_of or (lambda item: getattr(item, "doc_id", None) or str(item))
        items = list(items)
        results: list = []
        total = len(items)

        if not items:
            return results

        def _work(item):
            return self._run_one(func, item, id_of(item), job_id)

        for result in map_completed(_work, items, workers=self.max_workers):
            results.append(result)
            if on_result is not None:
                try:
                    on_result(result)
                except Exception:
                    pass
            self._emit(Event(
                type=JOB_PROGRESS, job_id=job_id,
                payload={"completed": len(results), "total": total,
                         "status": result.status},
            ))
        return results

    # ------------------------------- internals ------------------------------
    def _emit(self, event: Event) -> None:
        if self.bus is not None:
            self.bus.publish(event)

    def _run_one(self, func, item, item_id: str, job_id: str) -> TaskResult:
        started = time.monotonic()
        if self.cancel_token.cancelled:
            return TaskResult(item_id, "cancelled", attempts=0)

        self._emit(Event(type=DOC_STARTED, job_id=job_id, doc_id=item_id))
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 2):
            try:
                self.cancel_token.raise_if_cancelled()
                if self.rate_limiter is not None:
                    self.rate_limiter.acquire()
                value = func(item)
                duration = time.monotonic() - started
                self._emit(Event(type=DOC_COMPLETED, job_id=job_id, doc_id=item_id,
                                 payload={"attempts": attempt,
                                          "duration": round(duration, 3)}))
                return TaskResult(item_id, "ok", result=value, attempts=attempt,
                                  duration=duration)
            except Skip as skip:
                duration = time.monotonic() - started
                self._emit(Event(type=DOC_SKIPPED, job_id=job_id, doc_id=item_id,
                                 message=skip.reason))
                return TaskResult(item_id, "skipped", result=skip.result,
                                  error=skip.reason, attempts=attempt, duration=duration)
            except Cancelled:
                return TaskResult(item_id, "cancelled", attempts=attempt,
                                  duration=time.monotonic() - started)
            except self.dont_retry_on as exc:  # type: ignore[misc]
                last_error = exc
                break
            except self.retry_on as exc:  # type: ignore[misc]
                last_error = exc
                if attempt > self.max_retries or self.cancel_token.cancelled:
                    break
                delay = self.backoff_base ** (attempt - 1)
                delay += random.uniform(0, self.backoff_jitter * delay)
                time.sleep(delay)

        duration = time.monotonic() - started
        message = f"{type(last_error).__name__}: {last_error}"
        self._emit(Event(type=DOC_FAILED, job_id=job_id, doc_id=item_id, message=message))
        return TaskResult(
            item_id, "error", error=message,
            trace="".join(traceback.format_exception_only(type(last_error), last_error)),
            attempts=self.max_retries + 1, duration=duration,
        )
