"""Execution primitives — the one place that decides how work is run.

Every stage that processes items independently (documents, chunks, figures, the
text and vision passes) goes through this module instead of driving its own
execution. Concentrating that decision here means the rest of the codebase
describes *what* to do and never how it is scheduled.

Three primitives cover every call site:

``map_ordered``    apply a function to each item, results in input order;
``map_completed``  the same, yielding each result as it becomes available;
``run_both``       run two independent stages and collect both results.

An optional accelerated core (``_exec_impl``) is used when it is installed;
otherwise the sequential implementation below runs. Both honour the same
contract, so nothing else in the package changes with the backend.
``llm-extract check`` prints which one is active.
"""
from __future__ import annotations

__all__ = ["BACKEND", "BACKEND_DETAIL", "map_completed", "map_ordered", "run_both"]

try:
    from ._exec_impl import map_completed, map_ordered, run_both  # type: ignore

    BACKEND, BACKEND_DETAIL = "accelerated", "compiled"

except ImportError:
    BACKEND, BACKEND_DETAIL = "sequential", "sequential"

    def map_ordered(fn, items, workers=1) -> list:
        """Apply ``fn`` to every item and return results in the original order."""
        return [fn(item) for item in items]

    def map_completed(fn, items, workers=1):
        """Yield ``fn(item)`` for each item as it becomes available."""
        for item in items:
            yield fn(item)

    def run_both(fn_a, fn_b, workers=2):
        """Run two independent zero-argument stages and return ``(a, b)``.

        Exceptions are returned rather than raised, because the caller records
        a failed stage and continues with whatever the other one produced.
        """
        return _capture(fn_a), _capture(fn_b)

    def _capture(fn):
        try:
            return fn()
        except Exception as exc:  # returned to the caller, which records it
            return exc
