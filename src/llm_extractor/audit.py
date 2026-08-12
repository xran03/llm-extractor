"""Cache revalidation — "is what we cached still correct?"

A cache that is never checked is a liability: model versions change, prompts
drift, and a bad extraction that was cached once is served forever. Because
every cache entry stores its original request, any entry can be **replayed**.

The auditor:

1. draws a sample from the cache index (uniform, oldest-first, largest-first,
   or never-audited-first — stratifiable by stage);
2. replays each request with the cache bypassed, optionally against a stronger
   *referee* model, which turns the audit into a cross-model check rather than
   a self-consistency check;
3. scores the old and new answers field by field;
4. writes a verdict back onto the index row (``confirmed`` / ``drifted`` /
   ``suspect`` / ``error``), so results accumulate and repeat audits can skip
   what has already been confirmed;
5. optionally invalidates drifted entries so the next run re-extracts them.

The sampling estimate is reported with a Wilson score interval, so "we checked
40 of 5,000 entries" comes with an honest confidence range.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

from .bus import AUDIT_PROGRESS, Event
from .cache import (VERDICT_CONFIRMED, VERDICT_DRIFTED, VERDICT_ERROR,
                    VERDICT_SUSPECT)
from .normalize import token_set_similarity
from .parsing import extract_json_array, extract_json_object
from .scheduler import Scheduler

#: Agreement at or above this is "confirmed"; below `SUSPECT_THRESHOLD` the
#: entry is treated as outright wrong rather than merely drifted.
CONFIRM_THRESHOLD = 0.9
SUSPECT_THRESHOLD = 0.5

#: Audit flags are pipeline metadata, not model output; never compare them.
IGNORED_FIELDS = {"doc_id", "_grounded", "_value_grounded", "_judge"}


@dataclass
class EntryAudit:
    key: str
    stage: str = ""
    doc_id: str = ""
    model: str = ""
    verdict: str = ""
    agreement: float = 0.0
    n_cached: int = 0
    n_fresh: int = 0
    added: int = 0
    removed: int = 0
    changed_fields: dict = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class AuditReport:
    sampled: int = 0
    population: int = 0
    confirmed: int = 0
    drifted: int = 0
    suspect: int = 0
    errors: int = 0
    mean_agreement: float = 0.0
    referee_model: str = ""
    strategy: str = ""
    started_at: float = field(default_factory=time.time)
    duration_s: float = 0.0
    entries: list = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        scored = self.sampled - self.errors
        return round(self.confirmed / scored, 4) if scored else 0.0

    def to_dict(self) -> dict:
        low, high = wilson_interval(self.confirmed, max(self.sampled - self.errors, 0))
        return {
            "sampled": self.sampled,
            "population": self.population,
            "confirmed": self.confirmed,
            "drifted": self.drifted,
            "suspect": self.suspect,
            "errors": self.errors,
            "pass_rate": self.pass_rate,
            "pass_rate_ci95": [low, high],
            "mean_agreement": round(self.mean_agreement, 4),
            "referee_model": self.referee_model,
            "strategy": self.strategy,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 2),
            "entries": [e.to_dict() for e in self.entries],
        }


def wilson_interval(successes: int, total: int, z: float = 1.96):
    """95% Wilson score interval — honest bounds for small audit samples."""
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (round(max(0.0, (centre - margin) / denominator), 4),
            round(min(1.0, (centre + margin) / denominator), 4))


# ----------------------------- comparison logic -----------------------------
def value_agreement(a, b) -> float:
    """Agreement between two field values in [0, 1]."""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        if a == b:
            return 1.0
        scale = max(abs(a), abs(b))
        return 1.0 if scale and abs(a - b) / scale < 0.001 else 0.0
    if isinstance(a, (list, dict)) or isinstance(b, (list, dict)):
        return 1.0 if json.dumps(a, sort_keys=True, default=str) == \
            json.dumps(b, sort_keys=True, default=str) else 0.0
    sa, sb = str(a).strip().lower(), str(b).strip().lower()
    if sa == sb:
        return 1.0
    return token_set_similarity(sa, sb)


def record_signature(record: dict) -> str:
    """Match records across two runs by their evidence span, then their content."""
    span = str(record.get("source_span") or "").strip().lower()[:120]
    if span:
        return span
    return json.dumps(
        {k: v for k, v in sorted(record.items()) if k not in IGNORED_FIELDS and v is not None},
        ensure_ascii=False, default=str,
    )[:200]


def compare_records(cached: list, fresh: list):
    """Field-level agreement between two record sets.

    Returns ``(agreement, detail)`` where detail counts added/removed records
    and which fields changed — the actionable part of an audit.
    """
    cached_map = {record_signature(r): r for r in cached if isinstance(r, dict)}
    fresh_map = {record_signature(r): r for r in fresh if isinstance(r, dict)}
    shared = set(cached_map) & set(fresh_map)
    added = sorted(set(fresh_map) - set(cached_map))
    removed = sorted(set(cached_map) - set(fresh_map))

    changed: dict = {}
    scores: list = []
    for signature in shared:
        old, new = cached_map[signature], fresh_map[signature]
        fields = sorted((set(old) | set(new)) - IGNORED_FIELDS)
        for name in fields:
            score = value_agreement(old.get(name), new.get(name))
            scores.append(score)
            if score < 1.0:
                changed[name] = changed.get(name, 0) + 1

    field_agreement = sum(scores) / len(scores) if scores else 0.0
    total = len(cached_map) | 0
    union = len(set(cached_map) | set(fresh_map))
    set_agreement = len(shared) / union if union else 1.0
    # Both "did we find the same facts" and "do the facts say the same thing".
    agreement = round(0.5 * set_agreement + 0.5 * field_agreement, 4) if scores \
        else round(set_agreement, 4)
    return agreement, {
        "n_cached": len(cached),
        "n_fresh": len(fresh),
        "added": len(added),
        "removed": len(removed),
        "changed_fields": changed,
        "set_agreement": round(set_agreement, 4),
        "field_agreement": round(field_agreement, 4),
        "population": total,
    }


def parse_payload(text: str):
    """Parse a cached/fresh response into a list of comparable dict records.

    Extraction stages answer with a record array, but the OCR and aggregation
    stages answer with a single JSON object — those are compared as one record
    rather than being pulled apart into their internal lists.
    """
    try:
        parsed = extract_json_array(text)
    except ValueError:
        parsed = []
    records = [item for item in parsed if isinstance(item, dict)]
    if records:
        return records
    obj = extract_json_object(text)
    return [obj] if obj else []


# -------------------------------- the audit --------------------------------
def audit_cache(cache, provider, n: int = 20, stage: str = "", strategy: str = "random",
                seed=None, referee_model: str = "", max_workers: int = 4,
                invalidate_drifted: bool = False, bus=None,
                only_unverified: bool = False) -> AuditReport:
    """Replay a sample of cached calls and score them against the stored answers.

    ``provider`` must be a cache-bypassing provider (see
    :func:`llm_extractor.providers.build_provider` with ``bypass_cache=True``),
    otherwise the replay would simply return the cached answer.
    """
    rows = cache.sample(
        n, stage=stage,
        strategy="unverified" if only_unverified else strategy,
        seed=seed, replayable_only=True,
    )
    report = AuditReport(
        sampled=len(rows),
        population=len(cache.query(stage=stage, limit=1000000)),
        referee_model=referee_model,
        strategy="unverified" if only_unverified else strategy,
    )
    if not rows:
        return report

    started = time.monotonic()

    def _audit_one(row) -> EntryAudit:
        key = row["key"]
        outcome = EntryAudit(key=key, stage=row.get("stage") or "",
                             doc_id=row.get("doc_id") or "",
                             model=referee_model or (row.get("model") or ""))
        entry = cache.load_entry(key)
        request = (entry or {}).get("request")
        if not request:
            outcome.verdict = VERDICT_ERROR
            outcome.error = "entry has no stored request; cannot replay"
            return outcome
        try:
            completion = provider.complete(
                request["messages"],
                model=referee_model or request["model"],
                temperature=request.get("temperature", 0.0),
                max_tokens=request.get("max_tokens"),
                json_schema=request.get("json_schema"),
                meta={"stage": f"audit:{row.get('stage') or 'unknown'}",
                      "doc_id": row.get("doc_id") or ""},
            )
        except Exception as exc:
            outcome.verdict = VERDICT_ERROR
            outcome.error = f"{type(exc).__name__}: {exc}"
            return outcome

        cached_records = parse_payload(entry.get("text", ""))
        fresh_records = parse_payload(completion.text)
        agreement, detail = compare_records(cached_records, fresh_records)

        outcome.agreement = agreement
        outcome.n_cached = detail["n_cached"]
        outcome.n_fresh = detail["n_fresh"]
        outcome.added = detail["added"]
        outcome.removed = detail["removed"]
        outcome.changed_fields = detail["changed_fields"]
        if agreement >= CONFIRM_THRESHOLD:
            outcome.verdict = VERDICT_CONFIRMED
        elif agreement >= SUSPECT_THRESHOLD:
            outcome.verdict = VERDICT_DRIFTED
        else:
            outcome.verdict = VERDICT_SUSPECT
        return outcome

    scheduler = Scheduler(max_workers=max_workers, max_retries=0, bus=None)
    results = scheduler.run(_audit_one, rows, id_of=lambda r: r["key"])

    agreements = []
    drifted_keys = []
    for task in results:
        outcome = task.result if task.status == "ok" else EntryAudit(
            key=task.item_id, verdict=VERDICT_ERROR, error=task.error)
        report.entries.append(outcome)
        cache.mark(outcome.key, outcome.verdict, outcome.agreement,
                   verified_by=referee_model or "self",
                   detail={"added": outcome.added, "removed": outcome.removed,
                           "changed_fields": outcome.changed_fields,
                           "error": outcome.error})
        if outcome.verdict == VERDICT_CONFIRMED:
            report.confirmed += 1
        elif outcome.verdict == VERDICT_DRIFTED:
            report.drifted += 1
            drifted_keys.append(outcome.key)
        elif outcome.verdict == VERDICT_SUSPECT:
            report.suspect += 1
            drifted_keys.append(outcome.key)
        else:
            report.errors += 1
        if outcome.verdict != VERDICT_ERROR:
            agreements.append(outcome.agreement)
        if bus is not None:
            bus.publish(Event(type=AUDIT_PROGRESS, doc_id=outcome.doc_id,
                              stage=outcome.stage,
                              payload={"verdict": outcome.verdict,
                                       "agreement": outcome.agreement}))

    report.mean_agreement = sum(agreements) / len(agreements) if agreements else 0.0
    report.duration_s = time.monotonic() - started

    if invalidate_drifted and drifted_keys:
        cache.invalidate(drifted_keys)
    return report
