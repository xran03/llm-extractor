"""Reconciling the channels: one measurement, one record.

Three readers look at the same document, and where they overlap they will
report the same fact more than once — a titer stated in the prose, plotted in a
figure, and measured off that figure's geometry is one measurement, not three.
Concatenating the passes therefore inflates every count taken from the corpus,
and does so invisibly: each duplicate is individually correct, so nothing looks
wrong until someone counts.

Two different situations hide in that overlap and they must not be treated
alike:

*duplicates*  the channels agree. One record survives, and it records which
              other channels saw the same thing — agreement between
              independent readers is evidence, and throwing it away wastes the
              best signal in the pipeline.

*conflicts*   the channels disagree about the same measurement. The more
              reliable reading survives, but the other value is kept on the
              record and flagged, because a disagreement is a finding. Silently
              choosing a winner would hide exactly the cases a reviewer needs.

Individual data points are deliberately exempt from conflict handling. Two
hundred subjects in one group share every key field and differ only in their
values; that is the shape of the data, not a contradiction.
"""
from __future__ import annotations

from collections import defaultdict

#: How much a reading may be trusted with a number, most reliable first.
#:
#: ``vector_geometry`` is a measurement of the publisher's own coordinates.
#: ``llm`` quotes a number whose digits were checked against the document.
#: ``multimodal`` is the same, decided with more context in front of it.
#: ``ocr`` is a model's estimate of where a mark sat, and is the fallback for
#: the figures nothing else can read — useful, but outranked wherever anything
#: else saw the same measurement.
CHANNEL_TRUST = {
    "vector_geometry": 4,
    "llm": 3,
    "multimodal": 2,
    "ocr": 1,
    "na": 0,
}

#: Two readings of one measurement agree within this relative difference.
#: Published values are rounded and a figure is read to two significant figures
#: at best, so exact equality is the wrong test.
VALUE_TOLERANCE = 0.02

#: Value kinds where many records legitimately share every key field.
PER_SUBJECT_KINDS = ("individual",)


def channel_of(record: dict) -> str:
    return str(record.get("extraction_mode") or "na")


def trust(record: dict) -> int:
    return CHANNEL_TRUST.get(channel_of(record), 0)


def identity(record: dict, key_fields: list) -> tuple:
    """What makes two records the same measurement, regardless of who read it.

    Deliberately excludes the evidence: a quoted sentence and a calibrated axis
    are different justifications for the same fact, so including them would
    make every cross-channel duplicate look unique — which is precisely why the
    existing chunk-level dedupe never collapsed them.
    """
    return tuple(str(record.get(field) or "").strip().lower()
                 for field in [*key_fields, "value_kind", "value_unit"])


def _comparable(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def values_agree(a, b, tolerance: float = VALUE_TOLERANCE) -> bool:
    first, second = _comparable(a), _comparable(b)
    if first is None or second is None:
        return first == second
    if first == second:
        return True
    scale = max(abs(first), abs(second))
    return scale > 0 and abs(first - second) / scale <= tolerance


def reconcile(records: list, template, tolerance: float = VALUE_TOLERANCE) -> tuple:
    """Collapse cross-channel duplicates and flag cross-channel conflicts.

    Returns the reconciled records and a summary of what was done, so a run can
    report how much of its output was corroborated rather than merely counted
    twice.
    """
    key_fields = list(template.key_fields or template.field_names[:2])
    groups: dict = defaultdict(list)
    for record in records:
        groups[identity(record, key_fields)].append(record)

    kept: list = []
    stats = {"duplicates_merged": 0, "conflicts_flagged": 0, "corroborated": 0}

    for group in groups.values():
        for cluster in _cluster_by_value(group, tolerance):
            winner = max(cluster, key=trust)
            others = [r for r in cluster if r is not winner]
            if others:
                channels = sorted({channel_of(r) for r in others}
                                  - {channel_of(winner)})
                if channels:
                    winner["corroborated_by"] = ", ".join(channels)
                    stats["corroborated"] += 1
                stats["duplicates_merged"] += len(others)
            kept.append(winner)

        _flag_conflicts(group, tolerance, stats)

    return kept, stats


def _cluster_by_value(group: list, tolerance: float) -> list:
    """Split one identity into the distinct values reported for it."""
    clusters: list = []
    for record in group:
        for cluster in clusters:
            if values_agree(cluster[0].get("value"), record.get("value"), tolerance):
                cluster.append(record)
                break
        else:
            clusters.append([record])
    return clusters


def _flag_conflicts(group: list, tolerance: float, stats: dict) -> None:
    """Mark a measurement that two channels read differently.

    Only across channels, and never for per-subject values: a group of
    individual titers shares every key field by construction, and calling that
    a contradiction would flag the whole distribution.
    """
    kinds = {str(r.get("value_kind") or "").lower() for r in group}
    if kinds & set(PER_SUBJECT_KINDS):
        return

    by_channel: dict = {}
    for record in group:
        by_channel.setdefault(channel_of(record), []).append(record)
    if len(by_channel) < 2:
        return

    readings = [(channel, r) for channel, rs in by_channel.items() for r in rs]
    for channel, record in readings:
        rival = next((other for other_channel, other in readings
                      if other_channel != channel
                      and not values_agree(record.get("value"),
                                           other.get("value"), tolerance)), None)
        if rival is None or record.get("modality_conflict") == "value_mismatch":
            continue
        record["modality_conflict"] = "value_mismatch"
        record["notes"] = _append_note(
            record.get("notes"),
            f"{channel_of(rival)} read {rival.get('value')} for the same "
            f"measurement; this record is the {channel_of(record)} reading")
        stats["conflicts_flagged"] += 1


def _append_note(existing, addition: str) -> str:
    existing = str(existing or "").strip()
    return f"{existing}; {addition}" if existing else addition


def reconciliation_summary(stats: dict, before: int, after: int) -> dict:
    return {
        "records_before_reconcile": before,
        "records_after_reconcile": after,
        "duplicates_merged": stats.get("duplicates_merged", 0),
        "conflicts_flagged": stats.get("conflicts_flagged", 0),
        "corroborated_records": stats.get("corroborated", 0),
    }
