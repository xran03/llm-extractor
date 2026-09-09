"""The figure-digitisation stage: measure, name, audit, and hand back records.

This is the thin layer that runs the three parts in order and keeps their
outputs together:

1. :mod:`~llm_extractor.chartdigit` measures every plotted marker it can
   calibrate, and refuses the pages it cannot;
2. :mod:`~llm_extractor.chartlabel` asks a model to name the panels, groups and
   series — and only those, never a value;
3. :mod:`~llm_extractor.chartaudit` draws the overlay and, when the text pass
   has something to compare against, checks the two channels agree.

A failure anywhere is contained to one page. The stage is worth nothing if it
can lose a document, since the documents it is best at are precisely the ones
whose data exists nowhere but in the figures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .chartaudit import GMT_TOLERANCE, check_agreement, render_overlay
from .chartdigit import MAX_TICK_RESIDUAL_PT, digitize_page
from .chartlabel import (chart_records, empty_labels, label_page,
                         methods_excerpt)

#: A page with more markers than this is not a dot plot — it is a heat map or a
#: rendering artefact, and a row per marker would swamp the corpus. The limit is
#: deliberately generous: a grid of thirteen serotype panels, each holding three
#: age groups measured before and after vaccination, legitimately carries well
#: over ten thousand points, and that page is the whole reason for this channel.
MAX_POINTS_PER_PAGE = 20000


@dataclass
class ChartResult:
    records: list = field(default_factory=list)
    digitizations: list = field(default_factory=list)   # kept for the late audit
    overlays: dict = field(default_factory=dict)        # page number -> image path
    agreement: list = field(default_factory=list)
    agreement_ran: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: list = field(default_factory=list)

    @property
    def pages(self) -> list:
        return [d.to_dict() for d in self.digitizations]

    @property
    def audits(self) -> list:
        checks: dict = {}
        for check in self.agreement:
            checks.setdefault(check.page, []).append(check)
        return [{
            "page_number": d.page_number,
            "overlay_path": self.overlays.get(d.page_number, ""),
            "worst_residual_pt": round(d.worst_residual_pt, 4),
            "panels": len(d.panels),
            "points": len(d.points),
            "agreement": [c.to_dict() for c in checks.get(d.page_number, [])],
            "agreement_ran": self.agreement_ran,
        } for d in self.digitizations if d.ok]

    def summary(self) -> dict:
        digitized = [d for d in self.digitizations if d.ok]
        return {
            "chart_pages": len(digitized),
            "chart_panels": sum(len(d.panels) for d in digitized),
            "chart_points": len(self.records),
            "chart_worst_residual_pt": round(
                max((d.worst_residual_pt for d in digitized), default=0.0), 4),
            "chart_prompt_tokens": self.prompt_tokens,
            "chart_completion_tokens": self.completion_tokens,
            # Reported as ``None`` rather than 0 until the comparison has
            # actually been made: "checked and agreed" and "never checked" must
            # not look the same to whoever reads these numbers.
            "chart_agreement_checks": len(self.agreement) if self.agreement_ran else None,
            "chart_disagreements": (sum(1 for c in self.agreement
                                        if c.verdict == "disagrees")
                                    if self.agreement_ran else None),
        }


def digitize_document(provider, path, doc_id: str = "", model: str = "",
                      overlay_dir=None, max_pages: int = 0,
                      max_points: int = MAX_POINTS_PER_PAGE,
                      max_residual_pt: float = MAX_TICK_RESIDUAL_PT,
                      label: bool = True) -> ChartResult:
    """Digitise and name every figure page of one PDF.

    The cross-channel audit is deliberately *not* run here. It compares what
    geometry measured against what the text pass read, and the text pass has
    not happened yet — so it is left to :func:`audit_against_text`, which the
    pipeline calls once both channels have answered.
    """
    result = ChartResult()
    try:
        import pymupdf
    except ImportError:                       # pragma: no cover - optional dep
        try:
            import fitz as pymupdf
        except ImportError:
            result.errors.append("figure digitisation needs PyMuPDF: "
                                 "pip install 'llm-extractor[render]'")
            return result
    try:
        document = pymupdf.open(str(path))
    except Exception as exc:
        result.errors.append(f"{Path(path).name}: {type(exc).__name__}: {exc}")
        return result

    with document:
        methods = ""
        if label and provider is not None and model:
            try:
                methods = methods_excerpt("\n".join(
                    document.load_page(i).get_text() or ""
                    for i in range(min(len(document), 12))))
            except Exception:
                methods = ""
        for index in range(len(document)):
            if max_pages and index >= max_pages:
                break
            try:
                page = document.load_page(index)
                _digitize_one(provider, page, index + 1, result, doc_id=doc_id,
                              model=model, overlay_dir=overlay_dir,
                              max_points=max_points,
                              max_residual_pt=max_residual_pt, label=label,
                              source_path=str(path), methods_text=methods)
            except Exception as exc:
                result.errors.append(
                    f"page {index + 1}: {type(exc).__name__}: {exc}")
    return result


def reported_values(records: list) -> list:
    """The numbers the text pass read, as candidates to compare against."""
    reported = []
    for record in records or []:
        value = record.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        label = " ".join(str(record.get(key)) for key in
                         ("serotype", "group_label", "timepoint", "endpoint")
                         if record.get(key))
        reported.append({"value": value, "label": label})
    return reported


def audit_against_text(result: ChartResult, records: list,
                       tolerance: float = GMT_TOLERANCE) -> ChartResult:
    """Compare the digitised panels with what the text pass reported.

    This is the only check in the set that is genuinely independent: prose and
    geometry share no code, no model and no input, so their agreeing is real
    evidence. It runs after extraction because it cannot run before — and it
    records that it ran, so a document with nothing to compare against is not
    mistaken for one that was compared and passed.
    """
    reported = reported_values(records)
    result.agreement = []
    for digitization in result.digitizations:
        if digitization.ok:
            result.agreement.extend(
                check_agreement(digitization, reported, tolerance=tolerance))
    result.agreement_ran = bool(reported) and bool(result.agreement)
    return result


def _digitize_one(provider, page, page_number: int, result: ChartResult,
                  doc_id: str, model: str, overlay_dir, max_points: int,
                  max_residual_pt: float, label: bool, source_path: str,
                  methods_text: str = "") -> None:
    digitization = digitize_page(page, page_number=page_number,
                                 max_residual_pt=max_residual_pt)
    if not digitization.ok:
        result.digitizations.append(digitization)
        return
    if len(digitization.points) > max_points:
        digitization.rejected = (f"{len(digitization.points)} markers exceeds the "
                                 f"{max_points} allowed for one page")
        digitization.panels = []
        result.digitizations.append(digitization)
        return

    labels = empty_labels()
    if label and provider is not None and model:
        try:
            labels, usage = label_page(provider, page, digitization, model=model,
                                       page_text=page.get_text() or "",
                                       doc_id=doc_id, methods_text=methods_text)
            result.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            result.completion_tokens += int(usage.get("completion_tokens") or 0)
        except Exception as exc:
            # The measurements are already made and are the valuable half; an
            # unnamed point is still a point, so a failed naming call must not
            # discard it.
            result.errors.append(
                f"page {page_number} labelling: {type(exc).__name__}: {exc}")

    if overlay_dir:
        Path(overlay_dir).mkdir(parents=True, exist_ok=True)
        overlay = Path(overlay_dir) / f"{doc_id or 'doc'}-p{page_number:03d}.overlay.png"
        written = render_overlay(page, digitization, overlay)
        if written:
            result.overlays[page_number] = written

    result.records.extend(chart_records(digitization, labels, doc_id=doc_id,
                                        source_path=source_path))
    result.digitizations.append(digitization)


def digitized_pages(result: ChartResult) -> set:
    """Page numbers this stage measured, so the vision pass can skip them."""
    return {d.page_number for d in result.digitizations if d.ok}
