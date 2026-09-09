"""Auditing the digitised figures — three checks that fail in different ways.

Reconstructing a scatter plot from the recovered values and comparing it with
the original is the obvious audit, and it is worth having, but on its own it
proves nothing: the values came from inverting the axis model, so replotting
them through the same model reproduces the original by construction. It is a
tautology dressed as evidence.

What follows are three checks chosen so that each one can fail while the others
pass, which is the only property that makes an audit informative:

``overlay``     draws the detected markers back onto the *rendered page*. This
                audits **detection** — whether the shapes picked up were data
                points at all, rather than error-bar caps, legend keys or the
                pieces of an outlined glyph. It is checked against the image,
                not against the model, so it is not circular.

``calibration`` reports how far the fitted axis puts each tick from where the
                tick actually is. Because the fit is overdetermined — three or
                more ticks constraining two parameters — a mis-paired or
                mis-scaled axis cannot hide. This is the check that catches the
                single most dangerous failure, a plausible wrong scale.

``agreement``   recomputes the summary statistics from the recovered points and
                compares them with what the *text* pass read from the prose and
                tables. The two channels share no code, no model and no input,
                so agreement is real evidence and disagreement names a suspect.

Only the third can catch a figure that was digitised perfectly but belongs to a
different group than the one it was matched to, which is why the cheap checks
do not replace it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Colours used to mark up the overlay, as RGB triples in the 0-1 range.
POINT_COLOUR = (0.0, 0.75, 0.1)
TICK_COLOUR = (1.0, 0.0, 0.85)
PANEL_COLOUR = (0.0, 0.45, 1.0)

#: A recovered geometric mean this far from the reported one is a disagreement.
#: Figures are read to two significant figures at best, and a published GMT is
#: rounded, so a few percent is expected; ten percent is not.
GMT_TOLERANCE = 0.10


@dataclass
class AgreementCheck:
    """One comparison of a digitised panel against a reported summary value."""

    panel: int
    page: int = 0
    series: str = ""
    n_points: int = 0
    recovered_gmt: float = 0.0
    reported_value: float | None = None
    reported_label: str = ""
    relative_error: float | None = None
    verdict: str = "unchecked"        # agrees | disagrees | unchecked

    def to_dict(self) -> dict:
        return {
            "panel": self.panel,
            "page": self.page,
            "series": self.series,
            "n_points": self.n_points,
            "recovered_gmt": round(self.recovered_gmt, 6),
            "reported_value": self.reported_value,
            "reported_label": self.reported_label,
            "relative_error": (None if self.relative_error is None
                               else round(self.relative_error, 6)),
            "verdict": self.verdict,
        }


@dataclass
class DigitizationAudit:
    page_number: int = 0
    overlay_path: str = ""
    worst_residual_pt: float = 0.0
    panels: int = 0
    points: int = 0
    agreement: list = field(default_factory=list)

    @property
    def disagreements(self) -> list:
        return [c for c in self.agreement if c.verdict == "disagrees"]

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "overlay_path": self.overlay_path,
            "worst_residual_pt": round(self.worst_residual_pt, 4),
            "panels": self.panels,
            "points": self.points,
            "agreement": [c.to_dict() for c in self.agreement],
            "disagreements": len(self.disagreements),
        }


# --------------------------------------------------------------------------
# 1. Detection — mark what was found onto the page as rendered
# --------------------------------------------------------------------------
def render_overlay(page, digitization, out_path, zoom: float = 2.0) -> str:
    """Draw the detected markers and ticks onto a render of the page.

    A reviewer should be able to answer "did it find the dots?" in one glance,
    without reading a number. Each detected point gets a ring, each calibration
    tick a bar at the position the fit believes it occupies, and each panel the
    box its axis governs.
    """
    try:
        import pymupdf
    except ImportError:                       # pragma: no cover - optional dep
        try:
            import fitz as pymupdf
        except ImportError:
            return ""

    shape = page.new_shape()
    for panel in digitization.panels:
        top, bottom = panel.axis.span
        xs = [p.x for p in panel.points]
        shape.draw_rect(pymupdf.Rect(min(xs) - 4, top - 4, max(xs) + 4, bottom + 4))
        shape.finish(color=PANEL_COLOUR, width=0.7, dashes="[3 3] 0")

        for tick in panel.axis.ticks:
            shape.draw_line(pymupdf.Point(panel.axis.x_right + 1, tick.pos),
                            pymupdf.Point(panel.axis.x_right + 9, tick.pos))
        shape.finish(color=TICK_COLOUR, width=1.1)

        for point in panel.points:
            radius = max(point.size, 1.2)
            shape.draw_circle(pymupdf.Point(point.x, point.y), radius)
        shape.finish(color=POINT_COLOUR, width=0.5)
    shape.commit()

    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    pixmap.save(str(out_path))
    return str(out_path)


# --------------------------------------------------------------------------
# 3. Agreement — the only genuinely independent check
# --------------------------------------------------------------------------
def geometric_mean(values: list) -> float:
    """GMT of the recovered points, which is what these papers report."""
    positive = [v for v in values if v and v > 0]
    if not positive:
        return 0.0
    return 10 ** (sum(math.log10(v) for v in positive) / len(positive))


def summarise_panel(panel) -> dict:
    """Per-series summary statistics of one digitised panel."""
    series: dict = {}
    for point in panel.points:
        series.setdefault(point.colour, []).append(point.value)
    return {
        colour: {
            "n": len(values),
            "gmt": geometric_mean(values),
            "min": min(values),
            "max": max(values),
        }
        for colour, values in series.items()
    }


def check_agreement(digitization, reported: list,
                    tolerance: float = GMT_TOLERANCE) -> list:
    """Compare each digitised series against the values the text pass read.

    A reported geometric mean is matched to the series whose recovered mean is
    closest *in log space*, because a figure's series cannot be identified from
    geometry alone — that is the labelling model's job, and this check must not
    depend on it having been right. Matching to the nearest candidate is the
    weakest assumption that still lets the two channels be compared: if no
    series lands near the reported value, every candidate is flagged.
    """
    candidates = []
    for value in reported:
        try:
            number = float(value.get("value"))
        except (TypeError, ValueError):
            continue
        if number > 0:
            candidates.append((number, str(value.get("label") or "")))

    checks = []
    for panel in digitization.panels:
        for colour, stats in summarise_panel(panel).items():
            check = AgreementCheck(panel=panel.index,
                                   page=digitization.page_number,
                                   series=colour, n_points=stats["n"],
                                   recovered_gmt=stats["gmt"])
            if candidates and stats["gmt"] > 0:
                number, label = min(
                    candidates,
                    key=lambda c: abs(math.log10(c[0]) - math.log10(stats["gmt"])))
                check.reported_value = number
                check.reported_label = label
                check.relative_error = abs(number - stats["gmt"]) / number
                check.verdict = ("agrees" if check.relative_error <= tolerance
                                 else "disagrees")
            checks.append(check)
    return checks


def audit_page(page, digitization, reported: list | None = None,
               overlay_path=None, tolerance: float = GMT_TOLERANCE) -> DigitizationAudit:
    """Run all three checks over one digitised page."""
    audit = DigitizationAudit(
        page_number=digitization.page_number,
        worst_residual_pt=digitization.worst_residual_pt,
        panels=len(digitization.panels),
        points=len(digitization.points),
    )
    if overlay_path is not None and digitization.panels:
        audit.overlay_path = render_overlay(page, digitization, overlay_path)
    if reported:
        audit.agreement = check_agreement(digitization, reported, tolerance=tolerance)
    return audit
