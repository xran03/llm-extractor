"""Vector chart digitisation — read plotted points from the PDF's own geometry.

A dot plot of per-subject OPA titers holds its data in the drawing operators of
the page: every marker is a path whose coordinates the publisher wrote exactly.
Asking a vision model to eyeball those positions throws that precision away,
and cannot scale — the strict OCR schema costs ~29 output tokens per point, so
a figure with three groups of twenty-five subjects does not fit in one reply.
This module reads the coordinates instead, so the recovered value is limited by
the PDF's own rounding rather than by what a model can see.

The channel is only as trustworthy as its axis calibration, so calibration is
**overdetermined on purpose**: the axis model is fitted to every tick found and
the residual is reported. A mis-scaled or mis-paired axis then shows up as a
large residual rather than as plausible-looking wrong numbers — which is the
one failure mode that a reader cannot catch by eye.

The division of labour with the LLM is deliberate and is what keeps this
channel honest:

* geometry supplies the **numbers** — never a model;
* the LLM supplies only the **semantics** (unit, group, serotype, timepoint),
  reading the caption and the surrounding text.

Scope: vector figures, which is most of the modern literature. A scanned figure
has no drawing operators, so nothing is emitted and the vision pass keeps it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

#: Markers are small and roughly as wide as they are tall. Bars, axis rules and
#: error bars all fail one of these two tests.
MARKER_MIN_SIZE = 0.8
MARKER_MAX_SIZE = 14.0
MARKER_MIN_ASPECT = 0.55
MARKER_MAX_ASPECT = 1.8

#: A tick mark is a short stub attached to the axis line; a gridline is the
#: same rule drawn across the panel. Both stand at the tick's true position.
TICK_MAX_LENGTH = 12.0
TICK_MAX_THICKNESS = 1.6
#: How much longer than thick a rule must be before it counts as a tick mark.
TICK_MIN_ASPECT = 3.0

#: A genuine tick mark sits exactly where its axis says, so a snapped fit is
#: near-perfect. Anything looser means the stubs paired with the labels were
#: grid lines or a panel border, and the label fit — biased but sound — is the
#: safer of the two.
SNAP_MAX_RESIDUAL_PT = 0.5

#: Tick labels of one axis share a right edge; this is how much they may differ.
LABEL_COLUMN_TOLERANCE = 6.0

#: An axis needs this many ticks before its fit can be called overdetermined.
MIN_TICKS = 3

#: Reject an axis whose fitted model cannot reproduce its own ticks this well.
#: Expressed in axis-position units (points) so it means the same on a log and
#: on a linear axis.
MAX_TICK_RESIDUAL_PT = 2.5

#: Markers of one figure are drawn at one size; anything far from the modal
#: size is a different kind of object that passed the shape filter.
SIZE_TOLERANCE = 0.35

#: A legend key sits immediately to the left of its caption; a data marker does
#: not. This is how far to the right a word may start and still be the marker's
#: caption rather than unrelated text.
LEGEND_TEXT_GAP = 14.0

#: A dot plot has many markers. A handful of identical squares is a legend, a
#: set of bullets, or a bar chart's key — never a distribution worth rescuing.
MIN_POINTS = 8

#: Gap between x-clusters, as a fraction of the panel width, that separates one
#: plotted group from the next.
CLUSTER_GAP_FRACTION = 0.06

_NUMBER = re.compile(
    r"^[<>~≥≤]?\s*(-?\d+(?:[.,]\d+)?)\s*$|^10\s*[⁻-]?\s*(\d+)$"
)
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


@dataclass
class Tick:
    value: float
    pos: float          # page coordinate along the axis


@dataclass
class Axis:
    """A calibrated value axis: page position in, data value out."""

    kind: str                       # "linear" or "log"
    intercept: float
    slope: float
    ticks: list = field(default_factory=list)
    residual_pt: float = 0.0        # worst tick misfit, in page points
    label: str = ""
    x_right: float = 0.0            # right edge of its tick-label column
    calibrated_on: str = "tick labels"
    #: Distance from the tick labels to the rules they name, once measured.
    tick_offset: float | None = None

    def value_at(self, pos: float) -> float:
        raw = self.intercept + self.slope * pos
        return 10 ** raw if self.kind == "log" else raw

    @property
    def span(self) -> tuple:
        positions = [t.pos for t in self.ticks]
        return (min(positions), max(positions)) if positions else (0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "x_right": round(self.x_right, 3),
            "calibrated_on": self.calibrated_on,
            "tick_offset": (None if self.tick_offset is None
                            else round(self.tick_offset, 4)),
            "ticks": [{"value": t.value, "pos": round(t.pos, 3)} for t in self.ticks],
            "residual_pt": round(self.residual_pt, 4),
        }


@dataclass
class Point:
    """One plotted marker, with everything needed to trace it back."""

    x: float
    y: float
    value: float
    size: float
    colour: str = ""       # the marker's fill, which is how series are told apart
    cluster: int = 0
    panel: int = 0

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "value": self.value,
            "size": round(self.size, 3),
            "colour": self.colour,
            "cluster": self.cluster,
            "panel": self.panel,
        }


@dataclass
class Panel:
    """One sub-plot: an axis and the markers it governs."""

    index: int
    axis: Axis
    points: list = field(default_factory=list)
    clusters: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "axis": self.axis.to_dict(),
            "clusters": [round(c, 3) for c in self.clusters],
            "series": sorted({p.colour for p in self.points}),
            "points": [p.to_dict() for p in self.points],
        }


@dataclass
class Digitization:
    """The digitised content of one page, plus the evidence to audit it."""

    page_number: int = 0
    panels: list = field(default_factory=list)
    rejected: str = ""                             # why nothing was emitted

    @property
    def points(self) -> list:
        return [p for panel in self.panels for p in panel.points]

    @property
    def ok(self) -> bool:
        return bool(self.points)

    @property
    def worst_residual_pt(self) -> float:
        return max((p.axis.residual_pt for p in self.panels), default=0.0)

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "panels": [p.to_dict() for p in self.panels],
            "rejected": self.rejected,
        }


# --------------------------------------------------------------------------
# Tick labels
# --------------------------------------------------------------------------
def parse_tick_value(text: str):
    """Read a tick label, including the ``10^n`` form used on log axes."""
    s = (text or "").strip().translate(_SUPERSCRIPT)
    if not s:
        return None
    match = _NUMBER.match(s)
    if not match:
        return None
    plain, exponent = match.group(1), match.group(2)
    if exponent is not None:
        power = float(exponent)
        # A four-figure "exponent" is a year or a sample count, not an axis.
        return 10.0 ** power if -30 <= power <= 30 else None
    try:
        value = float(plain.replace(",", "."))
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def find_tick_labels(page) -> list:
    """Numeric words on the page, with the centre of their bounding box.

    Scientific log axes are labelled ``10²``, which a PDF stores as two words —
    the base and a raised exponent. Read separately they become the values 10
    and 2, which is how a page of panels ends up proposing an axis that fits
    beautifully and means nothing. So exponents are recombined with their base
    before anything else looks at them.
    """
    words = [w for w in page.get_text("words") if w[4].strip()]
    consumed: set = set()
    labels = []

    for index, (x0, y0, x1, y1, word, *_) in enumerate(words):
        if index in consumed:
            continue
        exponent_index = _superscript_after(words, index)
        if exponent_index is not None:
            base = _plain_number(word)
            power = _plain_number(words[exponent_index][4])
            if base is not None and power is not None and -30 <= power <= 30:
                consumed.add(exponent_index)
                labels.append(_label(base ** power, x0, x1, y0, y1))
                continue
        value = parse_tick_value(word)
        if value is not None:
            labels.append(_label(value, x0, x1, y0, y1))
    return labels


def _label(value: float, x0: float, x1: float, y0: float, y1: float) -> dict:
    return {
        "value": value,
        "x_right": x1,
        "x_centre": (x0 + x1) / 2,
        "y_centre": (y0 + y1) / 2,
    }


def _superscript_after(words: list, index: int):
    """Index of the word that is a raised exponent of ``words[index]``."""
    x0, y0, x1, y1, text, *_ = words[index]
    height = y1 - y0
    for other in range(max(0, index - 4), min(len(words), index + 5)):
        if other == index:
            continue
        ox0, oy0, ox1, oy1, otext, *_ = words[other]
        raised = (y0 + y1) / 2 - (oy0 + oy1) / 2
        if (abs(ox0 - x1) <= 2.5                       # sits against the base
                and 0.1 * height <= raised <= 0.9 * height   # and above it
                and (oy1 - oy0) <= height + 0.5              # in a smaller type
                and _plain_number(otext) is not None):
            return other
    return None


def _plain_number(text: str):
    """A bare number, accepting the minus signs typography actually uses."""
    s = (text or "").strip().translate(_SUPERSCRIPT).replace("−", "-").replace("–", "-")
    try:
        value = float(s)
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def candidate_columns(labels: list) -> list:
    """Every set of numeric labels that share a right edge.

    Tick labels of a y axis are right-aligned against it, so a column is the
    right shape for an axis. Body text, table cells and reference numbers also
    form columns, which is why a column is only a *candidate* — it has to earn
    the name by fitting a monotonic axis.
    """
    columns = {}
    for anchor in labels:
        column = [l for l in labels
                  if abs(l["x_right"] - anchor["x_right"]) <= LABEL_COLUMN_TOLERANCE]
        # One label per height: a value repeated at the same y is noise.
        seen = {}
        for label in column:
            seen.setdefault(round(label["y_centre"], 1), label)
        column = sorted(seen.values(), key=lambda l: l["y_centre"])
        if len(column) >= MIN_TICKS:
            columns[tuple(id(l) for l in column)] = column
    return sorted(columns.values(), key=len, reverse=True)


def _model_from_pair(a: dict, b: dict, kind: str):
    """The axis model determined by two ticks, or ``None`` if degenerate."""
    if a["y_centre"] == b["y_centre"] or a["value"] == b["value"]:
        return None
    if kind == "log":
        if a["value"] <= 0 or b["value"] <= 0:
            return None
        va, vb = math.log10(a["value"]), math.log10(b["value"])
    else:
        va, vb = a["value"], b["value"]
    slope = (vb - va) / (b["y_centre"] - a["y_centre"])
    if slope == 0:
        return None
    return va - slope * a["y_centre"], slope


def _inliers(labels: list, kind: str, intercept: float, slope: float,
             tolerance: float) -> list:
    """Labels the model places within ``tolerance`` points of where they are."""
    kept = []
    for label in labels:
        value = label["value"]
        if kind == "log":
            if value <= 0:
                continue
            target = math.log10(value)
        else:
            target = value
        predicted_pos = (target - intercept) / slope
        if abs(predicted_pos - label["y_centre"]) <= tolerance:
            kept.append(label)
    return kept


def calibrate(labels: list, tolerance: float = MAX_TICK_RESIDUAL_PT) -> Axis | None:
    """Fit an axis to a set of tick labels, tolerating stray numbers.

    A panel letter, a sample size or a footnote marker can sit in the same
    column as the tick labels, and demanding that *every* label fit would throw
    away a perfectly good axis because of one intruder. So the model is decided
    by consensus: each pair of labels proposes an axis, and the proposal
    supported by the most other labels wins.

    The residual is measured in page points — how far the fitted axis puts a
    tick from where it actually is — so linear and log candidates compare on
    the same scale, and the number means something physical to whoever reads
    the audit.
    """
    if len(labels) < MIN_TICKS:
        return None

    best = None                       # (inlier count, -residual, kind, fit, inliers)
    for kind in ("log", "linear"):
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                model = _model_from_pair(a, b, kind)
                if model is None:
                    continue
                inliers = _inliers(labels, kind, *model, tolerance=tolerance)
                if len(inliers) < MIN_TICKS:
                    continue
                if not _monotonic(inliers):
                    continue
                refined = _refit(inliers, kind)
                if refined is None:
                    continue
                intercept, slope = refined
                residual = _residual_pt(inliers, kind, intercept, slope)
                if residual > tolerance:
                    continue
                key = (len(inliers), -residual)
                if best is None or key > best[0]:
                    best = (key, kind, (intercept, slope), inliers, residual)

    if best is None:
        return None
    _, kind, (intercept, slope), inliers, residual = best
    ticks = [Tick(value=l["value"], pos=l["y_centre"]) for l in inliers]
    return Axis(kind=kind, intercept=intercept, slope=slope, ticks=ticks,
                residual_pt=residual,
                x_right=max(l["x_right"] for l in inliers))


def _monotonic(labels: list) -> bool:
    """A value axis runs one way: page y grows downward, so values fall."""
    ordered = sorted(labels, key=lambda l: l["y_centre"])
    values = [l["value"] for l in ordered]
    return all(b < a for a, b in zip(values, values[1:]))


def _refit(labels: list, kind: str):
    positions = [l["y_centre"] for l in labels]
    values = [l["value"] for l in labels]
    targets = [math.log10(v) for v in values] if kind == "log" else values
    fit = _fit(positions, targets)
    return fit if fit and fit[1] != 0 else None


def _residual_pt(labels: list, kind: str, intercept: float, slope: float) -> float:
    positions = [l["y_centre"] for l in labels]
    values = [l["value"] for l in labels]
    targets = [math.log10(v) for v in values] if kind == "log" else values
    return max(abs(p - (t - intercept) / slope) for p, t in zip(positions, targets))


def snap_to_tick_marks(axis: Axis, stubs: list, search: float = 22.0,
                       tolerance: float = 4.0) -> Axis:
    """Re-calibrate on the tick marks rather than on the tick labels.

    A label's bounding box is centred on its glyphs, not on the tick it names:
    ascenders and descenders pull the centre away from the axis by a fraction
    of a point. The offset is the same for every tick, so the fitted slope is
    right and the residual stays near zero — and the values come out uniformly
    wrong by about one percent, which is precisely the kind of error that
    survives every check that only looks at self-consistency.

    The tick marks themselves are drawn at the true positions, so pairing each
    label with its stub removes the bias entirely. When the stubs cannot be
    found the label fit is kept, because a one percent bias is still far better
    than no reading — but the axis records which basis it used.
    """
    near = [y for x, y in stubs if axis.x_right - 2.0 <= x <= axis.x_right + search]
    paired = _ladder(axis.ticks, near, tolerance)
    if paired is None:
        return axis

    positions = [p for _, p in paired]
    if len(set(round(p, 3) for p in positions)) != len(positions):
        return axis

    values = [v for v, _ in paired]
    targets = [math.log10(v) for v in values] if axis.kind == "log" else values
    fit = _fit(positions, targets)
    if fit is None or fit[1] == 0:
        return axis
    intercept, slope = fit
    residual = max(abs(p - (t - intercept) / slope) for p, t in zip(positions, targets))
    if residual > SNAP_MAX_RESIDUAL_PT:
        # The stubs paired with these labels do not lie on one straight scale,
        # so they were not this axis's tick marks.
        return axis
    offsets = [p - t.pos for (_, p), t in zip(paired, axis.ticks)]
    return Axis(kind=axis.kind, intercept=intercept, slope=slope,
                ticks=[Tick(value=v, pos=p) for v, p in paired],
                residual_pt=residual, label=axis.label, x_right=axis.x_right,
                calibrated_on="tick marks",
                tick_offset=sum(offsets) / len(offsets))


def _ladder(ticks: list, stubs: list, tolerance: float,
            epsilon: float = 0.25) -> list | None:
    """Pair every tick with its mark, using one shared offset for all of them.

    Matching each tick to its nearest stub independently is wrong on a dense
    figure: a data point can lie closer to a tick than the tick's own mark
    does, and one mispaired tick is enough to bend the whole scale. But a set
    of tick marks stands at a *constant* distance from its labels — that is
    what typography guarantees — so the right pairing is the offset that
    explains every tick at once, which no scatter of data points will do.
    """
    if not stubs:
        return None
    best = None
    for candidate in stubs:
        offset = candidate - ticks[0].pos
        if abs(offset) > tolerance:
            continue
        matched = []
        for tick in ticks:
            wanted = tick.pos + offset
            nearest = min(stubs, key=lambda y: abs(y - wanted))
            if abs(nearest - wanted) > epsilon:
                matched = []
                break
            matched.append((tick.value, nearest))
        if matched and (best is None or abs(offset) < best[0]):
            best = (abs(offset), matched)
    return best[1] if best else None


def share_tick_offset(axes: list, tolerance: float = 0.4) -> list:
    """Give the axes with no rule of their own the offset measured on the rest.

    A panel whose tick marks are missing — clipped, or drawn as part of a
    border — would otherwise keep the tick-label bias while its neighbours are
    exact, so one figure would be read on two different scales. The distance
    from a label's centre to its tick is a property of the type, not of the
    panel, so a page that anchors any of its axes has measured that distance
    for all of them.

    It is applied only when the anchored axes agree on the offset, and the
    result says where it came from, because a value corrected by inference must
    never be mistaken for one measured directly.
    """
    anchored = [a for a in axes if a.calibrated_on == "tick marks"]
    loose = [a for a in axes if a.calibrated_on != "tick marks"]
    if not anchored or not loose:
        return axes

    offsets = sorted(a.tick_offset for a in anchored if a.tick_offset is not None)
    if not offsets or offsets[-1] - offsets[0] > tolerance:
        return axes
    offset = offsets[len(offsets) // 2]

    corrected = []
    for axis in axes:
        if axis.calibrated_on == "tick marks":
            corrected.append(axis)
            continue
        ticks = [Tick(value=t.value, pos=t.pos + offset) for t in axis.ticks]
        values = [t.value for t in ticks]
        targets = [math.log10(v) for v in values] if axis.kind == "log" else values
        fit = _fit([t.pos for t in ticks], targets)
        if fit is None or fit[1] == 0:
            corrected.append(axis)
            continue
        intercept, slope = fit
        corrected.append(Axis(
            kind=axis.kind, intercept=intercept, slope=slope, ticks=ticks,
            residual_pt=axis.residual_pt, label=axis.label, x_right=axis.x_right,
            calibrated_on="tick offset shared from this page"))
    return corrected


def find_tick_stubs(page) -> list:
    """Horizontal rules that stand at a tick value: tick marks and gridlines.

    Both are drawn by the plotting library at the exact position of a tick, so
    either can calibrate an axis, and having both means a figure that omits one
    can still be read. They are collected once per page and shared by every
    axis: a library commonly emits a whole axis's ticks as a single path whose
    bounding box spans the axis, so the subpaths must be split out, and doing
    that per axis on a page of sixteen panels re-walks thousands of drawings to
    no purpose.
    """
    stubs = []
    for drawing in page.get_drawings():
        for rect in subpath_rects(drawing) or [tuple(drawing["rect"])]:
            width, height = rect[2] - rect[0], rect[3] - rect[1]
            if height > TICK_MAX_THICKNESS or width < 1.0:
                continue
            # A rule is far wider than it is thick; a data marker is not, and
            # letting one in lets a dot be mistaken for a tick.
            if width < TICK_MIN_ASPECT * max(height, 0.05):
                continue
            stubs.append((rect[0], (rect[1] + rect[3]) / 2))
    return stubs


def _fit(positions: list, values: list) -> tuple:
    """Least-squares fit of ``value = intercept + slope * position``."""
    n = len(positions)
    mean_p = sum(positions) / n
    mean_v = sum(values) / n
    denominator = sum((p - mean_p) ** 2 for p in positions)
    if denominator == 0:
        return None
    slope = sum((p - mean_p) * (v - mean_v) for p, v in zip(positions, values)) / denominator
    return mean_v - slope * mean_p, slope


# --------------------------------------------------------------------------
# Markers
# --------------------------------------------------------------------------
def _points_of(item) -> list:
    """Every coordinate referenced by one drawing item."""
    coords = []
    for part in item[1:]:
        if hasattr(part, "x") and hasattr(part, "y"):
            coords.append((part.x, part.y))
        elif hasattr(part, "x0"):
            coords.extend([(part.x0, part.y0), (part.x1, part.y1)])
    return coords


def subpath_rects(drawing) -> list:
    """Split one drawing into its separate shapes.

    Some producers emit every marker of a series as one path object, so the
    drawing's own bounding box covers the whole plot and tells us nothing. The
    shapes are separated by walking the items and starting a new shape wherever
    the pen jumps rather than continues.
    """
    rects = []
    current: list = []
    last_end = None
    for item in drawing.get("items") or []:
        coords = _points_of(item)
        if not coords:
            continue
        start = coords[0]
        if last_end is not None and math.dist(start, last_end) > 0.6 and current:
            rects.append(_bbox(current))
            current = []
        current.extend(coords)
        last_end = coords[-1]
    if current:
        rects.append(_bbox(current))
    return rects


def _bbox(coords: list) -> tuple:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (min(xs), min(ys), max(xs), max(ys))


def _shape_is_marker(rect: tuple) -> bool:
    width, height = rect[2] - rect[0], rect[3] - rect[1]
    if not (MARKER_MIN_SIZE <= width <= MARKER_MAX_SIZE):
        return False
    if not (MARKER_MIN_SIZE <= height <= MARKER_MAX_SIZE):
        return False
    aspect = width / height if height else 0
    return MARKER_MIN_ASPECT <= aspect <= MARKER_MAX_ASPECT


def marker_shapes(drawing) -> list:
    """The marker-shaped pieces of one drawing.

    A circle is emitted as a couple of dozen Bézier segments, and the pen lifts
    inside it often enough that splitting the path yields the same marker two
    or four times over — which inflates every count taken from the figure while
    leaving the values themselves looking perfectly correct.

    So a drawing whose own bounding box is already marker-sized *is* one
    marker, and is never split. Splitting is for the other case, where a
    library packs a whole series into a single path whose bounding box spans
    the plot and describes nothing.
    """
    rect = tuple(drawing["rect"])
    if _shape_is_marker(rect):
        return [rect]
    shapes = [r for r in subpath_rects(drawing) if _shape_is_marker(r)]
    return _drop_coincident(shapes)


def _drop_coincident(rects: list, fraction: float = 0.5) -> list:
    """Collapse subpaths of one drawing that describe the same shape.

    Distinct data points are distinct drawings, so two shapes this close
    together within a *single* path are one marker traced twice — not two
    subjects who happen to share a value.
    """
    kept: list = []
    for rect in rects:
        cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
        size = max(rect[2] - rect[0], rect[3] - rect[1], 0.1)
        if any(abs(cx - kx) <= fraction * size and abs(cy - ky) <= fraction * size
               for kx, ky in kept):
            continue
        kept.append((cx, cy))
    by_centre = {}
    for rect in rects:
        cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
        for kx, ky in kept:
            size = max(rect[2] - rect[0], rect[3] - rect[1], 0.1)
            if abs(cx - kx) <= fraction * size and abs(cy - ky) <= fraction * size:
                by_centre.setdefault((kx, ky), rect)
                break
    return list(by_centre.values())


#: Two same-coloured shapes whose centres are this close are one marker drawn
#: twice — the disc and the ring of a bordered point, emitted as two separate
#: paths. The threshold sits in a wide empty gap: measured on a real figure,
#: such pairs are ~0.0005 pt apart while the nearest genuinely distinct point
#: is 0.12 pt away, two orders of magnitude further.
COINCIDENT_EPSILON_PT = 0.02


def deduplicate_markers(candidates: list,
                        epsilon: float = COINCIDENT_EPSILON_PT) -> list:
    """Collapse the shapes that a bordered marker is drawn from.

    A plotting library draws a filled circle with an outline as two paths at
    one centre, so counting shapes counts every subject twice. The values are
    unaffected — both copies sit at the same height — which is what makes this
    dangerous: n doubles, the distribution gains phantom mass, and nothing in
    the numbers looks wrong.

    Only same-coloured shapes are merged. Two series overlapping at a point are
    two measurements and both are kept.
    """
    kept: list = []
    index: dict = {}
    for candidate in sorted(candidates, key=lambda c: -c["size"]):
        cell = (candidate["colour"],
                round(candidate["x"] / epsilon), round(candidate["y"] / epsilon))
        neighbours = [index.get((cell[0], cell[1] + dx, cell[2] + dy))
                      for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        if any(other is not None
               and abs(other["x"] - candidate["x"]) <= epsilon
               and abs(other["y"] - candidate["y"]) <= epsilon
               for other in neighbours):
            continue
        index[cell] = candidate
        kept.append(candidate)
    return kept


def find_markers(page) -> list:
    """Every shape on the page that could be a plotted data marker.

    Markers are small, filled and roughly as wide as they are tall. Bars, axis
    rules and error bars each fail one of those tests. The fill is kept because
    it is how a figure distinguishes its series — pre-dose from post-dose — and
    that mapping is not recoverable from position alone.
    """
    candidates = []
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        if fill is None and drawing.get("type") not in ("f", "fs"):
            continue
        for rect in marker_shapes(drawing):
            candidates.append({
                "x": (rect[0] + rect[2]) / 2,
                "y": (rect[1] + rect[3]) / 2,
                "size": max(rect[2] - rect[0], rect[3] - rect[1]),
                "colour": _colour_key(fill),
                "rect": rect,
            })
    return drop_legend_keys(page, deduplicate_markers(candidates))


def _colour_key(fill) -> str:
    if not fill:
        return ""
    try:
        return "#" + "".join(f"{int(round(c * 255)):02x}" for c in fill[:3])
    except (TypeError, ValueError):
        return str(fill)


def drop_legend_keys(page, candidates: list) -> list:
    """Remove swatches that are captioned, which is what makes them a legend.

    A legend key is a small filled shape with its series name set immediately
    to its right; a plotted point has nothing beside it. Without this test a
    bar chart's legend reads as a five-point distribution — small, identical,
    square and inside the axis range, so every other filter passes it.
    """
    words = [(x0, (y0 + y1) / 2) for x0, y0, x1, y1, word, *_ in page.get_text("words")
             if word.strip()]
    kept = []
    for candidate in candidates:
        rect = candidate["rect"]
        right, half = rect[2], (rect[3] - rect[1]) / 2 + 1.0
        captioned = any(right <= wx <= right + LEGEND_TEXT_GAP
                        and abs(wy - candidate["y"]) <= half
                        for wx, wy in words)
        if not captioned:
            kept.append(candidate)
    return kept


def _keep_modal_size(candidates: list) -> list:
    """Keep the shapes drawn at the series' own marker size.

    One panel draws its data points at one size. Anything materially larger or
    smaller is a different object — a bullet, an arrowhead, a fragment of an
    outlined glyph — that happened to be small and square.
    """
    if not candidates:
        return []
    sizes = sorted(c["size"] for c in candidates)
    median = sizes[len(sizes) // 2]
    if median <= 0:
        return []
    return [c for c in candidates
            if abs(c["size"] - median) / median <= SIZE_TOLERANCE]


def cluster_x(points: list, width: float) -> list:
    """Group markers into the plotted groups by their x position.

    ``width`` is the width of the *panel*, not the page: a grid of small
    multiples packs three groups into eighty points, so a gap measured against
    the page would never separate them.
    """
    if not points:
        return []
    gap = max(width * CLUSTER_GAP_FRACTION, 1.0)
    ordered = sorted(points, key=lambda p: p.x)
    cluster = 0
    ordered[0].cluster = 0
    for previous, current in zip(ordered, ordered[1:]):
        if current.x - previous.x > gap:
            cluster += 1
        current.cluster = cluster
    centres = []
    for index in range(cluster + 1):
        members = [p.x for p in ordered if p.cluster == index]
        centres.append(sum(members) / len(members))
    return centres


# --------------------------------------------------------------------------
# Page digitisation
# --------------------------------------------------------------------------
def find_axes(labels: list, tolerance: float = MAX_TICK_RESIDUAL_PT) -> list:
    """Every calibratable value axis on the page.

    A results page in this literature is a grid of small multiples — one panel
    per serotype — and each panel carries its own axis. Calibrating only the
    best one and applying it page-wide is the worst available error: it reads
    every panel through a neighbour's scale and the numbers stay plausible.

    Panels stacked vertically share the x of their tick labels, so one column
    holds several axes' worth of labels and no single fit can describe it. Each
    column is therefore peeled: the best axis is taken, its ticks are removed,
    and what is left is offered another chance until nothing more fits.
    """
    axes = []
    for column in candidate_columns(labels):
        remaining = list(column)
        while len(remaining) >= MIN_TICKS:
            axis = calibrate(remaining, tolerance=tolerance)
            if axis is None:
                break
            axes.append(axis)
            used = {round(t.pos, 3) for t in axis.ticks}
            remaining = [l for l in remaining if round(l["y_centre"], 3) not in used]
    return _deduplicate_axes(axes)


def _deduplicate_axes(axes: list) -> list:
    """Collapse candidates that describe the same physical axis.

    Overlapping label columns propose the same axis repeatedly; the richest fit
    of each group is the one to keep.
    """
    kept: list = []
    for axis in sorted(axes, key=lambda a: (len(a.ticks), -a.residual_pt), reverse=True):
        top, bottom = axis.span
        duplicate = False
        for other in kept:
            other_top, other_bottom = other.span
            overlap = min(bottom, other_bottom) - max(top, other_top)
            if (abs(axis.x_right - other.x_right) <= LABEL_COLUMN_TOLERANCE
                    and overlap > 0.5 * (bottom - top)):
                duplicate = True
                break
        if not duplicate:
            kept.append(axis)
    return kept


def panel_bounds(axes: list, page_width: float) -> dict:
    """Where each panel stops, so a missed axis cannot steal its neighbour.

    Assigning a marker to the nearest axis on its left is right only while
    every axis was found. When one is missed — its labels clipped, rotated or
    outlined — that panel's points would silently join the panel beside it and
    be read on the wrong scale. Bounding each panel at the next axis on its row
    turns that silent corruption into a visible drop.
    """
    bounds = {}
    for axis in axes:
        top, bottom = axis.span
        height = bottom - top
        right = page_width
        for other in axes:
            if other is axis or other.x_right <= axis.x_right:
                continue
            other_top, other_bottom = other.span
            overlap = min(bottom, other_bottom) - max(top, other_top)
            if height > 0 and overlap > 0.5 * height:
                right = min(right, other.x_right)
        bounds[id(axis)] = right
    return bounds


def assign_to_axis(x: float, y: float, axes: list, bounds: dict | None = None):
    """The panel a marker belongs to: the nearest axis on its left.

    Panels are laid out side by side, so the axis governing a point is the one
    whose labels sit immediately to its left and whose tick range brackets it
    vertically. Points that match no axis are dropped rather than guessed.
    """
    best = None
    for axis in axes:
        top, bottom = axis.span
        margin = 0.08 * (bottom - top) if bottom > top else 0.0
        if not (top - margin <= y <= bottom + margin):
            continue
        if x < axis.x_right:
            continue
        if bounds is not None and x > bounds.get(id(axis), float("inf")):
            continue
        distance = x - axis.x_right
        if best is None or distance < best[0]:
            best = (distance, axis)
    return best[1] if best else None


def digitize_page(page, page_number: int = 0,
                  max_residual_pt: float = MAX_TICK_RESIDUAL_PT) -> Digitization:
    """Recover the plotted values of one page, or explain why it was refused.

    Refusing is a first-class outcome. A figure this module cannot calibrate is
    left to the vision pass, which is the correct trade: an unreadable figure
    costs one vision call, whereas a wrongly calibrated one silently poisons
    every statistic computed from it.
    """
    result = Digitization(page_number=page_number)

    labels = find_tick_labels(page)
    if len(labels) < MIN_TICKS:
        result.rejected = f"only {len(labels)} numeric labels on the page"
        return result

    axes = find_axes(labels, tolerance=max_residual_pt)
    if not axes:
        result.rejected = (f"no monotonic axis fitted within {max_residual_pt} pt "
                           f"among {len(labels)} numeric labels")
        return result
    stubs = find_tick_stubs(page)
    axes = share_tick_offset([snap_to_tick_marks(axis, stubs) for axis in axes])

    markers = find_markers(page)
    if not markers:
        result.rejected = "no marker-shaped objects on the page"
        return result

    page_width = float(page.rect.width) if hasattr(page, "rect") else 600.0
    bounds = panel_bounds(axes, page_width)
    grouped: dict = {}
    for marker in markers:
        axis = assign_to_axis(marker["x"], marker["y"], axes, bounds)
        if axis is not None:
            grouped.setdefault(id(axis), (axis, []))[1].append(marker)

    for index, (axis, members) in enumerate(
            sorted(grouped.values(), key=lambda g: (g[0].span[0], g[0].x_right))):
        members = _keep_modal_size(members)
        if len(members) < MIN_POINTS:
            continue
        points = [Point(x=m["x"], y=m["y"], value=axis.value_at(m["y"]),
                        size=m["size"], colour=m["colour"], panel=index)
                  for m in members]
        panel_width = bounds.get(id(axis), page_width) - axis.x_right
        clusters = cluster_x(points, panel_width)
        result.panels.append(Panel(index=index, axis=axis, points=points,
                                   clusters=clusters))

    if not result.panels:
        result.rejected = (f"{len(markers)} marker-shaped objects, but no panel "
                           f"reached {MIN_POINTS} points under a calibrated axis")
    return result


def digitize_pdf(path, max_pages: int = 0, max_residual_pt: float = MAX_TICK_RESIDUAL_PT) -> list:
    """Digitise every page of a PDF that holds a calibratable plot."""
    try:
        import pymupdf
    except ImportError:                       # pragma: no cover - optional dep
        try:
            import fitz as pymupdf
        except ImportError:
            return []
    try:
        document = pymupdf.open(str(path))
    except Exception:
        return []

    results = []
    with document:
        for index in range(len(document)):
            if max_pages and index >= max_pages:
                break
            try:
                page = document.load_page(index)
                results.append(digitize_page(page, page_number=index + 1,
                                             max_residual_pt=max_residual_pt))
            except Exception as exc:
                results.append(Digitization(
                    page_number=index + 1,
                    rejected=f"{type(exc).__name__}: {exc}"))
    return results


def digitization_summary(results: list) -> dict:
    usable = [r for r in results if r.ok]
    return {
        "pages_examined": len(results),
        "pages_digitized": len(usable),
        "panels": sum(len(r.panels) for r in usable),
        "points": sum(len(r.points) for r in usable),
        "worst_residual_pt": round(
            max((r.worst_residual_pt for r in usable), default=0.0), 4),
    }
