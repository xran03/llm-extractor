"""Naming what geometry cannot know.

Digitisation recovers a number for every plotted marker, but a number alone is
not a record: 4.7 is meaningless until it is 4.7 µg/mL of serotype 6B IgG in
older adults 28 days after PCV13. Those facts are written in the panel titles,
the axis captions, the legend and the surrounding prose — which is exactly what
a language model is good at reading, and exactly what geometry cannot see.

So the two channels are split along their competences, and the split is the
anti-hallucination guarantee of this pipeline:

* **geometry decides every number.** The model is never shown a value, never
  asked for one, and its answer has nowhere to put one — the response schema
  has no numeric field at all. A hallucinated titer therefore has no path into
  the output; the worst a wrong answer can do is mislabel a real measurement,
  which the audit's cross-channel check is built to catch.
* **the model decides every name.** Which serotype a panel shows, which group a
  column of dots belongs to, what the axis unit is.

The skeleton handed over is deliberately small — panels, clusters and series,
not points — so a figure with twelve thousand markers still costs one short
prompt. That is the same asymmetry that makes this channel worth having: the
expensive part scales with the *structure* of a figure, not with its data.
"""
from __future__ import annotations

import json
import math

from .parsing import extract_json_object

LABEL_SYSTEM_PROMPT = (
    "You label the parts of a scientific figure whose data has already been "
    "measured from the page geometry. You are given the figure's structure and "
    "the text printed around it. You name panels, groups and series. You never "
    "report, estimate or invent a measurement — the numbers are not yours to "
    "produce, and there is nowhere in your response to put one."
)

LABEL_INSTRUCTIONS = (
    "Each panel of this figure has been located, its axis calibrated, and its "
    "markers counted. Name the parts.\n"
    "- panels: what the panel plots. Give the serotype (or analyte), the assay "
    "(opa | igg | igm | other | na), the endpoint as printed, and the unit taken "
    "from the axis label.\n"
    "- assay_platform: the technology that produced these numbers, from the "
    "methods text supplied. An IgG concentration from a reference ELISA, from a "
    "direct Luminex immunoassay and from an electrochemiluminescence assay are "
    "not the same quantity, and a multiplexed OPA titer is not a single-serotype "
    "one — so this decides whether two studies can be compared at all. Put the "
    "paper's own wording in assay_platform_detail and the reference serum in "
    "assay_standard. Use 'na' if the methods do not say; never infer it from the "
    "manufacturer.\n"
    "- clusters: one per column of markers, in the left-to-right order given. "
    "Name the study group and, if the x axis encodes it, the timepoint.\n"
    "- series: one per marker colour. Name what the legend says that colour "
    "means, and its timepoint if that is what the colour encodes.\n"
    "Use null wherever the figure does not say. Do not guess a serotype that is "
    "not printed. Return JSON only."
)

LABEL_JSON_SCHEMA = {
    "name": "figure_labels",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "figure_label": {"type": ["string", "null"],
                             "description": "The figure's own name, e.g. 'Figure 1A'."},
            "panels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "title": {"type": ["string", "null"]},
                        "serotype": {"type": ["string", "null"]},
                        "assay": {"type": ["string", "null"],
                                  "enum": ["opa", "igg", "igm", "other", "na", None]},
                        "endpoint": {"type": ["string", "null"]},
                        "unit": {"type": ["string", "null"]},
                        "species": {"type": ["string", "null"]},
                        "assay_platform": {
                            "type": ["string", "null"],
                            "enum": ["elisa", "luminex", "ecl", "mopa", "opa_single",
                                     "flow_cytometry", "agglutination", "other",
                                     "na", None]},
                        "assay_platform_detail": {"type": ["string", "null"]},
                        "assay_standard": {"type": ["string", "null"]},
                    },
                    "required": ["index", "title", "serotype", "assay", "endpoint",
                                 "unit", "species", "assay_platform",
                                 "assay_platform_detail", "assay_standard"],
                    "additionalProperties": False,
                },
            },
            "clusters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "panel": {"type": "integer"},
                        "cluster": {"type": "integer"},
                        "group_label": {"type": ["string", "null"]},
                        "timepoint": {"type": ["string", "null"]},
                    },
                    "required": ["panel", "cluster", "group_label", "timepoint"],
                    "additionalProperties": False,
                },
            },
            "series": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "colour": {"type": "string"},
                        "label": {"type": ["string", "null"]},
                        "timepoint": {"type": ["string", "null"]},
                    },
                    "required": ["colour", "label", "timepoint"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["figure_label", "panels", "clusters", "series"],
        "additionalProperties": False,
    },
}

#: Text printed this far outside a panel still describes it — the title sits
#: above, the group names below, the axis caption to the left.
CONTEXT_MARGIN_PT = 34.0

#: Cap on the surrounding prose handed over, so a dense page stays affordable.
MAX_CONTEXT_CHARS = 4000


def panel_context(page, panel, margin: float = CONTEXT_MARGIN_PT) -> list:
    """The words printed in and around one panel, in reading order."""
    top, bottom = panel.axis.span
    xs = [p.x for p in panel.points] or [panel.axis.x_right]
    left, right = min(xs) - margin, max(xs) + margin
    words = []
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        if not word.strip():
            continue
        if (left <= (x0 + x1) / 2 <= right
                and top - margin <= (y0 + y1) / 2 <= bottom + margin):
            words.append((round(y0, 1), round(x0, 1), word))
    return [w for _, _, w in sorted(words)]


def build_skeleton(page, digitization) -> dict:
    """The structure of the figure, with no measurement in it.

    Marker counts are included because they are structural — they say how many
    subjects a column holds — while the values they were measured from are not,
    so that nothing in this prompt can anchor the model toward a number.
    """
    panels = []
    for panel in digitization.panels:
        series: dict = {}
        clusters: dict = {}
        for point in panel.points:
            series[point.colour] = series.get(point.colour, 0) + 1
            clusters[point.cluster] = clusters.get(point.cluster, 0) + 1
        top, bottom = panel.axis.span
        panels.append({
            "index": panel.index,
            "axis": {
                "scale": panel.axis.kind,
                "from": min(t.value for t in panel.axis.ticks),
                "to": max(t.value for t in panel.axis.ticks),
                "ticks": [t.value for t in panel.axis.ticks],
            },
            "position": {"x": round(panel.axis.x_right, 1),
                         "y_top": round(top, 1), "y_bottom": round(bottom, 1)},
            "clusters": [{"cluster": index, "x": round(x, 1),
                          "markers": clusters.get(index, 0)}
                         for index, x in enumerate(panel.clusters)],
            "series": [{"colour": colour, "markers": count}
                       for colour, count in sorted(series.items())],
            "text_nearby": " ".join(panel_context(page, panel))[:600],
        })
    return {"page": digitization.page_number, "panels": panels}


def build_label_messages(skeleton: dict, page_text: str = "",
                         methods_text: str = "") -> list:
    payload = {"figure_structure": skeleton,
               "page_text": (page_text or "")[:MAX_CONTEXT_CHARS]}
    if methods_text:
        # The platform is named in the methods, which is rarely the page the
        # figure sits on, so it has to be carried here explicitly.
        payload["methods_text"] = methods_text[:MAX_METHODS_CHARS]
    return [
        {"role": "system", "content": LABEL_SYSTEM_PROMPT},
        {"role": "user",
         "content": f"{LABEL_INSTRUCTIONS}\n\n{json.dumps(payload, ensure_ascii=False)}"},
    ]


#: Section headings that introduce the description of how a value was measured.
METHODS_HEADINGS = ("materials and methods", "methods", "methodology",
                    "experimental procedures", "study design")

#: Assay wording worth carrying even when no methods heading was found.
ASSAY_HINTS = ("elisa", "luminex", "dlia", "electrochemiluminescen", "opa",
               "opsonophagocyt", "mopa", "assay", "reference serum", "89-sf",
               "007sp", "calibrat")

MAX_METHODS_CHARS = 3000


def methods_excerpt(text: str, limit: int = MAX_METHODS_CHARS) -> str:
    """The part of a paper that says how its numbers were produced.

    The whole document would answer the question and would also cost more than
    the rest of this channel put together, so the methods section is located by
    its heading and, failing that, by the sentences that actually mention an
    assay.
    """
    body = text or ""
    lowered = body.lower()
    for heading in METHODS_HEADINGS:
        start = lowered.find(f"\n{heading}")
        if start == -1:
            start = lowered.find(heading)
        if start != -1:
            return body[start:start + limit]

    sentences = [line.strip() for line in body.splitlines()
                 if any(hint in line.lower() for hint in ASSAY_HINTS)]
    return "\n".join(sentences)[:limit]


def label_page(provider, page, digitization, model: str, page_text: str = "",
               doc_id: str = "", max_tokens: int = 4000,
               methods_text: str = "") -> tuple:
    """Ask the model to name the panels, groups and series of one page.

    Returns the labels and the call's token usage, so a stage whose cost scales
    with the number of *figures* rather than the number of points is still
    visible in the per-document accounting.
    """
    if not digitization.panels:
        return empty_labels(), {}
    skeleton = build_skeleton(page, digitization)
    completion = provider.complete(
        build_label_messages(skeleton, page_text, methods_text),
        model=model, temperature=0.0, max_tokens=max_tokens,
        json_schema=LABEL_JSON_SCHEMA,
        meta={"stage": "chartlabel", "doc_id": doc_id},
    )
    return normalize_labels(extract_json_object(completion.text)), completion.usage.to_dict()


def empty_labels() -> dict:
    return {"figure_label": None, "panels": [], "clusters": [], "series": []}


def normalize_labels(payload: dict) -> dict:
    labels = empty_labels()
    if not isinstance(payload, dict):
        return labels
    if isinstance(payload.get("figure_label"), str):
        labels["figure_label"] = payload["figure_label"]
    for key in ("panels", "clusters", "series"):
        value = payload.get(key)
        if isinstance(value, list):
            labels[key] = [v for v in value if isinstance(v, dict)]
    return labels


def index_labels(labels: dict) -> tuple:
    """Turn the label lists into lookups keyed the way records need them."""
    panels = {int(p["index"]): p for p in labels.get("panels", [])
              if str(p.get("index", "")).lstrip("-").isdigit()}
    clusters = {}
    for entry in labels.get("clusters", []):
        try:
            clusters[(int(entry["panel"]), int(entry["cluster"]))] = entry
        except (KeyError, TypeError, ValueError):
            continue
    series = {str(s.get("colour")): s for s in labels.get("series", [])}
    return panels, clusters, series


# --------------------------------------------------------------------------
# Records — one row per plotted point
# --------------------------------------------------------------------------
#: Recovered values are reported to this many significant figures. The geometry
#: is more precise than that, but a marker's centre is not exactly its datum and
#: no journal reports a titer to seven figures, so the extra digits would be
#: false confidence rather than information.
SIGNIFICANT_FIGURES = 4

#: Marks how a record earned its ``_grounded`` flag. A text record is grounded
#: by a verbatim quote; a digitised point is grounded by a calibrated axis. Both
#: are checkable, neither is the other, and conflating them would let one
#: channel's guarantee be claimed by the other.
GEOMETRY_GROUNDING = "geometry"


def round_significant(value: float, digits: int = SIGNIFICANT_FIGURES) -> float:
    if not value or not math.isfinite(value):
        return value
    magnitude = math.floor(math.log10(abs(value)))
    return round(value, -(magnitude - digits + 1))


def axis_evidence(axis) -> str:
    """What stands in for a quotation when the evidence is geometry.

    A text record quotes the sentence its number came from. A digitised point
    has no sentence, and inventing one would be a lie in the field a reviewer
    trusts most. What it has instead is its calibration, which is quotable in
    the same sense: the ticks are printed on the page and their positions can
    be measured again by anyone holding the PDF.
    """
    ticks = ", ".join(f"{t.value:g}@y={t.pos:.1f}" for t in axis.ticks)
    return (f"{axis.kind} axis calibrated on ticks [{ticks}] "
            f"residual {axis.residual_pt:.3f} pt")


def chart_records(digitization, labels: dict, doc_id: str = "",
                  source_path: str = "") -> list:
    """One record per digitised marker, carrying where it came from.

    Every row states its channel, its page, its panel, its cluster and its
    series colour, so the corpus can be split by provenance after the fact —
    geometry from vision, one figure from another — without re-reading a PDF.
    """
    panel_labels, cluster_labels, series_labels = index_labels(labels)
    figure_label = labels.get("figure_label") or ""
    records = []

    for panel in digitization.panels:
        panel_label = panel_labels.get(panel.index, {})
        evidence = axis_evidence(panel.axis)
        for order, point in enumerate(panel.points):
            cluster_label = cluster_labels.get((panel.index, point.cluster), {})
            series_label = series_labels.get(point.colour, {})
            records.append({
                "doc_id": doc_id,
                "assay": panel_label.get("assay") or "na",
                "assay_platform": panel_label.get("assay_platform") or "na",
                "assay_platform_detail": panel_label.get("assay_platform_detail"),
                "assay_standard": panel_label.get("assay_standard"),
                "endpoint": panel_label.get("endpoint"),
                "serotype": panel_label.get("serotype"),
                "species": panel_label.get("species"),
                "group_label": cluster_label.get("group_label")
                or series_label.get("label"),
                "timepoint": cluster_label.get("timepoint")
                or series_label.get("timepoint"),
                "value": round_significant(point.value),
                "value_unit": panel_label.get("unit"),
                "value_kind": "individual",
                "value_source": "figure",
                "extraction_mode": "vector_geometry",
                "figure": figure_label or panel_label.get("title") or "",
                "n_subjects": 1,
                "source_span": evidence,
                "geometry_ref": geometry_ref(digitization.page_number, panel.index,
                                             point.cluster, point.colour, order),
                "geometry_residual_pt": round(panel.axis.residual_pt, 4),
                "geometry_basis": panel.axis.calibrated_on,
                "source_path": source_path,
                "_grounded": True,
                "_grounded_by": GEOMETRY_GROUNDING,
                "_value_grounded": True,
                "_unit_grounded": bool(panel_label.get("unit")),
                "_ungrounded": [] if panel_label.get("unit") else ["value_unit"],
            })
    return records


def geometry_ref(page_number: int, panel: int, cluster: int, colour: str,
                 order: int) -> str:
    """A stable address for one plotted point, readable without the code."""
    return f"p{page_number}:panel{panel}:cluster{cluster}:{colour or 'nofill'}:#{order}"
