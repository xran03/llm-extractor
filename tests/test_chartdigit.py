"""Vector figure digitisation: measurement, refusal, labelling and audit.

The measurement tests build a PDF whose plotted values are known exactly, so
recovery can be asserted against ground truth rather than against a previous
run of the same code. The refusal tests matter just as much: this channel is
only safe because it declines the figures it cannot calibrate, and a silent
wrong answer is the failure it exists to prevent.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from llm_extractor.chartaudit import (check_agreement, geometric_mean,
                                      render_overlay, summarise_panel)
from llm_extractor.chartdigit import (MIN_POINTS, Axis, Tick, calibrate,
                                      digitize_page, find_tick_labels,
                                      parse_tick_value)
from llm_extractor.chartlabel import (axis_evidence, build_label_messages,
                                      build_skeleton, chart_records,
                                      empty_labels, round_significant)

try:
    import pymupdf
except ImportError:                            # pragma: no cover - optional dep
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

requires_pymupdf = unittest.skipIf(pymupdf is None, "PyMuPDF is not installed")

WIDTH, HEIGHT = 420, 320
PLOT_LEFT, PLOT_RIGHT = 80, 400
PLOT_TOP, PLOT_BOTTOM = 40, 260
DECADES = 4                                     # y axis 1 .. 10^4


def y_for(value: float) -> float:
    return PLOT_BOTTOM - (math.log10(value) / DECADES) * (PLOT_BOTTOM - PLOT_TOP)


def build_dot_plot(path: Path, titers, group_x=(150, 250, 350),
                   superscript: bool = False) -> list:
    """Draw a log-scale dot plot and return the values actually plotted."""
    document = pymupdf.open()
    page = document.new_page(width=WIDTH, height=HEIGHT)
    page.draw_line((PLOT_LEFT, PLOT_TOP), (PLOT_LEFT, PLOT_BOTTOM))
    page.draw_line((PLOT_LEFT, PLOT_BOTTOM), (PLOT_RIGHT, PLOT_BOTTOM))

    for decade in range(DECADES + 1):
        value = 10.0 ** decade
        y = y_for(value)
        page.draw_line((PLOT_LEFT - 4, y), (PLOT_LEFT, y))
        if superscript:
            page.insert_text((PLOT_LEFT - 30, y + 3), "10", fontsize=7)
            page.insert_text((PLOT_LEFT - 20, y - 1), str(decade), fontsize=4)
        else:
            page.insert_text((PLOT_LEFT - 34, y + 3), f"{value:g}", fontsize=7)

    plotted = []
    for index, titer in enumerate(titers):
        x = group_x[index % len(group_x)] + (index % 7) - 3
        page.draw_circle((x, y_for(titer)), 2, color=None, fill=(0, 0, 0))
        plotted.append(titer)
    document.save(str(path))
    document.close()
    return plotted


def sample_titers(count: int = 45) -> list:
    """Spread values over the axis without relying on a random generator."""
    return [10 ** (0.2 + 3.4 * (i / (count - 1))) for i in range(count)]


@requires_pymupdf
class CalibrationBasisTest(unittest.TestCase):
    """The bias that survives every self-consistency check.

    A tick label's bounding box is centred on its glyphs, not on the tick. The
    offset is identical for every tick, so a fit to the labels has a near-zero
    residual and is uniformly wrong — and on a small multiple, whose axis may
    be forty points tall, "uniformly wrong" is tens of percent.
    """

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.pdf = self.dir / "dots.pdf"
        self.truth = build_dot_plot(self.pdf, sample_titers())
        self.page = pymupdf.open(str(self.pdf)).load_page(0)

    def test_calibration_prefers_the_tick_marks(self):
        result = digitize_page(self.page, page_number=1)
        self.assertEqual(result.panels[0].axis.calibrated_on, "tick marks")

    def test_ticks_land_on_the_drawn_positions(self):
        result = digitize_page(self.page, page_number=1)
        for tick in result.panels[0].axis.ticks:
            self.assertAlmostEqual(tick.pos, y_for(tick.value), places=2)

    def test_a_dense_field_of_markers_cannot_capture_a_tick(self):
        """A data point may lie nearer a tick than the tick's own mark does."""
        crowded = self.dir / "crowded.pdf"
        build_dot_plot(crowded, sample_titers(300))
        page = pymupdf.open(str(crowded)).load_page(0)
        result = digitize_page(page, page_number=1)
        self.assertEqual(result.panels[0].axis.calibrated_on, "tick marks")
        for tick in result.panels[0].axis.ticks:
            self.assertAlmostEqual(tick.pos, y_for(tick.value), places=2)


@requires_pymupdf
class MeasurementTest(unittest.TestCase):
    """What the channel claims: the values are read, not estimated."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.pdf = self.dir / "dots.pdf"
        self.truth = build_dot_plot(self.pdf, sample_titers())
        self.page = pymupdf.open(str(self.pdf)).load_page(0)
        self.result = digitize_page(self.page, page_number=1)

    def test_every_plotted_point_is_recovered(self):
        self.assertTrue(self.result.ok, self.result.rejected)
        self.assertEqual(len(self.result.points), len(self.truth))

    def test_recovered_values_match_the_plotted_ones(self):
        recovered = sorted(p.value for p in self.result.points)
        for got, want in zip(recovered, sorted(self.truth)):
            self.assertAlmostEqual(got / want, 1.0, places=3)

    def test_axis_is_recognised_as_logarithmic(self):
        self.assertEqual(self.result.panels[0].axis.kind, "log")

    def test_calibration_residual_is_negligible(self):
        self.assertLess(self.result.panels[0].axis.residual_pt, 0.1)

    def test_groups_are_separated_into_clusters(self):
        self.assertEqual(len(self.result.panels[0].clusters), 3)

    def test_geometric_mean_of_recovery_matches_the_truth(self):
        recovered = geometric_mean([p.value for p in self.result.points])
        self.assertAlmostEqual(recovered / geometric_mean(self.truth), 1.0, places=3)


@requires_pymupdf
class SuperscriptAxisTest(unittest.TestCase):
    """``10²`` is two words in a PDF, and reading them apart invents an axis."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.pdf = self.dir / "superscript.pdf"
        self.truth = build_dot_plot(self.pdf, sample_titers(), superscript=True)
        self.page = pymupdf.open(str(self.pdf)).load_page(0)

    def test_exponent_is_recombined_with_its_base(self):
        values = sorted({l["value"] for l in find_tick_labels(self.page)})
        self.assertIn(10000.0, values)
        self.assertNotIn(4.0, values)

    def test_values_are_recovered_from_a_superscript_axis(self):
        result = digitize_page(self.page, page_number=1)
        self.assertTrue(result.ok, result.rejected)
        recovered = sorted(p.value for p in result.points)
        for got, want in zip(recovered, sorted(self.truth)):
            self.assertAlmostEqual(got / want, 1.0, places=2)


@requires_pymupdf
class RefusalTest(unittest.TestCase):
    """Refusing is the feature: a wrong scale is worse than no reading."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def _page(self, build) -> object:
        path = self.dir / "page.pdf"
        document = pymupdf.open()
        page = document.new_page(width=WIDTH, height=HEIGHT)
        build(page)
        document.save(str(path))
        document.close()
        return pymupdf.open(str(path)).load_page(0)

    def test_a_page_without_an_axis_is_refused(self):
        def build(page):
            for index in range(20):
                page.draw_circle((100 + index * 8, 100), 2, color=None, fill=(0, 0, 0))
        result = digitize_page(self._page(build), page_number=1)
        self.assertFalse(result.ok)
        self.assertIn("numeric labels", result.rejected)

    def test_prose_numbers_alone_do_not_become_an_axis(self):
        def build(page):
            page.insert_text((60, 80), "In 2019 we enrolled 245 infants and 87 adults.",
                             fontsize=9)
            for index in range(20):
                page.draw_circle((100 + index * 8, 150), 2, color=None, fill=(0, 0, 0))
        result = digitize_page(self._page(build), page_number=1)
        self.assertFalse(result.ok)

    def test_a_legend_is_not_a_distribution(self):
        """A bar chart's key is small, square, uniform and inside the axis."""
        def build(page):
            for decade in range(5):
                y = y_for(10.0 ** decade)
                page.insert_text((PLOT_LEFT - 34, y + 3), f"{10 ** decade:g}", fontsize=7)
            for index in range(5):
                page.draw_rect(pymupdf.Rect(200, 60 + index * 12, 204, 64 + index * 12),
                               color=None, fill=(0, 0, 0))
                page.insert_text((208, 64 + index * 12), f"series {index}", fontsize=7)
        result = digitize_page(self._page(build), page_number=1)
        self.assertFalse(result.ok)

    def test_too_few_markers_is_refused(self):
        path = self.dir / "sparse.pdf"
        build_dot_plot(path, sample_titers(MIN_POINTS - 1))
        page = pymupdf.open(str(path)).load_page(0)
        self.assertFalse(digitize_page(page, page_number=1).ok)


class CalibrationTest(unittest.TestCase):
    """The overdetermined fit is what makes a bad axis visible."""

    @staticmethod
    def _labels(pairs):
        return [{"value": v, "x_right": 50.0, "x_centre": 45.0, "y_centre": y}
                for v, y in pairs]

    def test_a_clean_log_axis_fits(self):
        axis = calibrate(self._labels([(1, 200), (10, 150), (100, 100), (1000, 50)]))
        self.assertIsNotNone(axis)
        self.assertEqual(axis.kind, "log")
        self.assertLess(axis.residual_pt, 0.01)

    def test_a_clean_linear_axis_fits(self):
        axis = calibrate(self._labels([(0, 200), (5, 150), (10, 100), (15, 50)]))
        self.assertEqual(axis.kind, "linear")

    def test_an_intruder_does_not_break_a_good_axis(self):
        """A panel letter in the tick column must not cost the whole axis."""
        axis = calibrate(self._labels(
            [(1, 200), (10, 150), (100, 100), (1000, 50), (7, 125)]))
        self.assertIsNotNone(axis)
        self.assertEqual(len(axis.ticks), 4)

    def test_non_monotonic_labels_are_not_an_axis(self):
        """Two stacked panels share a tick column; their labels repeat."""
        axis = calibrate(self._labels(
            [(1, 200), (10, 150), (1, 100), (10, 50)]), tolerance=0.5)
        if axis is not None:
            values = [t.value for t in axis.ticks]
            self.assertEqual(values, sorted(values, reverse=True))

    def test_too_few_ticks_cannot_be_overdetermined(self):
        self.assertIsNone(calibrate(self._labels([(1, 200), (10, 150)])))

    def test_value_at_inverts_the_axis(self):
        axis = calibrate(self._labels([(1, 200), (10, 150), (100, 100), (1000, 50)]))
        self.assertAlmostEqual(axis.value_at(150), 10.0, places=3)
        self.assertAlmostEqual(axis.value_at(100), 100.0, places=3)


class TickLabelTest(unittest.TestCase):
    def test_plain_numbers_parse(self):
        self.assertEqual(parse_tick_value("0.5"), 0.5)
        self.assertEqual(parse_tick_value("1000"), 1000.0)

    def test_a_year_sized_exponent_is_not_an_axis(self):
        self.assertIsNone(parse_tick_value("10 2024"))

    def test_words_are_not_values(self):
        self.assertIsNone(parse_tick_value("serotype"))
        self.assertIsNone(parse_tick_value(""))


class LabellingBoundaryTest(unittest.TestCase):
    """The model names things; it must have nowhere to put a number."""

    def test_the_label_schema_admits_no_measurement(self):
        from llm_extractor.chartlabel import LABEL_JSON_SCHEMA

        types = set()

        def walk(node):
            if isinstance(node, dict):
                declared = node.get("type")
                for item in ([declared] if isinstance(declared, str) else declared or []):
                    types.add(item)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(LABEL_JSON_SCHEMA["schema"]["properties"]["panels"])
        walk(LABEL_JSON_SCHEMA["schema"]["properties"]["clusters"])
        self.assertNotIn("number", types)

    def test_the_prompt_carries_structure_but_no_values(self):
        axis = Axis(kind="log", intercept=0.0, slope=1.0,
                    ticks=[Tick(1.0, 200.0), Tick(10.0, 150.0), Tick(100.0, 100.0)],
                    x_right=50.0)
        panel = _panel(axis, values=[3.14159, 2.71828, 1.41421])
        skeleton = {"page": 1, "panels": [{
            "index": 0, "axis": {"scale": "log", "ticks": [1, 10, 100]},
            "clusters": [{"cluster": 0, "x": 100.0, "markers": 3}],
            "series": [{"colour": "#000000", "markers": 3}],
            "text_nearby": "Serotype 6B",
        }]}
        content = build_label_messages(skeleton, "some prose")[1]["content"]
        self.assertIn("Serotype 6B", content)
        self.assertNotIn("3.14159", content)
        self.assertNotIn("2.71828", content)


class RecordTest(unittest.TestCase):
    """One point, one row, and the row says where it came from."""

    def setUp(self):
        axis = Axis(kind="log", intercept=0.0, slope=1.0,
                    ticks=[Tick(1.0, 200.0), Tick(10.0, 150.0), Tick(100.0, 100.0)],
                    residual_pt=0.004, x_right=50.0)
        self.digitization = _digitization(_panel(axis, [12.5, 40.0, 3.0]))
        self.labels = {
            "figure_label": "Figure 2A",
            "panels": [{"index": 0, "title": "Serotype 6B", "serotype": "6B",
                        "assay": "opa", "endpoint": "OPA titer",
                        "unit": "OPA titer", "species": "human infant"}],
            "clusters": [{"panel": 0, "cluster": 0, "group_label": "PCV13",
                          "timepoint": "post-dose 3"}],
            "series": [{"colour": "#000000", "label": "PCV13", "timepoint": None}],
        }
        self.records = chart_records(self.digitization, self.labels, doc_id="d1")

    def test_one_record_per_plotted_point(self):
        self.assertEqual(len(self.records), 3)

    def test_the_channel_is_named_in_every_record(self):
        self.assertTrue(all(r["extraction_mode"] == "vector_geometry"
                            for r in self.records))
        self.assertTrue(all(r["value_kind"] == "individual" for r in self.records))
        self.assertTrue(all(r["value_source"] == "figure" for r in self.records))

    def test_each_point_is_individually_addressable(self):
        refs = [r["geometry_ref"] for r in self.records]
        self.assertEqual(len(set(refs)), len(refs))
        self.assertTrue(all(ref.startswith("p1:panel0:") for ref in refs))

    def test_semantics_come_from_the_labels(self):
        self.assertEqual(self.records[0]["serotype"], "6B")
        self.assertEqual(self.records[0]["assay"], "opa")
        self.assertEqual(self.records[0]["group_label"], "PCV13")

    def test_values_come_from_geometry_not_from_the_labels(self):
        self.assertEqual(sorted(r["value"] for r in self.records), [3.0, 12.5, 40.0])

    def test_the_evidence_is_the_calibration_not_a_quotation(self):
        span = self.records[0]["source_span"]
        self.assertIn("log axis calibrated on ticks", span)
        self.assertIn("residual", span)

    def test_grounding_declares_its_own_basis(self):
        self.assertEqual(self.records[0]["_grounded_by"], "geometry")
        self.assertEqual(self.records[0]["geometry_residual_pt"], 0.004)

    def test_unlabelled_units_are_reported_as_ungrounded(self):
        records = chart_records(self.digitization, empty_labels(), doc_id="d1")
        self.assertEqual(records[0]["_ungrounded"], ["value_unit"])
        self.assertEqual(records[0]["assay"], "na")

    def test_rounding_keeps_significant_figures_not_decimals(self):
        self.assertEqual(round_significant(0.000123456), 0.0001235)
        self.assertEqual(round_significant(123456.0), 123500.0)


class AgreementTest(unittest.TestCase):
    """The cross-channel check: independent source, so real evidence."""

    def setUp(self):
        axis = Axis(kind="log", intercept=0.0, slope=1.0,
                    ticks=[Tick(1.0, 200.0), Tick(100.0, 100.0)], x_right=50.0)
        self.digitization = _digitization(_panel(axis, [10.0, 100.0, 1000.0]))

    def test_a_matching_reported_mean_agrees(self):
        checks = check_agreement(self.digitization, [{"value": 100.0, "label": "GMT"}])
        self.assertEqual(checks[0].verdict, "agrees")

    def test_a_distant_reported_mean_disagrees(self):
        checks = check_agreement(self.digitization, [{"value": 4.0, "label": "GMT"}])
        self.assertEqual(checks[0].verdict, "disagrees")

    def test_nothing_to_compare_against_is_not_a_pass(self):
        checks = check_agreement(self.digitization, [])
        self.assertEqual(checks[0].verdict, "unchecked")

    def test_the_recovered_mean_is_geometric(self):
        summary = summarise_panel(self.digitization.panels[0])
        self.assertAlmostEqual(summary["#000000"]["gmt"], 100.0, places=6)


class AgreementReportingTest(unittest.TestCase):
    """"Checked and agreed" and "never checked" must not look the same.

    The chart stage runs before text extraction, so the comparison cannot
    happen inside it. Reporting zero disagreements before anything has been
    compared would claim evidence that was never gathered.
    """

    def setUp(self):
        from llm_extractor.chartstage import ChartResult

        axis = Axis(kind="log", intercept=0.0, slope=1.0,
                    ticks=[Tick(1.0, 200.0), Tick(100.0, 100.0)], x_right=50.0)
        self.result = ChartResult()
        self.result.digitizations = [_digitization(_panel(axis, [10.0, 100.0, 1000.0]))]

    def test_an_uncompared_document_reports_unknown_not_zero(self):
        summary = self.result.summary()
        self.assertIsNone(summary["chart_disagreements"])
        self.assertIsNone(summary["chart_agreement_checks"])

    def test_comparing_against_agreeing_text_records_reports_zero(self):
        from llm_extractor.chartstage import audit_against_text

        audit_against_text(self.result, [{"value": 100.0, "serotype": "6B"}])
        summary = self.result.summary()
        self.assertEqual(summary["chart_disagreements"], 0)
        self.assertGreater(summary["chart_agreement_checks"], 0)

    def test_a_conflicting_text_record_is_reported(self):
        from llm_extractor.chartstage import audit_against_text

        audit_against_text(self.result, [{"value": 3.0, "serotype": "6B"}])
        self.assertEqual(self.result.summary()["chart_disagreements"], 1)

    def test_text_records_without_numbers_leave_it_uncompared(self):
        from llm_extractor.chartstage import audit_against_text

        audit_against_text(self.result, [{"value": None, "serotype": "6B"}])
        self.assertIsNone(self.result.summary()["chart_disagreements"])


class OcrPolicyPrecedenceTest(unittest.TestCase):
    """An explicit ``--ocr always`` outranks a defaulted chart policy."""

    def _document(self):
        from llm_extractor.ingest import Document
        from llm_extractor import formats

        document = Document(doc_id="d", text="x" * 500, fmt=formats.FORMATS["pdf"])
        document.figures = [Path(f"doc-p{n:03d}.png") for n in (1, 2, 3)]
        return document

    def test_pages_are_withdrawn_from_vision_under_auto(self):
        from llm_extractor.pipeline import _page_of

        document = self._document()
        measured = {1, 2}
        remaining = [f for f in document.figures if _page_of(f) not in measured]
        self.assertEqual([_page_of(f) for f in remaining], [3])

    def test_page_numbers_survive_zero_padding(self):
        from llm_extractor.pipeline import _page_of

        self.assertEqual(_page_of(Path("paper-2024-p007.png")), 7)
        self.assertEqual(_page_of(Path("no-page-marker.png")), 0)


class TokenAccountingTest(unittest.TestCase):
    """A stage that calls a model has to appear in the bill."""

    def test_labelling_usage_is_reported(self):
        from llm_extractor.chartstage import ChartResult

        result = ChartResult()
        result.prompt_tokens, result.completion_tokens = 1200, 340
        summary = result.summary()
        self.assertEqual(summary["chart_prompt_tokens"], 1200)
        self.assertEqual(summary["chart_completion_tokens"], 340)

    def test_label_page_returns_its_usage(self):
        from llm_extractor.chartlabel import label_page

        class _Provider:
            def complete(self, *_args, **_kwargs):
                from llm_extractor.providers.base import Completion, Usage

                return Completion(text='{"figure_label": "Figure 1", "panels": [],'
                                       ' "clusters": [], "series": []}',
                                  usage=Usage(prompt_tokens=90, completion_tokens=12))

        axis = Axis(kind="log", intercept=0.0, slope=1.0,
                    ticks=[Tick(1.0, 200.0), Tick(100.0, 100.0)], x_right=50.0)
        digitization = _digitization(_panel(axis, [10.0, 100.0]))

        class _Page:
            def get_text(self, *_args):
                return []

        labels, usage = label_page(_Provider(), _Page(), digitization, model="m")
        self.assertEqual(labels["figure_label"], "Figure 1")
        self.assertEqual(usage["prompt_tokens"], 90)


@requires_pymupdf
class OverlayTest(unittest.TestCase):
    """The overlay is checked against the page image, so it is not circular."""

    def test_an_overlay_image_is_written(self):
        directory = Path(tempfile.mkdtemp())
        pdf = directory / "dots.pdf"
        build_dot_plot(pdf, sample_titers())
        page = pymupdf.open(str(pdf)).load_page(0)
        result = digitize_page(page, page_number=1)
        out = directory / "overlay.png"
        self.assertEqual(render_overlay(page, result, out), str(out))
        self.assertGreater(out.stat().st_size, 0)


def _panel(axis, values):
    from llm_extractor.chartdigit import Panel, Point

    points = [Point(x=100.0 + index, y=axis.ticks[0].pos, value=value, size=2.0,
                    colour="#000000", cluster=0, panel=0)
              for index, value in enumerate(values)]
    return Panel(index=0, axis=axis, points=points, clusters=[100.0])


def _digitization(panel):
    from llm_extractor.chartdigit import Digitization

    return Digitization(page_number=1, panels=[panel])


if __name__ == "__main__":                      # pragma: no cover
    unittest.main()
