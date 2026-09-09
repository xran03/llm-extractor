"""Regenerate `demo/results/` without calling a real model.

The demo ships reference output so you can see the artifact shapes before
spending a token. That output is produced by running the **real pipeline** —
real ingest, real schema coercion, real grounding checks, real CSV writing —
against a stub that returns fixed answers instead of a model.

Two consequences worth knowing:

* the numbers below were read out of the source document by hand, so the
  reference output is factually correct about NACA Report 1372;
* grounding is still computed for real, so if a quoted span were not actually
  present in the PDF, `_grounded` would come back false. Nothing here is
  asserted by the pipeline that the source does not support.

Run:  python demo/_make_reference_output.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO.parent / "src"))

from llm_extractor import runner  # noqa: E402
from llm_extractor.providers.base import Completion, Usage  # noqa: E402
from llm_extractor.runner import run_job  # noqa: E402
from llm_extractor.settings import Settings  # noqa: E402

# --- facts taken verbatim from the excerpt's text layer ---------------------
PDF_RECORDS = [
    {
        "subject": "example 1(b) copper wall",
        "attribute": "wall thickness",
        "value": 3, "unit": "inches",
        "qualifier": "example 1(b)",
        "value_source": "text",
        "notes": "Same conditions as example 1(a) but a thicker wall.",
        "source_span": "the copper wall is 3 inches thick, or 1=% foot",
    },
    {
        "subject": "example 2 heating history",
        "attribute": "adiabatic-wall temperature swing",
        "value": 5000, "unit": "degrees",
        "qualifier": "over 10 seconds",
        "value_source": "text",
        "notes": "Temperature rises and falls across this range during the run.",
        "source_span": "rising and falling over 5,000",
    },
    {
        "subject": "example 3 adiabatic-wall temperature",
        "attribute": "first value of the assigned time series",
        "value": 1365, "unit": "degrees",
        "qualifier": "0.5-second intervals",
        "value_source": "text",
        "notes": "Start of the assigned Taw series used to drive the solution.",
        "source_span": "following time series: Tam=1,365",
    },
    {
        "subject": "example 3 wall",
        "attribute": "wall thickness",
        "value": 3, "unit": "inches",
        "qualifier": "example 3",
        "value_source": "text",
        "notes": "Example 3 repeats example 2 with a thicker wall.",
        "source_span": "that the wall is 3 inches thick",
    },
    {
        "subject": "example 3 computing interval",
        "attribute": "time step used for the thick-wall solution",
        "value": None, "value_text": "one half second", "unit": "seconds",
        "value_source": "text",
        "notes": "Written as a fraction in the source, so no digits appear in the text layer.",
        "source_span": "Solution (a) (thick-wall solution)",
    },
]

# --- values that exist only in the chart, not in the text layer -------------
FIGURE_OCR = {
    "figure_type": "chart",
    "caption": "FIGURE 2.-Example 2. Temperatures of 1/2-inch copper wall heated "
               "according to assigned history of h and Taw.",
    "axis_x": "Time, sec",
    "axis_y": "Wall surface temperature, deg F",
    "items": [
        {"label": "outer surface at t=2 s", "series": "Outer surface (present method)",
         "value": 40, "value_text": None, "unit": "deg F", "note": "read from the curve"},
        {"label": "outer surface at t=4 s", "series": "Outer surface (present method)",
         "value": 120, "value_text": None, "unit": "deg F", "note": "read from the curve"},
        {"label": "outer surface at t=6 s", "series": "Outer surface (present method)",
         "value": 217, "value_text": None, "unit": "deg F", "note": "read from the curve"},
        {"label": "outer surface at t=10 s", "series": "Outer surface (present method)",
         "value": 300, "value_text": None, "unit": "deg F", "note": "curve plateau"},
        {"label": "inner surface at t=6 s", "series": "Inner surface (present method)",
         "value": 185, "value_text": None, "unit": "deg F", "note": "read from the curve"},
        {"label": "inner surface at t=10 s", "series": "Inner surface (present method)",
         "value": 297, "value_text": None, "unit": "deg F",
         "note": "converges with the outer surface"},
    ],
    "tables": [],
    "text_blocks": [
        "Outer surface | Present method; d=1/2 sec",
        "Inner surface | Present method; d=1/2 sec",
        "Outer surface | Exact theory",
        "Inner surface | Exact theory",
        "Time, sec",
        "Wall surface temperature, deg F",
    ],
    "notes": "Axis labels and the caption appear only in the image; the PDF text "
             "layer does not contain them.",
}

AGGREGATE = {
    "summary": "NACA Report 1372 presents a method for computing transient "
               "temperatures of thick walls from an arbitrary history of "
               "adiabatic-wall temperature and heat-transfer coefficient. The "
               "excerpt works through examples with 1/2-inch and 3-inch copper "
               "walls and compares the method against exact theory.",
    "key_findings": [
        "Example 1(b) repeats example 1(a) with a 3-inch copper wall.",
        "The assigned adiabatic-wall temperature rises and falls over 5,000 degrees in 10 seconds.",
        "Example 3 drives the solution with a tabulated Taw series starting at 1,365 degrees.",
    ],
    "figure_insights": [
        {"image": "naca-figure-2.png",
         "finding": "Outer-surface temperature reaches about 300 deg F at 10 s",
         "value": 300, "unit": "deg F"},
        {"image": "naca-figure-2.png",
         "finding": "Inner and outer surface curves converge by 10 s",
         "value": 297, "unit": "deg F"},
    ],
    "conflicts": [],
    "coverage_gaps": [
        "The computing interval is written as a fraction, so no numeric value "
        "appears in the text layer.",
        "Axis labels and figure captions are absent from the text layer and were "
        "recovered only by the vision pass.",
    ],
}


# --- the scatter plot: numbers that are drawn, never written ---------------
#: What ``opa-scatter.png`` actually prints. It is generated by
#: ``_make_opa_scatter.py``, so this reading is correct by construction rather
#: than by someone reading values off a chart.
OPA_FIGURE_OCR = {
    "figure_type": "chart",
    "caption": "Figure 1. Serotype 6B OPA titers 28 days after vaccination.",
    "axis_x": "Vaccine group",
    "axis_y": "OPA titer (1/dilution)",
    "items": [
        {"label": "PCV13", "series": "Day 28", "value": 814,
         "value_text": "GMT 814", "unit": "OPA titer",
         "note": "geometric mean printed above the column"},
        {"label": "PCV20", "series": "Day 28", "value": 437,
         "value_text": "GMT 437", "unit": "OPA titer",
         "note": "geometric mean printed above the column"},
        {"label": "Placebo", "series": "Day 28", "value": 15,
         "value_text": "GMT 15", "unit": "OPA titer",
         "note": "geometric mean printed above the column"},
    ],
    "tables": [],
    "text_blocks": [
        "Figure 1. Serotype 6B OPA titers 28 days after vaccination.",
        "OPA titer (1/dilution)",
        "GMT 814", "GMT 437", "GMT 15",
        "PCV13", "PCV20", "Placebo",
        "n=34", "n=34", "n=34",
        "10000", "1000", "100", "10", "1",
    ],
    "notes": "Each group plots 34 individual subjects. Only the printed group "
             "means are reported: the individual points carry no labels, and a "
             "vision model cannot enumerate them reliably.",
}

OPA_AGGREGATE = {
    "summary": "A dot plot of serotype 6B OPA titers 28 days after vaccination "
               "in three arms of 34 subjects each. Both conjugate groups "
               "responded and the placebo group did not.",
    "key_findings": [
        "PCV13 reached a geometric mean OPA titer of 814.",
        "PCV20 reached a geometric mean OPA titer of 437.",
        "The placebo group stayed near the assay floor at 15.",
    ],
    "figure_insights": [
        {"image": "opa-scatter.png", "finding": "PCV13 group geometric mean",
         "value": 814, "unit": "OPA titer"},
        {"image": "opa-scatter.png", "finding": "PCV20 group geometric mean",
         "value": 437, "unit": "OPA titer"},
        {"image": "opa-scatter.png", "finding": "Placebo group geometric mean",
         "value": 15, "unit": "OPA titer"},
    ],
    "conflicts": [],
    "coverage_gaps": [
        "The 102 individual subject titers are plotted but never written, so "
        "only the three printed group means were recovered.",
    ],
}


# --- a real published figure: what a live model actually returned ----------
#: Unlike every other payload here, this one was **not** written by hand. It is
#: the verbatim reply of a live vision model (gpt-4.1) on
#: ``h5-titre-histogram-scatter.jpg``, recorded once and replayed, so the demo
#: can show real performance on a real published figure without costing a token
#: to look at.
#:
#: Two things in it are worth reading closely. The model transcribed "Mean RMSE
#: (detectable titers)" as "Mean RMSE (recodable titres)"; the misreading is
#: kept, because editing it out is how a demo starts flattering itself. And
#: ``items`` is empty — the panels were recognised and their labels read, but
#: not one data point was recovered from the histogram or either scatter plot.
H5_FIGURE_OCR = {
    "figure_type": "mixed",
    "caption": None,
    "axis_x": None,
    "axis_y": None,
    "items": [],
    "tables": [],
    "text_blocks": [
        "a", "Number of data points", "Log2 standard deviation",
        "b", "Mean RMSE (recodable titres)", "Dimension",
        "c", "R2", "Dimension",
        "d", "Total map stress", "Dimension",
        "e", "4D map distances", "3D map distances", "R2 = 0.93",
        "f", "4D map antigen stress", "3D map antigen stress", "R2 = 0.84",
    ],
    "notes": None,
}

#: The aggregation agent's verbatim reply on the same figure. It lifted the two
#: printed R² values out of the loose text blocks — which is the only place the
#: numbers this figure states ever reached.
H5_AGGREGATE = {
    "summary": "The document presents a mixed figure containing several panels "
               "labeled a–f, each apparently showing different statistical "
               "metrics related to map dimensions and antigen stress. The "
               "figure includes references to metrics such as Log2 standard "
               "deviation, Mean RMSE, R2 values, and map stress, comparing 3D "
               "and 4D map analyses. No textual records were extracted from "
               "the document's main text.",
    "key_findings": [
        "R2 = 0.93 for 4D vs 3D map distances (panel e).",
        "R2 = 0.84 for 4D vs 3D map antigen stress (panel f).",
    ],
    "figure_insights": [
        {"image": "h5-titre-histogram-scatter.jpg",
         "finding": "R2 value for 4D vs 3D map distances",
         "value": 0.93, "unit": None},
        {"image": "h5-titre-histogram-scatter.jpg",
         "finding": "R2 value for 4D vs 3D map antigen stress",
         "value": 0.84, "unit": None},
    ],
    "conflicts": [],
    "coverage_gaps": [
        "Exact numerical values for Log2 standard deviation, Mean RMSE, and "
        "Total map stress are referenced but not extracted.",
        "Underlying data points or sample sizes for each panel are not "
        "captured.",
        "No textual records are available to corroborate or expand on figure "
        "findings.",
    ],
}


# --- the vaccine corpus: OPA titres and a regulatory serotype list ----------
#: What the PCV13/EHPC paper reports for serotype 6B, read off the results
#: text. The control arm matters as much as the vaccine arm: a run that only
#: recovers the 7132 has not understood the experiment.
PCV13_OPA = [
    ("PCV13-vaccinated adults", "baseline", 154,
     "geometric mean OI for all PCV-vaccinees at baseline: 154 [95% CI 59–404]"),
    ("PCV13-vaccinated adults", "post-vaccination", 7132,
     "post- vaccination: 7132, [3268–15565]"),
    ("PCV13-vaccinated adults", "post-challenge", 3122,
     "geometric mean OI post-challenge: 3122 [1340–7273]"),
    ("HepA-vaccinated controls", "baseline", 293,
     "293 [40–2135]; post-vaccination: 382 [91–1598]; post-challenge: 164 [27–985]"),
    ("HepA-vaccinated controls", "post-vaccination", 382,
     "293 [40–2135]; post-vaccination: 382 [91–1598]; post-challenge: 164 [27–985]"),
    ("HepA-vaccinated controls", "post-challenge", 164,
     "293 [40–2135]; post-vaccination: 382 [91–1598]; post-challenge: 164 [27–985]"),
]

PCV13_RECORDS = [
    {
        "assay": "opa",
        "endpoint": "opsonic index",
        "group_label": f"{group}, {timepoint}",
        "value": value,
        "value_unit": "opsonic index",
        "value_kind": "geometric_mean",
        "value_source": "text",
        "serotype": "6B",
        "figure": "Fig. 2",
        "notes": "Multiplexed opsonophagocytic killing assay; 6B was the challenge serotype.",
        "source_span": span,
    }
    for group, timepoint, value, span in PCV13_OPA
]

#: The serotypes the FDA letter licenses, in the order the sentence lists them.
#: One record each, because the template asks for one atomic fact per field —
#: and because fifteen serotypes is the cheapest way to show `repeat_unit`
#: resolving a whole valency at once.
PCV15_SEROTYPES = ["1", "3", "4", "5", "6A", "6B", "7F", "9V", "14", "18C",
                   "19A", "19F", "22F", "23F", "33F"]

PCV15_SPAN = (
    "invasive disease caused by Streptococcus pneumoniae serotypes 1, 3, 4, 5, "
    "6A, 6B, 7F, 9V, 14, 18C, 19A, 19F, 22F, 23F and 33F in adults 18 years of "
    "age and older"
)

FDA_RECORDS = [
    {
        "assay": "na",
        "endpoint": "licensed indication",
        "factor_type": "serotype",
        "group_label": "Pneumococcal 15-valent Conjugate Vaccine",
        "value": None,
        "value_source": "text",
        "serotype": serotype,
        "notes": "Named in the approved indication for adults 18 years and older.",
        "source_span": PCV15_SPAN,
    }
    for serotype in PCV15_SEROTYPES
] + [
    # Valency gets its own record because the indication sentence never says
    # "15" — asserting it there would be a number its own evidence cannot show.
    {
        "assay": "na",
        "endpoint": "valency",
        "factor_type": "valency",
        "group_label": "Pneumococcal 15-valent Conjugate Vaccine",
        "value": None,
        "value_source": "text",
        "valency": 15,
        "notes": "Stated in the licensing paragraph.",
        "source_span": "We have approved your BLA for Pneumococcal 15-valent "
                       "Conjugate Vaccine effective this date.",
    },
]

VACCINE_AGGREGATE = {
    "summary": "Two documents about pneumococcal conjugate vaccines: a challenge "
               "study reporting serotype 6B opsonophagocytic killing in "
               "PCV13-vaccinated adults against a hepatitis A-vaccinated "
               "control arm, and the FDA letter licensing a 15-valent "
               "conjugate vaccine.",
    "key_findings": [
        "PCV13 raised the serotype 6B geometric mean opsonic index from 154 to 7132.",
        "The HepA control arm stayed between 164 and 382 throughout.",
        "The 15-valent vaccine is licensed for fifteen named serotypes in adults.",
    ],
    "figure_insights": [],
    "conflicts": [],
    "coverage_gaps": [
        "Individual subject opsonic indices are plotted per subject but never "
        "written, so only the group geometric means were recovered.",
    ],
}


class StubProvider:
    """Replays fixed answers instead of calling a model.

    Every payload but one was read out of the source by hand; ``H5_*`` is a
    recorded reply from a live model, kept verbatim.
    """

    name = "stub"
    API_STYLE = "stub"

    #: doc_id prefix -> the answer for each stage. A prefix that is not listed
    #: falls back to ``DEFAULT``, which is the NACA report.
    DOCUMENTS = {
        "naca-figure": {"extract": [], "ocr": FIGURE_OCR, "aggregate": AGGREGATE},
        "opa-scatter": {"extract": [], "ocr": OPA_FIGURE_OCR, "aggregate": OPA_AGGREGATE},
        "h5-titre": {"extract": [], "ocr": H5_FIGURE_OCR, "aggregate": H5_AGGREGATE},
        "pcv13-opa-colonisation": {"extract": PCV13_RECORDS, "ocr": FIGURE_OCR,
                                   "aggregate": VACCINE_AGGREGATE},
        "fda-pcv15-approval-letter": {"extract": FDA_RECORDS, "ocr": FIGURE_OCR,
                                      "aggregate": VACCINE_AGGREGATE},
    }
    DEFAULT = {"extract": PDF_RECORDS, "ocr": FIGURE_OCR, "aggregate": AGGREGATE}

    def list_models(self):
        return ["stub-model"]

    def _answers(self, doc_id: str) -> dict:
        for prefix, answers in self.DOCUMENTS.items():
            if doc_id.startswith(prefix):
                return answers
        return self.DEFAULT

    def complete(self, messages, model, temperature=0.0, max_tokens=None,
                 json_schema=None, meta=None, **kwargs):
        stage = (meta or {}).get("stage", "extract")
        answers = self._answers((meta or {}).get("doc_id", ""))
        if stage in ("ocr", "aggregate"):
            payload = json.dumps(answers[stage])
        else:
            payload = json.dumps({"records": answers["extract"]})
        return Completion(text=payload,
                          usage=Usage(prompt_tokens=0, completion_tokens=0))

    def complete_text(self, messages, model, **kwargs):
        return self.complete(messages, model, **kwargs).text


# --- keep the committed output machine-independent --------------------------
#: Fields whose value depends on when and where the run happened.
UNSTABLE_FIELDS = {"generated_at": 0, "duration_s": 0.0}
REPO_ROOT = DEMO.parent


def _relativise(value: str) -> str:
    """Rewrite any path pointing into this checkout to a repo-relative one."""
    root = str(REPO_ROOT)
    for prefix in (REPO_ROOT.as_uri() + "/", root + "\\", root + "/", root):
        if value.startswith(prefix):
            return value[len(prefix):].replace("\\", "/").lstrip("/")
    return value


def _stabilise(node):
    if isinstance(node, dict):
        return {key: (UNSTABLE_FIELDS[key] if key in UNSTABLE_FIELDS else _stabilise(value))
                for key, value in node.items()}
    if isinstance(node, list):
        return [_stabilise(item) for item in node]
    return _relativise(node) if isinstance(node, str) else node


def stabilise_results(results: Path) -> None:
    """Strip absolute paths and timings so the output regenerates identically.

    Without this the reference artifacts carry the author's home directory and
    a fresh timestamp, so every regeneration shows a diff and the committed
    output is not reproducible on another machine.
    """
    for path in sorted(results.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(_stabilise(data), ensure_ascii=False, indent=2),
                        encoding="utf-8")


#: The demo is two corpora, because the template decides what a record even is.
#: The general one asks "what does this document state"; the vaccine one asks
#: for assay, serotype and censoring, which is what makes `repeat_unit` resolve.
CORPORA = (
    {"name": "general", "template": "generic", "out": "results",
     "input": ".", "exclude": ["vaccine", "results"]},
    {"name": "vaccine", "template": "immunogenicity", "out": "results/vaccine",
     "input": "vaccine", "exclude": []},
)


def main() -> int:
    results = DEMO / "results"
    if results.exists():
        shutil.rmtree(results)

    runner.build_provider = lambda settings, **kwargs: StubProvider()

    total_ok = total_docs = total_records = 0
    for corpus in CORPORA:
        out_dir = DEMO / corpus["out"]
        settings = Settings(
            api="stub", base_url="https://stub", api_key="stub",
            model="stub-model", ocr_model="stub-vision", agent_model="stub-mini",
            cache_dir=str(DEMO / ".cache"), cache_enabled=False,
            template=corpus["template"], ocr="always", max_workers=1,
        )
        summary = run_job(
            settings, source_name="folder",
            source_params={"input_dir": str(DEMO / corpus["input"]),
                           "extensions": [".pdf", ".png", ".jpg"],
                           "exclude": corpus["exclude"]},
            out_dir=str(out_dir), resume=False, job_id=f"demo-{corpus['name']}",
        )
        stabilise_results(out_dir)
        total_ok += summary.ok
        total_docs += summary.total
        total_records += summary.records
        print(f"[{corpus['name']:<8}] {summary.ok}/{summary.total} documents  "
              f"{summary.records} records  {summary.figures} figures  "
              f"({corpus['template']})")

    shutil.rmtree(DEMO / ".cache", ignore_errors=True)
    print(f"\ndocuments {total_ok}/{total_docs}   records {total_records}")
    for path in sorted(results.rglob("*")):
        print(f"  {path.relative_to(results).as_posix()}")
    return 0 if total_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
