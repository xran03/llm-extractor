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


class StubProvider:
    """Returns fixed, hand-verified answers instead of calling a model."""

    name = "stub"
    API_STYLE = "stub"

    def list_models(self):
        return ["stub-model"]

    def complete(self, messages, model, temperature=0.0, max_tokens=None,
                 json_schema=None, meta=None, **kwargs):
        stage = (meta or {}).get("stage", "extract")
        if stage == "ocr":
            payload = json.dumps(FIGURE_OCR)
        elif stage == "aggregate":
            payload = json.dumps(AGGREGATE)
        elif (meta or {}).get("doc_id", "").startswith("naca-figure"):
            payload = json.dumps({"records": []})
        else:
            payload = json.dumps({"records": PDF_RECORDS})
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


def main() -> int:
    results = DEMO / "results"
    if results.exists():
        shutil.rmtree(results)

    runner.build_provider = lambda settings, **kwargs: StubProvider()

    settings = Settings(
        api="stub", base_url="https://stub", api_key="stub",
        model="stub-model", ocr_model="stub-vision", agent_model="stub-mini",
        cache_dir=str(DEMO / ".cache"), cache_enabled=False,
        template="generic", ocr="always", max_workers=1,
    )
    summary = run_job(
        settings, source_name="folder",
        source_params={"input_dir": str(DEMO), "extensions": [".pdf", ".png"]},
        out_dir=str(results), resume=False, job_id="demo",
    )
    shutil.rmtree(DEMO / ".cache", ignore_errors=True)
    stabilise_results(results)

    print(f"documents {summary.ok}/{summary.total}   records {summary.records}   "
          f"figures {summary.figures}")
    for path in sorted(results.iterdir()):
        print(f"  {path.name}")
    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
