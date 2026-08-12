# Demo

Two real inputs and the output they produce, so you can see the artifact shapes
before spending a token.

| File | What it is |
|---|---|
| `naca-report-1372-excerpt.pdf` | 3-page excerpt of a 1958 scanned technical report |
| `naca-figure-2.png` | one chart cropped out of that report |
| `results/` | reference output for both inputs |

Both inputs come from **NACA Report 1372**, *A Method of Computing the Transient
Temperature of Thick Walls from Arbitrary Variation of Adiabatic-Wall
Temperature and Heat-Transfer Coefficient* (P. R. Hill, 1958), retrieved from
the [NASA Technical Reports Server](https://ntrs.nasa.gov/citations/19930091019).
It is a work of the U.S. Government and is **in the public domain**. The excerpt
is pages 887–889; the PNG is Figure 2 from page 888.

## Why this document

It is deliberately awkward, in the way real archives are:

- it is a **scan**, so the text layer is OCR-damaged (`Tawis O.`, `~J484,3J3~6`);
- axis labels and figure captions **exist only in the image** — search the text
  layer for "Wall surface temperature" and you get nothing;
- the numbers that matter are split between prose and charts.

That is exactly the case where a text-only extractor quietly under-reports, and
where the vision pass earns its cost.

## Run it

```bash
# from the repository root, with credentials configured (see ../.env.example)
llm-extract -i ./demo -o ./demo/out --api llmhub --ocr always
```

Or use the wrappers:

```bash
./demo/run.sh          # macOS / Linux
.\demo\run.ps1         # Windows
```

Then open `demo/out/records.csv`.

## What comes out

```
results/
  records.csv                              every record from every document
  figures.csv                              every value read out of a figure
  summary.json                             run totals, tokens, cache statistics
  <doc>.records.jsonl                      lossless per-document records
  <doc>.records.csv                        the same rows as a table
  <doc>.ocr.json                           structured vision output per figure
  <doc>.figures.csv                        figure readings as a table
  <doc>.document.json                      records + figures + aggregate + stats
```

`records.csv` from the PDF:

| attribute | value | unit | _grounded | _value_grounded |
|---|---|---|---|---|
| wall thickness | 3 | inches | true | true |
| adiabatic-wall temperature swing | 5000 | degrees | true | true |
| first value of the assigned time series | 1365 | degrees | true | true |
| wall thickness | 3 | inches | true | true |
| time step used for the thick-wall solution | | seconds | true | |

The last row is the interesting one: the source writes the interval as a
fraction ("½ second"), so there is no number in the text layer. The record is
kept with the value empty rather than a number being invented, and
`_value_grounded` is blank because the check does not apply.

`figures.csv` holds what only the chart knows — `Wall surface temperature, deg F`
against `Time, sec`, and the plotted values — none of which appear in the PDF's
text layer.

## About `results/`

It was produced by running the **real pipeline** (real ingest, schema coercion,
grounding checks and CSV writing) against a stub that returns fixed answers
instead of a model, so the demo is reproducible and costs nothing:

```bash
python demo/_make_reference_output.py
```

The numbers were read out of the source by hand, and grounding is still computed
for real — if a quoted span were not actually in the PDF, `_grounded` would come
back false. Every record in `results/` is grounded, which is the pipeline's own
verdict, not a claim.

A live model on the same inputs will find more records and word them
differently. The **shape** is what this folder is showing you.
