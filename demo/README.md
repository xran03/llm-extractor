# Demo

Four inputs and the output they produce, so you can see the artifact shapes
before spending a token.

| File | What it is |
|---|---|
| `naca-report-1372-excerpt.pdf` | 3-page excerpt of a 1958 scanned technical report |
| `naca-figure-2.png` | one chart cropped out of that report |
| `opa-scatter.png` | a scatter plot of vaccine titers, drawn for this demo |
| `h5-titre-histogram-scatter.jpg` | a real published figure: a titre histogram and two scatter plots |
| `results/` | reference output for all four |

The first two come from **NACA Report 1372**, *A Method of Computing the Transient
Temperature of Thick Walls from Arbitrary Variation of Adiabatic-Wall
Temperature and Heat-Transfer Coefficient* (P. R. Hill, 1958), retrieved from
the [NASA Technical Reports Server](https://ntrs.nasa.gov/citations/19930091019).
It is a work of the U.S. Government and is **in the public domain**. The excerpt
is pages 887–889; the PNG is Figure 2 from page 888.

`opa-scatter.png` is ours: `_make_opa_scatter.py` draws it, so it can be
redistributed freely and what is printed on it is known exactly rather than
squinted at.

`h5-titre-histogram-scatter.jpg` is Extended Data Fig. 1 of Kok, A. *et al.*,
*A vaccine central in A(H5) influenza antigenic space confers broad immunity*,
**Nature** (2025), [doi:10.1038/s41586-025-09626-3](https://doi.org/10.1038/s41586-025-09626-3)
([PMC12657240](https://europepmc.org/article/PMC/PMC12657240)). It is reproduced
unmodified under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It
is here because the three files above it were all made or chosen by us, and a
demo that only ever meets figures of its own making is not evidence of
anything.

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

## The scatter plot, and what a vision model can honestly do with it

`opa-scatter.png` is the case this literature is full of: three groups of 34
subjects, every titer plotted as a dot, and the group geometric means printed
above the columns. A text extractor gets nothing from it at all — there is no
text layer, and even in a born-digital paper the numbers are drawn rather than
written.

The vision pass reads what is *written on* the figure:

| label | series | value | unit | value_text |
|---|---|---|---|---|
| PCV13 | Day 28 | 814 | OPA titer | GMT 814 |
| PCV20 | Day 28 | 437 | OPA titer | GMT 437 |
| Placebo | Day 28 | 15 | OPA titer | GMT 15 |

— plus the caption, both axis labels and the tick values, all as structured
JSON rather than a paragraph, which is what lets these rows sit in the same
table as records read out of prose.

What it does **not** do is give you the 102 individual subject titers. Those
points carry no labels, and one call reads one figure under a 4,000-token cap,
which is about 150 values. This channel reads the values a figure *states* —
printed means, bar heights, table cells, axis annotations — and the reference
output says so in its own `notes` and `coverage_gaps` rather than quietly
returning three rows and letting you assume that was everything.

## The same channel on a figure we did not draw

`opa-scatter.png` is the favourable case, and it is favourable because *we*
drew it: the group means are printed on it as text, so reading them is reading
words. `h5-titre-histogram-scatter.jpg` is the ordinary case — a real published
figure, six panels, a histogram of log₂ standard deviation of HI titres in
panel **a** and scatter plots in panels **e** and **f**.

Here is the whole of what the vision pass returned for it:

| what came back | count |
|---|---|
| `items` (the structured readings) | **0** |
| `tables` | 0 |
| `text_blocks` | 20 |

`figures.csv` therefore gets one row for the figure with every value column
empty, and `records.csv` gets nothing at all. Not one of the several hundred
plotted points was recovered, and neither were the histogram bar heights.

The two numbers this figure actually states in print, `R2 = 0.93` and
`R2 = 0.84`, *were* read — but they arrived as loose entries in `text_blocks`
rather than as items with a value, so the only artifact that carries them
forward is the aggregation agent's `figure_insights`. Anyone reading
`figures.csv` never sees them.

One label came back wrong: the figure prints "Mean RMSE (detectable titers)"
and the model returned "Mean RMSE (recodable titres)". It is kept in
`results/` as returned. A vision pass makes transcription errors that look
exactly like correct answers, and a demo that quietly corrects them is telling
you about its author rather than about the tool.

So the honest summary of this channel: it reads what a figure **writes** —
printed means, table cells, axis labels, captions — and it does not measure
what a figure **draws**. When the numbers exist only as ink at a position, as
in a scatter plot or a histogram, expect labels back and no data.



It was produced by running the **real pipeline** (real ingest, schema coercion,
grounding checks and CSV writing) against a stub that returns fixed answers
instead of a model, so the demo is reproducible and costs nothing:

```bash
python demo/_make_reference_output.py
```

The numbers were read out of the source by hand — with one deliberate
exception. The payload for `h5-titre-histogram-scatter.jpg` is the **verbatim
reply of a live vision model**, recorded once and replayed, so that at least
one document in this folder shows what really comes back rather than what we
would have written down. Grounding is still computed for real either way — if a
quoted span were not actually in the PDF, `_grounded` would come back false.
Every record in `results/` is grounded, which is the pipeline's own verdict,
not a claim.

A live model on the other inputs will find more records and word them
differently. The **shape** is what this folder is showing you.
