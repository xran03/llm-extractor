# Demo

Five documents and the output they produce, so you can see the artifact shapes
before spending a token. Everything here is redistributable, and everything is
about conjugate vaccines — the case this was built for.

| File | What it is |
|---|---|
| `pcv13-opa-colonisation.pdf` | a published PCV13 challenge study reporting serotype 6B OPA titres |
| `pcv13-opa-figure2.png` | Figure 2 of that paper: per-subject dot plots and three scatter plots |
| `fda-pcv15-approval-letter.pdf` | the FDA letter licensing a 15-valent pneumococcal conjugate vaccine |
| `opa-scatter.png` | a scatter plot of OPA titres, drawn for this demo |
| `h5-titre-histogram-scatter.jpg` | a real published figure: a titre histogram and two scatter plots |
| `results/` | reference output for all five |

## Where they came from

`pcv13-opa-colonisation.pdf` and the `pcv13-opa-figure2.png` crop taken from it
are Wolf, A.-S. *et al.*, *Quality of antibody responses by adults and young
children to 13-valent pneumococcal conjugate vaccination and Streptococcus
pneumoniae colonisation*, **Vaccine** 40(50): 7201–7210 (2022),
[doi:10.1016/j.vaccine.2022.09.069](https://doi.org/10.1016/j.vaccine.2022.09.069)
([PMC10615833](https://europepmc.org/article/PMC/PMC10615833)). Reproduced under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the figure is cropped
but otherwise unmodified.

`fda-pcv15-approval-letter.pdf` is the FDA's BLA approval letter for
Pneumococcal 15-valent Conjugate Vaccine (STN BL 125741/0, 16 July 2021),
retrieved from [fda.gov](https://www.fda.gov/media/150820/download). It is a
work of the U.S. Government and is **in the public domain**.

`h5-titre-histogram-scatter.jpg` is Extended Data Fig. 1 of Kok, A. *et al.*,
*A vaccine central in A(H5) influenza antigenic space confers broad immunity*,
**Nature** (2025), [doi:10.1038/s41586-025-09626-3](https://doi.org/10.1038/s41586-025-09626-3)
([PMC12657240](https://europepmc.org/article/PMC/PMC12657240)), reproduced
unmodified under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

`opa-scatter.png` is ours: `_make_opa_scatter.py` draws it, so what is printed
on it is known exactly rather than squinted at.

## Why these five

Each one is here to fail differently.

**The trial paper** carries the shape that matters. Its hepatitis A control arm
sits between 164 and 382 while the PCV13 arm reaches 7132, so a run that
recovers only the impressive number is visibly wrong rather than plausibly
right.

**Figure 2** is the case for looking at images at all. Its R² and p values —
`R2= 0.351`, `p= 0.012` — are drawn into the panels, and its axis labels
(`Opsonic index`, `IgG1 titre (ug/ml)`) exist nowhere else. Panels A and B plot
one dot per subject, and those values are *not* recoverable: they are drawn and
never written, so the reference output reports the printed statistics and says
so in `coverage_gaps` instead of inventing per-subject numbers.

**The FDA letter** is the opposite kind of document — no figures, no assay, one
regulatory sentence naming fifteen serotypes. It is the cheapest way to watch
the `repeat_unit` column resolve a whole valency.

**The two figures we did not draw** are there because a demo that only ever
meets figures of its own making is not evidence of anything. The published H5
figure keeps the model's misreading of an axis label verbatim; editing it out is
how a demo starts flattering itself.

## Run it

```bash
# from the repository root, with credentials configured (see ../.env.example)
llm-extract -i ./demo -o ./demo/out --template immunogenicity --ocr always
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
  records.csv        every record from every document — the file to open first
  figures.csv        every value read out of a figure
  summary.json       run totals, tokens, cache statistics
  documents/         per-document detail, one set of files each
    <doc_id>.records.jsonl    every record, nested fields intact
    <doc_id>.records.csv      the same rows, flat
    <doc_id>.figures.csv      values read out of that document's figures
    <doc_id>.ocr.json         the raw structured OCR reading
    <doc_id>.document.json    records, figures, aggregate and stats together
```

The combined tables sit at the top because that is what a run is usually opened
for; everything per-document goes one level down, which keeps a 200-document run
readable and makes two runs easy to stitch together.

## About the reference output

`results/` is produced by `_make_reference_output.py`, which runs the **real
pipeline** — real ingest, real schema coercion, real grounding checks, real CSV
writing — against a stub that returns fixed answers instead of a model. So it
costs nothing to regenerate:

```bash
python demo/_make_reference_output.py
```

Two consequences worth knowing. The values were read out of the sources by hand,
so the reference output is factually correct about these documents. And
grounding is still computed for real: if a quoted span were not actually present
in the source, `_grounded` would come back false. Nothing here is asserted that
the documents do not support.
