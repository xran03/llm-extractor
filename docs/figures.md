# Reading figures

## Reading figures: measure first, look second

A dot plot of per-subject titers is the hardest thing in a paper to extract and
the most valuable, because those numbers usually appear nowhere else. Asking a
vision model to read it does not work at scale: under the strict OCR schema each
point costs about 29 output tokens, so one reply holds at most ~140 points, and
the values are eyeballed positions rather than measurements.

But in a born-digital PDF those dots are not a picture. They are drawing
operators, with coordinates the publisher wrote exactly. So the pipeline
measures them:

```bash
llm-extract -i ./docs -o ./out --chart auto     # default
llm-extract -i ./docs -o ./out --chart never    # vision pass only
```

Each plotted point becomes one record, with `extraction_mode = vector_geometry`,
`value_kind = individual`, and a `geometry_ref` of the form
`p5:panel0:cluster2:#cd3333:#41` that addresses the exact marker it came from —
so the corpus can be split by provenance later without reopening a PDF.

The split of labour is what keeps it honest:

| | supplies | never supplies |
|---|---|---|
| geometry | every **number** | any name |
| the model | every **name** — serotype, group, timepoint, unit | any number |

The labelling model is not shown a single value, is not asked for one, and its
response schema contains no numeric field at all. A hallucinated titer therefore
has no path into the output.

### Refusing is a feature

A figure that cannot be calibrated is left to the vision pass rather than
guessed at. That is the cheaper mistake by far: an unread figure costs one
vision call, while a wrongly calibrated one silently poisons every statistic
computed from it. Pages are refused when there is no monotonic axis, when too
few markers are found to be a distribution, or when the axis fit is loose.

### The three audits

Replotting the recovered values and comparing them with the original proves
nothing — the values came from inverting the axis model, so replotting them
through the same model reproduces it by construction. The checks that carry
information are the ones that can fail independently:

| check | catches | why it is not circular |
|---|---|---|
| **overlay** — detected markers drawn onto the rendered page | picking up error-bar caps, legend keys, glyph fragments | it is compared with the image, not with the model |
| **calibration residual** — how far the fit puts each tick from where it is | a mis-scaled or mis-paired axis | the fit is overdetermined: 3+ ticks constrain 2 parameters |
| **cross-channel agreement** — recovered GMT and n vs what the text pass read | the whole channel being wrong | different source, different reader, no shared code |

Calibration prefers the rules the plotting library drew at each tick over the
tick label text, and every record says which it got in `geometry_basis`. This
matters more than it sounds: a label's bounding box is centred on its glyphs,
not on the tick, and that constant offset produces a fit with a near-zero
residual whose values are uniformly wrong — on a small multiple with a 41-point
axis, by tens of percent. It is the one error that self-consistency cannot see.

### One measurement, one record

The channels overlap. A titer stated in the prose, plotted in a figure and
measured off that figure's geometry is one fact read three times, and simply
concatenating the passes inflates every count taken from the corpus — silently,
because each duplicate is individually correct.

So records are reconciled before they are written. Two records are the same
measurement when the template's key fields, the value kind, the unit *and* the
value all match; the evidence is deliberately not part of that identity, since
a quoted sentence and a calibrated axis are different justifications for the
same fact. The surviving record names the channels that agreed with it in
`corroborated_by` — agreement between independent readers is the best signal in
the pipeline and is worth keeping.

Where the channels *disagree* about one measurement, both records survive,
flagged `modality_conflict = value_mismatch` with a note naming the other
reading. A disagreement is a finding; quietly picking a winner would hide
exactly the cases worth reviewing.

Individual data points are exempt from conflict handling: two hundred subjects
in one group share every key field and differ only in their values, which is
the shape of the data rather than a contradiction.

`assay_platform` is one of the key fields, so an IgG GMC measured by reference
ELISA is never merged with one measured by dLIA or by electrochemiluminescence.
Those are different quantities that differ systematically, and treating them as
one measurement would be a worse error than counting them twice.

### Using the vision pass on its own

Collaborators who only want figures read by a vision model — no geometry, no
text extraction to pay for — can run that channel alone:

```bash
# vision only: every figure goes to the vision model, nothing is digitised
llm-extract -i ./docs -o ./out --ocr always --chart never --no-aggregate
```

What each switch is doing, and why you may want the other setting:

| switch | effect | when to change it |
|---|---|---|
| `--ocr always` | every figure is sent to the vision model, even when the text pass already succeeded | `auto` (default) only calls it when the text pass came back empty or ungrounded — much cheaper |
| `--chart never` | turns off vector digitisation entirely | leave it at `auto` if you want exact values where they are available; the pages it measures are then withdrawn from the vision pass |
| `--no-aggregate` | skips the reconciliation agent, saving one call per document | drop it if you want a per-document summary and conflict list |

The vision pass writes its own artifacts, independent of the record table:

```
out/
  <doc>.ocr.json     the structured reading of each figure: items, tables,
                     axis labels, caption, text blocks
  <doc>.figures.csv  the same thing flattened — one row per value read
  figures.csv        every figure value from every document
```

Pick the model with `--ocr-model` (or `LLM_EXTRACTOR_OCR_MODEL`); it does not
have to be the model doing the text extraction:

```bash
llm-extract -i ./docs -o ./out --ocr always --chart never --ocr-model gpt-4.1
```

Two limits are worth knowing before you rely on this channel. Its output is
capped by `--max-output-tokens`, and under the strict figure schema each value
costs roughly 29 tokens, so a very dense figure will be cut off — when that
happens the partial reading is kept and the figure is marked `truncated` in
`<doc>.ocr.json` rather than being silently reported as empty. And `--max-figures`
(default 20) bounds how many figures per document are sent at all.

If what you actually want is per-point data out of a vector figure, the vision
pass is the wrong tool and `--chart` is the right one — see above.

---


---

[Back to the README](../README.md)
