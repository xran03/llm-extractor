# Output

## Output

Per document, plus run-level tables:

```
out/
  records.csv                  every record from every document  <- start here
  figures.csv                  every value read out of a figure
  summary.json                 run totals, tokens, cache statistics
  <doc>.records.csv            the same rows, one file per document
  <doc>.records.jsonl          lossless records (nested values, audit flags)
  <doc>.ocr.json               structured vision output per figure
  <doc>.figures.csv            figure readings as a table
  <doc>.document.json          records + figures + aggregate + stats
  overlays/<doc>-pNNN.overlay.png   digitised points drawn back onto the page
```

`records.csv` is the analysis artifact: one row per fact, columns fixed by the
template, written with a UTF-8 BOM so Excel renders `µg/mL` correctly. JSONL is
the lossless machine format. Choose with `--format jsonl|csv|both` (default
`both`).

A record always carries its evidence and two audit flags:

```json
{
  "subject": "group A",
  "attribute": "antibody concentration",
  "value": 12.5,
  "unit": "µg/mL",
  "direction": "higher",
  "significant": "yes",
  "p_value": "<0.01",
  "source_span": "Group A reached 12.5 ug/mL, higher than group B (p<0.01).",
  "doc_id": "report",
  "_grounded": true,
  "_value_grounded": true,
  "_unit_grounded": true,
  "_ungrounded": []
}
```

Four deterministic checks run on every record. They cost no tokens and no API
calls, so they run on everything rather than on a sample:

| Flag | Meaning |
|---|---|
| `_grounded` | the quoted span was really found in the document |
| `_value_grounded` | **every** number in the record appears inside that span |
| `_unit_grounded` | the units declared match the units the span actually writes |
| `_ungrounded` | the field names that failed, so review can go straight to them |

`_value_grounded` and `_unit_grounded` are `null` when the check does not apply
(no number, or no unit stated in the evidence) — an unstated unit is reported as
unknown, never as wrong.

### Derived columns

Two columns are resolved rather than extracted, and sit apart from the model's
own fields for that reason:

| Column | Meaning |
|---|---|
| `repeat_unit` | the capsular repeat unit for the record's serotype |
| `repeat_unit_source` | `reference table (...)` or `document` |

A capsular measurement is only half-interpretable without the structure it was
measured on: `12.5 µg/mL` means different things for serotype 3 and 19F, and
the difference is the repeat unit. Records naming a serotype are annotated from
the shipped harmonised Danish-type table (105 serotypes):

```
serotype=19F  src=reference table (full_structure)  unit=→4)-β-D-ManpNAc-(1→4)-α-D-Glcp-(1→2)-α-L-Rhap-…
serotype=3    src=reference table (full_structure)  unit=→3)-β-D-GlcpA-(1→4)-β-D-Glcp-(1→
```

A structure the document itself states wins over the table — a paper reporting
a novel or corrected structure is more current than a static reference — and is
marked `document`. Lookup is exact after normalising the token (`19f`, `19 F`,
`serotype 19F` all resolve); an unknown serotype is left empty rather than
matched to a near neighbour, because attaching 6B's structure to a 6C record
would be worse than attaching nothing.

The span check is not a substring test: a quote that reproduces a real sentence
and then appends an invented clause is rejected, as is one that swaps a group
label or a number, while whitespace, case and OCR damage such as `ug/rnL` for
`ug/mL` are tolerated. The unit check is what catches a value reported in
`mg/mL` when the paper said `µg/mL` — a thousand-fold error that a plain number
comparison passes.


---

[Back to the README](../README.md)
