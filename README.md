# llm-extractor

Turn a pile of documents into one CSV of checked, quoted facts.

Point it at a folder of PDFs, Word files, slides and images, or at a literature
search, and it returns a table where every row carries the sentence it came
from — and a flag saying whether that sentence actually supports it.

```bash
llm-extract -i ./papers -o ./out --template immunogenicity
open ./out/records.csv
```

---

## Four things people actually do with it

### 1. A folder of mixed private documents

The common case: a directory someone has been filling for two years, with PDFs
next to `.docx` protocols, a `.pptx` readout and a few screenshots of figures.
Give it the folder.

```bash
llm-extract -i "S:/programs/pcv/reports" -o ./out --template immunogenicity
```

Format is detected from **content**, not the file extension, so a `.pdf` that is
really a scan, a mislabelled file and an extension-less export all route
correctly. Each format declares its own path — text layer, embedded media, or
rendered pages — so tables in `.docx`/`.xlsx` keep their rows, PowerPoint
speaker notes are read alongside slide text, and an image or scanned page goes
to the vision model. Nothing leaves your machine except the model calls.

| Kind | Formats |
|---|---|
| Documents | `.pdf` `.docx` `.doc` `.odt` `.rtf` |
| Slides | `.pptx` `.odp` |
| Spreadsheets | `.xlsx` `.ods` `.csv` `.tsv` |
| Text & markup | `.txt` `.md` `.rst` `.html` `.xml` (JATS/PMC) `.json` `.jsonl` |
| Mail & books | `.eml` `.mbox` `.epub` |
| Images | `.png` `.jpg` `.jpeg` `.gif` `.webp` `.tif` `.tiff` `.bmp` |

`llm-extract formats` lists them and how each is processed.

### 2. A literature search by keyword

No folder needed. Search Europe PMC or OpenAlex directly — neither needs a
credential:

```bash
llm-extract run --source europepmc \
  --param search="pneumococcal conjugate vaccine opsonophagocytic" \
  --fulltext -o ./out --template immunogenicity
```

`--fulltext` is the difference between reading abstracts and reading papers.
Without it you get the abstract; with it, open-access articles are fetched in
full — as JATS XML from Europe PMC, which the pipeline parses natively, so
sections and tables survive as structure rather than being recovered from a
picture of themselves. On one PCV13 trial paper that is 1,485 characters versus
51,971, and the opsonic index values exist only in the second.

Only what the API itself marks as open access is ever fetched, what arrives is
checked against what was promised, and a paper whose full text cannot be had
falls back to its abstract rather than failing the run.

One question is rarely one phrase, so a search can be a **list of terms**:

```bash
llm-extract run --source openalex --param search_file=terms.txt \
                --param max_records=500 --fulltext -o ./out
```

Every term is searched separately and the results are unioned by record id, so
terms are meant to overlap. Each document records the term that found it.

To connect a database that is not built in — a patent API, a licensed service,
an internal store — describe it in JSON instead of writing code:

```bash
llm-extract sources --init myapi.json     # a working Crossref example to edit
llm-extract run --source-config myapi.json -o ./out
```

### 3. Stitching several sources into one table

Runs share an output directory, and the combined `records.csv` **accumulates**
rather than being replaced. So two folders, or a keyword search plus a folder,
merge into one table by pointing them at the same place:

```bash
llm-extract -i ./trials-2024 -o ./out --template immunogenicity
llm-extract -i ./trials-2025 -o ./out --template immunogenicity
llm-extract run --source europepmc --param search="serotype 19F OPA" \
                --fulltext   -o ./out --template immunogenicity
```

One `out/records.csv`, every source in it, `doc_id` saying where each row came
from.

Two rules make this safe rather than lucky. Documents already extracted are
skipped by content hash, so re-running is cheap and nothing is duplicated. And
the template is part of that identity — asking a *different* question re-runs
the corpus instead of reporting everything as already done. If the columns
change, the combined table is rewritten rather than appended to, because new
columns under an old header would misdescribe every earlier row.

**Sharing a template is what makes the rows comparable.** Use the same
`--template` for everything you intend to stitch; that is the contract the
columns come from.

### 4. Checking the table before anyone trusts it

Every row already carries its evidence and three deterministic flags. For a
second opinion on whether a row *means* what it says:

```bash
llm-extract batch review -i ./papers -o ./out
llm-extract batch fetch -o ./out --wait
```

That adds `_review_value`, `_review_unit` and `_review_row` — the last being
whether the fields describe one measurement or several stitched together.

---

## Live, parallel, or batch

Three ways to spend the same work, answering different questions.

| | What it is | Use when |
|---|---|---|
| **live** | one document at a time | iterating on a template, or a handful of files |
| **live + parallel** | documents, chunks and figures overlapped | you want results now and the corpus is real |
| **batch** | queued with the provider, collected later | thousands of documents, results can wait |

Live is the default and needs no flag. **Parallel comes from the accelerated
wheel** — the compiled execution core published in the [release
assets](https://github.com/xran03/llm-extractor/releases). Same CLI, same
output, same tests; it only changes how the work is driven. Install the asset
matching your platform *and* Python version, then confirm:

```bash
pip install https://github.com/xran03/llm-extractor/releases/download/v0.1.0/llm_extractor-0.1.0-cp312-cp312-manylinux1_x86_64.manylinux_2_5_x86_64.whl
llm-extract check     # execution : accelerated (compiled)
```

Installing from source runs sequentially, which is fully supported.

Batch is a different shape: two calls, then close the laptop.

```bash
llm-extract batch preflight               # can this gateway do it at all?
llm-extract batch submit -i ./papers -o ./out --figures
llm-extract batch fetch  -o ./out --wait
```

It does not make a document faster, it stops you waiting for it — the caller's
cost stays at one upload and one create whether the corpus is 30 documents or
30,000. The trade is latency: a 24-hour completion window is a real 24 hours.
Answers come back identified only by an id, so `submit` writes a manifest and
`fetch` reassembles through it; grounding still runs at collect time against the
document re-read from disk.

---

## Models

Each backend has its own defaults, because a gateway only serves the models it
hosts — naming one it has never heard of turns a first run into a 404 hunt.

| Role | `aimodelhub` | `llmhub` |
|---|---|---|
| extraction, OCR, aggregation | `gpt-5.6-sol` | `gpt-4.1` |
| review (judging records) | `claude-fable-5` | `gpt-4.1-mini` |

Review deliberately uses a different family: a second opinion from the model
that wrote the answer mostly restates it. `--model` overrides every extraction
role at once; individual roles can still be pinned with `--ocr-model` and
`--agent-model`.

---

## Install

> **Not on PyPI.** That name belongs to an unrelated project, so
> `pip install llm-extractor` will **not** get you this package.

```bash
pip install "llm-extractor[all] @ git+https://github.com/xran03/llm-extractor"
```

Extras: `[pdf]` adds PDF text, `[all]` also adds page rendering for scanned
PDFs. The core itself has no dependencies. For the parallel build, see above.

Credentials take one command — `llm-extract login` pastes a key once and stores
it outside the project — or use a `.env`.

---

## What comes out

```
out/
  records.csv     one row per fact, with its quote and audit flags
  figures.csv     every value read out of a chart
  summary.json    totals, tokens, cache statistics
  documents/      per-document detail, one set of files each
```

The table you want is at the top; per-document intermediates go one level down,
which keeps a 200-document run readable and two runs easy to merge.

Every record carries `source_span` — a verbatim quote — plus `_grounded`,
`_value_grounded` and `_unit_grounded`. A number whose digits do not appear in
its own quote is flagged, not silently kept. Records naming a pneumococcal
serotype also get `repeat_unit`, resolved from a shipped reference table,
because a titre means different things for serotype 3 and 19F.

---

## Try it without spending anything

[`demo/`](demo/) ships five redistributable vaccine documents and the output
they produce — a PCV13 trial paper, its Figure 2, an FDA approval letter, and
two published figures. The reference output is committed, so you can read the
artifact shapes before configuring a key.

```bash
llm-extract -i ./demo -o ./demo/out --template immunogenicity --ocr always
```

The figure is the interesting one: `R2= 0.351` and `p= 0.012` are *drawn into
the image* and appear in no text layer, and the demo recovers them — while
reporting in `coverage_gaps` that the per-subject dots, which are drawn and
never written, were not. See [demo/README.md](demo/README.md).

---

## Documentation

| | |
|---|---|
| [Credentials](docs/credentials.md) | `login`, `.env`, OAuth, and what `check` reports |
| [Sources](docs/sources.md) | folders, literature APIs, full text, custom connectors |
| [Templates](docs/templates.md) | defining your own schema |
| [Output](docs/output.md) | every artifact and column, and what the flags mean |
| [Reading figures](docs/figures.md) | the vision pass, chart digitisation, and when it refuses |
| [Batch and review](docs/batch.md) | queueing a corpus, and verifying the table afterwards |
| [Caching and auditing](docs/caching.md) | cost control, cache revalidation, the HTTP API |
| [Architecture](docs/architecture.md) | how the pieces fit, and how to extend them |

---

## Why it is built this way

An extractor that is confidently wrong is worse than one that finds less.
Everything here follows from that: records quote their evidence, numbers are
checked against the quote they claim, figures are digitised before they are
described, values below a detection limit are recorded as censored rather than
dropped, and a document that yields nothing says so instead of producing
plausible filler.

## License

MIT. See [LICENSE](LICENSE).


