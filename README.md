# llm-extractor

Folder in, JSON out. Point it at a directory of PDFs, XML, DOCX, PPTX and images
and get back structured, evidence-grounded records — extracted by an LLM, with
figure OCR, an on-disk response cache, and a way to re-check the cache later.

```bash
llm-extract -i ./docs -o ./out --api llmhub
```

---

## Why it is built this way

| Need | How it is met |
|---|---|
| Two API backends | `llmhub` speaks `/v1/chat/completions`; `aimodelhub` speaks the newer `/v1/responses`. One interface, one command. |
| A table at the end | Records come out as **CSV** (one row per fact, columns fixed by the template) alongside lossless JSONL. |
| Your own schema | An **extraction template** is a JSON file you write; it becomes the strict JSON Schema sent to the model and the columns of the CSV. |
| Figures carry the numbers | The vision pass returns **structured JSON** (items, tables, axes), so it merges with text records instead of being prose. |
| One answer per document | An aggregation agent reconciles the text pass and the OCR pass, flags conflicts, and never invents records. |
| Cost | Every call is cached by content hash. Re-runs, added files and code iteration are free. |
| Trust | Records carry `_grounded` / `_value_grounded` / `_unit_grounded`; `llm-extract audit` replays a sample of cached calls and scores them. |
| Growth | Sources, providers and templates are registries with entry points — a patent or literature connector is a separate pip package. |
| Frontends | `llm-extract serve` exposes jobs, progress (SSE) and results over HTTP. |

---

## Install

> **Not on PyPI.** The name `llm-extractor` there belongs to an unrelated
> project, so `pip install llm-extractor` will **not** get you this package.
> Install from this repository or from a release asset.

From source — works on every platform, executes sequentially:

```bash
pip install "llm-extractor[all] @ git+https://github.com/xran03/llm-extractor"
```

Or for development:

```bash
git clone https://github.com/xran03/llm-extractor
cd llm-extractor
pip install -e ".[all]"
```

Extras: `[pdf]` adds PDF text (pypdf), `[all]` also adds page rendering for
scanned PDFs (PyMuPDF). The core itself has no dependencies.

### Accelerated build

The [releases page](https://github.com/xran03/llm-extractor/releases) carries
wheels with a compiled execution core that overlaps API calls across documents,
chunks and figures. Same CLI, same output, same tests — it only changes how the
work is driven, and it is a large difference on a folder of thousands of files.

Install the asset matching your platform and Python version by URL:

```bash
# Linux, CPython 3.12 (glibc 2.5 or newer, so any current distribution)
pip install https://github.com/xran03/llm-extractor/releases/download/v0.1.0/llm_extractor-0.1.0-cp312-cp312-manylinux1_x86_64.manylinux_2_5_x86_64.whl

# Windows, CPython 3.13
pip install https://github.com/xran03/llm-extractor/releases/download/v0.1.0/llm_extractor-0.1.0-cp313-cp313-win_amd64.whl
```

A wheel is built for one platform and one Python version, so the tag has to
match on both counts — a Linux asset cannot be installed on Windows, and pip
will say so rather than install something that cannot load.

With the optional extras:

```bash
pip install "llm_extractor[all] @ https://github.com/xran03/llm-extractor/releases/download/v0.1.0/llm_extractor-0.1.0-cp312-cp312-manylinux1_x86_64.manylinux_2_5_x86_64.whl"
```

Platforms and versions without a published wheel install from source and run
sequentially, which is fully supported. Check which core is active:

```bash
llm-extract check          # -> execution : accelerated (compiled) | sequential (sequential)
```

## Credentials

### The easy way: paste it once

If you do not want to edit configuration files, run:

```bash
llm-extract login
```

It asks for the gateway URL and then for your API key (**input is hidden**),
checks the key against the gateway, and only saves it if it actually works — so
a mistyped or expired key is reported immediately rather than halfway through a
run. From then on, just extract:

```bash
llm-extract -i ./docs -o ./out
```

The key is written to a file that only your account can read
(`0600`, inside a `0700` directory):

| Platform | Location |
|---|---|
| Windows | `%APPDATA%\llm-extractor\credentials.json` |
| macOS / Linux | `$XDG_CONFIG_HOME` (or `~/.config`) `/llm-extractor/credentials.json` |

It is kept outside your project folder on purpose, so a key can never be swept
into a commit. To remove it again:

```bash
llm-extract logout            # forget the selected backend
llm-extract logout --all      # forget every backend
```

### The file way: `.env`

For servers, CI and shared installations, copy `.env.example` to `.env` and fill
in the backend you use:

```ini
LLM_HUB_BASE_URL=https://your-gateway
LLM_HUB_API_KEY=...
# or OAuth2 client credentials, which mint a short-lived token:
LLM_HUB_CLIENT_ID=...
LLM_HUB_CLIENT_SECRET=...
LLM_HUB_TOKEN_URL=...
```

Resolution order, first hit wins:

`--api-key` → environment variable → nearest `.env` → saved login → paste prompt

A saved login therefore never overrides an explicit flag, a real environment
variable or a `.env` file, which keeps servers and CI behaving exactly as
before. To paste a one-off key without saving it, pass `-`:

```bash
llm-extract -i ./docs -o ./out --api-key -      # prompts, input hidden
```

Verify everything before spending tokens:

```bash
llm-extract check
```

## Usage

```bash
# save your key once (hidden input, verified before it is stored)
llm-extract login

# extract a folder
llm-extract -i ./docs -o ./out --api llmhub --model gpt-4.1

# the newer Responses API instead
llm-extract -i ./docs -o ./out --api aimodelhub

# only some formats, capped, with a rate limit
llm-extract -i ./docs -o ./out --extensions .pdf,.docx --limit 100 \
            --rate-limit 300

# force or disable the figure/vision pass
llm-extract -i ./docs -o ./out --ocr always
llm-extract -i ./docs -o ./out --ocr never --no-aggregate

# an external database instead of a folder
llm-extract run --source europepmc --param query="pneumococcal conjugate" \
                --param max_records=50 -o ./out

# ...or a list of search terms instead of one
llm-extract run --source openalex \
                --param search_file=templates/keywords-example.txt \
                --param max_records=500 -o ./out
```

### Searching a literature database

`europepmc` and `openalex` are built in and need no credential; both fetch title
and abstract and hand them to the same pipeline a folder would use. OpenAlex
ships abstracts as an inverted index, which the connector rebuilds into reading
order rather than passing on as unusable JSON.

A literature question is rarely one phrase, so a search can be a **list of
terms** instead of a single string:

```bash
llm-extract run --source openalex \
                --param search_file=my-terms.txt \
                --param max_records=500 -o ./out
```

The file is whatever you already have — `.txt` with one term per line, or the
first column of a `.csv`/`.tsv` export:

```text
# blank lines and '#' comments are ignored
pneumococcal conjugate vaccine immunogenicity
ExPEC conjugate vaccine immunogenicity
opsonophagocytic killing assay conjugate vaccine titer
```

A header row named `query`/`term`/`search`/`keyword` is skipped, and duplicate
terms are dropped before anything is fetched.

Every term is searched **separately** and the results are unioned by record id,
so terms are meant to overlap: broad wording finds the obvious papers, narrow
wording reaches the ones a single phrase misses, and a paper several terms
return is still extracted once. Each document records the term that found it in
its `search_term` metadata, so a result set can be traced back to the wording
that produced it.

Terms are deliberately not joined into one boolean query, because every API
spells boolean syntax differently — a wrong join fails silently by returning
the wrong set rather than an error. `--param max_records` caps the run as a
whole, not each term.

[`templates/keywords-example.txt`](templates/keywords-example.txt) is a working
list covering three pathogens and the assay wordings. Point either connector at
an internal mirror with `--param base_url=...`, and see `llm-extract sources`
for everything registered.

Wrappers are provided for convenience: `./bin/llm-extract` and
`./bin/llm-extract.ps1`.

Supported inputs — the format is detected from **content**, so mislabelled and
extension-less files still work:

| Kind | Formats |
|---|---|
| Documents | `.pdf` `.docx` `.doc` `.odt` `.rtf` |
| Slides | `.pptx` `.odp` |
| Spreadsheets | `.xlsx` `.ods` `.csv` `.tsv` |
| Text & markup | `.txt` `.md` `.rst` `.html` `.xml` (JATS/PMC) `.json` `.jsonl` |
| Mail & books | `.eml` `.mbox` `.epub` |
| Images | `.png` `.jpg` `.jpeg` `.gif` `.webp` `.tif` `.tiff` `.bmp` |

Each format declares its own path rather than the pipeline checking extensions:
whether it has a text layer, and where its figures come from (the file itself,
media embedded in the container, or rendered pages). So an image and a scanned
PDF both route to the vision pass, tables in `.docx`/`.xlsx`/`.ods` keep their
rows intact, and PowerPoint speaker notes are read alongside slide text.

```bash
llm-extract formats          # every format, its kind and how it is processed
```

Adding a format is one entry in `formats.BUILTIN` plus a reader; nothing
downstream changes.

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

The span check is not a substring test: a quote that reproduces a real sentence
and then appends an invented clause is rejected, as is one that swaps a group
label or a number, while whitespace, case and OCR damage such as `ug/rnL` for
`ug/mL` are tolerated. The unit check is what catches a value reported in
`mg/mL` when the paper said `µg/mL` — a thousand-fold error that a plain number
comparison passes.

## The vision pass on its own

Numbers in this kind of literature often live only in a figure, so there is a
second reader: each figure is sent to a vision model that must answer under
`OCR_JSON_SCHEMA`, returning structured data — items, tables, axis labels, a
caption, text blocks — rather than prose. That is what lets a figure reading be
merged with text records instead of ending up as a paragraph nobody can query.

Most runs get this automatically. If you want *only* this channel, turn the
other work off:

```bash
# vision only: read every figure, skip the aggregation agent
llm-extract -i ./docs -o ./out --ocr always --no-aggregate

# pick the vision model separately from the extraction model
llm-extract -i ./docs -o ./out --ocr always --ocr-model gpt-4.1
```

The three policies trade cost against recall:

| `--ocr` | when the vision model is called |
|---|---|
| `never` | not at all — text only, cheapest |
| `auto` (default) | when the document has no text layer, when too little text was recovered to be prose, or when the text pass returned nothing or nothing grounded |
| `always` | for every figure of every document, even when the text pass already succeeded |

`auto` is the one to leave alone for a mixed folder: a born-digital paper whose
text extracted cleanly never pays for a vision call, while a scanned one falls
through to it automatically. Reach for `always` when you know the numbers you
want are plotted rather than written.

### What it writes

The vision pass has its own artifacts, independent of the record table:

```
out/
  <doc>.ocr.json     the structured reading of each figure — items, tables,
                     axis labels, caption, text blocks, notes
  <doc>.figures.csv  the same thing flattened: one row per value read, with
                     the image, figure type, axis labels, series and unit
  figures.csv        every figure value from every document in the run
```

`figures.csv` is the one to open first; `<doc>.ocr.json` keeps the full nested
reading for anything the flat table cannot express.

Where figures come from depends on the format, not on the extension: an image
file is itself the figure, `pptx`/`docx`/`odp`/`epub` have their embedded media
unpacked, and a PDF has its pages rasterised so a scan can still be read. PDF
pages are triaged before rendering — a page that draws a graph, embeds a
picture, or holds too little text to be prose is rendered, and one that is
plainly prose the text pass already read verbatim is skipped.

### Limits worth knowing

Each figure is read by a single call, and that call is capped at 4,000 output
tokens. Under the strict figure schema every value costs roughly thirty tokens
once its label, series and unit are included, which works out at about 150
values per figure. A very dense figure — a dot plot with a point per subject,
say — does not fit, and what comes back is the part the model chose to report
rather than everything that is plotted. Treat this channel as a reader of
labelled values (bar heights, table cells, plotted means, axis annotations)
rather than as a way to recover a whole distribution.

Two other bounds apply per document: `--max-figures` (20 by default) caps how
many figures are sent at all, and any image over 12 MB is skipped rather than
uploaded, with the reason recorded in that figure's `notes`.

A figure that fails is contained: it is recorded with an empty reading and the
document keeps going, because one unreadable figure should never cost a paper.

Every call goes through the same cache as everything else, so re-running a
folder after changing only the text-side template costs nothing on this side —
and `llm-extract cache entries --stage ocr` lists what the vision pass has
already answered.

## Try it

[`demo/`](demo/) ships a public-domain scanned report, one chart cropped out of
it, and the output both produce — including a value the text layer does not
contain and only the vision pass recovers.

```bash
llm-extract -i ./demo -o ./demo/out --ocr always
```

See [demo/README.md](demo/README.md).

## Schemas you define

A template is the JSON contract: fields, types, enums, prompt, and the strict
JSON Schema sent to the model. Two ship built in — `generic` (subject /
attribute / value / evidence, works on anything) and `immunogenicity` — and
anything else is a JSON file you write.

```bash
llm-extract templates                          # list built-ins
llm-extract templates --show generic           # print one with its JSON schema
llm-extract templates --init my-template.json  # scaffold a valid starting point
llm-extract templates --validate my-template.json
llm-extract -i ./docs -o ./out --template my-template.json
```

Minimal template:

```json
{
  "name": "patent_claims",
  "instructions": "Extract each claim. Preserve the claim's own wording.",
  "key_fields": ["claim_number"],
  "fields": [
    {"name": "claim_number", "type": "integer", "description": "Claim number"},
    {"name": "claim_type",   "type": "string",  "description": "Claim kind",
     "enum": ["independent", "dependent", "na"]},
    {"name": "claim_text",   "type": "string",  "description": "Claim wording, verbatim"},
    {"name": "source_span",  "type": "string",  "description": "Verbatim evidence"}
  ]
}
```

Rules the validator enforces, with a message naming the offending field:

- field `type` is one of `string`, `number`, `integer`, `boolean`;
- `enum` is a non-empty list, and only on string fields;
- `key_fields` must name fields that exist;
- `doc_id`, `doc_title`, `_grounded`, `_value_grounded`, `_unit_grounded` and `_ungrounded` are reserved;
- a `source_span` field is **required** — it is what makes a record checkable.

Worked examples live in [`templates/`](templates/). The CSV columns follow the
template, so changing the schema changes the table.

A frontend can send a schema inline instead of shipping a file:

```bash
curl -X POST localhost:8080/v1/templates/validate \
  -d '{"template": {"name": "t", "fields": [...]}}'

curl -X POST localhost:8080/v1/jobs \
  -d '{"source": "folder", "params": {"input_dir": "./docs"},
       "template": {"name": "t", "fields": [...]}}'
```

## Caching and cost

```bash
llm-extract cache stats            # entries, bytes, hit rate, tokens saved
llm-extract cache entries --stage ocr
llm-extract cache clear
```

The cache key covers backend, model, full message content (images included),
temperature, token budget and schema — so any real change misses, and nothing
else does. Documents unchanged since a previous successful run are skipped
entirely (`--no-resume` to force).

## Auditing what was cached

A cache nobody checks is a liability. Each entry stores its original request, so
it can be replayed:

```bash
# replay 25 sampled calls and score them against what was cached
llm-extract audit --n 25 --strategy oldest

# cross-check with a stronger referee model, drop whatever fails
llm-extract audit --n 50 --referee-model gpt-4.1 --invalidate-drifted -o audit.json
```

Sampling strategies: `random`, `oldest`, `newest`, `largest`, `unverified`.
Each entry gets a verdict (`confirmed` / `drifted` / `suspect` / `error`) written
back to the index, and the report includes a Wilson 95% confidence interval so a
small sample is not over-read.

## HTTP API

```bash
llm-extract serve --port 8080            # optional: --token <shared secret>
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/v1/capabilities` | providers, sources (+ parameters), templates |
| GET | `/v1/templates` | built-in templates + a starter schema |
| GET | `/v1/templates/{name}` | template with its JSON Schema |
| POST | `/v1/templates/validate` | check a user-authored schema |
| POST | `/v1/jobs` | start a job, returns `202` + `job_id` |
| GET | `/v1/jobs/{id}` | status, counters, progress |
| GET | `/v1/jobs/{id}/tasks` | per-document rows |
| GET | `/v1/jobs/{id}/events` | Server-Sent Events progress stream |
| GET | `/v1/documents/{doc_id}` | aggregated document JSON |
| GET/DELETE | `/v1/cache` | statistics / clear |
| POST | `/v1/cache/audit` | run an audit |

```bash
curl -X POST localhost:8080/v1/jobs -H 'Content-Type: application/json' \
  -d '{"source":"folder","params":{"input_dir":"./docs"},"api":"llmhub"}'
```

Capabilities are generated from the registries, so an installed plugin shows up
in the API — and in a frontend's forms — without touching the service code.

## Architecture

```
sources/       where documents come from   folder | rest | patents | literature
    |
ingest         format readers              pdf xml docx pptx png jpeg txt md html
    |
pipeline       per document:  text extraction  ->  figure OCR  ->  aggregation agent
    |          every model call goes through providers/ and the cache
runner         scheduler (rate limit, retry, isolation) + job store + event bus
    |
cli / service  the CLI and the HTTP API drive the same runner
```

Extension points, all registries with entry-point discovery:

| Axis | Registry | Entry point group |
|---|---|---|
| API backends | `providers.BACKENDS` | — |
| Document sources | `sources.SOURCES` | `llm_extractor.sources` |
| Templates | `templates.BUILTIN_TEMPLATES` | — (or a JSON file) |

Adding a patent database in a separate package:

```python
from llm_extractor.sources import SOURCES, RestSource

@SOURCES.register("my-patents")
class MyPatents(RestSource):
    name = "my-patents"
    description = "Internal patent store"
    defaults = {
        "base_url": "https://patents.internal",
        "path": "/api/search",
        "records_path": "hits",
        "id_field": "docId",
        "text_fields": ["abstract", "claims"],
        "auth": "bearer",
        "auth_env": "PATENT_API_KEY",
    }
```

```toml
[project.entry-points."llm_extractor.sources"]
my-patents = "my_package:MyPatents"
```

It is then available to the CLI (`--source my-patents`) and the HTTP API with no
changes here.

## Development

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests -t .
```

356 offline tests: no network, no credentials. Model calls run through fakes and
the end-to-end suite drives the real CLI against a local fake gateway that
implements both API styles.

## License

MIT — see [LICENSE](LICENSE).
