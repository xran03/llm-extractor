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
| Figures carry the numbers | Vector figures are **measured** from the PDF's own drawing coordinates — one record per plotted point, exact to the file. Where that cannot be calibrated, the vision pass returns **structured JSON** (items, tables, axes) so it merges with text records instead of being prose. |
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

## Models

Each backend has its own defaults, because a gateway only serves the models it
hosts — naming one it has never heard of turns a first run into a 404 hunt.

| Role | `aimodelhub` | `llmhub` |
|---|---|---|
| extraction, OCR, aggregation | `gpt-5.6-sol` | `gpt-4.1` |
| review (judging records) | `claude-fable-5` | `gpt-4.1-mini` |

Review deliberately uses a different model family: a second opinion from the
same model that wrote the answer mostly restates it.

Passing `--model` overrides all the extraction roles at once, so the whole run
uses what you asked for. Individual roles can still be pinned with
`--ocr-model`, `--agent-model`, or `LLM_EXTRACTOR_REVIEW_MODEL`.

`llm-extract check` reports which models are configured and whether the gateway
lists them:

```
model     : gpt-5.6-sol
review_model: claude-fable-5
connectivity: OK (195 models visible)
models    : claude-fable-5, gpt-5.6-sol all listed
```

Listing is advisory — some gateways serve aliases and deployments they do not
list, and a key is often scoped to a subset of what it can see. If a model is
refused at call time the gateway says which ones the key may use.

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

# hand the whole corpus to the gateway's batch queue instead of waiting
llm-extract batch submit -i ./docs -o ./out
llm-extract batch fetch  -o ./out --wait

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

Two of those examples are meant to be used together on the same corpus:
`conjugate-titer.json` extracts immunogenicity measurements, and
`conjugate-characterization.json` extracts the chemistry of the lots that
produced them — polysaccharide size, degree of activation, saccharide/protein
input and product ratios, free saccharide and O-acetylation. They are separate
runs rather than one wide table because a record is only checkable against its
own evidence: a titer is quoted from the results, while a degree of activation
is quoted from the methods, so folding both into one record would leave every
chemistry number unverifiable. Join the two tables afterwards on `study_batch`
and `serotype`, which both templates instruct the model to copy verbatim.

```bash
llm-extract -i ./papers -o ./out/titer --template templates/conjugate-titer.json
llm-extract -i ./papers -o ./out/cmc   --template templates/conjugate-characterization.json
```

A frontend can send a schema inline instead of shipping a file:

```bash
curl -X POST localhost:8080/v1/templates/validate \
  -d '{"template": {"name": "t", "fields": [...]}}'

curl -X POST localhost:8080/v1/jobs \
  -d '{"source": "folder", "params": {"input_dir": "./docs"},
       "template": {"name": "t", "fields": [...]}}'
```

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

## Batch API: hand over the corpus and come back

The live path spends one request per chunk and waits for each. For a few
thousand documents that is the wrong shape — it burns rate limit, pins a
process open for hours, and loses the run if the machine sleeps. The
OpenAI-compatible batch API inverts it: upload every request at once, let the
gateway work through them within a completion window, then collect.

```bash
llm-extract batch preflight --api aimodelhub          # can this gateway do it?
llm-extract batch submit -i ./docs -o ./out --figures # queue text + figures
llm-extract batch status -o ./out
llm-extract batch fetch  -o ./out --wait              # collect, write artifacts
```

`fetch` writes exactly the artifacts a live run writes — same records, same
CSV columns, same grounding flags — so nothing downstream has to know which
route produced them.

**Preflight first.** A gateway can expose the batch routes and still be unable
to run one, because queued input is uploaded as a file and file storage is a
separate setting:

```
list batches : ok
upload files : failed
detail       : HTTP 500 ... files_settings is not set, set it on your config.yaml
batch ready  : no
```

That is a gateway deployment setting, not a client problem, and the command
says so rather than failing later with a confusing error.

### Why the answers are the hard part

A batch answer arrives hours later carrying nothing but a `custom_id`. The
document it belonged to, which chunk of it, whether it was text or a figure —
none of that is in the response. So `submit` writes a manifest recording, for
every queued request, where its answer belongs, and `fetch` reassembles
through it. A document split into ten chunks comes back as one record set in
the original order; a figure comes back attached to its image; one failed
request is reported without losing the other nine.

Grounding still happens at collect time against the document re-read from
disk, so a fabricated value is caught in batch mode exactly as it is live.

### What batch is and is not for

Batch does not make a document faster, it stops you waiting for it. The caller's
cost stays at two calls — one upload, one create — whether the corpus is 30
documents or 30,000, and the answers land within the gateway's completion
window. Run the corpus live when you want the results now; queue it when you
want the machine back.

---

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
pipeline       per document:  text extraction  ->  figure digitisation  ->  figure OCR  ->  aggregation agent
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
