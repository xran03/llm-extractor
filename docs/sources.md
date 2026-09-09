# Sources: folders, literature APIs and custom connectors

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

### Connecting a database we do not ship

A patent office, a licensed literature service and an internal store differ from
Europe PMC only in URL, paging, auth and field names — so those are
configuration, not code. `rest` takes them as parameters, and since a real
connector needs a dozen of them, it takes them as a **file** you can review,
version and hand to someone else:

```bash
llm-extract sources --init crossref.json     # a working example to edit
llm-extract run --source-config crossref.json -o ./out
```

```json
{
  "source": "rest",
  "base_url": "https://api.crossref.org",
  "path": "/works",
  "query_param": "query",
  "search": "pneumococcal conjugate vaccine",
  "records_path": "message.items",
  "id_field": "DOI",
  "title_field": "title.0",
  "text_fields": ["abstract"],
  "uri_field": "URL",
  "paging": "offset",
  "offset_param": "offset",
  "size_param": "rows",
  "auth": "none"
}
```

Auth is `none`, `bearer`, `header` or `query`, with the credential read from an
environment variable named by `auth_env` — so a key lives in `.env` next to the
model credentials rather than in the connector file you are about to share.
`--param` overrides any single setting without restating the rest, which is what
makes one shared file usable against a staging host.

Paging covers `page`, `offset`, `cursor` and `none`. If an API needs more than
that, a connector is a subclass that sets defaults (see
[`sources/patents.py`](src/llm_extractor/sources/patents.py)) or a separate pip
package publishing an `llm_extractor.sources` entry point — installing it is
then the whole integration.

The same flags work on `batch submit`, so a literature search can be queued
rather than run live.

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


---

[Back to the README](../README.md)
