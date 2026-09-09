# Architecture and development

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


---

[Back to the README](../README.md)
