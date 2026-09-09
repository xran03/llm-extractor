# Caching, auditing and the HTTP API

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


---

[Back to the README](../README.md)
