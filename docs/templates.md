# Schemas you define

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


---

[Back to the README](../README.md)
