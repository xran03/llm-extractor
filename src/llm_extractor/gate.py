"""Relevance gate — decide what is worth extracting before paying to extract it.

A literature harvest is mostly noise for any given question: a search for
conjugate-vaccine papers returns reviews, editorials, epidemiology and vaccine
policy alongside the handful of studies that actually report titers. Running
full-text extraction over all of it is the expensive way to find that out.

This stage reads only what is already on hand — title and abstract — and asks a
cheap model one question per record: could this paper plausibly contain the data
we are after? Records that pass are the only ones worth fetching a PDF for.

Records are judged in small batches so the instructions are paid for once per
batch rather than once per record, and each verdict is tied back by ``id`` so a
model that returns them out of order, or drops one, cannot silently shift every
answer by one row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ._exec import map_completed
from .parsing import extract_json_array

#: Records per model call. Large enough to amortise the instructions, small
#: enough that one malformed reply costs little and stays inside the context.
BATCH_SIZE = 10

GATE_SYSTEM_PROMPT = (
    "You are screening scientific abstracts for a data-extraction pipeline. "
    "You decide only whether a paper is worth reading in full. You are "
    "deliberately permissive: a false exclusion loses data permanently, while a "
    "false inclusion only costs one extraction. When an abstract is ambiguous "
    "or too short to judge, include it."
)

GATE_INSTRUCTIONS = (
    "For each record decide whether the full text could plausibly report "
    "QUANTITATIVE ANTIBODY RESPONSE data (OPA titers, IgG/IgM concentrations, "
    "GMT/GMC, seroconversion rates) for a GLYCOCONJUGATE VACCINE against "
    "pneumococcus, E. coli, Group B Streptococcus, meningococcus or Hib.\n\n"
    "INCLUDE: clinical trials, animal immunogenicity studies, assay development "
    "or bridging studies, carrier/chemistry comparisons - anything that measures "
    "an antibody response.\n"
    "EXCLUDE: pure epidemiology, carriage or surveillance without immunogenicity, "
    "cost-effectiveness and policy papers, narrative reviews and editorials, "
    "papers about unrelated diseases or unconjugated vaccines only.\n\n"
    "Return ONE JSON array, one object per input record, each with:\n"
    "  id       - the record id exactly as given\n"
    "  include  - true or false\n"
    "  pathogen - pneumococcus | e_coli | gbs | meningococcus | hib | other | na\n"
    "  study_type - clinical | animal | in_vitro | review | other | na\n"
    "  confidence - high | medium | low\n"
    "  reason   - one short clause justifying the decision\n"
    "Return every id you were given, exactly once. JSON only."
)

GATE_JSON_SCHEMA = {
    "name": "gate_verdicts",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "include": {"type": "boolean"},
                        "pathogen": {"type": "string"},
                        "study_type": {"type": "string"},
                        "confidence": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "include"],
                },
            }
        },
        "required": ["verdicts"],
    },
}


@dataclass
class GateResult:
    verdicts: dict = field(default_factory=dict)   # id -> verdict dict
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_calls: int = 0
    errors: list = field(default_factory=list)

    @property
    def included(self) -> list:
        return [v for v in self.verdicts.values() if v.get("include")]

    def summary(self) -> dict:
        included = sum(1 for v in self.verdicts.values() if v.get("include"))
        return {
            "judged": len(self.verdicts),
            "included": included,
            "excluded": len(self.verdicts) - included,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_calls": self.cached_calls,
            "errors": len(self.errors),
        }


def build_batch_messages(batch: list) -> list:
    """Render one batch of records into a screening prompt."""
    payload = [
        {
            "id": str(record.get("id")),
            "title": str(record.get("title") or "")[:400],
            "abstract": str(record.get("abstract") or "")[:2500],
        }
        for record in batch
    ]
    return [
        {"role": "system", "content": GATE_SYSTEM_PROMPT},
        {"role": "user",
         "content": f"{GATE_INSTRUCTIONS}\n\n{json.dumps(payload, ensure_ascii=False)}"},
    ]


def _parse_verdicts(text: str) -> list:
    """Accept either a bare array or the schema's {"verdicts": [...]} envelope."""
    try:
        return extract_json_array(text)
    except ValueError:
        payload = json.loads(text)
        if isinstance(payload, dict) and isinstance(payload.get("verdicts"), list):
            return payload["verdicts"]
        raise


def gate_batch(provider, batch: list, model: str, max_tokens: int = 2000) -> dict:
    """Screen one batch; returns ``{id: verdict}`` plus a ``_usage`` entry."""
    completion = provider.complete(
        build_batch_messages(batch), model=model, temperature=0.0,
        max_tokens=max_tokens, json_schema=GATE_JSON_SCHEMA,
        meta={"stage": "gate", "doc_id": f"gate:{batch[0].get('id')}"},
    )
    verdicts = {}
    for raw in _parse_verdicts(completion.text):
        if isinstance(raw, dict) and raw.get("id") is not None:
            verdicts[str(raw["id"])] = raw
    return {"verdicts": verdicts, "usage": completion.usage}


def gate_records(provider, records, model: str, workers: int = 8,
                 batch_size: int = BATCH_SIZE, on_progress=None) -> GateResult:
    """Screen every record. Unjudged records are included, never dropped.

    A record the model failed to answer for is treated as a pass: the gate
    exists to save money, and silently discarding a paper because a batch came
    back malformed would trade a permanent data loss for a few cents.
    """
    records = list(records)
    batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
    result = GateResult()

    def _one(batch):
        try:
            return batch, gate_batch(provider, batch, model)
        except Exception as exc:
            return batch, exc

    for batch, outcome in map_completed(_one, batches, workers=workers):
        if isinstance(outcome, Exception):
            result.errors.append(f"{type(outcome).__name__}: {outcome}")
            for record in batch:
                result.verdicts[str(record.get("id"))] = _unjudged(record, str(outcome))
        else:
            usage = outcome["usage"]
            result.prompt_tokens += usage.prompt_tokens
            result.completion_tokens += usage.completion_tokens
            result.cached_calls += 1 if usage.cached else 0

            for record in batch:
                key = str(record.get("id"))
                verdict = outcome["verdicts"].get(key)
                result.verdicts[key] = verdict if verdict else _unjudged(
                    record, "no verdict returned")
        if on_progress is not None:
            on_progress(len(result.verdicts), len(records))
    return result


def _unjudged(record, reason: str) -> dict:
    return {
        "id": str(record.get("id")),
        "include": True,
        "pathogen": "na",
        "study_type": "na",
        "confidence": "low",
        "reason": f"not judged ({reason}); included so it is not lost",
        "_unjudged": True,
    }
