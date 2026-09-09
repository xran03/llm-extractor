# Batch API and review

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

### Choosing between them

They are not competing implementations of one thing, which is why there is no
single `--fast` switch. Live decides *how you spend the wait*; batch decides
*whether you wait at all*. That makes the choice a question about the corpus and
the deadline:

| Situation | Use | Why |
|---|---|---|
| Iterating on a template or prompt | live | You need to see records to know if the schema is right. |
| A few hundred documents, needed today | live | Bounded wait, results in the same session, cache warms as it goes. |
| Thousands of documents, needed tomorrow | `batch submit` | Two calls, then close the laptop; no rate-limit pressure. |
| Rate limit is the binding constraint | `batch submit` | Queued work is scheduled by the gateway, not by your retry loop. |
| The gateway has no batch support | live | `batch preflight` tells you before you build a corpus. |

Two practical cautions. Batch trades latency for throughput, so a 24-hour
completion window is a real 24 hours — it is the wrong tool for anything
interactive. And its answers arrive detached from their questions, which is why
`submit` writes a manifest; a queued run cannot be reassembled without it, so
the output directory matters more than in a live run.

Both paths share the extraction code, the template, the grounding checks and the
cache, so a corpus queued today and a document run live tomorrow produce the
same columns and the same verdicts. Switching is a change of command, not a
change of pipeline.

### Reviewing what was extracted

Grounding proves a record's quoted span exists and that its digits appear in
that span. That catches invention, but not **misattribution**: a value can be
real, its span real, and the row still wrong because the number was reported
under a different assay, group or timepoint than the row names.

`batch review` queues a second opinion over records that already exist, asking
three questions per record and only these three:

| Column | Question |
|---|---|
| `_review_value` | Does the document report this value for *this* row's subject, assay, group and timepoint? |
| `_review_unit` | Is the unit the one the document states, rather than a converted or assumed one? |
| `_review_row` | Do the fields describe **one** measurement, or has the row been stitched together from several? |

`_review_row` is the one the deterministic checks cannot reach, and it is why
the whole row is shown to the reviewer rather than the value alone.

```bash
llm-extract run -i ./docs -o ./out --template immunogenicity
llm-extract batch review -i ./docs -o ./out        # queue the check
llm-extract batch fetch -o ./out --wait            # merge verdicts into the CSV
```

Three things make the result trustworthy rather than decorative:

- **Records are judged on the text they came from.** Each record is routed to
  the chunk its evidence span belongs to, so a record from page 40 is reviewed
  against page 40 rather than a truncated head of the document.
- **Abstention is allowed.** A reviewer that cannot settle a question from the
  text answers `null`, not `false`. An unsupported rejection costs as much as
  the error it claims to find, so empty and rejected are different columns.
- **Verdicts cannot slide.** The answer names records by the index its request
  handed out, and the manifest maps that back to a row. A reply that comes back
  short, reordered, or citing an index nobody asked about annotates only what it
  legitimately covers.

Review defaults to a different model family from extraction (`review_model`,
`claude-fable-5` on the AI Model Hub), because a second opinion from the model
that wrote the answer mostly restates it. `review.json` carries the run totals,
and the four columns are only added once a review has actually run — an
un-reviewed run does not ship empty columns implying a check nobody performed.

---


---

[Back to the README](../README.md)


