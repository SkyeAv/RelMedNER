# DocRED

`thunlp/docred` is the document-level relation-extraction benchmark built from Wikipedia and
Wikidata: each row is one tokenized document with gold entity clusters and gold relations
between them. Four splits are declared as four `hf_json` ingests, one per raw
`data/<split>.json.gz` file, because the hub repo ships no parquet conversion at all -- only
the raw gzipped JSON files, `data/rel_info.json.gz`, and a loading script (`docred.py`), so
the plain `hf` source cannot stream it. The `hf_json` route (`load_dataset("json",
data_files="hf://datasets/thunlp/docred/data/<file>.json.gz")`) transparently decompresses
`.gz` and reads all four files; this route is measured, not assumed (probe receipts
`DOC_PROBE_ROUTE` on wenceslaus, 2026-09-22).

All numbers below are full-split measurements from that probe round (logs
`~/docred_probe.log`, `~/docred_probe2.log`, `~/docred_relprobe.log` on wenceslaus). To
re-run them: `load_dataset("json", data_files=...)` over each file and replay
`DocredScript`'s drop rules (the throwaway probes in those logs show how).

## Row shape and span convention

One row carries `title`, `sents` (a list of tokenized sentences), `vertexSet` (entity
clusters: each cluster a list of `{name, pos, sent_id, type}` mentions), and `labels` (gold
relations keyed `h`/`t`/`r`/`evidence` -- NOT `head`/`tail`/`relation`). `test.json.gz` rows
carry no `labels` key at all, so the test entry declares `outputs: [entities]` and ships
entity-only examples; 26 of 3,053 / 293 of 101,873 / 13 of 998 labeled-split docs also have
no labels and ship entities only under the declared-outputs contract.

Mention `pos` is sentence-relative and END-EXCLUSIVE: 77,515 exclusive vs 145 inclusive
surface matches on the full train_annotated split (the 145 are single-token mentions that
satisfy both readings). `DocredScript` flattens the sentences into one token stream, converts
each mention to a global end-inclusive token span, and emits the TOKEN-SPAN JOIN as the
surface -- never the annotator `name` string, which disagrees with the token join on
2.3-5.3% of mentions purely through punctuation spacing from tokenization ("Worker-Peasant"
vs `['Worker', '-', 'Peasant']`; "People's" vs `['People', "'", 's']`). That mismatch is
annotation style, not corruption, so it never drives a drop; emitting the token join also
guarantees every surface occurs in the emitted text by construction.

Out-of-bounds mentions (measured `b > len(sentence)`): 149 of 79,481 in train_annotated
(0.19%), 14,319 of 2,558,350 in train_distant (0.56%), 55 of 26,141 in dev (0.21%), 36 of
26,704 in test (0.13%). Degenerate, non-int, bool-typed, wrong-arity, or bad-`sent_id` spans
drop the same way (skip-don't-coerce); every rule has a negative test with a surviving
well-formed sibling.

## Entities: trust gold, six types

The corpus is general-domain Wikipedia text, so `DocredScript` takes the trust-gold stance
(same philosophy as `SuperGlueRecordScript`): nothing is re-resolved through fullmap, and
mentions ride under biolink classes derived from the corpus's own closed six-type
vocabulary: PER -> `Human`, ORG -> `Agent`, LOC -> `GeographicLocation`, MISC -> `NamedThing`
(import-validated). TIME and NUM have no honest biolink class and stay raw TitleCase
(`Time`, `Num`) -- the deliberate-omission stance, matching the unmappable MeSH headings in
`PubmedAbstractsScript`; a wrong bucket would silently mislabel training data.

## Relations: the frozen Wikidata map

Gold relations name Wikidata properties (`P17`, `P131`, ...). The dataset's own
`data/rel_info.json.gz` maps all 96 to English names; that table is frozen verbatim into
`DocredScript.RELATION_MAP` (96 entries, provenance comment in the class), because the file
is a top-level dict the json builder cannot project as rows. Each name resolves through
`ScriptUtils.resolve_predicate`, which keeps native snake_case for the names that are not
biolink predicates (the large majority, the `SentenceRexScript` precedent: native predicates
train zero-shot breadth).

One `Relation` ships per gold label, head and tail citing each endpoint cluster's FIRST valid
mention surface (DocRED relations hold between clusters, and the earliest surviving mention
is the canonical reference). The measured label census is clean: zero self-loops, zero bad
keys, zero out-of-range indices, zero unmapped P-ids, and -- replaying the drop rules over
the full cached tables -- zero relations lost to an endpoint cluster with no surviving
mention, across all 38,180 / 1,505,638 / 12,275 gold labels (train_annotated /
train_distant / dev). The guards exist for drift, not for the observed data.

## Shipment evidence

Declared-entry probes through the production path (`probe.py --declared
'thunlp/docred:data/<file>.json.gz' --limit 2000 --script DocredScript --outputs ...`):
every row emits in all four splits (100.0%): train_annotated 2,000/2,000 (2,000 entities +
1,981 relations rows, 42,533 mentions, 25,037 relations), train_distant 2,000/2,000 (2,000 +
1,995, 42,291 mentions, 30,068 relations), dev 998/998 (998 + 985, 21,377 mentions, 12,275
relations), test 1,000/1,000 entities-only (21,559 mentions). Dev and test were probed over
their FULL splits; the labeled splits average ~12.5 gold relations per document.

## Related

- [README index](../README.md) -- the ingest-table rows for the four splits and the full docs index
- [ingests](ingests.md) -- the shared resolution chain and gates
- [output](output.md) -- the records the four splits emit
