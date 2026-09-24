# FewRel

The six `source: local` FewRel ingests. The hub repo `thunlp/few_rel` is a loading-script
dataset and cannot be loaded under this repo's `datasets>=5.0.1` pin, so the raw JSON lives on
the thunlp/FewRel GitHub (`data/*.json`, dict keyed by relation id) and
`scripts/build_fewrel_avro.py` converts it to six avro containers, one per split, built
out-of-band on wenceslaus only:

    cd ~/Code/RelMedNER-worktrees/few-rel && /home/sgoetz/bin/uv run python scripts/build_fewrel_avro.py

The containers land at `src/relmedner/data/fewrel/<split>.avro`, are gitignored, and are
rsync-excluded, so they exist only where built. `LocalAvroDataStream` ships each whole record
to the declared script; there is no `columns_out` projection. The record contract is
`{"tokens": [str], "label": str, "h_indices": [[int]], "t_indices": [[int]]}`.

## Dataset-format notes (`FewRelScript`)

Delivery: the hub repo `thunlp/few_rel` is a loading-script dataset, unusable under the repo's
`datasets>=5.0.1` pin, so the corpus arrives as raw JSON from
`https://raw.githubusercontent.com/thunlp/FewRel/master/data/` (six split files plus
`pid2name.json`, 744 entries, ~22.5MB total) and is converted out-of-band by
`scripts/build_fewrel_avro.py` on wenceslaus into six avro containers
(`src/relmedner/data/fewrel/<split>.avro`). Build receipts (2026-09-23): train_wiki 44,800
records / 7.0MB, val_wiki 11,200 / 1.8MB, val_nyt 2,500 / 560K, val_semeval 8,851 / 1.2MB,
val_pubmed 1,000 / 252K, pubmed_unsupervised 2,500 / 584K. Total 70,851 records, ~11.8MB.
Measured row census (full split, raw GitHub JSON): train_wiki 44,800 (64 relations x 700),
val_wiki 11,200 (16 x 700), val_nyt 2,500 (25 x 100), val_semeval 8,851 (17 labels),
val_pubmed 1,000 (10 x 100), pubmed_unsupervised 2,500 (no relation label).

Span encoding: `h_indices` / `t_indices` are lists of lists of python ints; a run is one
contiguous token span, and a multi-run entity is the same entity mentioned more than once in
one sentence. Measured over the full corpus: 0 out-of-bounds indices, 0 unsorted runs, 0
non-int elements. Multi-run entity mentions on train_wiki: 1-run 87,903, 2-run 1,624, 3-run
68, 4-run 5 of 89,600. Each contiguous run emits one mention under NamedThing.

Surfaces are derived from tokens+indices and are therefore always verbatim substrings of the
re-joined token stream (`ScriptUtils.join_tokens`). The raw `h_text`/`t_text` fields are
detokenized lowercase surfaces and are never used: 91.8% of wiki entity surfaces
case-mismatch the token slice, 6.1% mismatch even case-insensitively, so the converter omits
them from the container entirely. The Wikidata Q-id / UMLS C-id / SemEval type fields are
deliberately unused as labels.

Trust-gold stance: the spans are dataset gold, so nothing is re-resolved through fullmap.
Every gold mention rides under the single catch-all biolink class NamedThing (the same
philosophy as `CtkpInterventionsScript` and `SuperGlueRecordScript`); the Wikidata P-id,
UMLS, and SemEval label vocabularies would not survive a biomedical re-resolution anyway.

Relation resolution: the converter resolves each row's gold relation label through the pure
helpers in `relmedner.scripts.fewrel` -- strip a SemEval participant suffix (`(e1,e2)` /
`(e2,e1)`), resolve a bare `P<digits>` id through `pid2name[pid][0]`, then
`ScriptUtils.normalize_predicate` (biolink-shaped snake_case; native names kept when no
biolink member matches, the `SentenceRexScript` precedent). Measured: 0 of 97 distinct P-ids
are missing from pid2name and 0 normalized names collide; val_pubmed's 10 labels are already
snake_case native (0 are biolink members) and val_semeval's 17 labels carry the suffix on
9 + 8 labels. An unresolvable P-id records an empty `label` and the script ships entities
only; pubmed_unsupervised's 2,500 rows carry no relation by construction and ship entities
only.

Drop rules (skip-don't-coerce): a malformed index run (wrong container shape, non-int or bool
element, empty run, any index outside `0 <= i < len(tokens)`, or non-contiguous indices)
drops that run alone while sibling runs survive; empty tokens yield the empty text-only
example the declared-outputs filter drops; a self-loop (head and tail first surviving run
equal case-insensitively) drops the relation but keeps its entities -- 9 measured rows, all
val_semeval, 0 elsewhere; head/tail token-index overlap is 0 rows. Measured dispatch yield
through the production stream values (2026-09-23, every row of every split): train_wiki
44,800/44,800 emit entities+relations, val_wiki 11,200/11,200, val_nyt 2,500/2,500,
val_semeval 8,851 rows emit with 8,842 relations + 9 entities-only (the self-loops),
val_pubmed 1,000/1,000, pubmed_unsupervised 2,500/2,500 entities-only. rows_out equals
rows_in for every split.

Declared outputs are `[entities, relations]` for all six entries; pubmed_unsupervised rows
and the 9 val_semeval self-loops ship fewer shapes, which the permitted-shapes contract
allows.

## Related

- [README index](../README.md) -- the ingest-table rows for the six splits and the full docs index.
- [bc5cdr](bc5cdr.md) -- the other out-of-band local-avro delivery.
- [output.md](output.md) -- the records the container rows emit.
