# bigbio/ehr_rel

EHR concept relatedness (Schulz et al., COLING 2020, apache-2.0): 3,741 SNOMED concept pairs
sampled from real EHRs, rated 0-3 for relatedness by human raters (5 raters in batch `a`,
3 raters in batch `b`; the hub ships the mean as a string). Four declared subsets over one
repo: `ehr_rel_a_source` (111 rows), `ehr_rel_b_source` (3,630), `ehr_rel_source` (3,741),
`ehr_rel_bigbio_pairs` (3,741) -- all counts from the probe `PROBE_ROWS` receipt on the
datasets-server `splits` endpoint.

Why a separate source kind: the repo is a script-era hub dataset whose `main` branch carries
only loading-script files (`ehr_rel.py`, `bigbiohub.py`), which current `datasets` refuses
outright. The hub's auto-converted `refs/convert/parquet` branch has one parquet file per
config, and a whole-revision load fails because the four configs publish different schemas
(mixed-schema union). Each entry therefore reads exactly one per-config file through the
`hf_parquet` source kind, which pins the revision in the `hf://datasets/{dataset}@refs/convert/parquet/{file}`
URL. The route is proved end to end by the wenceslaus probes and the live-row test.

## Row mapping

`EhrRelScript` is column-agnostic: all four entries project `(surface1, surface2, rating)` --
the three `*_source` subsets as `snomed_label_1`, `snomed_label_2`, `mean_rating`, the pairs
subset as `text_1`, `text_2`, `label`. The emitted text is exactly the two surfaces joined by
one space, so every relation field value is a substring of text (gliner2's validator). Each
shipping row carries one `Relation` under the biolink member `related_to` with
`evidence="asserted"` and `negated=False`. Trust-gold stance like `SuperGlueRecordScript`: the
corpus's own concept labels ship as relation field values and nothing is re-resolved, because
there is no prose for fullmap to resolve against; no `LABEL_MAP` either, the predicate is fixed.

The subsets overlap by construction: the pair set of `ehr_rel_source` equals
`ehr_rel_bigbio_pairs` exactly (3,717 distinct pairs), `a + b` covers the same set, and 24
pairs appear inside both `a` and `b`. Every pair therefore enters the stream up to four times;
`RunConfig`'s default `NEAR` dedup collapses the duplicates to the 3,717 unique examples at
export.

## Measured gates (wenceslaus, 2026-09-22)

- Ship bar `mean_rating >= 1.0`: 2,535 of 3,741 ship in the source/pairs rendering (67.8%),
  a_source 21 of 111 (18.9%), b_source 2,514 of 3,630 (69.3%) -- `PROBE_DISPATCH_*` receipts
  from `probe.py --declared`, matching the full-split census exactly. Ratings below the bar
  drop (1,206 rows in the source/pairs rendering): the pipeline never emits `negated=True`, so
  a weak-mean pair can neither ship as a positive nor as a negation.
- Rating grids: `a` on 0.2 steps (mean of 5), `b` on 1/3 steps (mean of 3); both parse through
  one float path that rejects bools and unparseable strings (0 measured).
- Measured zeros with guards kept: 0 blank surfaces, 0 case-insensitive self-loops, 0
  unparseable ratings, 0 empty rows (the declared `drop_empty` filter is a no-op today).
- Text shape: median 44 chars, max 184; no context sentence exists in the corpus.
- Deliberate omissions: the SNOMED id columns and `CUI_1`/`CUI_2` never stream (provenance
  only; `CUI_1` is empty on 252 rows and `CUI_2` on 250), and `document_id` is a deidentified
  hash unusable as text.

## Related

- [README index](../README.md) -- the ingest-table rows for the four subsets and the full docs index
- [yaml config](yaml-config.md) -- the `hf_parquet` source kind these entries use
- [ingests](ingests.md) -- the resolution chain and gates
