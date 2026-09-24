# GAD (bigbio/gad)

Dataset-format notes (GadBlurbScript): bigbio/gad is the Genetic Association Database
sentence corpus (cc-by-4.0, not gated). The repo's own builder is a loading script
(`gad.py`) that the installed `datasets` refuses ("Dataset scripts are no longer
supported"), so rows stream from the hub's auto parquet conversion branch
(`refs/convert/parquet`) through the `hf_parquet` source. Each split is a single
auto-convert shard, so each entry declares `file: gad_blurb_bigbio_text/<split>/0000.parquet`
(checked against the datasets-server `parquet` listing, 2026-09-23). The declared subset is `gad_blurb_bigbio_text` only, all three
splits: train 4,261 / validation 535 / test 534 rows (source: datasets-server
`splits` listing plus a full-census probe over every parquet file, wenceslaus
2026-09-23). A row is (`text`, `labels`): `text` is a short anonymized sentence
(7-81 tokens, median 25) carrying literal `@GENE$` / `@DISEASE$` placeholders that
ship as-is; `labels` is a one-element list of `"1"` (gene-disease association
reported) or `"0"`, ~52.6% `"1"` corpus-wide, no third label, no nulls. Measured
over the first 200 declared-stream rows per split: 100% emit exactly one
classification, zero drops. The repo's other 21 subset-splits (10 cross-validation
folds in two variants) re-partition the same 5,122 unique texts (111,930 rows
total); declaring them would upweight GAD about 22x in the mix, so they stay out.
Probe invocation: `probe.py --declared 'bigbio/gad:gad_blurb_bigbio_text/train/0000.parquet'
--limit 200 --script GadBlurbScript --outputs classifications --text-column text`
(plus the full-census throwaway probe kept at wenceslaus:~/probe_gad.py).

## Related

- [README index](../README.md) -- the ingest-table rows for the three GadBlurbScript entries and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates the GadBlurbScript rows flow through.
- [EHR-rel](ehr-rel.md) -- the `hf_parquet` source kind this corpus loads through.
- [Weighting](weighting.md) -- the silver-tier prior the three splits share.
