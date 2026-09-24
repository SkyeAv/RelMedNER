# TrialPanorama database

`TrialPanorama/TrialPanorama-database` ships 11 subsets (~27.4M rows total) keyed by
`study_id`. This ingest takes the `studies` subset (1,332,141 rows) and mines the
`abstract` column, the only long free text in the database. The structured subsets carry
MeSH / MedDRA / RxNorm identifiers the pipeline cannot yet translate into biolink classes,
so those subsets are deferred until an ontology-mapping pass exists. Empty or null
abstracts degrade to bare examples and are dropped by the declared-outputs filter,
matching fullmap behavior on textless rows.

Measured row count: 1,332,141 (source: HF datasets-server info endpoint, split `all`).

Mined rows flow through the same resolution chain and quality gates as the
`anthonyyazdaniml/gliner-biomed-curated-corpus` fullmap entries above; the miner itself
is documented under [How fullmap mining works](fullmap-mining.md) and not
re-described here.

## Related

- [README index](../README.md) -- the ingest-table row and the full docs index
- [fullmap mining](fullmap-mining.md) -- the miner its abstracts flow through
- [weighting](weighting.md) -- the tier and trust prior for its row key
