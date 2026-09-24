# Text-Clinical-Records

`hackint0sh/Text-Clinical-Records` (MIT): 31,489 clinical-record texts, the EHR-shaped prose
register. Measured over the first 300 train rows (wenceslaus 2026-09-24): mean 842 chars (median
513), 78 of 300 rows empty (dropped on the declared `drop_empty` filter), 195 of the 222
surviving docs produce mentions at 9.57 mentions per doc, 61 relations. The kjappelbaum
chemnlp-chemdner candidate was rejected before probing: no license declared on its card (unknown
= reject). Clinical-record prose is tiered silver in `docs/weighting.md` (distant labels).

## Related

- [README index](../README.md) -- the ingest-table row(s) for this corpus and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates these rows flow through.
- [Weighting](weighting.md) -- the tier prior this corpus is declared at.
