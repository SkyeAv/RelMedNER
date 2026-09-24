# ClinicalTrials.gov summaries + eligibility

`rjac/clinicaltrials.gov-summary_and_eligibility` (MIT; upstream ClinicalTrials.gov registry data
is public domain): 3,002 trial records pairing structured registry fields (nct id, status, title,
summary, dates) with the eligibility criteria prose, the relational trial-database shape. One
fullmap ingest mines `eligibility` (the densest clinical text; a second text column would collide
on the fullmap entry key, so `brief_summary` stays undeclared). Measured over the first 300
train rows (wenceslaus 2026-09-24): 300 of 300 docs with mentions at 17.87 mentions per doc, 295
relations, 0 empty criteria strings. Criteria prose is tiered silver in `docs/weighting.md`
(distant labels, unverified provenance).

## Related

- [README index](../README.md) -- the ingest-table row(s) for this corpus and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates these rows flow through.
- [Weighting](weighting.md) -- the tier prior this corpus is declared at.
