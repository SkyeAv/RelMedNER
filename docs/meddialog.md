# MedDialog

`OpenMed/MedDialog` (apache-2.0): 251,731 patient-doctor consultations total; the declared train
split carries 226,557 rows (datasets-server size endpoint), and the validation split stays
undeclared. The declared column is `doctor_response`, the clinician reply (the
`dialogue_context` column is empty on every sampled row). Measured over the first 300 train
rows (wenceslaus 2026-09-24): mean 93.7 tokens per reply (median 83), 0 empty, 285 of 300 docs
produce mentions at 6.19 mentions per doc, 84 relations. Consultation replies are informal
clinical register, so the corpus is tiered silver in `docs/weighting.md` (distant labels).

## Related

- [README index](../README.md) -- the ingest-table row(s) for this corpus and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates these rows flow through.
- [Weighting](weighting.md) -- the tier prior this corpus is declared at.
