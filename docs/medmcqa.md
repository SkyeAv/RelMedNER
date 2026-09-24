# MedMCQA

`openlifescienceai/medmcqa` (apache-2.0): medical entrance-exam questions; the train split alone
is 182,822 rows (validation 4,183, test 6,150, datasets-server size endpoint, undeclared). The
declared column is `exp`, the expert explanation: measured over the first 300 train rows, 36
rows (12%) carry no explanation and drop on the declared `drop_empty` filter, 264 ship, 234
docs produce mentions at 8.1 mentions per doc, and 158 relations over the 300-row sample.
Explanations are short (mean 532 chars), so the yield per row is the lowest of the fullmap
mining bases; the volume is what makes the corpus useful. Exam prose is tiered silver in
`docs/weighting.md` (distant labels, unverified provenance).

## Related

- [README index](../README.md) -- the ingest-table row(s) for this corpus and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates these rows flow through.
- [Weighting](weighting.md) -- the tier prior this corpus is declared at.
