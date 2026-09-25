# PubMedQA

`bigbio/pubmed_qa` (mit): yes/no/maybe biomedical question answering over PubMed abstract
contexts. Five declared `hf_parquet` ingests off one hub repo (one shared weight slot): the
expert-labeled folds (`pubmed_qa_labeled_fold{i}_source`, train declared, 450 rows each
measured on 2026-09-24, labels `yes`/`no`/`maybe` at 249/152/49 on fold 0), gold tier.

`PubmedQaScript` composes the example text as the question followed by its evidence-context
sentences (the CONTEXTS list), and emits exactly one `Classification` per row over the fixed
`['no', 'yes', 'maybe']` vocabulary. Labels outside the vocabulary (bools, ints, other
strings, None) ship as text-only examples: skip-don't-coerce, never guess a label
(SuperGlueMultiRCScript precedent).

The 200k-row artificial split stays undeclared on purpose: the repo-id row key may carry only
one weight (`weights_by_source`, models.py), and the artificial split's weak heuristic labels
would need silver, conflicting with the folds' gold. If it is ever wanted, it needs its own
mirror repo id, not a conflicting weight.

The knowledgator/PubmedQA hub mirror was probed and rejected: no license metadata on the card,
and the bigbio mirror already covers the corpus under mit. Declared outputs are
[classifications] only; no entities or relations are mined here.

## Related

- [README index](../README.md) -- the ingest-table row for this corpus and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates these rows flow through.
- [Weighting](weighting.md) -- the tier prior this corpus is declared at.
