# Augmented clinical notes

`AGBonnet/augmented-clinical-notes` (MIT): 30,000 clinical-note paragraphs (a streaming count of
the single `augmented_notes_30K.jsonl` file, wenceslaus 2026-09-24), an EHR-shaped unlabeled text
corpus mined through the shared fullmap path at the repo defaults (max_ngram=6, taxon=9606).
Measured over the first 200 rows: mean 343.8 tokens per note (median 343, min 341, max 360), 0
empty notes, every one of the 200 notes produced mentions, 21.58 mentions per note, and 214
relations over 200 notes. The rows are LLM-augmented (the `note` column paraphrases a source
document), so the corpus is tiered silver in `docs/weighting.md`: mining scores the fiction, and
the labels are distant, not gold. The declared column is `note` (not `full_note`), the
paragraph-length clinical note itself.

## Related

- [README index](../README.md) -- the ingest-table row(s) for this corpus and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates these rows flow through.
- [Weighting](weighting.md) -- the tier prior this corpus is declared at.
