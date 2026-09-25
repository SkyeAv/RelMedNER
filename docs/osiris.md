# OSIRIS

`bigbio/osiris` (cc-by-3.0): the OSIRIS corpus of genetic-variant mentions in PubMed abstracts,
105 documents with gold `gene` (689) and `variant` (541) mentions, 1,230 entities measured over
the full single `osiris_bigbio_kb` parquet file (refs/convert/parquet, laptop 2026-09-24). The
hub repo is script-only, so the declared read path is the bigbio/ehr_rel pattern: one
`hf_parquet` entry over the canonical config. Every row carries exactly two passages (title +
abstract, inter-passage gap 1 char on 105/105) with global char offsets, so `OsirisScript`
reconstructs the document text by placing passages at their declared offsets and gap-filling,
which keeps both title-relative and abstract-relative entity offsets addressable. 1,228/1,230
spans verify end-EXCLUSIVE (text[start:end] == surface); the remaining 2 are sloppy-source
trailing-space offsets that the shared char bridge clamps or drops. Every entity is single-part
(1,230/1,230). The `normalized` dbSNP/Entrez ids are provenance, not training signal (the
pipeline emits no entity normalization), so they are read and dropped.

Trust-gold stance (LinnaeusScript precedent): the `gene`/`variant` labels are gold, so
snap-to-token spans go through NO fullmap re-resolution; each surviving span ships under its own
mapped biolink class (`gene` -> Gene, `variant` -> SequenceVariant), and an entity type with no
honest biolink target drops instead of a guessed label. The corpus ships no relations (empty on
every row) and none are fabricated: the declared outputs are [entities] only. Gold spans are
tiered gold in `docs/weighting.md` (human annotations).

## Related

- [README index](../README.md) -- the ingest-table row for this corpus and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates these rows flow through.
- [Weighting](weighting.md) -- the tier prior this corpus is declared at.
