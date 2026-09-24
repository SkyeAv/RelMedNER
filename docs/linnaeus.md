# LINNAEUS

`bigbio/linnaeus` (cc-by-4.0): the LINNAEUS species corpus, 100 PMC full-text articles with gold
species mentions and NCBI taxon normalization. The hub repo is script-only, so the declared read
path is the bigbio/ehr_rel pattern over `refs/convert/parquet`: one `hf_parquet` entry for the
canonical `linnaeus_bigbio_kb` config. Measured over the full single parquet file on wenceslaus
2026-09-24: 95 documents, 4,259 gold species mentions, every span verified end-EXCLUSIVE
(4,259/4,259 text[start:end] == surface), every entity single-part, exactly one gapless passage
per row. The `_source` and `filtered_*` configs re-render the same documents and are deliberately
undeclared (the agentlans/json-extraction double-stream precedent). `LinnaeusScript` runs the
shared char bridge (out-of-bounds and degenerate spans drop, whitespace slop clamps) and ships the
re-joined token stream, but the snap-to-token spans go through NO re-resolution: the `species`
labels are gold, and the shared fullmap-first chain measurably corrupts them (the surface
"patients" fullmap-maps to UMLS C0030705 "Patients" -> Cohort on the mounted bundle, wenceslaus
2026-09-24). This is the trust-gold stance of CtkpInterventionsScript / SuperGlueRecordScript:
each surviving span ships under its own mapped biolink class (species -> OrganismTaxon), and an
entity type with no honest biolink target drops instead of a guessed label. The entities'
`normalized` taxon ids are provenance, not training signal (the pipeline emits no entity
normalization), so they are read and dropped. The corpus ships no relations and none are
fabricated: the declared outputs are [entities] only. Declared-ingest probe over the full split:
95/95 rows emitting (100.0%), 924 deduped mention groups (group_entities dedups surfaces per
label), 0 relations. Gold spans are tiered gold in `docs/weighting.md` (human annotations).

## Related

- [README index](../README.md) -- the ingest-table row for this corpus and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates these rows flow through.
- [Weighting](weighting.md) -- the tier prior this corpus is declared at.
