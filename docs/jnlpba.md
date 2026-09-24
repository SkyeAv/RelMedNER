# JNLPBA

`commanderstrife/jnlpba` (apache-2.0): the classic JNLPBA biomedical NER corpus (GENIA abstracts,
IOB gold spans for protein, DNA, RNA, cell line, cell type and species-flavored labels). The hub
repo ships an unsupported `jnlpba.py` loader script, but its `refs/convert/parquet` branch carries
the splits as parquet files, so the declared read path is the bigbio/ehr_rel pattern: one
`hf_parquet` file entry per split. The raw parquet loses the ClassLabel name table, so
`JnlpbaScript` carries the fixed 19-tag vocabulary and decodes the int `ner_tags` itself; an
out-of-vocabulary index decodes as background, never as a guessed span. The corpus's complete
vocabulary (9 labels) is mapped: chemical -> ChemicalEntity, gene -> Gene, protein -> Protein,
disease -> Disease, dna/rna -> NucleicAcidEntity, cell_line -> CellLine, cell_type -> Cell,
species -> OrganismTaxon. Measured on the parquet refs (wenceslaus 2026-09-24): train 37,094
rows / 985,102 tokens / 102,602 gold spans, validation 7,714 rows / 202,078 tokens / 17,324 gold
spans. The repo's test parquet is byte-identical to its validation parquet (md5
e446a9191dbf474bdb79f37b08183fb3 on both), so it is deliberately undeclared: declaring it would
double-stream 7,714 duplicate rows (the agentlans/json-extraction precedent). Gold IOB spans are
tiered gold in `docs/weighting.md` (human annotations).

## Related

- [README index](../README.md) -- the ingest-table row(s) for this corpus and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates these rows flow through.
- [Weighting](weighting.md) -- the tier prior this corpus is declared at.
