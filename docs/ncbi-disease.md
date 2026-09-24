# NCBI Disease

`ncbi/ncbi_disease` (public domain, US government work): the NCBI disease corpus (Dogan et al.
2014), 793 PubMed abstracts sentence-split with dual-annotator gold disease mentions normalized
to MeSH / OMIM upstream. The hub repo ships an unsupported `ncbi_disease.py` loader script, but
its `refs/convert/parquet` branch carries the splits as parquet files, so the declared read path
is the bigbio/ehr_rel pattern: one `hf_parquet` file entry per split. The raw parquet loses the
ClassLabel name table, so `NcbiDiseaseScript` carries the fixed 3-tag vocabulary (O, B-DISEASE,
I-DISEASE) and decodes the int `ner_tags` itself; an out-of-vocabulary index decodes as
background, never as a guessed span, and an orphaned I- run promotes to a single span (the
`iob_spans` contract, pinned in tests). The corpus's complete vocabulary (one closed Disease
class) maps disease -> Disease. The corpus ships no relations; the 22 gazetteer relations in the
declared-probe sample are derived signal, the gliner_biomed/jnlpba convention. Measured over the
full parquet splits on wenceslaus 2026-09-24 (probe: `probe_parquet.py`-style read of the refs
branch): train 5,433 / validation 924 / test 941 sentences, tag census O 169,361 / B-DISEASE
6,892 / I-DISEASE 8,299, max sentence 123 tokens. Declared-ingest probe over the first 500 train
rows: 275 rows emitting (55.0%), 477 mentions, 22 gazetteer relations. Gold IOB spans are tiered
gold in `docs/weighting.md` (human annotations).

## Related

- [README index](../README.md) -- the ingest-table row for this corpus and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates these rows flow through.
- [Weighting](weighting.md) -- the tier prior this corpus is declared at.
