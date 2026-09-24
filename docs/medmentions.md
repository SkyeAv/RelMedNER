# MedMentions ST21pv

`chanzuckerberg/MedMentions` ST21pv (CC0): 4,392 PubMed documents with 203,282 gold UMLS-linked
entity spans over exactly 21 semantic types. The declared read path is the bc5cdr pattern: three
local avro ingests (train 2,635 docs / 122,241 spans, dev 878 / 40,884, test 879 / 40,157), built
out-of-band from the PubTator corpus with document-absolute offsets over title + newline +
abstract; the converter's census slice-matched 203,282 of 203,282 annotations. The zameji hub
mirror of this corpus was probed first and REJECTED: its token-index re-encoding left 33.6
percent of spans unrecoverable on any shift 0-5 (wenceslaus 2026-09-24). `MedMentionsScript`
trusts each span's own UMLS CUI as the curie (trust-gold, the bc5cdr MESH pattern), maps all 21
semantic types through LABEL_MAP (T005 Virus, T007 Bacterium, T017/T022/T031 AnatomicalEntity,
T033 ClinicalFinding, T037 Disease, T038 BiologicalProcess, T058 ClinicalIntervention, T062
Study, T074 Device, T082 GeographicLocation, T091 Activity, T092 Agent, T097/T098
PopulationOfIndividualOrganisms, T103 ChemicalEntity, T168 Food, T170 InformationContentEntity,
T201 ClinicalAttribute, T204 OrganismTaxon), and ships no relations (the corpus annotates
entities only). Gold spans are tiered gold in `docs/weighting.md`.

## Related

- [README index](../README.md) -- the ingest-table row(s) for this corpus and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates these rows flow through.
- [Weighting](weighting.md) -- the tier prior this corpus is declared at.
