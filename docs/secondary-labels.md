# Secondary labels and multi-class fan-out

Two mechanisms give one span more than one label. Both ride the shared `ScriptUtils.group_entities`
grouping, so every script (and fullmap mining) emits them without per-script wiring; both are
native gliner2 supervision (per-query boundary targets, overlapping spans train fine).

## Morphological secondary labels

`secondary_labels()` (utils) maps a mention SURFACE to zero or more secondary classes the biolink
categories cannot express (pharmacological stem classes, disease morphology, cells, procedures).
Rules live in constants (`SECONDARY_SUFFIX_LABELS` end-anchored suffixes, `SECONDARY_TAIL_LABELS`
exact last-token / last-two-word phrases, `SECONDARY_SUFFIX_STOP` stopwords) and are
membership-only (`str.endswith` + dict lookups), because the measured 37-regex loop costs
~46us/span vs ~1.8us/span this way (203,282 MedMentions spans, wenceslaus 2026-09-25).

Every rule was validated against MedMentions ST21pv gold before landing: the kept table fires at
>= 0.90 dominant-gold-type concentration (INN drug stems mostly 1.000: -nib, -statin, -cycline,
-dipine, -prazole, -cillin, -floxacin, -olol, -mab; disease -itis 0.991, -oma 0.960; procedures
-ectomy 0.992, -plasty 0.988; cells -cyte 0.985). Measured rejects stay OFF: generic -ase
('disease'/'database'/'Embase' all fire it, 0.705 -- resurrect as an explicit enzyme-stem
whitelist if ever needed), -algia 0.714, -cept 0.700 ('percept'). Structural requirements every
rule obeys: end-anchored LAST-token matching ('melanoma cells' fires no -oma; 'insulin
resistance' fires no InsulinDrug), plural variants as explicit keys ('lymphocytes' -> cytes), and
measured stopwords (trauma/comma for -oma, myostatin for -statin, glycemia/academia for -emia,
caenorhabditis for -itis, encephalitides for -tide). Tests pin all of it
(tests/test_secondary_labels.py).

Secondary labels are never biolink classes (a test enforces it), so they never merge with primary
labels or take biolink descriptions, and they carry NO description at all: the fullmap CURIE
evidences the ontology class the surface resolved to, not the morphological one, so the evidence
string stays on the primary label. Shipping it on the secondary label would put per-row
provenance inside a gliner2 label prompt, which the end-to-end contract in `tests/test_outputs.py`
forbids for every non-biolink label.

## Multi-class fan-out (one entity, two vocabularies)

`_fullmap_best` no longer keeps one best-ranked row per term: `_fanout_rows` keeps the winner
plus one identity-agreeing cross-category row (cap 2), where identity agreement means the
preferred names match case-insensitively. This is the measured seam in the fullmap redb: 5,400 of
the probed CTKP/MedMentions surfaces carry same-name rows in distinct biolink categories, 98% of
those pairs cross namespaces (CHEBI `SmallMolecule` + UMLS `Protein` for one "semaglutide";
UMLS `Protein` + CHEBI `SmallMolecule` for "corticotropin"). The KP-side analog (CTKP
multi-category rows per matched_text) never fires: the 20260920 snapshot gives every
`matched_text` exactly one category, measured over all 1,020,749 records -- see
docs/ctkp-interventions.md. Rows without name agreement (an 'insulin measurement' Procedure row
next to 'Insulin' Protein) never fan out; neither do ancestor-descendant pairs: ancestor
propagation was measured and rejected (every mention gains a structural parent like
MolecularEntity, ~100% correlated, zero disambiguation).

## CURIE code-structure facts (measured, redb rows 2026jul22)

Prefix alone is deterministic for whole vocabularies: `NCBIGene` -> Gene 1.000,
`PUBCHEM.COMPOUND` -> SmallMolecule 0.996, `NCBITaxon` -> OrganismTaxon 1.000,
`UniProtKB` -> Protein 1.000, `RXCUI` -> Drug 1.000, `MONDO` -> Disease 1.000,
`HP` -> PhenotypicFeature 1.000, `ComplexPortal` -> MacromolecularComplex 1.000,
`FB`/`MGI`/`ZFIN` -> Gene 1.000, `PR` -> Protein, `HMDB` -> SmallMolecule 0.998,
`GTOPDB` -> SmallMolecule, `CHEMBL.COMPOUND` -> ChemicalEntity 0.981. Code structure splits
beyond the prefix: `MESH:C*` (supplementary-chemical records) -> ChemicalEntity 0.976 while bare
`MESH` sits at 0.79, `MESH:D*` is heterogeneous (0.52) and unusable, and `UMLS:C##########` is
opaque (0.213, no rule). `CHEBI` is prefix-level 0.82 with per-ID-length splits (6-digit
0.914, 5-digit 0.69) and stays resolved by rank, not pattern. These facts gate the fan-out:
identity-agreeing rows almost always disagree on category exactly because their CURIE namespaces
encode different ontological views of one entity.

## Related

- [README index](../README.md) -- the full docs index
- [ctkp-interventions](ctkp-interventions.md) -- the KP-side multi-class measurement (never fires)
- [output](output.md) -- the record shapes these labels land in
