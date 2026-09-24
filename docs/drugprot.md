# DrugProt

DrugProt gold (Krallinger et al. 2021, BioCreative VII; cc-by-4.0): 4,250 PubMed abstracts
(3,500 train / 750 validation) with expert-annotated chemical and gene mentions plus gold
mention-level chemical-gene relations. The declared source is `OpenMed/drugprot-parquet`, not
`bigbio/drugprot` (script-only hub loading, the same datasets-5.x problem BioRED documents); the
mirror's rows carry exactly the DrugProt tables and the license is stated on its card. Measured
full-split census (wenceslaus, 2026-09-23): 108,387 entities, all end-EXCLUSIVE char offsets
against the pre-joined `text` column with exact surface matches on 108,387/108,387, 0
out-of-bounds, 0 self-loops, 0 dangling relation args. Entity census: train CHEMICAL 46,274 /
GENE-Y 28,421 / GENE-N 14,834, validation CHEMICAL 9,853 / GENE 9,005; `DrugprotScript` maps
CHEMICAL to ChemicalEntity, GENE-Y and the plain GENE twin to Gene, and GENE-N to GeneFamily per
the ChemprotScript precedent (the -N mark names nonspecific mentions such as "kinase").

Relations: 13 gold labels over 21,035 measured pairs (INHIBITOR 6,538, DIRECT-REGULATOR 2,705,
SUBSTRATE 2,497, INDIRECT-UPREGULATOR 1,680, ACTIVATOR 1,674, INDIRECT-DOWNREGULATOR 1,661,
PRODUCT-OF 1,078, PART-OF 1,142, ANTAGONIST 1,190, AGONIST 789, AGONIST-ACTIVATOR 39,
SUBSTRATE_PRODUCT-OF 27, AGONIST-INHIBITOR 15). The corpus-native labels are the ChemProt CPR
codes unabbreviated, so the directional families reuse the landed predicate choices (ACTIVATOR
and INDIRECT-UPREGULATOR -> increases_amount_or_activity_of, INHIBITOR and
INDIRECT-DOWNREGULATOR -> decreases_amount_or_activity_of, DIRECT-REGULATOR -> regulates,
PART-OF -> part_of, SUBSTRATE -> is_substrate_of); AGONIST, AGONIST-ACTIVATOR,
AGONIST-INHIBITOR, ANTAGONIST, PRODUCT-OF, and SUBSTRATE_PRODUCT-OF stay native snake_case
(biolink has no honest slot; resolve_predicate convention). 1,275 of 4,250 rows (1,067 train /
208 validation) carry entities with zero relations and ship entities-only under the
permitted-shapes contract. Gold relations emit `evidence="asserted"`, never negated. Declared
probe over the first 300 train rows: rows_in = rows_out = 300, 100% emit, 4,184 entity mentions,
1,565 relations, 219 rows shipping the relations shape.

## Related

- [README index](../README.md) -- the ingest-table row(s) for this corpus and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates these rows flow through.
- [Weighting](weighting.md) -- the tier prior this corpus is declared at.
