# BioRED

BioRED gold (Luo et al. 2022, `ftp.ncbi.nlm.nih.gov/pub/lu/BioRED`): 600 PubMed abstracts
(400 train / 100 validation / 100 test) with gold entities and gold document-level relations.
The declared source is `wcole3/biored-parquet`, not the `bigbio/biored` the corpus is usually
reached through: that hub repo is script-only, and its loading script needs `bioc` plus the
removed `trust_remote_code` path, which `datasets` 5.x no longer supports (the datasets-server
errors on it and its `refs/convert/parquet` branch is empty). The mirror is a faithful
conversion of the same data: the upstream zip holds 20,419 entity annotation lines
(13,351 / 3,533 / 3,535) and the parquet streams exactly 20,419 entities over 600 docs.

Each row is one abstract as two passages (title, abstract); 0 of 600 rows are empty. Entity
offsets are end-EXCLUSIVE and document-relative against the `title + " " + abstract` join:
20,419/20,419 sampled surfaces match (1,964 title-anchored, 18,455 abstract-anchored, 0
no-match), exactly one offset pair per entity, 0 discontinuous. The census behind the label
map: GeneOrGeneProduct 6,697, DiseaseOrPhenotypicFeature 5,545, ChemicalEntity 4,429,
OrganismTaxon 2,192, SequenceVariant 1,381, CellLine 175; all six map onto biolink Categories
one-for-one except GeneOrGeneProduct, which lands on Gene (these mentions carry NCBIGene ids,
unlike the Pile-NER nonspecific tail). Gold normalization ids (MESH 10,052, NCBIGene 7,406,
NCBITaxon 2,193, dbSNP 784, custom 597, Cellosaurus 175, OMIM 20) feed the relation dedupe key
but are deliberately not shipped on the emitted entities.

The upstream gold is 6,503 concept-pair REL lines; the bigbio conversion expands every concept
pair into ALL mention-pair combinations, inflating the stream to 128,460 raw relations (~20x).
`BioredScript` collapses each row back to concept level, keyed by (relation type, frozenset of
arg1 normalized db_ids, frozenset of arg2 normalized db_ids) and keeping the first mention pair
in row order: 128,460 raw -> 6,767 relations (4,390 train / 1,243 validation / 1,134 test).
121 mention-level self-loops drop before the dedupe; 8 rows carry entities but no relations and
ship entities-only under the permitted-shapes contract. Predicate map over the deduped census:
Association 3,510 -> associated_with, Positive_Correlation 1,854 -> positively_correlated_with,
Negative_Correlation 1,172 -> negatively_correlated_with, Bind 120 -> physically_interacts_with,
Drug_Interaction 13 -> pharmacologically_interacts_with; Comparison 39, Cotreatment 55, and
Conversion 4 stay native snake_case (biolink has no honest slot; resolve_predicate convention).
Gold relations emit `evidence="asserted"`, never negated. Declared probes measured rows_in =
rows_out and a 100% emit rate on every split.

## Related

- [README index](../README.md) -- the ingest-table rows for the three BioredScript entries and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates the BioredScript rows flow through.
- [Output](output.md) -- the permitted-shapes contract the entities-only rows ship under.
