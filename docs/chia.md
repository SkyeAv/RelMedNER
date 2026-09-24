# CHIA eligibility criteria

`bigbio/chia` is the CHIA corpus (Kury et al., Sci Data 2020): 12,409 expert-annotated
eligibility criteria from 1,000 Phase IV clinical trials. All five hub subsets are declared
together at weight 1.0 (one row key, `bigbio/chia`); the underlying data is public-domain
ClinicalTrials.gov text. Dataset-format notes (`ChiaScript`): the repo carries a builder
script (`chia.py`), which `datasets` >= 3 refuses to stream, so every entry loads through the
new `hf_parquet` source over the hub's auto-converted `refs/convert/parquet` branch
(`load_dataset("parquet", data_files="hf://datasets/bigbio/chia@refs/convert/parquet/<subset>/train/0000.parquet", streaming=True)`).
Entity offsets are char offsets, end-EXCLUSIVE, verified on 47,091/47,091 spans of both fixed
variants; entities can be multi-part (1,752 with 2 offsets, 53 with 3, 1 with 4, 1 with 10),
each part becoming its own token span and an entity's relation surface being the first
surviving part's token slice. The two `*_source` variants ship known-bad offsets (end-convention
neither-rate 58%, out-of-bounds 1.97% on `chia_source`): they are declared as asked, the
bridge's drop rules handle them, and their lower yield is a measured fact, not a bug.
Measured per full train split: 3.4% of rows carry no entities and 6.2% carry no relations;
those rows ship text-only or entities-only and fall out via the declared-outputs filter. The
16-type label census maps eight types with an honest biolink target (CONDITION 12,039 ->
DiseaseOrPhenotypicFeature, DRUG 3,801 -> Drug, PROCEDURE 3,595 -> Procedure, PERSON 1,666 ->
Human, DEVICE 386 -> Device, MEASUREMENT 3,305 -> ClinicalMeasurement, OBSERVATION 1,795 ->
ClinicalFinding, QUALIFIER 4,157 -> ClinicalModifier) and deliberately leaves eight
criterion-structure types unmapped as raw PascalCase tails (SCOPE 4,254, VALUE 4,002, TEMPORAL
3,044, REFERENCE_POINT 934, NEGATION 843, MULTIPLIER 671, MOOD 573, VISIT 165). Relations come
from the gold arg1_id/arg2_id links: Subsumes (1,871) maps to the biolink `superclass_of` with
arg1 as head, Has_temporal (3,083) to the symmetric `temporally_related_to`, the has_*
family (has_value 3,643, has_qualifier 3,040, has_index 829, has_negation 825, has_multiplier
602, has_mood 486) keeps native snake_case names, and the logical integrators AND (2,631) and
OR (7) drop as sentence combinatorics rather than relation semantics (all counts on the scope
subsets; the without_scope subsets differ by dropping Has_scope and by re-annotating temporal
scope). Drop rules with negative tests: dangling arg id (0), id self-loop (0), surface
self-loop (33), surface outside the emitted text. `chia_bigbio_kb` rows tile exactly one
gapless passage (2000/2000, median 291 chars); `events` and `coreferences` are always empty
and ignored. Every number above comes from `.pi/skills/add-dataset/scripts/chia-census.py`
run on wenceslaus over the full 2,000-row train split of each subset (receipt style
`PROBE_*`), and the per-subset dispatch numbers from `remote-gate.sh probe -- --declared
'bigbio/chia:<subset>/train/0000.parquet' --limit 200 --script ChiaScript --outputs entities,relations`; re-run those exact commands to re-derive any figure.

## Related

- [README index](../README.md) -- the ingest-table rows for the five ChiaScript entries and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates the ChiaScript rows flow through.
- [EHR-rel](ehr-rel.md) -- the `hf_parquet` source kind this corpus loads through.
- [Weighting](weighting.md) -- the gold-tier prior the five subsets share.
