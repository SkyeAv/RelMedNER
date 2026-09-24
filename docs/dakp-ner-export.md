# DAKP NER export

The DAKP pipeline's `dakp.ner.export.v1` bundle, written by its `export_ner` stage
(`dakp_pipeline/ner_export.py`, DAKP f61b4b8) from DailyMed, FAERS, and EMA corpora mined
by DAKP's composite GLiNER backend. One local avro ingest, `dakp-ner-export/examples.avro`,
built out-of-band like the bc5cdr containers: nothing streams from the hub at pipeline time.

## Source bundle

The export directory holds four files: `examples.avro` (the ingested artifact),
`examples.ndjson` (its gliner2 JSONL projection, not ingested), `manifest.json`, and
`ner_gold.json` (a 34-case NER eval benchmark, not ingested). The avro records mirror this
repo's `TrainingExample` under the same `relmedner.ingests` namespace, minus `weight` and
`Relation.description`.

Measured over the bundle generated 2026-09-24T05:46:40Z on wenceslaus:

- 187,267 records; tasks contraindication 108,863 / indication 78,404; families DailyMed
  167,126 (contraindication, boxed-warning / warnings-and-precautions, and
  indications-and-usage sections), FAERS 18,649 (distinct indication strings), EMA 1,492
  (therapeutic indications).
- Shapes: 103,210 entities + classifications + relations, 75,344 entities +
  classifications, 8,713 classification-only rows (DAKP keeps zero-span rows on purpose to
  train abstention).
- Text length: p50 472, p99 23,588, max 81,866 chars.
- Entity labels: Disease, PhenotypicFeature, PopulationOfIndividualOrganisms,
  AnatomicalEntity, BiologicalSex, OrganismTaxon, plus DAKP's qualifier-channel
  temporal_interval_qualifier, frequency_qualifier, temporal_context_qualifier. 0 mentions
  fall outside their text.
- Relations: 2,184,456 contraindicated_in, 168,274 treats, and 22 applied_to_treat
  (asserted, subject drug -> object mention), plus mined qualifier attachments
  (population, anatomical, temporal, frequency, species, sex, disease context).
- One classification per row: task `indication context classification` over
  indication / contraindication / prevention / observed_prevention.

## What DakpNerExportScript fixes

The raw export fails gliner2's exact in-text relation contract on 2,136,998 relation values.
All but 6 are the asserted relation head (the row's subject drug) differing only by case:
DAKP's subject guard is case-insensitive, so a FLUCONAZOLE head ships against text that
spells Fluconazole. The script:

- realigns a case-only value to the text's own first case-insensitive occurrence, and drops
  a relation whose value is absent even case-insensitively;
- drops 30,236 exact duplicate relations and 96 head == tail self-loops;
- drops `species_context_qualifier` (tablassert DISABLED_EDGE_FIELDS, the same slot
  `gazetteer.DISABLED_QUALIFIERS` never emits);
- keeps biolink-class entity labels and PascalCases the three qualifier-channel labels:
  their measured surfaces ("dosage", "history of", "14 days") have no honest biolink class,
  so they ride as raw tails (TemporalIntervalQualifier, FrequencyQualifier,
  TemporalContextQualifier) rather than an invented one;
- carries `negated` and `evidence` through (asserted for drug -> object relations, mined for
  qualifier attachments) and attaches each predicate's biolink slot description;
- keeps a classification only when its true label is one of its own labels.

## Measured yield

Full-container exercise through the production `LocalAvroDataStream` + `dispatch_row` path
on wenceslaus (207 s): rows_in 187,267, rows_out 186,698 (569 exceed the always-on
8192-token cap), all 186,698 ship under the permitted-shapes contract (101,109 with
relations, 76,876 entities + classifications, 8,713 classification-only), 2,603,203
relations (2,241,706 asserted, 361,497 mined), and 0 entity mentions or relation values
outside their text.

## Staging

The container is gitignored (`src/relmedner/data/dakp-ner-export/*.avro`, tracked dir
sentinel). Refresh it from the DAKP export on wenceslaus, then check its blake3 against
the manifest:

    ssh wenceslaus 'mkdir -p /local_raid1/sgoetz/operator-data/dakp-ner-export && cp -f /users/sgoetz/Code/dakp/tmp/store/ner-export/{examples.avro,manifest.json} /local_raid1/sgoetz/operator-data/dakp-ner-export/'

The remote gate stages every `/local_raid1/sgoetz/operator-data/<name>/` dir into
`src/relmedner/data/<name>/` automatically.

## Related

- [README index](../README.md) -- the ingest-table row for this corpus and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates these rows flow through.
- [Qualifiers](qualifiers.md) -- the repo's own qualifier slots and why the species slot is disabled.
- [BC5CDR](bc5cdr.md) -- the out-of-band local avro pattern this ingest follows.
- [Weighting](weighting.md) -- the silver-tier prior this corpus is declared at.
