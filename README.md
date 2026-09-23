# relmedner

Apache Beam pipeline that builds gliner2 training data from biomedical text corpora.
Datasets come from the HuggingFace hub (`source: hf`, or `hf_json` for one JSON file inside a
hub repo) or from files built out-of-band and shipped in the package data dir (`source: local`
for an avro container, see [CTKP interventions](docs/ctkp-interventions.md), and
`local_delimited` for a header-delimited TSV/CSV). Two task types share one declarative
pipeline:

- **script tasks** -- datasets that already carry gold spans
  (every `script` row in the [ingest table](#ingests), which is the list that stays current as
  corpora are added); spans are relabeled to biolink classes
  via tablassert `Categories` and local fullmap resolution, and relations are
  distant-supervised through a biolink-predicate gazetteer matched between mention surfaces.
  Two corpora deviate: the multi-task `anthonyyazdaniml/gliner-biomed-post-training`
  additionally carries native gold relations and sampled negatives, and splits into per-row
  task families (see docs/post-training-families.md), while `knowledgator/sentence_rex` carries gold relations with both
  participant spans marked inline (`<e1>`/`<e2>` tags) and labels kept native.
- **fullmap tasks** -- unlabeled text (e.g.
  `anthonyyazdaniml/gliner-biomed-curated-corpus` and the downsampled, class-balanced
  `anthonyyazdaniml/gliner-biomed-balanced-curated-corpus`, a strict 158,890-row subset of
  the curated corpus with an identical `text`-only schema); entities are *mined* by enumerating
  n-grams, resolving them in one batched round trip against the local fullmap redb, and
  keeping only spans that exactly match a normalized preferred name. Mined spans also feed
  the gazetteer for `evidence="distant"` relations. The gazetteer additionally emits
  statement-qualifier relations (see docs/qualifiers.md).
- **local sources** -- files on disk instead of a hub dataset. `source: local` reads an avro
  container and ships each record whole (see [CTKP interventions](docs/ctkp-interventions.md));
  `source: local_delimited` reads a header-delimited TSV/CSV with a `columns_out` projection
  (`LocalDelimitedDataStream`); both declared files of that kind, the qualifier corpus and the
  unannotated ADE tweets, are fullmap-mined.
  Relative paths resolve against the CWD first, then the package data dir, so in-repo corpora
  work from any CWD and inside the worker container.

## Ingests

Field-by-field schema, discriminator rules, and copy-paste templates for `ingests.yaml`: [docs/yaml-config.md](docs/yaml-config.md).
Machine-readable JSON Schemas for editor autocomplete and pre-validation: [schemas/ingests.schema.json](schemas/ingests.schema.json) (+ `schemas/cluster.schema.json`), regenerated with `uv run relmedner schema`.

| dataset | task | inputs | outputs | rows in |
| --- | --- | --- | --- | --- |
| `anthonyyazdaniml/gliner-biomed-pre-training` | `script` -> `GlinerBiomedScript` | `tokenized_text`, `ner` | entities, relations | 98,659 |
| `disi-unibo-nlp/Pile-NER-biomed-IOB` | `script` -> `PileNerBiomedScript` | `tokens`, `ner_tags` | entities | 58,861 |
| `Universal-NER/Pile-NER-type` | `script` -> `PileNerTypeScript` | `conversations` | entities | 45,889 |
| `knowledgator/sentence_rex` | `script` -> `SentenceRexScript` | `sentences`, `labels` | relations | 44,115 |
| `anthonyyazdaniml/gliner-biomed-pre-training` | `script` -> `GlinerBiomedScript` | `tokenized_text`, `ner` | entities, relations | 98,659 |
| `disi-unibo-nlp/Pile-NER-biomed-IOB` | `script` -> `PileNerBiomedScript` | `tokens`, `ner_tags` | entities | 58,861 |
| `knowledgator/biomed_NER` | `script` -> `KnowledgatorBiomedScript` | `text`, `entities` | entities | 4,840 |
| `knowledgator/gliner-multilingual-synthetic` | `script` -> `GlinerMultilingualScript` | `tokenized_text`, `ner` | entities | 96,606 |
| `anthonyyazdaniml/gliner-biomed-curated-corpus` | `fullmap` (max_ngram=6, taxon=9606) | `text` | entities, relations | 418,381 |
| `anthonyyazdaniml/gliner-biomed-balanced-curated-corpus` | `fullmap` (max_ngram=6, taxon=9606) | `text` | entities, relations | 158,890 |
| `TrialPanorama/TrialPanorama-database` (`studies`) | `fullmap` (max_ngram=6, taxon=9606) | `abstract` | entities, relations | 1,332,141 |
| `anthonyyazdaniml/gliner-biomed-post-training` | `script` -> `GlinerBiomedPostScript` | `tokenized_text`, `ner`, `negatives` | entities, classifications, structures, relations | -- |
| `interventions/interventions.avro` (local package data) | `script` -> `CtkpInterventionsScript` | whole avro record | entities | 1,020,749 |
| `qualifiers/qualifier_corpus.tsv` (local package data) | `fullmap` (max_ngram=6, taxon=9606) | `text` | entities, relations | 24 |
| `synthetic-ner-ade-tweets/ade_tweets.avro` (local package data) | `script` -> `SyntheticNerAdeTweetsScript` | whole avro record | entities | 17,000 (probe census, wenceslaus 2026-09-22) |
| `synthetic-ner-ade-tweets/ade_tweets_unannotated.tsv` (local package data) | `fullmap` (max_ngram=6, taxon=9606) | `text` | entities, relations | 8,502 (probe census, wenceslaus 2026-09-22) |
| `bc5cdr/train.avro` (local package data) | `script` -> `Bc5CdrScript` | whole avro record | entities, relations | 500 |
| `bc5cdr/dev.avro` (local package data) | `script` -> `Bc5CdrScript` | whole avro record | entities, relations | 500 |
| `bc5cdr/test.avro` (local package data) | `script` -> `Bc5CdrScript` | whole avro record | entities, relations | 500 |
| `aps/super_glue` (`multirc`) | `script` -> `SuperGlueMultiRCScript` | `paragraph`, `question`, `answer`, `label` | classifications | 27,243 |
| `aps/super_glue` (`record`) | `script` -> `SuperGlueRecordScript` | `passage`, `query`, `entities`, `entity_spans`, `answers` | entities, classifications | 100,730 |
| `knowledgator/PubMedAbstractsNER` | `script` -> `PubmedAbstractsScript` | `tokenized_text`, `ner` | entities, relations | 35,000 |
| `thunlp/docred` (`train_annotated`) | `script` -> `DocredScript` | `sents`, `vertexSet`, `labels` | entities, relations | 3,053 |
| `thunlp/docred` (`train_distant`) | `script` -> `DocredScript` | `sents`, `vertexSet`, `labels` | entities, relations | 101,873 |
| `thunlp/docred` (`dev`) | `script` -> `DocredScript` | `sents`, `vertexSet`, `labels` | entities, relations | 998 |
| `thunlp/docred` (`test`) | `script` -> `DocredScript` | `sents`, `vertexSet`, `labels` | entities | 1,000 |
| `nvidia/Nemotron-PII` (train + test) | `script` -> `NemotronPiiScript` | `text`, `spans` | entities | 200,000 |
| `bigbio/chemprot` (`chemprot_full_source`, 3 splits) | `script` -> `ChemprotScript` | `text`, `entities`, `relations` | entities, relations | 2,432 |
| `wcole3/biored-parquet` (train + validation + test) | `script` -> `BioredScript` | `passages`, `entities`, `relations` | entities, relations | 600 |
| `tensorshield/reddit_dataset_157` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 7,114,560 |
| `tensorshield/reddit_dataset_30` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 1,318,568 |
| `tensorshield/reddit_dataset_84` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 1,325,908 |
| `tensorshield/reddit_dataset_171` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 744,263 |
| `tensorshield/reddit_dataset_85` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 150,311 |
| `tensorshield/reddit_dataset_217` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 142,531 |
| `tensorshield/reddit_dataset_237` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 121,584 |
| `bigbio/chia` (five subsets: `chia_bigbio_kb`, `chia_fixed_source`, `chia_source`, `chia_without_scope_fixed_source`, `chia_without_scope_source`) | `script` -> `ChiaScript` | `text`/`passages`, `entities`, `relations` | entities, relations | 10,000 |
| `bigbio/ehr_rel` (4 subsets, see [docs/ehr-rel.md](docs/ehr-rel.md)) | `script` -> `EhrRelScript` | `snomed_label_1`, `snomed_label_2`, `mean_rating` / `text_1`, `text_2`, `label` | relations | 111 + 3,630 + 3,741 + 3,741 |
| `ruslan/bioleaflets-biomedical-ner` (`train`) | `script` -> `BioleafletsScript` | `Section_1`, `Section_2`, `Section_3`, `Section_4`, `Section_5`, `Section_6` | entities, relations | 1,068 |
| `ruslan/bioleaflets-biomedical-ner` (`test`) | `script` -> `BioleafletsScript` | `Section_1`, `Section_2`, `Section_3`, `Section_4`, `Section_5`, `Section_6` | entities, relations | 134 |
| `agentlans/json-extraction` (`owkin-medical_knowledge_from_extracts`) | `script` -> `JsonExtractionScript` | `text`, `json`, `source` | structures, entities | 1,383 |
| `agentlans/json-extraction` (`ProfessorBob-relation_extraction`) | `script` -> `JsonExtractionScript` | `text`, `json`, `source` | structures, relations | 6,920 |
| `agentlans/json-extraction` (`roborovski-dolly-entity-extraction`) | `script` -> `JsonExtractionScript` | `text`, `json`, `source` | structures, entities | 5,945 |
| `agentlans/json-extraction` (`sandeeppanem-resume-json-extraction-5k`) | `script` -> `JsonExtractionScript` | `text`, `json`, `source` | structures, entities | 4,879 |
| `agentlans/json-extraction` (`Jiraya-html_to_json_information_extraction_dataset`) | `script` -> `JsonExtractionScript` | `text`, `json`, `source` | structures, entities | 3,035 |
| `agentlans/json-extraction` (`HenriqueGodoy-extract-0`) | `script` -> `JsonExtractionScript` | `text`, `json`, `source` | structures | 2,606 |
| `Pennlaine/Medical-Entity-JSON-Extraction` (test) | `script` -> `MedicalEntityJsonScript` | `text` | entities | 50 |

Ingest resolution chains and gates (per-task relabeling, mining, distant supervision): [docs/ingests.md](docs/ingests.md).

## Chia eligibility criteria

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

## Documentation

- [docs/yaml-config.md](docs/yaml-config.md) -- field-by-field reference for the two pipeline YAML files and their pydantic validation.
- [docs/ingests.md](docs/ingests.md) -- ingest resolution chains and gates for every table row.
- [docs/ctkp-interventions.md](docs/ctkp-interventions.md) -- the `source: local` avro-container ingest and its schema.
- [docs/deduplication.md](docs/deduplication.md) -- the dedup stage every merged `TrainingExample` passes through before Avro write.
- [docs/docred.md](docs/docred.md) -- the `thunlp/docred` document-level relation ingests over the raw hub `json.gz` files.
- [docs/output.md](docs/output.md) -- the Avro `TrainingExample` records and their provenance fields.
- [docs/fullmap-mining.md](docs/fullmap-mining.md) -- how fullmap tasks mine n-gram spans against the local redb per batch.
- [docs/qualifiers.md](docs/qualifiers.md) -- the six DAKP qualifier slots on gazetteer relations and negation.
- [docs/post-training-families.md](docs/post-training-families.md) -- how the multi-task corpus splits into disjoint row families.
- [docs/super-glue-record.md](docs/super-glue-record.md) -- the `aps/super_glue` `record` ingest, the one that ships general-domain text.
- [docs/trialpanorama-database.md](docs/trialpanorama-database.md) -- the TrialPanorama `studies` subset this ingest mines.
- [docs/nemotron-pii.md](docs/nemotron-pii.md) -- the NVIDIA Nemotron-PII corpus, the one general-domain ingest.
- [docs/ehr-rel.md](docs/ehr-rel.md) -- the `bigbio/ehr_rel` concept-pair ingests and the `hf_parquet` source kind they need.
- [docs/bioleaflets.md](docs/bioleaflets.md) -- the EMA package-leaflet corpus and its measured ResolutionGate drugname bucket.
- [docs/synthetic-ner-ade-tweets.md](docs/synthetic-ner-ade-tweets.md) -- the synthetic ADE-tweets corpus, its BRAT-standoff decode, and its gold-plus-unannotated ingest pair.
- [docs/json-extraction.md](docs/json-extraction.md) -- the structured-extraction corpus, its six per-source ingests, and the honest-biolink label maps behind its structures-plus-entities shapes.
- [docs/weighting.md](docs/weighting.md) -- how to pick ingest weights, the `trust` band, and the offline `validate-trust` literature check.
- [docs/medical-entity-json-extraction.md](docs/medical-entity-json-extraction.md) -- the 50-row consumer-health vignette corpus and its JSON-in-assistant-turn decode.
- [docs/bc5cdr.md](docs/bc5cdr.md) -- the BC5CDR avro containers built out-of-band from the NCBI CDR BioC XML corpus (gold chemical/disease spans trusted at MeSH CURIEs, gold CID relations as `causes` surface pairs).
- [docs/quality-heuristics.md](docs/quality-heuristics.md) -- the C4/Gopher-style row-quality heuristic filters, their lineage, cost contract, and the measure-first threshold workflow.

## Reddit corpora

Seven general-Reddit dumps from the `tensorshield` hub org, all MIT-licensed. The
inclusion bar is corpus-wide `total_rows >= 100,000` in the org's stats.json: the kept
seven span 121,584 to 7,114,560 rows. `tensorshield/reddit_dataset_226` (22,572 rows)
carries the same license but fails the bar: after the health-community filter its
projected yield is noise, so it stays out of the pipeline.

None of these corpora are biomedical on their own, so every entry filters rows
client-side post-stream with `match_on`: exact-value membership on `communityName`
against a single yaml anchor `&biomed_communities` holding 42 provisional health
subreddits (`r/AskDocs`, `r/diabetes`, `r/CrohnsDisease`, ...; the full list lives in
`src/relmedner/data/ingests.yaml` and all seven entries alias the one anchor, so the
allowlists cannot drift apart). Casing must be the Reddit canonical form: `r/askdocs`
does not match, and neither does a non-health community like `r/madmen`. `columns_out`
is `text` alone.

The table's rows-in are stats.json `total_rows` for the whole corpus, before the
community filter. Mining runs the unchanged fullmap path of the previous section;
informal reddit prose (slang, misspellings, first-person narratives) is expected to
mine at lower unigram precision than the curated corpus, and the measured gates in
`relmedner/constants.py` are deliberately not loosened for it.

## BioRED

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

## Install

    uv sync

## Build the dataset

Smoke run over 5 sampled rows per dataset:

    uv run relmedner build-dataset -t -o ./relmedner-test.avro

Full run on the local DirectRunner -- no cluster, no VPN needed (`--direct` skips cluster
deployment, watching, and shard collection):

    uv run relmedner build-dataset --direct -o ./relmedner.avro

Full run on the LAN flink cluster -- run from the checkout **on the head host** (wenceslaus, in
tmux); the cluster is fully self-contained there (jobmanager, both taskmanagers, and the ssh
tunnels that carry the :22-only cross-host traffic), so the laptop needs no VPN and no cluster
ports:

    uv run relmedner build-dataset -o ./relmedner.avro

## Validate trust

Suggest a `trust:` value per source from literature evidence. No cluster and no VPN, but it
does need network egress: records stream through the same dispatch path as `build-dataset`,
and every emitted entity or relation becomes a PubMed E-utilities query. One source, two
records:

    uv run relmedner validate-trust --source Pennlaine/Medical-Entity-JSON-Extraction -t --sample-size 2 -o ./trust-report.jsonl

Nothing is written back to `ingests.yaml`. The command prints a suggested `trust:` snippet per
source and writes the per-record JSONL report for review. Defaults come from the optional
`x-trust:` section of `ingests.yaml` (50 records per source, `trust-report.jsonl`), overridden
flag-by-flag; an optional `NCBI_API_KEY` in the environment
lifts the client's own throttle from 3.4 to about 9 req/s (NCBI allows 3 unauthenticated, 10
authenticated). A key that is malformed or expired makes every query fail, and the printed
snippet says so rather than blaming the corpus. Reading the report and turning it into a
committed weight: [docs/weighting.md](docs/weighting.md).

## Testing

    make test        # full gate: every test, measured, 90% coverage floor
    make test-fast   # iteration loop: parallel, no coverage measurement
    make lint        # ruff check + format check

Pre-commit runs `make lint` only (sub-second); the suite runs at the pre-push boundary, so it gates
once per pushed batch instead of taxing every commit. Install both hook types once:

    uv run pre-commit install -t pre-commit -t pre-push

The default suite is fully offline: the ReCoRD live smoke (`tests/test_super_glue_record_live.py`)
streams one real hub row and runs only when `RELMEDNER_LIVE_HF=1` is set, so hub downloads never
happen in the default run.

The suite can also run on the wenceslaus box to keep the laptop unloaded. The remote shell is
bash-only and the snap `uv` on its PATH is broken, so sync with rsync and invoke uv by absolute
path; `PYTHONUTF8=1` is required for the remote pytest (UTF-8 locale gaps):

    ssh wenceslaus 'mkdir -p ~/Code/RelMedNER-worktrees/super-glue-record'
    rsync -a --delete --exclude .venv --exclude .ralph --exclude .git \
      /home/skyeav/Code/ISB/RelMedNER-worktrees/super-glue-record/ \
      wenceslaus:~/Code/RelMedNER-worktrees/super-glue-record/
    ssh wenceslaus 'cd ~/Code/RelMedNER-worktrees/super-glue-record && ~/.local/bin/uv sync'
    ssh wenceslaus 'cd ~/Code/RelMedNER-worktrees/super-glue-record && PYTHONUTF8=1 ~/.local/bin/uv run pytest -q'
    ssh wenceslaus 'cd ~/Code/RelMedNER-worktrees/super-glue-record && ~/.local/bin/uv run ruff check ./src ./tests && ~/.local/bin/uv run ruff format --check ./src ./tests'

### Agent skills

`.pi/skills/add-dataset/` is a repo-local skill for the pi coding agent: describe a corpus ("add
<nvidia/Nemotron-PII> to the training data", or `/skill:add-dataset`), and it runs the whole ingest
procedure -- remote probes and gates on wenceslaus, the `ingests.yaml`/`Script`/tuple-lock wiring,
README row plus measured format notes, one PR to `main`. It never streams data, builds, or runs the
full suite on the laptop; declaration-only adds run inline and adds needing new code hand off to the
`ralph` skill with a pre-filled constraint block. The remote loop it drives is exactly the
wenceslaus chain documented above, packaged as committed scripts
(`.pi/skills/add-dataset/scripts/remote-gate.sh`) with receipt lines.

One-time setup: `.pi/skills/` is a trust-requiring project resource, so pi asks whether to trust the
folder on first load. Answering "Trust parent folder" once covers this checkout and every
`RelMedNER-worktrees/*` worktree (pi resolves the nearest saved entry; the choice is written to
`~/.pi/agent/trust.json`). Headless runs (`pi -p`, and aoe/fleet workers) show no prompt and silently
ignore project resources unless that saved entry exists, `defaultProjectTrust` is set to `always` in
the global settings, or the invocation passes `--approve`.

Entity resolution reads a local fullmap database from the hardcoded
`FULLMAP_DIR` path in `src/relmedner/constants.py`; fullmap-dependent tests skip when that
mount is absent, and miner unit tests inject fake `lookup_rows` rows so they never need it.

## Cluster

`make deploy` runs from the head-host checkout (wenceslaus) and targets the LAN Flink cluster
declared in `src/relmedner/data/cluster.yaml`: a jobmanager + taskmanager + sdkworker on the head,
a taskmanager + sdkworker on hypatia, and per-host ssh `-L` tunnels (one tmux session per target,
`relmedner-tunnel-*`) because cross-host traffic is :22-only. Workers bind-mount the fullmap bundle
read-only at `/opt/fullmap` (the sdkworker reads it via `RELMEDNER_FULLMAP_DIR`). Cross-host ports
are firewalled, so everything -- deploy, submit, monitoring, shard collection -- happens on the head
host; from off-LAN, reach a head-host shell with a `ProxyJump` through the gateway in
`~/.ssh/config`, or run without the cluster entirely: `build-dataset --direct` (no pipeline
options) keeps output on the local filesystem.
