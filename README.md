# relmedner

Apache Beam pipeline that builds gliner2 training data from biomedical text corpora.
Datasets come from the HuggingFace hub (`source: hf`) or from a local avro container built
out-of-band (`source: local`, see [CTKP interventions](docs/ctkp-interventions.md)). Two ingest
types share one declarative pipeline:

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
  (`src/relmedner/data/qualifiers/qualifier_corpus.tsv`, read by `LocalDelimitedDataStream`).
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
| `aps/super_glue` (`multirc`) | `script` -> `SuperGlueMultiRCScript` | `paragraph`, `question`, `answer`, `label` | classifications | 27,243 |
| `aps/super_glue` (`record`) | `script` -> `SuperGlueRecordScript` | `passage`, `query`, `entities`, `entity_spans`, `answers` | entities, classifications | 100,730 |
| `knowledgator/PubMedAbstractsNER` | `script` -> `PubmedAbstractsScript` | `tokenized_text`, `ner` | entities, relations | 35,000 |
| `nvidia/Nemotron-PII` (train + test) | `script` -> `NemotronPiiScript` | `text`, `spans` | entities | 200,000 |
| `tensorshield/reddit_dataset_157` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 7,114,560 |
| `tensorshield/reddit_dataset_30` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 1,318,568 |
| `tensorshield/reddit_dataset_84` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 1,325,908 |
| `tensorshield/reddit_dataset_171` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 744,263 |
| `tensorshield/reddit_dataset_85` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 150,311 |
| `tensorshield/reddit_dataset_217` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 142,531 |
| `tensorshield/reddit_dataset_237` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 121,584 |
| `bigbio/ehr_rel` (4 subsets, see [docs/ehr-rel.md](docs/ehr-rel.md)) | `script` -> `EhrRelScript` | `snomed_label_1`, `snomed_label_2`, `mean_rating` / `text_1`, `text_2`, `label` | relations | 111 + 3,630 + 3,741 + 3,741 |

Ingest resolution chains and gates (per-task relabeling, mining, distant supervision): [docs/ingests.md](docs/ingests.md).

## Documentation

- [docs/yaml-config.md](docs/yaml-config.md) -- field-by-field reference for the two pipeline YAML files and their pydantic validation.
- [docs/ingests.md](docs/ingests.md) -- ingest resolution chains and gates for every table row.
- [docs/ctkp-interventions.md](docs/ctkp-interventions.md) -- the `source: local` avro-container ingest and its schema.
- [docs/deduplication.md](docs/deduplication.md) -- the dedup stage every merged `TrainingExample` passes through before Avro write.
- [docs/output.md](docs/output.md) -- the Avro `TrainingExample` records and their provenance fields.
- [docs/fullmap-mining.md](docs/fullmap-mining.md) -- how fullmap tasks mine n-gram spans against the local redb per batch.
- [docs/qualifiers.md](docs/qualifiers.md) -- the six DAKP qualifier slots on gazetteer relations and negation.
- [docs/post-training-families.md](docs/post-training-families.md) -- how the multi-task corpus splits into disjoint row families.
- [docs/super-glue-record.md](docs/super-glue-record.md) -- the `aps/super_glue` `record` ingest, the one that ships general-domain text.
- [docs/trialpanorama-database.md](docs/trialpanorama-database.md) -- the TrialPanorama `studies` subset this ingest mines.
- [docs/nemotron-pii.md](docs/nemotron-pii.md) -- the NVIDIA Nemotron-PII corpus, the one general-domain ingest.
- [docs/ehr-rel.md](docs/ehr-rel.md) -- the `bigbio/ehr_rel` concept-pair ingests and the `hf_parquet` source kind they need.

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

## Install

    uv sync

## Build the dataset

Smoke run over 5 sampled rows per dataset:

    uv run relmedner build-dataset -t -o ./relmedner-test.avro

Full run on the local DirectRunner -- no cluster, no VPN needed (`--direct` skips cluster
deployment, watching, and shard collection):

    uv run relmedner build-dataset --direct -o ./relmedner.avro

Full run on the LAN flink cluster (requires the VPN route to every `cluster.yaml` worker -- the SSH
gateway alone is not enough, because flink taskmanagers connect back to the laptop's jobmanager
ports and `YamlClusterParser.jobmanager()` resolves the laptop via a route probe toward the first
remote):

    uv run relmedner build-dataset -o ./relmedner.avro

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

`make deploy` targets the LAN Flink cluster declared in
`src/relmedner/data/cluster.yaml`; workers bind-mount the fullmap bundle read-only at
`/opt/fullmap`. The compute node is only reachable through the gateway SSH hop -- from
off-VPN, add a `ProxyJump` through the gateway in `~/.ssh/config`, or run without the
cluster entirely: `build-dataset --direct` (no pipeline options) keeps output on the local
filesystem.
