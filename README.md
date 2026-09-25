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
| `fewrel/train_wiki.avro` (local package data) | `script` -> `FewRelScript` | whole avro record | entities, relations | 44,800 |
| `fewrel/val_wiki.avro` (local package data) | `script` -> `FewRelScript` | whole avro record | entities, relations | 11,200 |
| `fewrel/val_nyt.avro` (local package data) | `script` -> `FewRelScript` | whole avro record | entities, relations | 2,500 |
| `fewrel/val_semeval.avro` (local package data) | `script` -> `FewRelScript` | whole avro record | entities, relations | 8,851 |
| `fewrel/val_pubmed.avro` (local package data) | `script` -> `FewRelScript` | whole avro record | entities, relations | 1,000 |
| `fewrel/pubmed_unsupervised.avro` (local package data) | `script` -> `FewRelScript` | whole avro record | entities | 2,500 |
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
| `OpenMed/drugprot-parquet` (train + validation) | `script` -> `DrugprotScript` | `text`, `entities`, `relations` | entities, relations | 4,250 |
| `AGBonnet/augmented-clinical-notes` | `fullmap` (max_ngram=6, taxon=9606) | `note` | entities, relations | 30,000 |
| `openlifescienceai/medmcqa` | `fullmap` (max_ngram=6, taxon=9606) | `exp` | entities, relations | 182,822 |
| `OpenMed/MedDialog` | `fullmap` (max_ngram=6, taxon=9606) | `doctor_response` | entities, relations | 226,557 |
| `commanderstrife/jnlpba` | `script` (JnlpbaScript) | `tokens`, `ner_tags` | entities, relations | 37,094 train + 7,714 validation |
| `ncbi/ncbi_disease` (train + validation + test, one row key) | `script` (NcbiDiseaseScript) | `tokens`, `ner_tags` | entities, relations | 5,433 + 924 + 941 (probe census of the refs/convert/parquet files, wenceslaus 2026-09-24) |
| `bigbio/linnaeus` | `script` (LinnaeusScript) | `passages`, `entities` | entities | 95 (probe census of the refs/convert/parquet file, wenceslaus 2026-09-24) |
| `bigbio/osiris` | `script` (OsirisScript) | `passages`, `entities` | entities | 105 (probe census of the refs/convert/parquet file, laptop 2026-09-24) |
| `bigbio/pubmed_qa` (5 labeled folds) | `script` (PubmedQaScript) | `QUESTION`, `CONTEXTS`, `final_decision` | classifications | 450 x 5 (probe census of the refs/convert/parquet files, laptop 2026-09-24) |
| `GBaker/MedQA-USMLE-4-options` | `script` (MedQaScript) | `question`, `options`, `answer_idx` | classifications | 10,178 (probe census of `phrases_no_exclude_train.jsonl`, wenceslaus 2026-09-24) |
| `rjac/clinicaltrials.gov-summary_and_eligibility` | `fullmap` (max_ngram=6, taxon=9606) | `eligibility` | entities, relations | 3,002 |
| `hackint0sh/Text-Clinical-Records` | `fullmap` (max_ngram=6, taxon=9606) | `text` | entities, relations | 31,489 |
| `chanzuckerberg/MedMentions` (ST21pv) | `script` (MedMentionsScript, local avro) | `title`, `abstract`, `entities` | entities | 2,635 train + 878 dev + 879 test |
| `tensorshield/reddit_dataset_157` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 7,114,560 |
| `tensorshield/reddit_dataset_30` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 1,318,568 |
| `tensorshield/reddit_dataset_84` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 1,325,908 |
| `tensorshield/reddit_dataset_171` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 744,263 |
| `tensorshield/reddit_dataset_85` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 150,311 |
| `tensorshield/reddit_dataset_217` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 142,531 |
| `tensorshield/reddit_dataset_237` | `fullmap` (max_ngram=6, taxon=9606) | communityName-filtered `text` | entities, relations | 121,584 |
| `bigbio/chia` (five subsets: `chia_bigbio_kb`, `chia_fixed_source`, `chia_source`, `chia_without_scope_fixed_source`, `chia_without_scope_source`) | `script` -> `ChiaScript` | `text`/`passages`, `entities`, `relations` | entities, relations | 10,000 |
| `bigbio/gad` (`gad_blurb_bigbio_text`, train + validation + test) | `script` -> `GadBlurbScript` | `text`, `labels` | classifications | 5,330 |
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
| `dakp-ner-export/examples.avro` (DAKP `dakp.ner.export.v1`) | `script` -> `DakpNerExportScript` | whole avro record | entities, classifications, relations | 187,267 |

Ingest resolution chains and gates (per-task relabeling, mining, distant supervision): [docs/ingests.md](docs/ingests.md).
## Documentation

- [docs/yaml-config.md](docs/yaml-config.md) -- field-by-field reference for the two pipeline YAML files and their pydantic validation.
- [docs/ingests.md](docs/ingests.md) -- ingest resolution chains and gates for every table row.
- [docs/ctkp-interventions.md](docs/ctkp-interventions.md) -- the `source: local` avro-container ingest and its schema.
- [docs/fewrel.md](docs/fewrel.md) -- the six FewRel local-avro ingests: delivery, span encoding, resolution chain.
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
- [docs/secondary-labels.md](docs/secondary-labels.md) -- the measured morphological secondary labels (INN stems, disease/procedure morphology), the fullmap multi-class fan-out, and the CURIE code-structure rule table.
- [docs/reddit-corpora.md](docs/reddit-corpora.md) -- the seven general-Reddit `tensorshield` dumps, their `match_on` health-community anchor, and the 100k-row inclusion bar.
- [docs/biored.md](docs/biored.md) -- the BioRED gold corpus over the `wcole3/biored-parquet` mirror, its end-exclusive span convention, and the concept-pair relation collapse.
- [docs/chia.md](docs/chia.md) -- the CHIA eligibility-criteria corpus, its five `hf_parquet` subsets, and the measured label-map and offset caveats.
- [docs/gad.md](docs/gad.md) -- the GAD gene-disease sentence corpus, its single declared subset, and the 21 subset-splits deliberately left out.
- [docs/drugprot.md](docs/drugprot.md) -- the DrugProt chemical-protein gold corpus over the `OpenMed/drugprot-parquet` mirror and its relation map.
- [docs/augmented-clinical-notes.md](docs/augmented-clinical-notes.md) -- the `AGBonnet/augmented-clinical-notes` fullmap ingest over clinical-note paragraphs.
- [docs/medmcqa.md](docs/medmcqa.md) -- the `openlifescienceai/medmcqa` fullmap ingest over exam-question explanations.
- [docs/meddialog.md](docs/meddialog.md) -- the `OpenMed/MedDialog` fullmap ingest over doctor replies.
- [docs/jnlpba.md](docs/jnlpba.md) -- the `commanderstrife/jnlpba` IOB script ingest over train and validation.
- [docs/ncbi-disease.md](docs/ncbi-disease.md) -- the `ncbi/ncbi_disease` gold disease-NER ingest over three parquet splits.
- [docs/linnaeus.md](docs/linnaeus.md) -- the `bigbio/linnaeus` gold species-NER ingest and its trust-gold label stance.
- [docs/osiris.md](docs/osiris.md) -- the `bigbio/osiris` gold variant-NER ingest and its offset-aware passage reconstruction.
- [docs/pubmed-qa.md](docs/pubmed-qa.md) -- the `bigbio/pubmed_qa` gold question-answering classification ingests.
- [docs/med-qa.md](docs/med-qa.md) -- the `GBaker/MedQA-USMLE-4-options` gold four-option multiple-choice QA ingest.
- [docs/medmentions.md](docs/medmentions.md) -- the MedMentions ST21pv gold entity corpus as a local avro container.
- [docs/clinicaltrials-gov.md](docs/clinicaltrials-gov.md) -- the `rjac/clinicaltrials.gov-summary_and_eligibility` fullmap ingest over eligibility criteria.
- [docs/text-clinical-records.md](docs/text-clinical-records.md) -- the `hackint0sh/Text-Clinical-Records` fullmap ingest over record text.
- [docs/dakp-ner-export.md](docs/dakp-ner-export.md) -- the DAKP GLiNER-mined DailyMed / FAERS / EMA export and the case-only relation drift its script realigns.
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

    make test              # full gate: every test, measured, 90% coverage floor
    make test-fast         # iteration loop: parallel, no coverage measurement
    make lint              # ruff check + format check
    make prepush           # what the push hook runs: lint + test-fast

Worker fan-out defaults to 4 (`addopts`), the safe ceiling for a 30 GB laptop (each worker
imports 150-335 MB of interpreter + relmedner + apache_beam state). A big host passes the
knob instead: `make test XDIST=auto`.

Pre-commit runs `ruff check` on the staged files only (sub-second); the push boundary runs
`make prepush` (lint + the fast suite, no coverage measurement), so tests gate once per
pushed batch instead of taxing every commit. The measured 90% coverage floor belongs to the
wenceslaus full gate (`make test` / the add-dataset remote gate in `cov` mode), never to the
laptop. Install both hook types once:

    uv run pre-commit install -t pre-commit -t pre-push

The default suite is fully offline: every live-hub smoke (the nine `tests/test_*_live.py`
files and the every-ingest pipeline smoke in `tests/test_outputs.py`) runs only when
`RELMEDNER_LIVE_HF=1` is set, and the fullmap-gated tests skip when the bundle is not
mounted, so hub downloads never happen in the default run.

The suite can also run on the wenceslaus box to keep the laptop unloaded. The remote shell is
bash-only and the snap `uv` on its PATH is broken, so sync with rsync and invoke uv by absolute
path; `PYTHONUTF8=1` is required for the remote pytest (UTF-8 locale gaps):

    ssh wenceslaus 'mkdir -p ~/Code/RelMedNER-worktrees/super-glue-record'
    rsync -a --delete --exclude .venv --exclude .ralph --exclude .git \
      /home/skyeav/Code/ISB/RelMedNER-worktrees/super-glue-record/ \
      wenceslaus:~/Code/RelMedNER-worktrees/super-glue-record/
    ssh wenceslaus 'cd ~/Code/RelMedNER-worktrees/super-glue-record && ~/.local/bin/uv sync'
    ssh wenceslaus 'cd ~/Code/RelMedNER-worktrees/super-glue-record && PYTHONUTF8=1 ~/.local/bin/uv run pytest -q -n auto'
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
