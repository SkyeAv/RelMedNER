# YAML configuration reference

Two YAML files drive the pipeline, both validated by the pydantic models in
`src/relmedner/models.py`:

- `src/relmedner/data/ingests.yaml` -> `YamlIngests`: which datasets enter the pipeline, with what task and mixing weight.
- `src/relmedner/data/cluster.yaml` -> `Cluster`: the Flink worker nodes for cluster runs.

This page is the field-by-field contract for both files. The drift guard in
`tests/test_ingests.py` parametrizes over every field of the models below and
asserts the field name appears here, so a model change that skips this page fails
the test suite.

## Tripwire: unknown keys fail validation

Every model on this page sets `extra="forbid"`. A typo'd, invented, or misplaced
key ANYWHERE (a dataset entry, a task block, a `match_on` item, a worker entry)
fails validation with a pydantic error. Nothing is silently ignored; the only
exception is the content of `x-defaults`, which is stored but never read (see
below). Do not add convenience keys: if a name is not in a table on this page,
the loader does not accept it.

## Discriminators

- `task.type` selects the task block: `script` -> `ScriptTask`, `fullmap` -> `FullmapTask`.
- `source` selects the dataset entry: `hf` -> `HuggingFaceDataset`, `local` -> `LocalDataset`.
- Any other value for either key is a validation error.

## ingests.yaml

### Top level (`YamlIngests`)

| field | required? | default | constraint | meaning |
| --- | --- | --- | --- | --- |
| `datasets` | yes | none | non-empty list of dataset entries (discriminated on `source`) | the declared datasets, in pipeline order |
| `x_defaults` (yaml key `x-defaults`) | no | absent | arbitrary map | anchor host only; validated but never semantically read (see below) |

### The `x-defaults` convention

`x-defaults` follows the compose-spec `x-` extension pattern: a top-level map
reserved for reusable YAML fragments, named so it never collides with a real
schema key.

- Anchors defined under it are document-global. YAML anchors are scoped to the
  whole document wherever they are defined, so `&name` under `x-defaults` can be
  aliased (`*name`) or merge-keyed (`<<: *name`) anywhere below it.
- Its content is never semantically interpreted. After parsing, the map lands in
  `YamlIngests.x_defaults` (a validation-only field, excluded from the generated
  avro schema); no loader code reads it. It exists solely to host anchors while
  keeping `extra="forbid"` intact everywhere else.
- Rules (from the header of `ingests.yaml`): alias depth is one level (never
  alias an alias); every entry still names its dataset id, its task (script name
  or `fullmap` type), and its `columns_out`; anything an entry merges
  (source/split/weight) stays visible under `x-defaults`.

### Task blocks

Selected by `task.type`:

| `task.type` | model | use |
| --- | --- | --- |
| `script` | `ScriptTask` | dataset already carries gold spans; a registered `Script` subclass consumes each row |
| `fullmap` | `FullmapTask` | unlabeled text; entities are mined by n-gram enumeration against the local fullmap redb |

`ScriptTask`:

| field | required? | default | constraint | meaning |
| --- | --- | --- | --- | --- |
| `type` | yes | none | literal `script` | selects `ScriptTask` |
| `name` | yes | none | must be a registered script name | the `Script` subclass that consumes each streamed row (undeclared names are rejected after parsing with the list of declared scripts) |
| `outputs` | yes | none | non-empty list of `entities`, `classifications`, `structures`, `relations` | permitted-shapes contract: a row ships when it produced at least one declared shape |

`FullmapTask`:

| field | required? | default | constraint | meaning |
| --- | --- | --- | --- | --- |
| `type` | yes | none | literal `fullmap` | selects `FullmapTask` |
| `max_ngram` | no | 6 | 1 <= n <= 10 | longest contiguous n-gram enumerated per tokenized batch for fullmap lookup |
| `taxon` | no | `"9606"` | string | NCBI taxon id passed to the fullmap `filter_and_rank` step; `9606` is human |
| `relations` | no | true | bool | when true, mined spans also feed the gazetteer for `evidence="distant"` relations |
| `outputs` | yes | none | non-empty list of `entities`, `classifications`, `structures`, `relations` | permitted-shapes contract (mined rows typically declare `[entities, relations]`) |

The measured quality gates of fullmap mining (function-word guards, GENELIKE
casing rule, strict unigram name agreement, junk-category gate, model-organism
CURIE exclusion) are NOT fields here; they live as constants (see "Deliberately
not configurable" below).

### Dataset entries

Selected by `source`. Both shapes inherit two fields from `DatasetBase`:

| field | required? | default | constraint | meaning |
| --- | --- | --- | --- | --- |
| `task` | yes | none | `script` or `fullmap` task block (see above) | what happens to each row of this dataset |
| `weight` | no | 1.0 | must be > 0 | per-source mixing weight stamped onto every `TrainingExample` the source emits; stock gliner2 has no per-example weight channel, so consumption is weighted duplication at avro->JSONL export. Two entries sharing one row key (hub repo id, or local path) must agree on the weight or stamping would be ambiguous |

`HuggingFaceDataset` (`source: hf`):

| field | required? | default | constraint | meaning |
| --- | --- | --- | --- | --- |
| `source` | yes | none | literal `hf` | selects `HuggingFaceDataset` |
| `dataset` | yes | none | hub repo id | the HuggingFace dataset; doubles as the row key for weight stamping |
| `subset` | no | absent | hub config/subset name | passed to `load_dataset` as `name` |
| `split` | no | absent | hub split name | which split streams; absent = the dataset default |
| `match_on` | no | absent | list of `MatchOn` (see below) | row filter: keep only rows whose `column` value is in `values` |
| `columns_out` | yes | none | list of column names | the columns streamed to the task, in declaration order |

`LocalDataset` (`source: local`):

| field | required? | default | constraint | meaning |
| --- | --- | --- | --- | --- |
| `source` | yes | none | literal `local` | selects `LocalDataset` |
| `path` | yes | none | filesystem path | avro container built out-of-band; `~` is expanded with `expanduser` at stream time. The whole record ships to the declared script; there is no `columns_out` projection because the file's own schema is the contract. Doubles as the row key for weight stamping |

### `match_on` entries (`MatchOn`)

| field | required? | default | constraint | meaning |
| --- | --- | --- | --- | --- |
| `column` | yes | none | string | dataset column to test |
| `values` | yes | none | list of strings | accepted values; rows whose `column` is not in this list are skipped |

### Copy-paste templates

Templates (a) through (c) each validate as-is against the current models. Prove a
template before committing it:

    uv run python -c "import yaml; from relmedner.models import YamlIngests; \
      YamlIngests.model_validate(yaml.load(open('template.yaml'), Loader=yaml.CSafeLoader))"

(a) hf dataset + script task, sharing the common hf fragment via `x-defaults`:

```yaml
x-defaults:
  hf-train: &hf-train {source: hf, split: train, weight: &weight 1.0}
datasets:
  - task: {type: script, name: GlinerBiomedScript, outputs: [entities, relations]}
    <<: *hf-train
    dataset: anthonyyazdaniml/gliner-biomed-pre-training
    columns_out: [tokenized_text, ner]
```

(b) hf dataset + fullmap task, aliasing the shared fullmap block so both corpora
stay on one declaration:

```yaml
x-defaults:
  fullmap-task: &fullmap-task {type: fullmap, outputs: [entities, relations]}
  hf-train: &hf-train {source: hf, split: train, weight: 1.0}
datasets:
  - task: *fullmap-task
    <<: *hf-train
    dataset: anthonyyazdaniml/gliner-biomed-curated-corpus
    columns_out: [text]
  - task: *fullmap-task
    <<: *hf-train
    dataset: anthonyyazdaniml/gliner-biomed-balanced-curated-corpus
    columns_out: [text]
```

(c) local avro dataset (no `columns_out`; the file schema is the contract):

```yaml
datasets:
  - task: {type: script, name: CtkpInterventionsScript, outputs: [entities]}
    source: local
    path: ~/Desktop/interventions.avro
```

`path` keeps the `~`; expansion happens at stream time, not at parse time.

(d) hf_json preview. LANDS VIA THE pubmed-abstracts-ner PR -- NOT VALID UNTIL
THEN: the current models have no `hf_json` source, and the `x-defaults` field
lands on this branch, so treat this as the planned shape, not a working entry:

```yaml
x-defaults:
  hf-json: &hf-json {source: hf_json, split: train}
datasets:
  - task: {type: script, name: PubmedAbstractsScript, outputs: [entities, relations]}
    <<: *hf-json
    dataset: knowledgator/PubMedAbstractsNER
    file: train.json
    columns_out: [tokenized_text, ner]
```

On the pubmed-abstracts-ner branch this shape is `HuggingFaceJsonDataset`
(`source: hf_json`, fields `dataset`, `file`, `split`, `match_on`,
`columns_out`), a json-builder ingest over one hub repo file kept out of the
`hf` source by two measured blockers (old-style `dataset_infos.json`,
cold-cache streaming corruption).

### Deliberately NOT configurable

- Fullmap quality gates stay constants. The measured gates (function-word
  guards, GENELIKE casing rule, strict unigram name agreement, junk-category
  gate, model-organism CURIE exclusion) live in `relmedner.constants` /
  `relmedner.fullmap_mine`, each fixed by measurement (`PLAN.md`); the
  constants carry an explicit "Never extend this list" comment. If a gate ever
  needs to move, add a field to `FullmapTask` and bring the measurement.
- Per-dataset limits are deferred (tuple-lock blast radius, architecture D5).
  Row caps for smoke runs are not YAML: `sample_limit`, `output`, and `run_id`
  live on `RunConfig` and are set by CLI flags (`build-dataset -t -o ...`).

## cluster.yaml

### Top level (`Cluster`)

| field | required? | default | constraint | meaning |
| --- | --- | --- | --- | --- |
| `ssh_user` | yes | none | login name | ssh user on every worker |
| `workers` | yes | none | non-empty list of `WorkerNode` | the Flink worker nodes; the laptop entry hosts the jobmanager |

### Worker entries (`WorkerNode`)

| field | required? | default | constraint | meaning |
| --- | --- | --- | --- | --- |
| `host` | yes | none | hostname or IP; `local` means the laptop | worker address (taskmanagers connect back to the laptop's jobmanager ports, so every worker needs a VPN route to it) |
| `slots` | yes | none | >= 1 | Beam SDK worker slots on this node |
| `memory` | yes | none | string like `16g` | memory given to the worker container |
| `fullmap` | yes | none | host directory | directory holding the fullmap redb bundle (primary redb plus `<stem>.s<N>.redb` shards); bind-mounted read-only into the sdkworker at `/opt/fullmap` |
| `outputs` | yes | none | host directory | directory the sdkworker writes avro shards into; shards on remotes are collected back to the laptop when the job finishes |

The shared path pair (`fullmap` + `outputs`) can be anchored once and merged
into sibling workers, as the wenceslaus/hypatia blocks in `cluster.yaml` do
with `<<: &wenceslaus-paths ...`.

## How the loader reads these files

- `YamlParser.parse` opens the file with an explicit `encoding="utf-8"`. This
  is a locale guard: the packaged yaml carries non-ascii comments, and on the
  compute box `LANG=en_US` resolves to ISO-8859-1, which would silently
  mis-decode them into control chars that CSafeLoader rejects.
- The document is parsed with `yaml.load(..., Loader=CSafeLoader)`. Anchors,
  aliases, and merge keys are resolved at parse time, before pydantic sees
  anything, so downstream an alias is indistinguishable from the expanded
  text. pydantic validation (with the `extra="forbid"` tripwire above) runs on
  the fully resolved mapping.
- After validation, `YamlIngestsParser` checks every `script` task `name`
  against `Script.REGISTRY`; an undeclared script raises with the list of
  declared scripts.
