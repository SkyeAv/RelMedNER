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
| `gazetteer` | no | absent | `GazetteerSpec` block (see below) | additive overlay over the builtin relation-gazetteer trigger tables; rebuilt from builtins + this section on every parse |

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

### The `gazetteer` section (`GazetteerSpec`)

Optional top-level overlay on the relation gazetteer (`relmedner.gazetteer`) that fullmap mining
scans for `evidence="distant"` relations. Parsed by `YamlIngestsParser.parse_ingests` and applied
through `configure_gazetteer`, which REBUILDS the module trigger tables from the builtin
constants plus this section on every parse (last-parse-wins, idempotent); omitting the section
entirely leaves the builtin tables exactly as shipped. A section that declares none of its arms
(`gazetteer: {}`) is a validation error, not a no-op.

Worker re-application: Beam serializes DoFn instances, not post-import module state, so the
parsed spec is additionally carried through the pipeline as data on the fullmap mining DoFn
(`ResolveMinedBatches` in `src/relmedner/pipeline.py`), which re-runs `configure_gazetteer`
on every worker at `setup()` time. A Flink sdkworker therefore rebuilds its trigger tables
from the builtins plus this section before its first batch, exactly like the driver; the
driver-side configure inside `parse_ingests` stays as the idempotent double-configuration.

Merge rules (fail-loud, enforced at parse/configure time):

- ADDITIVE only: YAML `triggers` join the builtin phrases of the predicate they name, and a
  predicate name with no builtin entry is added outright. Nothing here can remove or shadow a
  builtin phrase.
- One phrase, one owner. A phrase claimed by two owners (YAML-vs-builtin or YAML-vs-YAML) fails
  with a `ValidationError` naming both owners, because the longest-match scanner would otherwise
  make the winner depend on table order.
- `name` must be a member of `tablassert.biolink.Predicates`; phrases must be non-empty, with
  non-empty, all-lowercase tokens (matching is against the lowercased token stream). An unknown
  key anywhere in the section fails validation (`extra="forbid"`).
- Zero-emission loudness: every YAML-declared predicate gets an emission counter, and a predicate
  that emitted zero relations by the end of mining logs ONE WARNING on the stdlib logger
  `relmedner.gazetteer`, so a mistyped trigger phrase can never silently ship an empty relation
  arm. Builtin predicates are not tracked (their yield is pinned by `tests/test_gazetteer.py`).

| field | required? | default | constraint | meaning |
| --- | --- | --- | --- | --- |
| `predicates` | no | absent | list of `GazetteerPredicate` | additive predicate arms (the only arm wired to the scanner in this tree) |
| `qualifiers` | no | absent | list of `GazetteerQualifier` | LANDS WITH PR #22: structurally validated only until then (see below) |
| `negation_cues` | no | absent | list of phrase lists | LANDS WITH PR #22: structurally validated only until then (see below) |

`GazetteerPredicate`:

| field | required? | default | constraint | meaning |
| --- | --- | --- | --- | --- |
| `name` | yes | none | must be a `tablassert.biolink.Predicates` member | the predicate the phrases trigger |
| `triggers` | yes | none | non-empty list of phrase lists; phrases non-empty, tokens non-empty and all-lowercase | the gap n-grams (1-6 tokens, no sentence break) that fire this predicate |

`GazetteerQualifier`:

| field | required? | default | constraint | meaning |
| --- | --- | --- | --- | --- |
| `slot` | yes | none | non-empty string | the biolink qualifier slot these triggers fill |
| `range` | no | absent | string | optional value range for the slot |
| `triggers` | no | empty | same phrase rules as `GazetteerPredicate.triggers` | the phrases that fill this qualifier |

Qualifiers and negation: the qualifier/negation scanner machinery (`QUALIFIER_TRIGGERS`,
`QUALIFIER_RANGES`, `DISABLED_QUALIFIERS`, the biolink `Qualifiers` membership and disabled-slot
checks) lives on the `add-qualifiers-to-relationship-pipelines` branch and lands with PR #22.
Until then, `qualifiers` and `negation_cues` are STRUCTURALLY validated at parse time (shape,
phrase rules above) but `configure_gazetteer` raises a structured `NotImplementedError` naming
PR #22 the moment either arm is declared: fail-loud, never silent accept-and-ignore. The same
deferral applies to `GazetteerQualifier.slot`: only structural (non-empty string) validation
runs now; the biolink `Qualifiers`-membership validator for the slot lands WITH PR #22's
substrate (the scanner machinery above), not before it. When PR #22 merges, the same models
extend to full validation without a YAML-format change.

Copy-paste example (adds "cures" as a `treats` trigger):

```yaml
gazetteer:
  predicates:
    - name: treats
      triggers:
        - [cures]
```

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
| `path` | yes | none | filesystem path | avro container built out-of-band. Used as declared when it names an existing file (absolute, `~`-expanded, or relative to the caller's CWD); otherwise resolved against the package data dir (`relmedner.constants.DATA`). The whole record ships to the declared script; there is no `columns_out` projection because the file's own schema is the contract. Doubles as the row key for weight stamping |

### `match_on` entries (`MatchOn`)

| field | required? | default | constraint | meaning |
| --- | --- | --- | --- | --- |
| `column` | yes | none | string | dataset column to test |
| `values` | yes | none | list of strings | accepted values; rows whose `column` is not in this list are skipped |

### Row filters (`filters` / `RowFilters`)

Every dataset entry (hf and local) accepts an optional `filters` block: declarative row-quality
filters applied by the stream AFTER `match_on`, before the row reaches the declared task. Filters
sit outside the frozen payload tuple, so adding or changing a `filters` block never shifts any
tuple position.

Drop reasons evaluate in this fixed order; the first match drops the row:

1. `drop_empty` - every projected value is None, `""`, or an empty list/tuple/dict
2. `min_text_len` - the joined text is shorter than this
3. `max_text_len` - the joined text is longer than this
4. `max_tokens` - the joined text is longer than the always-on platform token cap (see below)
5. `include_regex` - the joined text does NOT match this pattern
6. `exclude_regex` - the joined text DOES match this pattern

| field | required? | default | constraint | meaning |
| --- | --- | --- | --- | --- |
| `drop_empty` | no | false | bool | drop rows where every projected value is empty (None, `""`, or an empty list/tuple/dict) |
| `min_text_len` | no | absent | int >= 0 | drop rows whose joined text is shorter than this |
| `max_text_len` | no | absent | int >= 0 | drop rows whose joined text is longer than this; must be >= `min_text_len` when both are set |
| `include_regex` | no | absent | string | drop rows whose joined text does not match (unanchored `re.search` semantics) |
| `exclude_regex` | no | absent | string | drop rows whose joined text matches |

The always-on `max_tokens` cap: a platform constant, NOT a field you declare. Every source,
even one with no `filters` block at all, drops rows whose joined text is longer than the cap.
Estimated tokens = `len(joined_text) // 4` (the ~4 chars-per-token rule of thumb for English
prose, https://help.openai.com/en/articles/4936856-understanding-and-counting-tokens); the row
drops iff `len(joined_text) > MAX_TEXT_TOKENS * CHARS_PER_TOKEN = 8192 * 4 = 32768` chars.
Medical text runs denser than average prose, so the estimate carries roughly a one-token-in-four
error margin in either direction. Semantics are drop, never truncate: an over-cap row is dropped
whole so no partial context ever reaches a task. The cap needs no YAML declaration and cannot be
waived per dataset; it sits after the declared length rules (a row that already fires
`max_text_len` keeps that attribution) and before the regexes (an over-cap row never attributes
to a regex).

The text rule: the text every length/regex filter measures is `" ".join` over the projected
payload values in order. A value contributes as itself when it is a str, as its elements joined
when it is a list/tuple of str (this is what makes `tokenized_text` columns filterable), and not
at all otherwise. For a local avro dataset the same rule applies over the record's field values.
A row whose values carry no str at all has text `""`.

Tripwires:

- Unknown keys inside `filters` fail validation (`extra="forbid"` applies there too), as does
  `min_text_len` > `max_text_len`.
- An invalid regex fails at stream construction with `re.error`, before any row streams.
- Fail-loud zero-yield guard: when a declared filter OR the always-on `max_tokens` cap
  drops 100% of the rows of a non-empty source, the run raises `ZeroYieldError` instead of
  silently shipping an empty training set.
  A genuinely empty source (0 rows in) is not an error. A dataset with no `filters` block keeps
  the historical unfiltered behavior exactly.

Copy-paste example (keep only the rows longer than 20 chars, drop empties):

```yaml
- task: {type: script, name: GlinerBiomedScript, outputs: [entities]}
  source: hf
  dataset: anthonyyazdaniml/gliner-biomed-pre-training
  split: train
  columns_out: [tokenized_text]
  filters: {drop_empty: true, min_text_len: 20}
```

### Reading ingest quality numbers

Every pass over a dataset's rows ends with exactly one INFO line on the stdlib logger
`relmedner.quality` (no logging setup needed: the Beam DirectRunner and the flink sdkworkers
capture logged records through their own handlers):

    ingest quality anthonyyazdaniml/gliner-biomed-pre-training: rows_in=1000 rows_out=842 dropped={match_on:150, min_text_len:8}

How to read it:

- `rows_in` counts every row read from the source, before `match_on` (for a local avro
  source: every record read).
- `rows_out` counts the rows handed to the declared task.
- `dropped` lists one `reason:count` pair per drop reason, in the order reasons first fired:
  `match_on` (which used to be silent) plus the filter reasons above. It is omitted entirely
  when nothing was dropped.
- The numbers reconcile: `rows_in - rows_out` equals the sum of the `dropped` counts. If they
  do not add up, the run was truncated (for example `sample_limit`) before the source was
  exhausted.

Honest limitation: the counts are per stream instance, which under Beam means per worker
process, not a run-global aggregate; a multi-worker run logs one line per worker. If a global
aggregate is ever needed, Beam's built-in Metrics counters are the path.

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
    path: interventions/interventions.avro
```

`path` is used as declared when it names an existing file -- absolute, `~`-expanded via
`expanduser`, or relative to the caller's CWD -- and resolves against the package data dir
(`relmedner.constants.DATA`) otherwise; `local_delimited` follows the same rule. Resolution
happens when the stream is built, not when the YAML is parsed, and a path that resolves to no
existing file raises `FileNotFoundError` when its rows are read, never a silent empty stream.

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

## Machine-readable schemas

Two JSON Schema artifacts (Draft 2020-12), generated from the models above,
are checked in at the repo root:

- `schemas/ingests.schema.json` -- the `ingests.yaml` contract (validation shape).
- `schemas/cluster.schema.json` -- the `cluster.yaml` contract.

They are GENERATED, never hand-edited: `uv run relmedner schema` regenerates
both (use `-o/--output-dir` to write elsewhere), and the drift guard in
`tests/test_schemas.py` fails CI if a model change lands without rerunning it.
The schemas describe the validation shape only; tuple packing
(`to_tuple`/`tuple_fields`) is internal and deliberately not represented.

Editor and agent recipe: point the yaml-language-server at the artifact as the
first line of the YAML file you are editing, and the editor autocompletes and
pre-validates against the same contract the loader enforces:

```yaml
# yaml-language-server: $schema=../schemas/ingests.schema.json
x-defaults:
  ...
```

Relative paths resolve against the edited file's directory, so a YAML file in
`src/relmedner/data/` uses the `../..` depth shown above. Agents can likewise
pre-validate a candidate config against `schemas/ingests.schema.json` before
the pipeline ever runs.
