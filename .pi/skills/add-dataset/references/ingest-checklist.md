# Ingest checklist: what adding a dataset actually touches

The change set below is derived from the landed ingests (`sentence_rex`, `knowledgator/biomed_NER`,
`Pile-NER-type`, `gliner-multilingual-synthetic`, `PubMedAbstractsNER`, `TrialPanorama`,
`SuperGLUE MultiRC`, `SuperGLUE ReCoRD`, CTKP interventions). Follow it in order; each step names the
guard that catches a mistake.

## Lane classification (decide this first)

| the add needs | lane |
| --- | --- |
| a `fullmap` task over a text column, no new code | 1 (inline) |
| a `script` task whose `Script` class already exists | 1 (inline) |
| a new `Script` subclass | 2 (ralph) |
| a new source kind (`models.py` + `registry.py` + a `DataStream`) | 2 (ralph) |
| a `ResolutionGate` / `PredicateRangeGate` change | 2 (ralph) |
| more than one split/subset declared, or a measured `LABEL_MAP` over the full split | 2 (ralph) |
| a gazetteer overlay (`gazetteer:` section in `ingests.yaml`) | 2 (ralph) |

Lane 2 hands off to the global `ralph` skill; see `ralph-handoff.md`. Lane 1 still runs every gate
below, it just does not need a backlog.

## Intake facts (ask the user only these)

- where the data lives: hub repo id, subset/config, split, file; or a local avro / delimited path
- what one row carries: unlabeled text, gold spans (and their encoding), gold relations, or labels
- which column is the text, and which columns are signal versus provenance or PII
- label vocabulary shape: closed or open, language, biomedical or general domain
- intent: mixing `weight`, row filters, subset/split restriction, license
- access: is the hub repo gated (needs a token on wenceslaus)?

Everything else is measured on wenceslaus with `scripts/probe.py`. A question the probe can answer is
a wasted round trip.

## The change set

### 1. `src/relmedner/data/ingests.yaml`

One entry per declared ingest. Reuse the `x-defaults` anchors already in the file (`&hf-train`,
`&weight`, `&fullmap-task`); alias depth stays one level and every entry still names its dataset id,
its task, and its `columns_out`.

```yaml
  - task: {type: script, name: <ScriptName>, outputs: [entities]}
    <<: *hf-train
    dataset: <org>/<name>
    subset: <config or omit>
    columns_out: [<text column>, <label column>]
```

Every model sets `extra="forbid"`: a typo'd or invented key anywhere fails validation, so there is no
such thing as a harmless convenience key. Field contract: `docs/yaml-config.md`.

Prove a candidate entry before committing it (docs/yaml-config.md's own recipe):

```bash
uv run python -c "import yaml; from relmedner.models import YamlIngests; \
  YamlIngests.model_validate(yaml.load(open('src/relmedner/data/ingests.yaml'), Loader=yaml.CSafeLoader))"
```

Two keys collide easily and both fail loud, which is the point:

- `row_key` (weight stamping) is the repo id for `hf`/`hf_json` and the path for the local sources, so
  two subsets of one repo share a weight slot and must declare the same `weight`
  (`YamlIngests.weights_by_source` raises otherwise).
- `entry_key` (the test lock) is the repo id plus a scalar discriminator, so two subsets of one repo
  are two distinct locks. A bare repo id would silently lose one of them.

### 2. `src/relmedner/scripts/<slug>.py` (script tasks only)

A `Script` subclass: `NAME: ClassVar[str]` matching the yaml `task.name`, and one
`run(values: ScriptValues) -> TrainingExample`. `__init_subclass__` self-registers it into
`Script.REGISTRY`; `YamlIngestsParser.parse_ingests` then rejects an undeclared `task.name` with the
list of declared scripts. Skeleton and the `ScriptUtils` map: `script-template.md`.

### 3. `src/relmedner/scripts/__init__.py`

Import the class and add it to `__all__`, alphabetically. `__all__` stays Script-only: dataset-local
helpers are imported from their module in tests (`from relmedner.scripts.sentence_rex import
parse_tagged_sentence`), never re-exported here.

### 4. `tests/test_ingests.py`: freeze the tuple lock

Add one additive `EXPECTED[entry_key]` block. Generate it, do not hand-write it:

```bash
bash .pi/skills/add-dataset/scripts/remote-gate.sh probe -- --freeze '<org>/<name>[:<subset|file>]'
```

Paste the printed block into `EXPECTED`, then let `ruff format` normalize quotes and layout. The
printed tuple comes from the live `generate_tuples()` output, so a declaration typo cannot be
laundered into a lock that asserts the wrong shape. Guards:
`test_the_declared_tuple_shape_is_locked_per_ingest` (parametrized over `EXPECTED`) and
`test_every_declared_ingest_is_accounted_for` (set equality, which is what notices the new row).

Tuple positions are frozen by `tuple_fields` on each dataset model. Adding a model field is an
explicit tuple change: append it to `tuple_fields`, update every `EXPECTED` block, and expect every
parallel dataset branch to need a rebase. That blast radius is why per-dataset limits and other
conveniences stay off the payload.

### 5. `tests/test_<slug>.py`

Real-shaped fixtures copied from the probe output, not invented ones. Required coverage:

- one test per drop rule, each asserting the row/span is DROPPED (skip-don't-coerce), plus the
  well-formed sibling that ships
- a nonzero-yield test over a handful of realistic rows (the silent-zero-yield bug is the failure
  mode this repo has already shipped once)
- every emitted mention occurs in the emitted `text` (gliner2's `InputExample.validate()` rule)
- label-map edges: a raw label that resolves, a variant that only the normalizing fallback resolves,
  and a label that deliberately stays unmapped
- if a live hub row is worth pinning, add `tests/test_<slug>_live.py` gated on
  `RELMEDNER_LIVE_HF=1` with every `datasets`-importing statement INSIDE the test body, so
  `pytest --collect-only` stays offline (`tests/test_super_glue_record_live.py` is the pattern)

### 6. `tests/test_outputs.py`: touch nothing

The live-smoke bound already derives from `len(Ingests.datasets) * TEST_ROW_LIMIT`, and the shape
assertion is a subset check. Two past branches re-hardcoded a dataset count here and re-staled it.
If this file needs a change, the change is wrong.

### 7. `README.md`

- one row in the `## Ingests` table: `| dataset | task | inputs | outputs | rows in |`. `rows in` is
  a measured count with its source named (probe `PROBE_ROWS` or the datasets-server size endpoint);
  write `--` only when a run genuinely has not measured it.
- one `Dataset-format notes (<ScriptName>)` paragraph (or a `## <corpus>` section for a big one)
  carrying the measured facts: row encoding, span convention and bounds, drop counts per rule, the
  label census and its coverage, the resolution-origin shares, and any deliberate omission. Plain
  ASCII, no em dashes or unicode arrows.
- the ingest table is the list that stays current: do not re-enumerate datasets in prose bullets.

### 8. New source kind only

`models.py` (dataset model + `tuple_fields` + `row_key` + the `Dataset` union), `registry.py`
(`SOURCE_REGISTRY`), a `DataStream` subclass, then:

```bash
uv run relmedner schema        # regenerate schemas/ingests.schema.json
```

Guards: `tests/test_schemas.py` fails if the checked-in schema drifts from the models, and
`tests/test_ingests.py::test_every_yaml_model_field_is_named_in_docs_yaml_config` fails if a new
model field is not documented in `docs/yaml-config.md` (backtick-anchored match, so prose does not
satisfy it). Add the field table row and a copy-paste template to `docs/yaml-config.md` in the same
commit.

## Invariants (each one is a landed decision, not a preference)

- **Skip, don't coerce.** A malformed row or span drops; nothing is guessed. Text-only rows ship and
  the declared-outputs filter drops them.
- **`outputs` is a permitted-shapes contract.** A row ships when it produced at least one declared
  shape; producing fewer is fine, producing extra (a gazetteer relation on an `[entities]`
  declaration) is free signal, not a violation.
- **One resolution chain.** fullmap -> lowercased `FALLBACK_LABEL_MAP` (dataset vocabulary rides via
  `resolve_mentions(label_map=...)`) -> raw PascalCase. `ResolutionGate` / `PredicateRangeGate`
  rejections fall through to the next tier; they never drop a mention.
- **Trust gold when gold is right.** A corpus whose spans are already gold, or whose text is outside
  the biomedical vocabularies fullmap resolves against, skips re-resolution
  (`CtkpInterventionsScript`, `SuperGlueRecordScript`). Say why in the README.
- **Emitted `text` must contain every mention surface.** Re-join the token stream rather than
  shipping raw text; a detached punctuation token otherwise fails gliner2's validator.
- **Mining gates and knobs in `constants.py` are fixed by measurement.** Ingest work never retunes
  them. If a gate must move, that is its own PR with its own measurement.
- **The default suite is offline.** Anything that streams from the hub is gated behind
  `RELMEDNER_LIVE_HF=1` and lives in a `*_live.py` file.
- **ASCII prose** in README, docs, docstrings, commit messages, and PR bodies.

## Failure modes already paid for

1. **Sample-sized label census.** A 901-row sample of `knowledgator/biomed_NER` missed `EVENTS`
   (23 spans) and understated two plural variants by two orders of magnitude. Census the FULL split
   before writing `LABEL_MAP` or the README numbers.
2. **Streaming corrupts a mixed-type column.** `knowledgator/PubMedAbstractsNER` promotes every `ner`
   cell to utf8 on a cold-cache streaming read, so `mention_spans` drops every span and the ingest
   silently yields nothing. When a probe shows stringified indices or quote-wrapped labels
   (`PROBE_SPAN_STRINGIFIED_INDICES > 0`), suspect the builder, and consider the `hf_json` route.
3. **Silent zero yield.** A declaration typo or an over-tight filter ships an empty training set with
   a green suite. `filters` now raises `ZeroYieldError` when a non-empty source drops 100%, and the
   smoke's `AVRO_RECORDS` plus the probe's `PROBE_DISPATCH_ROWS_EMITTING` are the evidence.
4. **Stale hardcoded bounds.** Any test that names a dataset count goes stale on the next add. Derive
   bounds from the declaration.
5. **A lock written from the intended shape rather than the real one.** Freeze it with `--freeze`.
6. **Skipping on the mount.** A remote suite where the fullmap tests skip still exits 0. Audit
   `FULLMAP_SKIPS:0`.

## Definition of done for the PR

- T0 clean on the laptop; T1 `VERDICT:PASS` on wenceslaus with `FULLMAP_SKIPS:0`
- T2 measured: probe census for every README number, `LIVE_EXIT:0` for a live-row test when one was
  added, `SMOKE_EXIT:0` with `AVRO_RECORDS > 0` and the new ingest's shape present in `AVRO_SHAPES`
- `EXPECTED` block frozen from real output; `tests/test_outputs.py` untouched
- README table row plus format notes, ASCII only; `docs/yaml-config.md` updated if a model field changed
- `uv run relmedner schema` re-run if any model changed, and the regenerated artifact committed
- One or more conventional commits with the measurement in the body (the landed ingests put their
  numbers and their "Landing notes" in the commit message, not only in the PR)
- PR body in the global `pr` skill's style, Testing section quoting the real receipts
