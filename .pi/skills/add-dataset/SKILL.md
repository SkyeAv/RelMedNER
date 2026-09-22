---
name: add-dataset
description: "Add a training-data dataset (an ingest) to this repo end to end. The user describes the data; the skill measures it on the wenceslaus box over ssh_bash, wires the declarative ingest (ingests.yaml, optional Script class, tuple lock, tests), runs the remote gate (full suite plus ruff plus smoke plus probe receipts), writes the README ingest-table row and measured format notes, and ends in one PR to main. Triggers: add a dataset, add an ingest, ingest <corpus>, new dataset, wire up <dataset>, add <dataset> to the training data, or /skill:add-dataset. Heavy work never runs on the laptop: declaration-only adds are driven inline, adds needing new code hand off to the ralph skill with a pre-filled constraint block."
license: MIT
metadata:
  repo: RelMedNER
  remote-host: wenceslaus
---

# add-dataset: wire a new corpus into the relmedner pipeline

The user says "here is a dataset, add it". This skill turns that into a landed PR: measured,
declaratively wired, remote-verified, documented. It exists because the procedure used to live only
in git history and in gitignored `.ralph/` notes, and every new loop re-derived it -- sometimes
wrongly (stale smoke bounds, sample-sized label censuses, a crashed laptop).

## Non-negotiables

1. **The laptop stays light.** Editing files, `uv run pytest tests/test_ingests.py
   tests/test_streams.py -q`, and `uv run ruff check src tests` are the only local commands. Dataset
   streaming, `build-dataset` in any mode, the full suite, coverage runs, and corpus downloads run on
   wenceslaus over `ssh_bash`/`ssh_copy`, or not at all.
2. **Measure, do not ask.** Row counts, dtypes, span encodings, label censuses, drop rates, and
   yields come from `scripts/probe.py` on wenceslaus. One batched question round to the user covers
   only what a probe cannot answer (see `references/ingest-checklist.md`).
3. **Evidence, not claims.** Every claim of "it works" is a receipt: `PYTEST_EXIT:0`,
   `RUFF_EXIT:0`, `FULLMAP_SKIPS:0`, `LIVE_EXIT:0`, `SMOKE_EXIT:0`, `AVRO_RECORDS:<n>`. Spot-check
   the remote log yourself; pasted output is untrusted.
4. **ASCII prose everywhere**, including the README paragraph and the PR body.
5. **Never retune `constants.py` mining gates; never re-hardcode the `tests/test_outputs.py` smoke
   bound.** Both are settled by earlier measurements or by design.

## Workflow

### Step 0: orient (cheap, local)

Read `references/ingest-checklist.md` (the change set, invariants, failure modes) and
`docs/yaml-config.md` (the field contract). Skim the README's `## Ingests` table so the new corpus is
understood relative to the declared ones. Run the T0 checks once before editing anything, so a
pre-existing breakage is not blamed on the add later.

### Step 1: intake

Ask the user the intake questions (checklist section "Intake facts") in ONE round: where the data
lives, what a row carries, which column is text, label vocabulary shape, intent (weight, filters,
splits, license), and whether the hub repo is gated. Everything else gets measured in step 3.

### Step 2: lane classification

- **Lane 1** (declaration-only): a `fullmap` task over a text column, or a `script` task whose class
  already exists. Stay inline.
- **Lane 2** (new code): a new `Script` subclass, a new source kind, a gate change, multi-split, or a
  full-split label map. Hand off to the global `ralph` skill with the generated one-liner and the
  constraint block -- `references/ralph-handoff.md` is the exact procedure. The handoff still starts
  with steps 1 and 3: ralph's expander needs the probe evidence.

### Step 3: probe on wenceslaus

Follow `references/probe-recipes.md`: `--info` for row counts, shape discovery at 20-200 rows, the
FULL split for any label census, `--fullmap` for mined-yield expectations, `--declared` after the
entry exists. Long probes go in remote tmux (`remote-gate.sh --tmux --wait ...`). A gated repo needs
a token on the box first; say so rather than reporting a zero-row census as a property.

### Step 4: implement

Checklist sections 1-8, in order: `ingests.yaml` entry (validated with the docs' one-liner), optional
`Script` subclass per `references/script-template.md`, `scripts/__init__.py`, the `EXPECTED` lock
frozen via `probe.py --freeze` (never hand-written), `tests/test_<slug>.py` with one negative test
per drop rule, README untouched until step 6. A new source kind additionally touches `models.py`,
`registry.py`, a `DataStream`, `relmedner schema`, and `docs/yaml-config.md`.

### Step 5: remote gates

```bash
bash .pi/skills/add-dataset/scripts/remote-gate.sh                        # T1, every change
bash .pi/skills/add-dataset/scripts/remote-gate.sh cov                    # when coverage moved
bash .pi/skills/add-dataset/scripts/remote-gate.sh live -- tests/test_<slug>_live.py
bash .pi/skills/add-dataset/scripts/remote-gate.sh --tmux --wait 2400 smoke   # T2, once per dataset
```

T0 (local, seconds) after every edit; T1 before any commit; T2 before the PR. Receipt grammar, the
scripts' contracts, tmux rules, and the full gotcha list: `references/remote-wenceslaus.md`. Read
receipts through `tr ':' '~'` -- the renderer collapses ripgrep-shaped output.

### Step 6: docs

README: one ingest-table row (`rows in` measured and sourced) plus the format-notes paragraph built
from the probe numbers, plain ASCII. If a model field changed: `uv run relmedner schema` and the
matching `docs/yaml-config.md` row (both drift-guarded by tests).

### Step 7: PR

One PR to `main`. Author the body with the global `pr` skill: conventional-commit title, concern
sections, a `### Testing` section quoting the real receipts (`VERDICT:PASS`, pass counts,
`AVRO_RECORDS`), accepted caveats, and any deferred subsets. No `gh stack` here unless the repo's
state has changed; this repo's landed flow is a plain `gh pr create --base main` after approval.
Commit messages carry the measurements (the landed ingests put their numbers in the commit body).

## First run in a fresh worktree (pi project trust)

`.pi/skills/` is a trust-requiring project resource. At pi's startup prompt, answer **"Trust parent
folder (/home/skyeav/Code/ISB)"** once -- it covers the main checkout and every worktree, and headless
runs (`pi -p`, aoe/fleet workers) then load this skill too. Full detail: step 3 of
`references/remote-wenceslaus.md`.

## File map

| file | read when |
| --- | --- |
| `references/ingest-checklist.md` | before writing anything: change set, invariants, failure modes, DoD |
| `references/probe-recipes.md` | measuring: recipes per task type, and what each measurement decides |
| `references/script-template.md` | writing a `Script` class: skeleton, variants, `ScriptUtils` map |
| `references/remote-wenceslaus.md` | anything remote: host facts, sync routine, scripts, receipts, tmux, gotchas |
| `references/ralph-handoff.md` | lane 2: one-liner, constraint block, story shapes, hazards |
