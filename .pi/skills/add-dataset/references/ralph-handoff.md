# Ralph handoff: when an add is too big to drive inline

Lane 2 of this skill hands off to the global `ralph` skill (`~/.pi/agent/skills/ralph/SKILL.md`):
terse one-liner in, autonomous expansion into specs + an executable-test backlog, story-by-story
implementation in fresh-context workers, evidence gates, one commit per story, full audit at the end.
This page is the interface contract: what this skill still owns, what ralph gets, and the constraint
block that keeps ralph from running the suite on the laptop.

## Lane-2 triggers (any one is enough)

- a new `Script` subclass
- a new source kind (`models.py` + `registry.py` + a `DataStream`)
- a `ResolutionGate` / `PredicateRangeGate` change
- more than one split/subset declared, or a `LABEL_MAP` that needs full-split measurement
- a gazetteer overlay section in `ingests.yaml`
- the user asks for it, or the add simply arrives with three coupled decisions

Lane 1 (declaration-only) never reaches ralph.

## Division of labor

| owned by | what |
| --- | --- |
| this skill, BEFORE the handoff | intake answers, the T2 probe evidence (`PROBE_*` receipts), lane classification, the generated one-liner + constraint block, and the final PR step (the global `pr` skill) after ralph's Phase 2 audit passes |
| ralph, DURING | Phase 0 expansion (`.ralph/prd.json`, `specs/`, `architecture.md`, `impact-map.md`), Phase 1 stories with fresh implementer workers, evidence gates, one atomic commit per verified story, recovery ladder, Phase 2 audit |
| this skill, AFTER | read ralph's audit + the commit series, then author the PR to `main` |

`.ralph/` is gitignored in this repo already; the scheduler (you, mid-handoff) is the only reader and
writer of it. Workers never touch it.

## Step 1: generate the one-liner

Shape: `add <dataset> as a <task kind> ingest (<the 1-3 load-bearing decisions>), verified only on
wenceslaus`. Terse is correct -- Phase 0 expands it -- but the decisions must be in it, because the
expander cannot infer intent you leave out.

Real examples from landed loops:

- `add nvidia/Nemotron-PII as a script ingest with slice-authoritative PII spans, a two-tier label
  policy over both splits, and additive ResolutionGate person-name buckets, all heavy runs on
  wenceslaus`
- `declare the seven tensorshield reddit health corpora as fullmap ingests filtered by one
  match_on anchor over communityName, inclusion bar MIT and >=100k rows, everything measured by
  remote streaming probes`

## Step 2: attach the constraint block

Invoke `/skill:ralph <one-liner>`, and put this block into the Phase 0 expansion's `USER_INTENT`
(alongside the one-liner, verbatim). It exists because ralph's default build/test posture is "run the
tests here", which is exactly what must not happen in this repo.

```text
CONSTRAINTS (non-negotiable, from .pi/skills/add-dataset):
- ALL heavy execution runs on wenceslaus over ssh_bash (host alias `wenceslaus`, login shell bash).
  The laptop NEVER streams a dataset, never runs `build-dataset`, never runs the full suite or a
  coverage run, never downloads a corpus. Laptop work is limited to: editing files,
  `uv run pytest tests/test_ingests.py tests/test_streams.py -q`, and `uv run ruff check src tests`.
- The canonical gate is `bash .pi/skills/add-dataset/scripts/remote-gate.sh` (T1: rsync push, remote
  FULLMAP_DIR sed patch, uv sync, full pytest, ruff check + format) -> receipts PYTEST_EXIT:0,
  RUFF_EXIT:0, FULLMAP_SKIPS:0, VERDICT:PASS. Spot-check by reading the remote log yourself; a
  worker's pasted output is untrusted evidence.
- T2 evidence (before the final story): `remote-gate.sh probe -- --declared <key> ...` for the README
  numbers, a RELMEDNER_LIVE_HF=1 live-row test if one is added, and
  `remote-gate.sh --tmux --wait 2400 smoke` -> SMOKE_EXIT:0 with AVRO_RECORDS > 0.
- Long remote jobs (probes over >1k rows, smoke) go in remote tmux, never in a raw ssh call.
- The change set and repo invariants: .pi/skills/add-dataset/references/ingest-checklist.md (read it
  before Phase 0 writes any spec; its "Invariants" and "Failure modes" sections are settled law).
- Script class mechanics and the ScriptUtils map: references/script-template.md.
- Probes and measurements: references/probe-recipes.md. Measure, do not ask.
- Never retune constants.py mining gates; never re-hardcode the tests/test_outputs.py smoke bound.
- Plain ASCII everywhere. No em dashes, unicode arrows, check marks, or bullets.
- Workers: leave work UNCOMMITTED; the scheduler commits one atomic commit per verified story,
  message `feat: [US-XXX] <title>` with the measured numbers in the body.
```

## Step 3: shape the backlog (tell Phase 0, or fix it in review)

The landed loops converged on this story order; ask the expander for it explicitly:

1. **US-001 declarations + locks**: `ingests.yaml` entries, `EXPECTED` freeze blocks
   (`probe.py --freeze`), registry wiring if a class exists. Acceptance:
   `remote-gate.sh gate` green with the new locks present.
2. **US-002 the script class** (lane 2 core): decode/coerce, drop rules, `LABEL_MAP`, emit path.
   Acceptance: new `tests/test_<slug>.py` + `test_scripts.py` additions green, nonzero yield asserted.
3. **US-003 shared-machinery changes, only if the add truly needs them** (a `ResolutionGate` bucket,
   a new source kind). Strictly additive; every pre-existing case passes unmodified.
4. **US-004 measurement + docs**: full-split census probes, README table row + format notes,
   `LIVE_EXIT:0` + smoke receipts, the full suite on wenceslaus.
5. Optional **US-005 casing/measurement corrections** when a probe disagrees with a seed (the reddit
   allowlist needed exactly this story; budget for it).

Every story's `acceptance_tests` must name REMOTE commands and receipts, not local ones, and the
`definition_of_done` should read like the landed nemotron-pii one: full suite green on wenceslaus via
ssh_bash, ruff clean, split-aware locks present, script registered with real-shaped fixtures and
nonzero-yield asserted, no swallowed malformed rows, README updated with plain ASCII measured
numbers, no refactors beyond the change set, independent audit approving the tree.

## Step 4: run the loop, then come back

While ralph runs: the scheduler spot-checks every story by reading the remote log itself
(`ssh wenceslaus 'tail -40 ~/relmedner-<branch>-gate.log' | tr ':' '~'`), because `FULLMAP_SKIPS:0`
and a real pass count are the evidence, not a worker's summary.

After Phase 2: this skill takes over again. Verify the branch (ralph worked in this worktree on the
branch named in `prd.json`), collect the receipts from the final gates, and author ONE PR to `main`
with the global `pr` skill (conventional-commit title, `### Testing` quoting the real receipts, one
one-time pi-trust note only if the PR adds the skill itself).

## Hazards seen in real handoffs

- **A worker ran `make test` on the laptop** despite the block above: the dispatch prompt for every
  story must repeat the laptop-forbidden line, not rely on Phase 0 having read it.
- **The expander invented a local test command** in `acceptance_tests`. Review `prd.json` before the
  loop starts and rewrite any acceptance test that does not name wenceslaus.
- **Two implementers racing the shared checkout** is ralph's own documented failure mode; it applies
  here unchanged. One at a time.
- **Parallel dataset worktrees rebasing over `EXPECTED`**: tuple-position changes (a new model field)
  rebase-conflict every parallel branch. If the add needs one, say so in the one-liner and expect a
  coordination story.
- **The remote tree is a copy, not a git repo.** If a story's acceptance test assumes `git` on
  wenceslaus, it is wrong.
