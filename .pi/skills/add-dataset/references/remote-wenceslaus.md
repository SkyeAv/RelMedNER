# Remote execution on wenceslaus

Everything heavy runs on wenceslaus. The laptop edits files and runs parse-scale checks only.
This page is the measured contract: host facts, the sync routine, the remote environment, the
receipt grammar, tmux discipline, and the gotchas that have already cost a loop.

## Why remote

- Dataset streaming, `build-dataset`, and the full suite are minutes-to-hours of CPU plus
  gigabytes of hub downloads. On the laptop they have crashed sessions before; wenceslaus has
  80 cores and a warm uv cache.
- The fullmap bundle the miner needs is not on the laptop path this repo hardcodes, so a remote
  run is also the only run that exercises fullmap-dependent tests instead of skipping them.

## Host facts (measured over `ssh_bash` on 2026-09-22)

| fact | value |
| --- | --- |
| pi ssh host name | `wenceslaus` (`ssh.json` -> `sgoetz@wenceslaus` via `gateway`) |
| canonical home | `/users/sgoetz` (`/home/sgoetz` is the same tree) |
| cores / disk | 80 cores; `/home` 1.5T with 405G free (74% used) |
| non-interactive shell | `/bin/bash` (`/usr/bin/bash`). `zsh` is NOT on PATH; it lives at `/users/sgoetz/linuxbrew/bin/zsh` and the older hand-written `~/run-*.zsh` runners use that absolute shebang |
| uv | `0.12.17` at both `/users/sgoetz/bin/uv` and `/users/sgoetz/.local/bin/uv`. The snap `uv` on the default PATH is broken: always invoke uv by absolute path |
| python3 | `3.12.3` (system). The suite runs under uv's own interpreter, not this one |
| tmux / rsync | `/usr/bin/tmux`, `/usr/bin/rsync` |
| fullmap bundle | `~/Desktop/fullmap` -> `/local_raid1/sgoetz/DBSTORE/FULLMAP/fullmap`, holding `data/fullmap.redb` plus `data/fullmap.s<N>.redb` shards |
| local-source fixture | `~/Desktop/interventions.avro`, 92,592,955 bytes (md5 `bee03c038faf3c5d25d4d77d338ad81c`) |
| hf cache | `~/.cache/huggingface/{datasets,hub,xet}` warm; NO token file, so gated/private hub repos cannot stream until one is added |
| uv cache | `~/.cache/uv`, 7.7G warm (`uv sync` on a re-synced tree takes seconds) |
| existing remote copies | `~/Code/RelMedNER`, `~/Code/RelMedNER-worktrees/{interventions-avro,reddit-dataset,super-glue-record}` |
| other live tmux sessions | `dakp1`, `relmed-smoke`, `relmedner-tunnel-10-2-9-19` (other agents' work: never kill them) |

Re-verify before trusting any of it in a new loop; the box is shared and its state moves:

```bash
ssh_bash wenceslaus '{ hostname; nproc; ~/.local/bin/uv --version; ls -ld ~/Desktop/fullmap; \
  ls -l ~/Desktop/interventions.avro; tmux ls; } 2>&1 | tr ":" "~" | base64 -w0'
```

## Directory convention

Remote trees are PLAIN rsynced copies, not git repos and not worktrees: one directory per laptop
worktree, named after it.

    laptop:  /home/skyeav/Code/ISB/RelMedNER-worktrees/<branch>
    remote:  ~/Code/RelMedNER-worktrees/<branch>

Never run `git` on the remote copy; there is no `.git` there (rsync excludes it). Commits happen on
the laptop only.

## Sync routine

`scripts/remote-gate.sh` (in this skill directory, run from the laptop) performs the whole routine.
The steps it performs, spelled out so they can be run or repaired by hand:

1. Push the tree, excluding anything that would poison the remote venv or bloat the transfer:

```bash
rsync -az --delete \
  --exclude='.git' --exclude='.venv' --exclude='.ralph' \
  --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.ruff_cache' \
  --exclude='.coverage' --exclude='dist' --exclude='*.avro' \
  /home/skyeav/Code/ISB/RelMedNER-worktrees/<branch>/ \
  wenceslaus:Code/RelMedNER-worktrees/<branch>/
```

   `--exclude='.venv'` is load-bearing: without it uv rebuilds the remote environment from the
   laptop's, which is a different platform. `--delete` keeps the copy honest, so a file deleted on
   the laptop cannot survive remotely and quietly satisfy an import.

2. Patch the remote copy's `FULLMAP_DIR` (remote copy only; the laptop tree and git history are
   never touched). `src/relmedner/constants.py` hardcodes `/home/skyeav/Desktop/fullmap`, which does
   not exist on wenceslaus, so out of the box every fullmap-dependent test skips
   (`FullmapMiner.available()` / `ScriptUtils.fullmap_available()`) and smoke mining mines nothing:

```bash
ssh wenceslaus 'cd ~/Code/RelMedNER-worktrees/<branch> && \
  sed -i "s#/home/skyeav/Desktop/fullmap#$HOME/Desktop/fullmap#" src/relmedner/constants.py && \
  grep -n FULLMAP_DIR src/relmedner/constants.py'
```

   The `sed` writes the remote `$HOME` form (`/users/sgoetz/...`), NOT a `/home/sgoetz` literal: the
   runner exports `RELMEDNER_FULLMAP_DIR=$HOME/Desktop/fullmap` and `tests/test_constants.py`
   compares the env value against the envless default as a STRING, so the two forms must match
   textually (a `/home/sgoetz` literal fails two FULLMAP_DIR tests; measured 2026-09-22 on the
   untouched main checkout too). The `sed` is idempotent (a second run finds no laptop literal). The
   `grep` is the receipt: it must print the `/users/sgoetz/Desktop/fullmap` path before any remote
   run. A branch that has adopted the `RELMEDNER_FULLMAP_DIR` env override still works, because the
   literal is inside the `environ.get` fallback and the runners also export the variable.

3. Run the remote executor and read receipts (next section).

## Remote environment

Every remote invocation exports the same block (the runners in `scripts/` do this; hand-written
`ssh` commands must too):

```bash
export PATH="/home/sgoetz/bin:$PATH"
export LC_ALL=C.UTF-8
export LANG=en_US.UTF-8
export PYTHONUTF8=1                       # required: the login locale has UTF-8 gaps
export UV_CACHE_DIR=/home/sgoetz/.cache/uv
export XDG_CACHE_HOME=/home/sgoetz/.cache
export RELMEDNER_FULLMAP_DIR=/home/sgoetz/Desktop/fullmap   # no-op until constants.py reads it
/home/sgoetz/bin/uv ...                    # absolute path, never bare `uv`
```

## Receipt grammar

A run either prints receipts or it did not happen. The runners append these to the log; the gate
greps them back:

| receipt | meaning |
| --- | --- |
| `FULLMAP_DIR:<path>` | the fullmap bundle the remote run resolves |
| `FULLMAP_REDB:present\|MISSING` | `data/fullmap.redb` exists under that path |
| `SYNC_EXIT:<n>` | exit code of `uv sync` |
| `PYTEST_EXIT:<n>` | exit code of the suite; `0` is the only pass |
| `SUITE_SUMMARY:<pytest tail line>` | e.g. `564 passed, 1 skipped in 243.63s`, so a pass-count regression is visible |
| `FULLMAP_SKIPS:<n>` | count of `fullmap database is not mounted` skips; must be `0` after the patch |
| `RUFF_CHECK_EXIT:<n>` / `RUFF_FORMAT_EXIT:<n>` / `RUFF_EXIT:<n>` | lint, format check, and their conjunction |
| `COV_EXIT:<n>` / `COVERAGE:<TOTAL line>` | the `--cov-fail-under=90` run and its measured total |
| `LIVE_EXIT:<n>` | exit code of the `RELMEDNER_LIVE_HF=1` live-row test |
| `PROBE_EXIT:<n>` plus `PROBE_*` | `probe.py` exit code and every measurement it printed |
| `SMOKE_EXIT:<n>` | exit code of `relmedner build-dataset -t --direct` |
| `AVRO_RECORDS:<n>` | record count in the smoke avro; must be > 0 |
| `AVRO_SHAPES:<shape=n,...>` | populated-shape census over the smoke records |
| `AVRO_WEIGHTS:<w=n,...>` | stamped mixing weights (all `1.0` today) |
| `QUALITY_LINES:<n>` | `relmedner.quality` lines captured from the smoke log; `0` is normal under Prism |
| `GATE_EXIT:<n>` | suite plus lint combined; `0` is the only pass |
| `VERDICT:PASS\|FAIL` | the runner's own final verdict, so a polling caller never parses the log |
| `RUN_ERROR:<why>` | the run never started (missing tree, missing uv, unknown mode) |

Skip audit matters as much as the exit code: a green `N passed` can hide the fullmap tests skipping,
which is exactly the silent-nothing-happened failure this routine exists to prevent.

`QUALITY_LINES:0` is expected, not a bug: the per-ingest `relmedner.quality` INFO records do not reach
a Prism `--direct` run's captured stdout (measured: 0 lines in a 62-record smoke). Per-ingest
`rows_in`/`rows_out` evidence comes from `probe.py --declared`, which reads the stream's own counters
and prints them as `PROBE_QUALITY`.

Reading receipts back through pi: the bash-result renderer collapses output shaped like ripgrep
results (`name: N: text`), which receipt lines and log tails trigger. `remote-gate.sh` already pipes
its receipt and tail output through `tr ':' '~'`; do the same for any hand-written `ssh` read.

## Committed scripts (this skill's `scripts/`, measured 2026-09-22)

| script | runs on | what it does |
| --- | --- | --- |
| `remote-gate.sh` | laptop | rsync push -> remote `FULLMAP_DIR` `sed` + grep receipt -> run the mode -> print receipts and a PASS/FAIL verdict. Options: `--host`, `--name`, `--base`, `--no-sync`, `--tmux`, `--session`, `--wait SECONDS`. Modes: `gate` (default), `cov`, `tests`, `lint`, `sync`, `live`, `smoke`, `probe`. Extra args go after `--` |
| `remote-runner.sh` | wenceslaus | the executor: env block, fullmap evidence, then the mode, appending every receipt to the log |
| `remote-smoke.sh` | wenceslaus | `relmedner build-dataset -t --direct -o OUT`, then a fastavro record count, shape census, and weight census |
| `probe.py` | wenceslaus | the measurement harness (see `probe-recipes.md`) |

Typical invocations, all from the laptop worktree root:

```bash
bash .pi/skills/add-dataset/scripts/remote-gate.sh                       # T1 gate
bash .pi/skills/add-dataset/scripts/remote-gate.sh cov                   # T1 plus coverage floor
bash .pi/skills/add-dataset/scripts/remote-gate.sh live -- tests/test_<slug>_live.py
bash .pi/skills/add-dataset/scripts/remote-gate.sh --tmux --wait 2400 smoke
bash .pi/skills/add-dataset/scripts/remote-gate.sh probe -- --declared 'aps/super_glue:multirc' --limit 20
bash .pi/skills/add-dataset/scripts/remote-gate.sh --no-sync probe -- --dataset <repo> --info
```

Measured costs on wenceslaus (this worktree, warm uv cache, 2026-09-22):

- `gate` on a FRESH remote copy: `uv sync` plus 564 passed / 1 skipped in 243.63s, `FULLMAP_SKIPS:0`,
  ruff check and format clean, `GATE_EXIT:0`, `VERDICT:PASS`. The 1 skip is the `RELMEDNER_LIVE_HF`
  gated live test.
- `smoke`: 62 avro records (`entities=56 relations=21 classifications=3`, every record `weight=1.0`)
  in about 3 minutes, `SMOKE_EXIT:0`, `VERDICT:PASS`.
- `probe --info`: seconds, no download. Validated against the README's published figures for
  `TrialPanorama/TrialPanorama-database`: `studies:all=1332141`, total `27446320`.

A smoke run WITHOUT `--direct` submits to the LAN flink cluster whose jobmanager is the laptop
(`10.2.9.11:18081`); from wenceslaus that job goes RUNNING -> RESTARTING -> FAILED unless a VPN route
plus an ssh tunnel to the jobmanager ports exist. `remote-smoke.sh` therefore defaults to `--direct`,
and cluster runs stay out of this skill's scope.

## Gate tiers

| tier | where | what | when |
| --- | --- | --- | --- |
| T0 | laptop | `uv run pytest tests/test_ingests.py tests/test_streams.py -q`, `uv run ruff check src tests` | every edit; seconds; parse/lock/lint only |
| T1 | wenceslaus | `uv sync` + full `pytest -q` + `ruff check` + `ruff format --check` via `scripts/remote-gate.sh` | every change, before any commit |
| T2 | wenceslaus, tmux | `scripts/probe.py` census, `RELMEDNER_LIVE_HF=1` live row, `scripts/remote-smoke.sh` (`build-dataset -t` + avro count) | once per dataset, before the PR |
| optional | wenceslaus, tmux | full `build-dataset --direct` (multi-hour) | only on explicit user request |

Forbidden on the laptop: hub streaming, `build-dataset` in any mode, the full suite, coverage runs,
and any probe that downloads a corpus.

## tmux discipline

The gateway jump path idle-kills SSH masters server-side, so a long job in a raw `ssh_bash` call dies
mid-run. Long work goes in remote tmux:

```bash
ssh wenceslaus 'tmux new-session -d -s add-dataset-smoke "cd ~/Code/RelMedNER-worktrees/<branch> && bash .pi/skills/add-dataset/scripts/remote-smoke.sh \$PWD \$HOME/smoke.log"'
ssh wenceslaus 'tmux capture-pane -p -t add-dataset-smoke | tail -30'
ssh wenceslaus 'tail -40 ~/smoke.log | tr ":" "~"'
```

`remote-gate.sh --tmux --wait SECONDS` wraps exactly this (launch, then poll for the `VERDICT`
receipt) and is the path to prefer.

Rules:

- Pick a session name unique to this dataset (`add-dataset-<slug>-smoke`). Other agents' sessions
  (`dakp1`, `relmed-smoke`, ...) are live work: never `tmux kill-server`, never `kill-session` on a
  name you did not create.
- The session dies with the command it wraps, so the command must `tee`/redirect to a log file and
  append its own receipt. Read the log, not just the pane.
- Poll with `tmux capture-pane -p` plus `tail -n` on the log; do not block a single `ssh_bash` call
  on a multi-minute job (the tool has its own timeout).

## Gotchas (each one already cost a loop)

1. **`rsync` without `--exclude='.venv'`** rebuilds the remote environment from a laptop-platform venv.
2. **Skipping the `sed`** leaves every fullmap test skipped and the smoke run mining nothing, while
   pytest still exits 0. Audit `FULLMAP_SKIPS:0`.
3. **`~/Desktop/interventions.avro` must exist remotely** or `tests/test_outputs.py`'s local-source
   ingest test fails inside Beam Prism with `FileNotFoundError`. It is present (92,592,955 bytes); if a
   fresh box lacks it, `ssh_copy` it up once rather than deleting the test.
4. **Bare `uv`** resolves to a broken snap. Absolute path only.
5. **No `PYTHONUTF8=1`** turns packaged non-ASCII reads into decode failures under the login locale.
6. **Receipt lines read back mangled** through pi's renderer; use `tr ':' '~'` or `base64`.
7. **No hub token on the box.** Public repos stream fine; a gated repo needs `hf auth login` (or
   `HF_TOKEN` exported in the runner) before the probe can measure anything. Say so loudly instead of
   reporting a zero-row census as a dataset property.
8. **`test_outputs.py` derives its smoke bound from the declaration.** Adding a dataset must not
   re-hardcode a dataset count there; two past branches re-staled that bound.
9. **Remote copies are not git repos.** No `git` on the remote; no commits from the remote.
10. **The box is shared.** Check `tmux ls` and load before launching a multi-hour build; a full
    `--direct` run competes with other agents' DAGs.

## pi project trust (one-time, user-level)

`.pi/skills/` is a trust-requiring project resource, so pi asks once per folder before loading this
skill. Answer the startup prompt with **"Trust parent folder (/home/skyeav/Code/ISB)"**: pi's trust
store resolves the nearest ancestor entry, so one answer covers the main checkout and every
`RelMedNER-worktrees/*` worktree. `/trust` in interactive mode does the same (restart pi afterwards;
the current session is not reloaded).

Headless runs (`pi -p`, `--mode json|rpc`, therefore aoe/fleet workers) show NO prompt and IGNORE
project resources unless one of these applies: a saved `~/.pi/agent/trust.json` entry from the above,
`defaultProjectTrust: "always"` in `~/.pi/agent/settings.json`, or `--approve` / `-a` on the
invocation. If a worker silently behaves as though this skill does not exist, that is why.
