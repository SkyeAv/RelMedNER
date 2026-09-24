#!/usr/bin/env bash
# Remote executor for the relmedner add-dataset gate. Runs ON wenceslaus (bash), never on the laptop.
# It writes every receipt into LOG so a run is either provable or it did not happen.
#
# usage: remote-runner.sh TREE LOG MODE [extra args...]
#   TREE  absolute path of the synced remote copy (a plain rsync copy, NOT a git repo)
#   LOG   absolute path of the receipt/output log (truncated at the start of every run)
#   MODE  gate   uv sync + full suite + ruff check + ruff format --check + skip audit   (T1)
#         cov    gate plus the coverage floor run (--cov-fail-under=90)
#         tests  full suite only
#         lint   ruff check + ruff format --check only
#         sync   uv sync only
#         live   RELMEDNER_LIVE_HF=1 pytest over [extra args] (default: the live test file)
#         probe  uv run python .pi/skills/add-dataset/scripts/probe.py [extra args]
#
# receipts appended to LOG: FULLMAP_DIR, FULLMAP_REDB, SYNC_EXIT, PYTEST_EXIT, SUITE_SUMMARY,
# FULLMAP_SKIPS, RUFF_EXIT, COV_EXIT, LIVE_EXIT, PROBE_EXIT. See references/remote-wenceslaus.md.
set -uo pipefail

TREE="${1:?usage: remote-runner.sh TREE LOG MODE [extra args...]}"
LOG="${2:?usage: remote-runner.sh TREE LOG MODE [extra args...]}"
MODE="${3:?usage: remote-runner.sh TREE LOG MODE [extra args...]}"
shift 3

UV_BIN="${UV_BIN:-$HOME/bin/uv}"
# default must string-equal the literal the remote-gate.sh sed patch writes into constants.py:
# tests/test_constants.py captures FULLMAP_DIR (this export) at collection and reloads it envless
# against the patched literal, so a diverging default fails the envless-stability tests (the sed
# writes the remote $HOME form, /users/sgoetz/Desktop/fullmap on this box)
FULLMAP="${RELMEDNER_FULLMAP_DIR:-$HOME/Desktop/fullmap}"
SKILL_DIR="$TREE/.pi/skills/add-dataset"

# the remote environment block from references/remote-wenceslaus.md: absolute uv (the snap uv on
# PATH is broken), UTF-8 (the login locale has gaps), warm caches, and the fullmap override for
# branches whose constants.py reads it (the sed patch in remote-gate.sh covers the rest)
export PATH="$HOME/bin:$PATH"
export LC_ALL=C.UTF-8
export LANG=en_US.UTF-8
export PYTHONUTF8=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export RELMEDNER_FULLMAP_DIR="$FULLMAP"

: > "$LOG"
receipt() { printf '%s:%s\n' "$1" "$2" >> "$LOG"; }

receipt RUN_MODE "$MODE"
receipt RUN_TREE "$TREE"
receipt RUN_DATE "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
receipt UV_BIN "$("$UV_BIN" --version 2>&1 | head -1)"

if [ ! -d "$TREE" ]; then
    receipt RUN_ERROR "tree missing: $TREE (rsync push first)"
    exit 2
fi
if [ ! -x "$UV_BIN" ]; then
    # the snap uv on PATH is broken on this box, so a missing absolute uv is a hard stop, not a fallback
    receipt RUN_ERROR "uv not executable at $UV_BIN"
    exit 2
fi
cd "$TREE" || exit 2

# fullmap evidence first: a suite that skips every fullmap test still exits 0, so the mount is
# proven before anything runs rather than inferred afterwards
receipt FULLMAP_DIR "$FULLMAP"
if [ -f "$FULLMAP/data/fullmap.redb" ]; then
    receipt FULLMAP_REDB present
else
    receipt FULLMAP_REDB MISSING
fi
grep -n 'FULLMAP_DIR' src/relmedner/constants.py >> "$LOG" 2>&1

run_suite() {  # run_suite [pytest args...]
    "$UV_BIN" run pytest -q --no-header "$@" >> "$LOG" 2>&1
    local code=$?
    receipt PYTEST_EXIT "$code"
    # the pytest tail line ("289 passed in 131.02s") is the pass-count regression signal
    receipt SUITE_SUMMARY "$(grep -E '^[0-9]+ (passed|failed)|passed|failed|error' "$LOG" | tail -1 | tr -d '\r')"
    # skip audit: "fullmap database is not mounted" skips mean the sed patch or the mount is wrong
    receipt FULLMAP_SKIPS "$(grep -c 'fullmap database is not mounted' "$LOG" 2>/dev/null || true)"
    return "$code"
}

run_lint() {
    "$UV_BIN" run ruff check ./src ./tests >> "$LOG" 2>&1
    local check=$?
    "$UV_BIN" run ruff format --check ./src ./tests >> "$LOG" 2>&1
    local format=$?
    receipt RUFF_CHECK_EXIT "$check"
    receipt RUFF_FORMAT_EXIT "$format"
    [ "$check" -eq 0 ] && [ "$format" -eq 0 ]
    local code=$?
    receipt RUFF_EXIT "$code"
    return "$code"
}

case "$MODE" in
    sync)
        "$UV_BIN" sync >> "$LOG" 2>&1
        receipt SYNC_EXIT "$?"
        ;;
    tests)
        run_suite "$@"
        ;;
    lint)
        run_lint
        ;;
    gate)
        "$UV_BIN" sync >> "$LOG" 2>&1
        receipt SYNC_EXIT "$?"
        suite=0
        run_suite "$@" || suite=1
        lint=0
        run_lint || lint=1
        receipt GATE_EXIT "$((suite + lint))"
        ;;
    cov)
        "$UV_BIN" sync >> "$LOG" 2>&1
        receipt SYNC_EXIT "$?"
        "$UV_BIN" run pytest -q --no-header --cov=relmedner --cov-report=term-missing --cov-fail-under=90 "$@" >> "$LOG" 2>&1
        receipt COV_EXIT "$?"
        receipt COVERAGE "$(grep -E '^TOTAL' "$LOG" | tail -1 | tr -s ' ' | tr -d '\r')"
        run_lint
        ;;
    live)
        if [ "$#" -eq 0 ]; then
            set -- tests/test_super_glue_record_live.py
        fi
        RELMEDNER_LIVE_HF=1 "$UV_BIN" run pytest -q --no-header --no-cov "$@" >> "$LOG" 2>&1
        receipt LIVE_EXIT "$?"
        ;;
    probe)
        "$UV_BIN" run python "$SKILL_DIR/scripts/probe.py" "$@" >> "$LOG" 2>&1
        receipt PROBE_EXIT "$?"
        ;;
    *)
        receipt RUN_ERROR "unknown mode: $MODE"
        exit 2
        ;;
esac

# one final grep-able verdict line so a polling caller never has to parse the whole log
if grep -qE '^(PYTEST_EXIT|RUFF_EXIT|COV_EXIT|LIVE_EXIT|PROBE_EXIT|SYNC_EXIT|GATE_EXIT):[1-9]' "$LOG"; then
    receipt VERDICT FAIL
    exit 1
fi
receipt VERDICT PASS
exit 0
