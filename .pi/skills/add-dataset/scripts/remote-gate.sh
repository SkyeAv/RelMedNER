#!/usr/bin/env bash
# Laptop-side orchestrator for the relmedner add-dataset remote gate. Pushes this worktree to
# wenceslaus, patches the REMOTE COPY's FULLMAP_DIR, runs the gate there, and prints receipts.
# Nothing heavy ever runs locally: this script only rsyncs and ssh's.
#
# usage: remote-gate.sh [options] [mode] [-- extra args for the mode]
#   modes: gate (default, T1) | cov | tests | lint | sync | live | smoke | probe
#   options:
#     --host HOST         ssh host (default: wenceslaus)
#     --name NAME         remote directory name (default: this worktree's directory name)
#     --base PATH         remote base dir, relative to home (default: Code/RelMedNER-worktrees)
#     --no-sync           skip the rsync push (reuse the remote copy as-is)
#     --tmux              run the mode in a detached remote tmux session (long jobs)
#     --session NAME      tmux session name (default: add-dataset-<name>-<mode>)
#     --wait SECONDS      with --tmux: poll until the receipt appears, up to SECONDS
#     -h, --help
#
# examples:
#   remote-gate.sh                                  # T1: sync + full suite + ruff on wenceslaus
#   remote-gate.sh cov                              # T1 plus the 90% coverage floor
#   remote-gate.sh live -- tests/test_x_live.py     # RELMEDNER_LIVE_HF=1 over one test file
#   remote-gate.sh smoke --tmux --wait 1800         # build-dataset -t in tmux, poll to completion
#   remote-gate.sh probe -- --dataset nvidia/Nemotron-PII --split train --limit 500
set -uo pipefail

HOST="wenceslaus"
BASE="Code/RelMedNER-worktrees"
# one canonical remote fullmap path: the runner's RELMEDNER_FULLMAP_DIR env override and the
# sed-patched constants.py fallback literal must name the SAME directory, or the env-capture
# constants tests see two different paths and the suite fails on the box's $HOME spelling.
REMOTE_FULLMAP="/home/sgoetz/Desktop/fullmap"
DO_SYNC=1
TMUX=0
SESSION=""
WAIT=0
MODE="gate"
EXTRA=()

usage() { sed -n '2,26p' "$0"; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --name) NAME_OVERRIDE="$2"; shift 2 ;;
        --base) BASE="$2"; shift 2 ;;
        --no-sync) DO_SYNC=0; shift ;;
        --tmux) TMUX=1; shift ;;
        --session) SESSION="$2"; TMUX=1; shift 2 ;;
        --wait) WAIT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; while [ "$#" -gt 0 ]; do EXTRA+=("$1"); shift; done ;;
        -*) echo "unknown option: $1" >&2; usage; exit 2 ;;
        *) MODE="$1"; shift ;;
    esac
done

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
NAME="${NAME_OVERRIDE:-$(basename "$ROOT")}"
REMOTE_REL="$BASE/$NAME"
LOG_REL="relmedner-$NAME-$MODE.log"
SESSION="${SESSION:-add-dataset-$NAME-$MODE}"
SKILL_REL=".pi/skills/add-dataset/scripts"

remote() { ssh "$HOST" "$@"; }

echo "== relmedner remote gate: host=$HOST tree=$ROOT"
echo "==   remote=$REMOTE_REL mode=$MODE tmux=$TMUX log=~/$LOG_REL"

if [ ! -f "$ROOT/$SKILL_REL/remote-runner.sh" ]; then
    echo "FATAL: $SKILL_REL/remote-runner.sh missing from this worktree; the skill directory must be committed or present" >&2
    exit 2
fi

# 1. push the tree. --exclude='.venv' is load-bearing: shipping a laptop-platform venv makes uv
#    rebuild the remote environment. --delete keeps the copy honest. --exclude='*.avro' keeps
#    operator-built containers off the wire, and the sentinel file keeps --delete from wiping the
#    whole package-data dir on a box where the containers have not been built yet (the runner
#    syncs real containers from a sibling tree before the suite runs).
if [ "$DO_SYNC" -eq 1 ]; then
    echo "== rsync push"
    remote "mkdir -p '$REMOTE_REL'" || exit 1
    rsync -az --delete \
        --exclude='.git' --exclude='.venv' --exclude='.ralph' \
        --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.ruff_cache' \
        --exclude='.coverage' --exclude='dist' --exclude='*.avro' \
        --exclude='.rsync-dir-sentinel' \
        "$ROOT/" "$HOST:$REMOTE_REL/" || exit 1
    remote "mkdir -p '$REMOTE_REL/src/relmedner/data/bc5cdr' '$REMOTE_REL/src/relmedner/data/synthetic-ner-ade-tweets' '$REMOTE_REL/src/relmedner/data/interventions' '$REMOTE_REL/src/relmedner/data/medical-entity-json-extraction' && touch '$REMOTE_REL/src/relmedner/data/bc5cdr/.rsync-dir-sentinel' '$REMOTE_REL/src/relmedner/data/synthetic-ner-ade-tweets/.rsync-dir-sentinel' '$REMOTE_REL/src/relmedner/data/interventions/.rsync-dir-sentinel' '$REMOTE_REL/src/relmedner/data/medical-entity-json-extraction/.rsync-dir-sentinel'" || exit 1
fi

# 2. patch the REMOTE COPY's hardcoded laptop fullmap path (idempotent; the laptop tree and git
#    history are never touched). The grep line printed back is the receipt.
echo "== remote FULLMAP_DIR patch"
remote "cd '$REMOTE_REL' && sed -i 's#/home/skyeav/Desktop/fullmap#$REMOTE_FULLMAP#' src/relmedner/constants.py && grep -n 'FULLMAP_DIR' src/relmedner/constants.py | tr ':' '~'" || exit 1

# 3. run the mode. Long jobs go in remote tmux because the gateway idle-kills raw ssh masters.
if [ "$MODE" = "smoke" ]; then
    RUNNER="$SKILL_REL/remote-smoke.sh"
    MODE_ARG=""   # remote-smoke.sh takes TREE LOG [build-dataset args], with no mode slot
else
    RUNNER="$SKILL_REL/remote-runner.sh"
    MODE_ARG="'$MODE'"
fi

if [ "$TMUX" -eq 1 ]; then
    echo "== launching remote tmux session '$SESSION'"
    remote "tmux kill-session -t '$SESSION' 2>/dev/null; tmux new-session -d -s '$SESSION' \"cd '$REMOTE_REL' && RELMEDNER_FULLMAP_DIR='$REMOTE_FULLMAP' bash '$RUNNER' \\\$PWD \\\$HOME/'$LOG_REL' $MODE_ARG ${EXTRA[*]}\"" || exit 1
    echo "   poll: ssh $HOST 'tmux capture-pane -p -t $SESSION | tail -30'"
    echo "   log:  ssh $HOST 'tail -40 ~/$LOG_REL | tr \":\" \"~\"'"
    if [ "$WAIT" -gt 0 ]; then
        deadline=$((SECONDS + WAIT))
        while [ "$SECONDS" -lt "$deadline" ]; do
            if remote "grep -qE '^(VERDICT|RUN_ERROR):' ~/'$LOG_REL' 2>/dev/null"; then
                break
            fi
            sleep 20
        done
    fi
else
    echo "== remote run ($MODE)"
    remote "cd '$REMOTE_REL' && RELMEDNER_FULLMAP_DIR='$REMOTE_FULLMAP' bash '$RUNNER' \"\$PWD\" \"\$HOME/$LOG_REL\" $MODE_ARG ${EXTRA[*]}"
    echo "== exit=$?"
fi

# 4. receipts. Colon-mangled on purpose: pi's renderer collapses ripgrep-shaped output.
echo "== receipts"
remote "grep -E '^(RUN_MODE|RUN_TREE|FULLMAP_DIR|FULLMAP_REDB|SYNC_EXIT|PYTEST_EXIT|SUITE_SUMMARY|FULLMAP_SKIPS|RUFF_CHECK_EXIT|RUFF_FORMAT_EXIT|RUFF_EXIT|COV_EXIT|COVERAGE|LIVE_EXIT|PROBE_[A-Z_]+|SMOKE_EXIT|AVRO_RECORDS|AVRO_SHAPES|QUALITY_LINES|GATE_EXIT|VERDICT|RUN_ERROR):' ~/'$LOG_REL' 2>/dev/null | tr ':' '~' | tr -d '\r'"
echo "== log tail"
remote "tail -25 ~/'$LOG_REL' 2>/dev/null | tr ':' '~' | tr -d '\r'"

if remote "grep -qE '^(VERDICT:FAIL|SMOKE_EXIT:[1-9]|RUN_ERROR:)' ~/'$LOG_REL' 2>/dev/null"; then
    echo "== GATE: FAIL"
    exit 1
fi
if remote "grep -qE '^VERDICT:PASS' ~/'$LOG_REL' 2>/dev/null"; then
    echo "== GATE: PASS"
    exit 0
fi
echo "== GATE: no verdict receipt (tmux job still running, or smoke receipts above are the evidence)"
exit 0
