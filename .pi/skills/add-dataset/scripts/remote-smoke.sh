#!/usr/bin/env bash
# Smoke build on wenceslaus. Runs ON the remote box, inside tmux for anything longer than a minute
# (the gateway jump path idle-kills raw ssh masters). Writes receipts to LOG.
#
# usage: remote-smoke.sh TREE LOG [extra build-dataset args...]
#   default run is the sampled DirectRunner smoke: `relmedner build-dataset -t --direct -o OUT`.
#   --direct is load-bearing on this box: without it the job submits to the LAN flink cluster
#   (cluster.yaml's jobmanager is the LAPTOP, 10.2.9.11:18081) and fails from wenceslaus unless a
#   VPN route plus an ssh tunnel to the jobmanager ports exist. Cluster runs are out of scope here.
#   pass your own args to override, e.g. `--direct -o /some/out.avro` for a full multi-hour build
#
# env: SMOKE_OUT (default ~/relmedner-add-dataset-test.avro)
# receipts: SMOKE_EXIT, AVRO_OUT, AVRO_RECORDS, AVRO_SHAPES, AVRO_WEIGHTS, QUALITY_LINES, VERDICT
set -uo pipefail

TREE="${1:?usage: remote-smoke.sh TREE LOG [extra build-dataset args...]}"
LOG="${2:?usage: remote-smoke.sh TREE LOG [extra build-dataset args...]}"
shift 2

UV_BIN="${UV_BIN:-$HOME/bin/uv}"
OUT="${SMOKE_OUT:-$HOME/relmedner-add-dataset-test.avro}"

export PATH="$HOME/bin:$PATH"
export LC_ALL=C.UTF-8
export LANG=en_US.UTF-8
export PYTHONUTF8=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export RELMEDNER_FULLMAP_DIR="${RELMEDNER_FULLMAP_DIR:-$HOME/Desktop/fullmap}"

: > "$LOG"
receipt() { printf '%s:%s\n' "$1" "$2" >> "$LOG"; }

receipt RUN_MODE smoke
receipt RUN_TREE "$TREE"
receipt RUN_DATE "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$TREE" || { receipt RUN_ERROR "tree missing: $TREE"; exit 2; }
rm -f "$OUT"

if [ "$#" -eq 0 ]; then
    set -- -t --direct -o "$OUT"
fi
echo "build-dataset $*" >> "$LOG"
"$UV_BIN" run relmedner build-dataset "$@" >> "$LOG" 2>&1
smoke_exit=$?
receipt SMOKE_EXIT "$smoke_exit"
receipt AVRO_OUT "$OUT"

if [ -f "$OUT" ]; then
    # record count plus a populated-shape census, read with the repo's own avro dependency
    "$UV_BIN" run python - "$OUT" >> "$LOG" 2>&1 <<'PY'
import sys
from collections import Counter
from fastavro import reader
from relmedner.models import TrainingExample

records = list(reader(open(sys.argv[1], "rb")))
print(f"AVRO_RECORDS:{len(records)}")
shapes: Counter[str] = Counter()
weights: Counter[float] = Counter()
for record in records:
    example = TrainingExample(**record)
    shapes.update(example.populated())
    weights[example.weight] += 1
print("AVRO_SHAPES:" + ",".join(f"{shape}={count}" for shape, count in sorted(shapes.items())))
print("AVRO_WEIGHTS:" + ",".join(f"{weight}={count}" for weight, count in sorted(weights.items())))
PY
else
    receipt AVRO_RECORDS 0
    echo "avro output missing: $OUT" >> "$LOG"
fi

# Per-ingest quality lines are BEST EFFORT: the relmedner.quality INFO records do not surface in a
# Prism --direct run's captured stdout (measured: 0 lines in a 62-record smoke), so QUALITY_LINES:0 is
# normal here and NOT a failure. Per-ingest rows_in/rows_out evidence comes from probe.py --declared,
# which reads the stream's own counters (PROBE_QUALITY).
tmp="$(mktemp)"
grep -o 'ingest quality .*' "$LOG" > "$tmp" 2>/dev/null || true
sed 's/^/QUALITY /' "$tmp" >> "$LOG"
rm -f "$tmp"
receipt QUALITY_LINES "$(grep -c '^QUALITY ' "$LOG" 2>/dev/null || true)"
receipt SMOKE_DONE "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$smoke_exit" -ne 0 ]; then
    receipt VERDICT FAIL
    exit 1
fi
receipt VERDICT PASS
exit 0
