#!/usr/bin/env python
"""build the six FewRel avro containers from the raw thunlp/FewRel GitHub JSONs.

Runs ON WENCESLAUS ONLY (the laptop never downloads a corpus):

    cd ~/Code/RelMedNER-worktrees/few-rel && /home/sgoetz/bin/uv run python scripts/build_fewrel_avro.py

For every split file the raw JSON is a dict keyed by relation id whose values are rows
{tokens, h: [text, type, indices], t: [text, type, indices]} (pubmed_unsupervised.json is a
bare list; its rows carry no relation). pid2name.json maps P-id -> [name, description] (744
entries). The converter writes one avro container per split to
src/relmedner/data/fewrel/<split>.avro with the record contract FewRelScript consumes:

    {"tokens": [str], "label": str, "h_indices": [[int]], "t_indices": [[int]]}

"label" is the RESOLVED predicate (strip participant suffix -> pid2name[0] ->
ScriptUtils.normalize_predicate) via the pure helpers in relmedner.scripts.fewrel, so the
streaming script never needs the mapping; an unresolvable P-id records "" and the script then
ships entities only. h_text/t_text (detokenized lowercase surfaces) and the Q-id/UMLS type
fields are deliberately omitted: surfaces derive from tokens+indices, and no label ever rides
those fields.

The avro files are gitignored (src/relmedner/data/fewrel/*.avro) and rsync-excluded, so they
exist only where they are built.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from fastavro import writer

from relmedner.scripts.fewrel import resolve_pid_label
from relmedner.utils import ScriptUtils

BASE_URL: str = "https://raw.githubusercontent.com/thunlp/FewRel/master/data/"
SPLITS: tuple[str, ...] = (
    "train_wiki",
    "val_wiki",
    "val_nyt",
    "val_semeval",
    "val_pubmed",
    "pubmed_unsupervised",
)
PID2NAME: str = "pid2name"
OUT_DIR: Path = Path("src/relmedner/data/fewrel")

# explicit schema, deterministic field order; list<list<int>> is exactly the
# FewRel indices encoding and avro round-trips it without coercion
SCHEMA: dict[str, Any] = {
    "type": "record",
    "name": "FewRelRow",
    "fields": [
        {"name": "tokens", "type": {"type": "array", "items": "string"}},
        {"name": "label", "type": "string"},
        {"name": "h_indices", "type": {"type": "array", "items": {"type": "array", "items": "int"}}},
        {"name": "t_indices", "type": {"type": "array", "items": {"type": "array", "items": "int"}}},
    ],
}


def fetch(name: str) -> Any:
    """download one raw JSON file and decode it; a missing or undecodable file fails loudly"""
    url = f"{BASE_URL}{name}.json"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def rows_of(data: dict[str, Any] | list[Any]) -> list[tuple[str, Any]]:
    """flatten the raw shapes: dict keyed by relation id -> rows, or a bare row list.
    Yields 2-tuples (relation, row dict); rows_of MUST yield tuples, never a dict of the
    shape {"relation": ..., "row": ...}, because a caller's `for relation, row in rows`
    unpacks a dict into its KEYS when the rows list itself is iterated bare."""
    rows: list[tuple[str, Any]] = []
    if isinstance(data, dict):
        for relation, items in data.items():
            for item in items:
                rows.append((relation, item))
    else:
        for item in data:
            rows.append(("", item))
    return rows


def record_of(relation: str, row: Any, pid2name: dict[str, Any]) -> dict[str, Any]:
    """one raw FewRel row -> one avro record; the only logic beyond IO is the shared
    resolution chain (suffix strip, pid2name, normalize) reused from the script module.
    A row that is not a dict fails loudly: the corpus shape is fixed upstream and a
    silently skipped row would understate the census the README publishes."""
    if not isinstance(row, dict):
        raise TypeError(f"{relation}: expected a dict row, got {type(row).__name__}")
    predicate, _ = ScriptUtils.resolve_predicate(resolve_pid_label(relation, pid2name))
    return {
        "tokens": [str(token) for token in row["tokens"]],
        "label": predicate,
        "h_indices": row["h"][2],
        "t_indices": row["t"][2],
    }


def main() -> int:
    pid2name_raw = fetch(PID2NAME)
    if not isinstance(pid2name_raw, dict):
        raise TypeError(f"pid2name.json: expected a dict, got {type(pid2name_raw).__name__}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        data = fetch(split)
        records = [record_of(relation, row, pid2name_raw) for relation, row in rows_of(data)]
        out_path = OUT_DIR / f"{split}.avro"
        with out_path.open("wb") as handle:
            writer(handle, SCHEMA, records)
        print(f"RECORDS:{split}:{len(records)}:{out_path}", flush=True)
        print(f"AVRO_BYTES:{split}:{out_path.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
