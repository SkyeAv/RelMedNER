#!/usr/bin/env python
"""Measurement probe for a relmedner dataset ingest. Runs ON wenceslaus, never on the laptop.

Every number a README "Dataset-format notes" section claims comes from a run of this probe (or from
an equivalent throwaway harness): row counts, the distinct-label census and its coverage, span
encoding and bounds, malformed-row rates, and mined or dispatched yield. Measure, do not ask, and
never let an unmeasured number reach the docs.

usage (from the synced remote tree, through uv so the repo deps resolve):

    uv run python .pi/skills/add-dataset/scripts/probe.py --dataset nvidia/Nemotron-PII \
        --split train --limit 500 --text-column text --label-column labels --json

    # probe an ingest that is ALREADY declared, through the repo's own production path
    uv run python .pi/skills/add-dataset/scripts/probe.py --declared 'aps/super_glue:record' \
        --limit 200 --script SuperGlueRecordScript --outputs entities,classifications

    # hub-side row counts only (no download; the datasets-server endpoints)
    uv run python .pi/skills/add-dataset/scripts/probe.py --dataset TrialPanorama/TrialPanorama-database --info

    # fullmap mining yield for an unlabeled-text candidate
    uv run python .pi/skills/add-dataset/scripts/probe.py --dataset <repo> --split train \
        --limit 200 --text-column text --fullmap

    # generate the tests/test_ingests.py EXPECTED block from the live tuple output (no download)
    uv run python .pi/skills/add-dataset/scripts/probe.py --freeze 'aps/super_glue:record'
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request
from collections import Counter
from typing import Any

PROBE = "PROBE"


def say(key: str, value: Any) -> None:
    """one grep-able receipt line per measurement"""
    print(f"{PROBE}_{key}:{value}", flush=True)


def heading(text: str) -> None:
    print(f"\n== {text}", flush=True)


# --------------------------------------------------------------------------- hub-side metadata --


def hub_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:  # https endpoint, no redirects followed
        return json.load(response)


def hub_info(dataset: str) -> dict[str, Any]:
    """per-config feature/dtype specs from the datasets-server info endpoint, no download"""
    return hub_json(f"https://datasets-server.huggingface.co/info?dataset={dataset}")


def hub_size(dataset: str) -> dict[str, Any]:
    """row counts per config and split from the datasets-server size endpoint, no download. This
    (not /info, whose num_rows is null for parquet-backed repos) is how the TrialPanorama
    1,332,141 and ~27.4M figures in the README were measured."""
    return hub_json(f"https://datasets-server.huggingface.co/size?dataset={dataset}")


def entry_base_key(payload: tuple[object, ...]) -> str:
    """repo id plus a scalar discriminator (subset for "hf", file for "hf_json"). Kept identical
    to tests/test_ingests.py entry_base_key, which locks and asserts on these keys"""
    dataset = str(payload[2])
    discriminator = payload[3] if len(payload) > 3 else None
    # only a SCALAR discriminator qualifies the key: for the local sources payload position 3 is
    # columns_out (a tuple), and there the declared path is already unique per file
    return f"{dataset}:{discriminator}" if isinstance(discriminator, str) else dataset


def declared_entry_keys(ingests: Any) -> list[tuple[Any, str, tuple[object, ...], str, str]]:
    """(dataset, source, payload, key, base) per declared ingest under the tests/test_ingests.py
    entry_key rule, kept identical: a base key declared on more than one ingest gets the split
    appended, and a key that still collides after that is a hard error, never a silent dict
    overwrite (the loser would vanish from both the freeze blocks and the locks)"""
    tuples: list[tuple[Any, str, tuple[object, ...]]] = []
    for dataset in ingests.datasets:
        source, payload = dataset.to_tuple()
        tuples.append((dataset, source, payload))
    base_counts: Counter[str] = Counter(entry_base_key(payload) for _dataset, _source, payload in tuples)
    keyed: dict[str, tuple[Any, str, tuple[object, ...], str, str]] = {}
    for dataset, source, payload in tuples:
        base = entry_base_key(payload)
        key = base
        if base_counts[base] > 1:
            split = payload[4] if len(payload) > 4 else None
            if not isinstance(split, str):
                raise ValueError(f"entry base {base!r} is declared on more than one ingest but its split {split!r} is not a str")
            key = f"{base}:{split}"
        if key in keyed:
            raise ValueError(f"entry key {key!r} is produced by more than one declared ingest")
        keyed[key] = (dataset, source, payload, key, base)
    return list(keyed.values())


def report_freeze(requested_key: str | None) -> None:
    """print the tests/test_ingests.py EXPECTED block for one declared ingest (or all of them),
    generated from the live generate_tuples() output rather than hand-written. Paste it into the
    local test file, then let `ruff format` normalize the layout.

    WHY generated: a hand-copied tuple lock is how a declaration typo survives review -- the lock
    would then assert the wrong shape and pass forever."""
    import pprint

    from relmedner.ingests import YamlIngestsParser

    ingests = YamlIngestsParser().parse_ingests()
    printed = 0
    # match on the entry key OR its base: a fully qualified key prints one block, a colliding
    # base (bigbio/chemprot:chemprot_full_source) prints every split sharing it
    for _dataset, source, payload, key, base in declared_entry_keys(ingests):
        if requested_key is not None and key != requested_key and base != requested_key:
            continue
        printed += 1
        print(f'    "{key}": (', flush=True)
        print(f'        "{source}",', flush=True)
        print(pprint.pformat(payload, width=110, indent=8, sort_dicts=False).rstrip() + ",", flush=True)
        print("    ),", flush=True)
    if printed == 0:
        say("FREEZE_ERROR", f"no declared ingest matches {requested_key!r}")
        raise SystemExit(1)
    say("FREEZE_BLOCKS", printed)


def report_info(dataset: str) -> None:
    heading(f"hub metadata {dataset}")
    try:
        size = (hub_size(dataset) or {}).get("size") or {}
    except Exception as exc:  # fail loud: a guessed row count is worse than none
        say("SIZE_ERROR", f"{type(exc).__name__}: {exc}")
        size = {}
    total = (size.get("dataset") or {}).get("num_rows")
    if total is not None:
        say("ROWS_TOTAL", f"{dataset}={total}")
    for entry in size.get("splits") or []:
        say("ROWS", f"{dataset}:{entry.get('config')}:{entry.get('split')}={entry.get('num_rows')}")
    try:
        payload = hub_info(dataset)
    except Exception as exc:
        say("INFO_ERROR", f"{type(exc).__name__}: {exc}")
        return
    for config, spec in (payload.get("dataset_info") or {}).items():
        features = spec.get("features") or {}
        for column, feature in features.items():
            say("FEATURE", f"{config}:{column}={json.dumps(feature)[:200]}")


# ------------------------------------------------------------------------------- row scanning --


def iter_declared(requested_key: str, limit: int) -> tuple[tuple[str, ...], list[Any], Any]:
    """rows through the repo's own production path: declared tuple -> build_stream -> rows().
    Proves the declaration itself, not just the hub schema."""
    from relmedner.ingests import YamlIngestsParser
    from relmedner.registry import build_stream

    ingests = YamlIngestsParser().parse_ingests()
    match = None
    columns: tuple[str, ...] = ()
    for dataset, _source, _payload, key, base in declared_entry_keys(ingests):
        if key == requested_key or base == requested_key:
            match = dataset.to_stream_args()
            # read the projection off the model, never off a payload position: local_delimited packs
            # match_on last, and a local avro source has no projection at all (the whole record ships)
            columns = tuple(getattr(dataset, "columns_out", ()) or ())
            break
    if match is None:
        raise SystemExit(f"no declared ingest matches entry_key {requested_key!r}")
    source, payload, filters, sample_rate, read_shards = match
    stream = build_stream(source, payload, filters=filters, sample_rate=sample_rate, read_shards=read_shards)
    rows: list[Any] = []
    for _, (_, values) in stream.rows():
        rows.append(values)
        if len(rows) >= limit:
            break
    return columns, rows, stream


def iter_hub(dataset: str, subset: str | None, split: str, columns: tuple[str, ...], limit: int) -> tuple[tuple[str, ...], list[Any], None]:
    """raw streaming probe for a corpus that is NOT declared yet (intake phase)"""
    from datasets import load_dataset

    stream = load_dataset(dataset, subset, split=split, streaming=True)
    rows: list[dict[str, Any]] = []
    for row in stream:
        rows.append(row)
        if len(rows) >= limit:
            break
    seen = tuple(columns) if columns else tuple(rows[0]) if rows else ()
    return seen, rows, None


def as_dict(row: Any) -> dict[str, Any] | None:
    """raw hub rows are dicts, and a local avro source ships the whole record as a 1-tuple dict"""
    if isinstance(row, dict):
        return row
    if isinstance(row, (list, tuple)) and len(row) == 1 and isinstance(row[0], dict):
        return row[0]
    return None


def value_at(row: Any, columns: tuple[str, ...], name: str) -> Any:
    """one accessor for every row shape: declared rows are tuples in columns_out order, raw hub
    rows are dicts, local avro rows are a whole record"""
    record = as_dict(row)
    if record is not None:
        return record.get(name)
    if name in columns:
        return row[columns.index(name)]
    return None


def project(row: Any, columns: tuple[str, ...]) -> tuple[Any, ...]:
    """the values tuple a Script expects: already projected for a declared stream, projected here
    for a raw hub row"""
    record = as_dict(row)
    if record is not None:
        return tuple(record.get(name) for name in columns)
    return tuple(row)


def type_name(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return "str"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, dict):
        return "dict{" + ",".join(sorted(value))[:60] + "}"
    if isinstance(value, (list, tuple)):
        inner = Counter(type_name(item) for item in value)
        return f"list[{len(value)}]of({','.join(f'{k}x{v}' for k, v in inner.most_common(4))})"
    return type(value).__name__


def report_columns(columns: tuple[str, ...], rows: list[Any]) -> None:
    heading(f"columns over {len(rows)} rows")
    first = as_dict(rows[0])
    if first is not None:
        columns = tuple(dict.fromkeys([*columns, *first]))
    for name in columns:
        shapes = Counter(type_name(value_at(row, columns, name)) for row in rows)
        empties = sum(1 for row in rows if not value_at(row, columns, name))
        say("COLUMN", f"{name} empty={empties}/{len(rows)} shapes={dict(shapes.most_common(6))}")


def report_lengths(rows: list[Any], columns: tuple[str, ...], text_column: str) -> None:
    heading(f"text lengths ({text_column})")
    lengths: list[int] = []
    token_counts: list[int] = []
    for row in rows:
        value = value_at(row, columns, text_column)
        if isinstance(value, str):
            lengths.append(len(value))
            token_counts.append(len(value.split()))
        elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            token_counts.append(len(value))
            lengths.append(len(" ".join(value)))
    if not lengths:
        say("TEXT_ERROR", f"no str or list-of-str values in {text_column!r}")
        return
    for label, series in (("chars", lengths), ("tokens", token_counts)):
        if not series:
            continue
        say(f"TEXT_{label.upper()}_MIN", min(series))
        say(f"TEXT_{label.upper()}_MEDIAN", int(statistics.median(series)))
        say(f"TEXT_{label.upper()}_MEAN", round(statistics.fmean(series), 1))
        say(f"TEXT_{label.upper()}_MAX", max(series))


def flatten_labels(value: Any, index: int) -> list[str]:
    """labels arrive as a str, a list of str, an int (ClassLabel index), or a list of span lists
    whose label sits at a fixed index (the [start, end, label] convention). All four are counted
    rather than assumed away: a wrong assumption here is a wrong LABEL_MAP."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [str(value)]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                if len(item) > index:
                    out.extend(flatten_labels(item[index], index))
                elif item:
                    out.extend(flatten_labels(item[-1], index))
            else:
                out.extend(flatten_labels(item, index))
        return out
    return []


def report_labels(rows: list[Any], columns: tuple[str, ...], label_columns: tuple[str, ...], index: int, top: int) -> None:
    for name in label_columns:
        heading(f"label census ({name})")
        census: Counter[str] = Counter()
        spans = 0
        for row in rows:
            labels = flatten_labels(value_at(row, columns, name), index)
            spans += len(labels)
            census.update(labels)
        total = sum(census.values())
        say("LABEL_COLUMN", name)
        say("LABEL_SPANS", total)
        say("LABEL_DISTINCT", len(census))
        if total:
            head = census.most_common(top)
            say("LABEL_TOP_COVERAGE_PCT", round(100 * sum(count for _, count in head) / total, 1))
            for label, count in head:
                say("LABEL", f"{label}={count}")
            tail = [label for label, count in census.items() if count == 1]
            say("LABEL_HAPAX", len(tail))


def report_spans(rows: list[Any], columns: tuple[str, ...], span_column: str, text_column: str | None, index: int) -> None:
    heading(f"span shapes ({span_column})")
    arity: Counter[int] = Counter()
    element_types: Counter[str] = Counter()
    out_of_bounds = 0
    stringified = 0
    checked = 0
    for row in rows:
        value = value_at(row, columns, span_column)
        extent: int | None = None
        if text_column:
            text = value_at(row, columns, text_column)
            if isinstance(text, str):
                extent = len(text)
            elif isinstance(text, (list, tuple)):
                extent = len(text)
        if not isinstance(value, (list, tuple)):
            arity[-1] += 1
            continue
        for span in value:
            if not isinstance(span, (list, tuple)):
                arity[-2] += 1
                continue
            arity[len(span)] += 1
            for item in span:
                element_types[type_name(item)] += 1
                if isinstance(item, str) and item.strip().lstrip("-").isdigit():
                    stringified += 1
            if extent is not None and len(span) > index:
                bounds = [int(item) for item in span[:index] if str(item).strip().lstrip("-").isdigit()]
                if bounds:
                    checked += 1
                    if max(bounds) >= extent or min(bounds) < 0:
                        out_of_bounds += 1
    say("SPAN_ARITY", dict(arity.most_common(8)))
    say("SPAN_ELEMENT_TYPES", dict(element_types.most_common(8)))
    say("SPAN_STRINGIFIED_INDICES", stringified)
    say("SPAN_BOUNDS_CHECKED", checked)
    say("SPAN_OUT_OF_BOUNDS", out_of_bounds)


def report_dispatch(rows: list[Any], columns: tuple[str, ...], script: str, outputs: tuple[str, ...]) -> None:
    heading(f"dispatch yield ({script}, outputs={list(outputs)})")
    from relmedner.models import TrainingExample
    from relmedner.types import Script

    shapes: Counter[str] = Counter()
    emitted = 0
    mentions = 0
    relations = 0
    for row in rows:
        values = project(row, columns)
        if len(values) != len(columns) and as_dict(row) is not None:
            say("DISPATCH_ERROR", "--script over a raw hub row needs --columns matching the intended columns_out")
            return
        _, example = Script.dispatch(script, (outputs, values))
        assert isinstance(example, TrainingExample)
        populated = example.populated()
        if populated:
            emitted += 1
        shapes.update(populated)
        mentions += sum(len(entity.mentions) for entity in example.entities)
        relations += len(example.relations)
    say("DISPATCH_ROWS", len(rows))
    say("DISPATCH_ROWS_EMITTING", emitted)
    say("DISPATCH_EMIT_PCT", round(100 * emitted / len(rows), 1) if rows else 0)
    say("DISPATCH_SHAPES", {str(shape): count for shape, count in shapes.most_common()})
    say("DISPATCH_ENTITY_MENTIONS", mentions)
    say("DISPATCH_RELATIONS", relations)


def report_fullmap(rows: list[Any], columns: tuple[str, ...], text_column: str, max_ngram: int, taxon: str) -> None:
    heading(f"fullmap mining yield ({text_column}, max_ngram={max_ngram}, taxon={taxon})")
    from relmedner.fullmap_mine import FullmapMiner
    from relmedner.models import FullmapTask

    if not FullmapMiner.available():
        say("FULLMAP_ERROR", "fullmap database is not mounted (patch FULLMAP_DIR on the remote copy)")
        return
    task = FullmapTask(type="fullmap", max_ngram=max_ngram, taxon=taxon, outputs=["entities", "relations"])
    texts: list[str] = []
    for row in rows:
        value = value_at(row, columns, text_column)
        if isinstance(value, str) and value.strip():
            texts.append(value)
        elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            texts.append(" ".join(value))
    examples = FullmapMiner.resolve_batch([(text, task) for text in texts])
    mentions = sum(sum(len(entity.mentions) for entity in example.entities) for example in examples)
    relations = sum(len(example.relations) for example in examples)
    say("FULLMAP_DOCS", len(texts))
    say("FULLMAP_MENTIONS", mentions)
    say("FULLMAP_MENTIONS_PER_DOC", round(mentions / len(texts), 2) if texts else 0)
    say("FULLMAP_RELATIONS", relations)
    say("FULLMAP_DOCS_WITH_MENTIONS", sum(1 for example in examples if example.entities))


# ---------------------------------------------------------------------------------------- main --


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", help="hub repo id to probe")
    parser.add_argument("--subset", default=None, help="hub config/subset name")
    parser.add_argument("--split", default="train", help="hub split (default: train)")
    parser.add_argument("--declared", default=None, help="probe an already-declared ingest by its tests/test_ingests.py entry_key")
    parser.add_argument("--columns", default="", help="comma-separated columns_out to project (raw probe)")
    parser.add_argument("--limit", type=int, default=200, help="rows to scan (default: 200)")
    parser.add_argument("--text-column", default=None, help="column holding the text or token list")
    parser.add_argument("--label-column", action="append", default=[], help="column holding labels; repeatable")
    parser.add_argument("--span-column", default=None, help="column holding span lists")
    parser.add_argument("--label-index", type=int, default=2, help="label index inside a span list (default: 2, the [start, end, label] convention)")
    parser.add_argument("--script", default=None, help="registered Script NAME to dispatch rows through")
    parser.add_argument("--outputs", default="entities", help="comma-separated declared outputs for --script (default: entities)")
    parser.add_argument("--fullmap", action="store_true", help="measure mining yield instead of dispatching a script")
    parser.add_argument("--max-ngram", type=int, default=6, help="fullmap max_ngram (default: 6, the repo's measured default)")
    parser.add_argument("--taxon", default="9606", help="fullmap taxon (default: 9606, human)")
    parser.add_argument("--info", action="store_true", help="hub row counts only, no download")
    parser.add_argument(
        "--freeze",
        nargs="?",
        const="",
        default=None,
        metavar="ENTRY_KEY",
        help="print the tests/test_ingests.py EXPECTED block for one entry_key (all ingests when no key is given); no download",
    )
    parser.add_argument("--sample", type=int, default=2, help="rows to pretty-print verbatim (default: 2)")
    parser.add_argument("--json", action="store_true", help="also dump the scanned rows' column shapes as json")
    args = parser.parse_args(argv)

    if not args.dataset and not args.declared and args.freeze is None:
        parser.error("one of --dataset, --declared, or --freeze is required")
    if args.freeze is not None:
        heading("EXPECTED tuple lock (paste into tests/test_ingests.py)")
        report_freeze(args.freeze or None)
        return 0
    if args.info:
        if not args.dataset:
            parser.error("--info needs --dataset")
        report_info(args.dataset)
        return 0

    say("TARGET", args.declared or f"{args.dataset}:{args.subset or '-'}:{args.split}")
    say("LIMIT", args.limit)
    columns: tuple[str, ...] = tuple(part.strip() for part in args.columns.split(",") if part.strip())
    if args.declared:
        columns, rows, stream = iter_declared(args.declared, args.limit)
    else:
        columns, rows, stream = iter_hub(args.dataset, args.subset, args.split, columns, args.limit)
    if not rows:
        say("ROWS", 0)
        say("ERROR", "source yielded no rows: check split/subset spelling, or a gated repo with no token on this box")
        return 1
    say("ROWS", len(rows))
    if stream is not None and getattr(stream, "stats", None) is not None:
        say("QUALITY", stream.stats.report())

    report_columns(columns, rows)
    for index in range(min(args.sample, len(rows))):
        heading(f"sample row {index}")
        row = rows[index]
        record = as_dict(row)
        names = columns or (tuple(record) if record is not None else ())
        for name in names:
            print(f"  {name} = {str(value_at(row, columns, name))[:600]}", flush=True)
    if args.text_column:
        report_lengths(rows, columns, args.text_column)
    if args.label_column:
        report_labels(rows, columns, tuple(args.label_column), args.label_index, top=60)
    if args.span_column:
        report_spans(rows, columns, args.span_column, args.text_column, args.label_index)
    if args.script:
        report_dispatch(rows, columns, args.script, tuple(part.strip() for part in args.outputs.split(",") if part.strip()))
    if args.fullmap:
        if not args.text_column:
            say("ERROR", "--fullmap needs --text-column")
            return 2
        report_fullmap(rows, columns, args.text_column, args.max_ngram, args.taxon)
    if args.json:
        heading("column shapes (json)")
        print(json.dumps({"columns": list(columns), "rows": len(rows)}, indent=2), flush=True)
    say("DONE", 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
