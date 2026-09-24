"""throwaway full-split census for bigbio/chia, run on wenceslaus in tmux only.

Receipt-shaped output (PROBE_*), one block per subset, over the FULL train split (2,000 rows
each). Decides: LABEL_MAP breadth for the entity types, the relation-type mapping, the char
end convention (exclusive vs inclusive), malformed-span rates, dangling/self-loop relation
rates, and the bigbio_kb passage-tiling question.
"""

from __future__ import annotations

import collections
from typing import Any

from datasets import load_dataset

SUBSETS: list[str] = [
    "chia_bigbio_kb",
    "chia_fixed_source",
    "chia_source",
    "chia_without_scope_fixed_source",
    "chia_without_scope_source",
]

def load(subset: str):
    # bigbio/chia is a script-based repo (chia.py); datasets>=3 refuses script datasets, so read
    # the hub's auto-converted parquet branch. This is the load route the new hf_parquet source
    # will own in the pipeline.
    return load_dataset(
        "parquet",
        data_files=f"hf://datasets/bigbio/chia@refs/convert/parquet/{subset}/train/0000.parquet",
        split="train",
        streaming=True,
    )

def row_text(row: dict[str, Any]) -> str:
    if isinstance(row.get("text"), str):
        return row["text"]
    parts: list[str] = []
    for passage in row.get("passages") or []:
        parts.extend(passage.get("text") or [])
    return "".join(parts)

def census(subset: str) -> None:
    ds = load(subset)
    rows = 0
    text_chars: list[int] = []
    etypes: collections.Counter[str] = collections.Counter()
    rtypes: collections.Counter[str] = collections.Counter()
    offsets_per_entity: collections.Counter[int] = collections.Counter()
    offset_elem_types: collections.Counter[str] = collections.Counter()
    text_offset_mismatch = 0
    end_excl = end_incl = end_neither = 0
    oob = 0
    empty_entities = 0
    empty_relations = 0
    dangling = 0
    surface_selfloop = 0
    rel_total = 0
    ent_total = 0
    text_type: collections.Counter[str] = collections.Counter()
    passages_per_row: collections.Counter[int] = collections.Counter()
    events_nonempty = 0
    corefs_nonempty = 0
    passage_gap_or_overlap = 0
    id_selfloop = 0

    for row in ds:
        rows += 1
        text = row_text(row)
        text_chars.append(len(text))
        if "text_type" in row:
            text_type[str(row["text_type"])] += 1
        if "passages" in row:
            passages_per_row[len(row["passages"])] += 1
            if row.get("events"):
                events_nonempty += 1
            if row.get("coreferences"):
                corefs_nonempty += 1
            # passages are expected to tile the doc in offset order; measure gaps/overlaps
            cursor: int | None = None
            for passage in row["passages"]:
                for pstart, pend in passage["offsets"]:
                    if cursor is not None and pstart != cursor:
                        passage_gap_or_overlap += 1
                    cursor = pend
        ents = row.get("entities") or []
        if not ents:
            empty_entities += 1
        by_id: dict[str, dict[str, Any]] = {}
        for ent in ents:
            etypes[str(ent["type"])] += 1
            ent_total += 1
            offsets = ent.get("offsets") or []
            texts = ent.get("text") or []
            if len(texts) != len(offsets):
                text_offset_mismatch += 1
            offsets_per_entity[len(offsets)] += 1
            for off in offsets:
                offset_elem_types[f"{type(off[0]).__name__}/{type(off[1]).__name__}"] += 1
            by_id[str(ent["id"])] = ent
            for (start, end), surface in zip(offsets, texts, strict=False):
                surface = str(surface)
                if start < 0 or end > len(text):
                    oob += 1
                    continue
                if text[start:end] == surface:
                    end_excl += 1
                elif text[start:end + 1] == surface:
                    end_incl += 1
                else:
                    end_neither += 1
        rels = row.get("relations") or []
        if not rels:
            empty_relations += 1
        for rel in rels:
            rtypes[str(rel["type"])] += 1
            rel_total += 1
            arg1 = str(rel["arg1_id"])
            arg2 = str(rel["arg2_id"])
            if arg1 not in by_id or arg2 not in by_id:
                dangling += 1
                continue
            if arg1 == arg2:
                id_selfloop += 1
                continue
            s1 = " ".join(by_id[arg1].get("text") or [])
            s2 = " ".join(by_id[arg2].get("text") or [])
            if s1.lower() == s2.lower():
                surface_selfloop += 1

    chars_sorted = sorted(text_chars)
    def pct(n: int, d: int) -> str:
        return f"{(100.0 * n / d):.2f}%" if d else "n/a"
    print(f"PROBE_SUBSET_ROWS:{subset}={rows}")
    p95 = chars_sorted[int(rows * 0.95)] if rows else 0
    print(
        f"PROBE_TEXT_CHARS:{subset} median={chars_sorted[rows // 2] if rows else 0}"
        f" p95={p95} max={chars_sorted[-1] if rows else 0}"
    )
    print(f"PROBE_ENTITY_TYPES:{subset} distinct={len(etypes)}")
    for label, count in etypes.most_common():
        print(f"PROBE_ENTITY_TYPE:{subset} {label}={count}")
    print(f"PROBE_RELATION_TYPES:{subset} distinct={len(rtypes)}")
    for label, count in rtypes.most_common():
        print(f"PROBE_RELATION_TYPE:{subset} {label}={count}")
    print(f"PROBE_OFFSETS_PER_ENTITY:{subset} {dict(sorted(offsets_per_entity.items()))}")
    print(f"PROBE_OFFSET_ELEM_TYPES:{subset} {dict(offset_offset_elem if (offset_offset_elem := offset_elem_types) else {})}")
    print(f"PROBE_TEXT_OFFSET_MISMATCH:{subset}={text_offset_mismatch}")
    print(
        f"PROBE_END_CONVENTION:{subset} excl={end_excl} incl={end_incl} neither={end_neither}"
        f" total={sum(offsets_per_entity.values())}"
    )
    print(f"PROBE_OOB_SPANS:{subset}={oob} ({pct(oob, sum(offsets_per_entity.values()))})")
    print(f"PROBE_EMPTY_ENTITIES_ROWS:{subset}={empty_entities} ({pct(empty_entities, rows)})")
    print(f"PROBE_EMPTY_RELATIONS_ROWS:{subset}={empty_relations} ({pct(empty_relations, rows)})")
    print(f"PROBE_REL_TOTAL:{subset}={rel_total} dangling={dangling} ({pct(dangling, rel_total)})"
          f" id_selfloop={id_selfloop} surface_selfloop={surface_selfloop}")
    if text_type:
        print(f"PROBE_TEXT_TYPE:{subset} {dict(text_type)}")
    if passages_per_row:
        print(f"PROBE_PASSAGES_PER_ROW:{subset} {dict(sorted(passages_per_row.items()))}")
        print(f"PROBE_PASSAGE_GAPS:{subset}={passage_gap_or_overlap}"
              f" events_nonempty={events_nonempty} corefs_nonempty={corefs_nonempty}")

def main() -> None:
    for subset in SUBSETS:
        census(subset)
    print("PROBE_CHIA_CENSUS_DONE:1")

if __name__ == "__main__":
    main()
