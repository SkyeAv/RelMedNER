from __future__ import annotations

import pathlib
import re
from collections import Counter

import pytest

from relmedner.ingests import YamlIngestsParser
from relmedner.models import (
    Cluster,
    FullmapTask,
    GazetteerPredicate,
    GazetteerQualifier,
    GazetteerSpec,
    HuggingFaceDataset,
    HuggingFaceJsonDataset,
    LocalAvroDataset,
    LocalDelimitedDataset,
    MatchOn,
    ScriptTask,
    WorkerNode,
    YamlIngests,
)

# per-ingest locks: each declared ingest's entry is asserted independently so adding a dataset is an
# additive block here rather than a rewritten literal (and a guaranteed merge conflict) across the
# parallel dataset worktrees
EXPECTED: dict[str, tuple[object, ...]] = {
    "anthonyyazdaniml/gliner-biomed-pre-training": (
        "hf",
        (
            ("script", "GlinerBiomedScript", ("entities", "relations")),
            1.0,
            "anthonyyazdaniml/gliner-biomed-pre-training",
            None,
            "train",
            None,
            ("tokenized_text", "ner"),
        ),
    ),
    "anthonyyazdaniml/gliner-biomed-curated-corpus": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "anthonyyazdaniml/gliner-biomed-curated-corpus",
            None,
            "train",
            None,
            ("text",),
        ),
    ),
    "anthonyyazdaniml/gliner-biomed-balanced-curated-corpus": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "anthonyyazdaniml/gliner-biomed-balanced-curated-corpus",
            None,
            "train",
            None,
            ("text",),
        ),
    ),
    "anthonyyazdaniml/gliner-biomed-post-training": (
        "hf",
        (
            ("script", "GlinerBiomedPostScript", ("entities", "classifications", "structures", "relations")),
            1.0,
            "anthonyyazdaniml/gliner-biomed-post-training",
            None,
            "train",
            None,
            ("tokenized_text", "ner", "negatives"),
        ),
    ),
    "disi-unibo-nlp/Pile-NER-biomed-IOB": (
        "hf",
        (
            ("script", "PileNerBiomedScript", ("entities",)),
            1.0,
            "disi-unibo-nlp/Pile-NER-biomed-IOB",
            None,
            "train",
            None,
            ("tokens", "ner_tags"),
        ),
    ),
    "Universal-NER/Pile-NER-type": (
        "hf",
        (
            ("script", "PileNerTypeScript", ("entities",)),
            1.0,
            "Universal-NER/Pile-NER-type",
            None,
            "train",
            None,
            ("conversations",),
        ),
    ),
    "TrialPanorama/TrialPanorama-database:studies": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "TrialPanorama/TrialPanorama-database",
            "studies",
            "all",
            None,
            ("abstract",),
        ),
    ),
    "aps/super_glue:multirc": (
        "hf",
        (
            ("script", "SuperGlueMultiRCScript", ("classifications",)),
            1.0,
            "aps/super_glue",
            "multirc",
            "train",
            None,
            ("paragraph", "question", "answer", "label"),
        ),
    ),
    # the one local-source entry: no subset/split/match_on/columns_out, just the avro path, so its
    # tuple is deliberately shorter than the hf ones above (it still carries the weight field, which
    # DatasetBase declares for every source)
    "interventions/interventions.avro": (
        "local",
        (
            ("script", "CtkpInterventionsScript", ("entities",)),
            1.0,
            "interventions/interventions.avro",
        ),
    ),
    "knowledgator/sentence_rex": (
        "hf",
        (
            ("script", "SentenceRexScript", ("relations",)),
            1.0,
            "knowledgator/sentence_rex",
            None,
            "train",
            None,
            ("sentences", "labels"),
        ),
    ),
    "knowledgator/biomed_NER": (
        "hf",
        (
            ("script", "KnowledgatorBiomedScript", ("entities",)),
            1.0,
            "knowledgator/biomed_NER",
            None,
            "train",
            None,
            ("text", "entities"),
        ),
    ),
    "knowledgator/gliner-multilingual-synthetic": (
        "hf",
        (
            ("script", "GlinerMultilingualScript", ("entities",)),
            1.0,
            "knowledgator/gliner-multilingual-synthetic",
            None,
            "train",
            None,
            ("tokenized_text", "ner"),
        ),
    ),
    "knowledgator/PubMedAbstractsNER:train.json": (
        "hf_json",
        (
            ("script", "PubmedAbstractsScript", ("entities", "relations")),
            1.0,
            "knowledgator/PubMedAbstractsNER",
            "train.json",
            "train",
            None,
            ("tokenized_text", "ner"),
        ),
    ),
    "aps/super_glue:record": (
        "hf",
        (
            ("script", "SuperGlueRecordScript", ("entities", "classifications")),
            1.0,
            "aps/super_glue",
            "record",
            "train",
            None,
            ("passage", "query", "entities", "entity_spans", "answers"),
        ),
    ),
    "qualifiers/qualifier_corpus.tsv": (
        "local_delimited",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "qualifiers/qualifier_corpus.tsv",
            ("text",),
            None,
        ),
    ),
    # nvidia/Nemotron-PII declares one ingest PER SPLIT off one repo id: entry_key qualifies on the
    # split when no subset is declared, so the two locks coexist instead of silently overwriting
    "nvidia/Nemotron-PII:train": (
        "hf",
        (
            ("script", "NemotronPiiScript", ("entities",)),
            1.0,
            "nvidia/Nemotron-PII",
            None,
            "train",
            None,
            ("text", "spans"),
        ),
    ),
    "nvidia/Nemotron-PII:test": (
        "hf",
        (
            ("script", "NemotronPiiScript", ("entities",)),
            1.0,
            "nvidia/Nemotron-PII",
            None,
            "test",
            None,
            ("text", "spans"),
        ),
    ),
    "bigbio/chemprot:chemprot_full_source:train": (
        "hf",
        (
            ("script", "ChemprotScript", ("entities", "relations")),
            1.0,
            "bigbio/chemprot",
            "chemprot_full_source",
            "train",
            None,
            ("text", "entities", "relations"),
        ),
    ),
    "bigbio/chemprot:chemprot_full_source:validation": (
        "hf",
        (
            ("script", "ChemprotScript", ("entities", "relations")),
            1.0,
            "bigbio/chemprot",
            "chemprot_full_source",
            "validation",
            None,
            ("text", "entities", "relations"),
        ),
    ),
    "bigbio/chemprot:chemprot_full_source:test": (
        "hf",
        (
            ("script", "ChemprotScript", ("entities", "relations")),
            1.0,
            "bigbio/chemprot",
            "chemprot_full_source",
            "test",
            None,
            ("text", "entities", "relations"),
        ),
    ),
}


def entry_base_key(payload: tuple[object, ...]) -> str:
    """repo id plus a scalar discriminator (subset for "hf", file for "hf_json"): two ingests
    reading different subsets of aps/super_glue already collide at this base, and a bare repo id
    would silently lose one lock in the EXPECTED literal"""
    dataset = str(payload[2])
    discriminator = payload[3] if len(payload) > 3 else None
    # only a SCALAR discriminator qualifies the key: for the local sources payload position 3 is
    # columns_out (a tuple), and there the declared path is already unique per file
    return f"{dataset}:{discriminator}" if isinstance(discriminator, str) else dataset


def entry_key(payload: tuple[object, ...], colliding_bases: set[str]) -> str:
    """one key per DECLARED INGEST, not per repo: a base key declared on more than one ingest
    (nvidia/Nemotron-PII once PER SPLIT, bigbio/chemprot sharing dataset AND subset) gets the
    split appended, so the locks coexist instead of one silently overwriting the other in
    tuples_by_ingest -- the loser would vanish from both locks and the failure would be silent.
    Kept identical to .pi/skills/add-dataset/scripts/probe.py (report_freeze, iter_declared),
    which freezes and matches on these keys"""
    base = entry_base_key(payload)
    if base not in colliding_bases:
        return base
    split = payload[4] if len(payload) > 4 else None
    if not isinstance(split, str):
        raise ValueError(f"entry base {base!r} is declared on more than one ingest but its split {split!r} is not a str")
    return f"{base}:{split}"


def tuples_by_ingest() -> dict[str, tuple[object, ...]]:
    # the repo id moved to payload position 2 when the weight field joined DatasetBase; a base
    # key appearing on more than one declared ingest gets split-qualified keys (see entry_key)
    entries: list[tuple[str, tuple[object, ...]]] = list(YamlIngestsParser().generate_tuples())
    base_counts: Counter[str] = Counter(entry_base_key(payload) for _, payload in entries)
    colliding: set[str] = {base for base, count in base_counts.items() if count > 1}
    keyed: dict[str, tuple[object, ...]] = {}
    for source, payload in entries:
        key = entry_key(payload, colliding)
        if key in keyed:
            raise ValueError(f"entry key {key!r} is produced by more than one declared ingest")
        keyed[key] = (source, payload)
    return keyed


@pytest.mark.parametrize("ingest", sorted(EXPECTED))
def test_the_declared_tuple_shape_is_locked_per_ingest(ingest: str) -> None:
    assert tuples_by_ingest()[ingest] == EXPECTED[ingest]


def test_every_declared_ingest_is_accounted_for() -> None:
    assert set(tuples_by_ingest()) == set(EXPECTED)


def test_entry_key_fails_loudly_for_a_colliding_non_string_split() -> None:
    """The fail-loud guard that keeps a colliding base key from silently overwriting its locks
    when the split slot is malformed; this branch is the entire point of the rule, so it must be
    exercised directly rather than only ever firing on real declarations."""
    payload = (
        ("script", "SyntheticScript", ("entities",)),
        1.0,
        "synthetic/dataset",
        "synthetic_subset",
        123,
        None,
        ("text", "ner"),
    )
    with pytest.raises(ValueError, match="split 123 is not a str"):
        entry_key(payload, {"synthetic/dataset:synthetic_subset"})


# doc drift guard: docs/yaml-config.md is the agent-facing schema reference, and it rots silently
# when a model field is added or renamed; parametrizing over the live model_fields (not a copied
# list) means the guard itself cannot go stale. Every name below must appear in the doc.
# Every concrete dataset source is listed: main split the single LocalDataset into the avro,
# delimited, and hf_json kinds, and a doc naming only one of them is exactly the silent rot
# this guard exists to catch.
_DOC_MODELS = (
    YamlIngests,
    ScriptTask,
    FullmapTask,
    HuggingFaceDataset,
    HuggingFaceJsonDataset,
    LocalAvroDataset,
    LocalDelimitedDataset,
    MatchOn,
    Cluster,
    WorkerNode,
    GazetteerPredicate,
    GazetteerQualifier,
    GazetteerSpec,
)

_DOC_FIELD_NAMES: tuple[str, ...] = tuple(sorted({name for model in _DOC_MODELS for name in model.model_fields}))


@pytest.mark.parametrize("field_name", _DOC_FIELD_NAMES)
def test_every_yaml_model_field_is_named_in_docs_yaml_config(field_name: str) -> None:
    text = pathlib.Path("docs/yaml-config.md").read_text(encoding="utf-8")
    # backtick-anchored match, not a bare substring: prose words like "name" or "range" must
    # not satisfy the guard -- only a code span (table cell or inline) naming the field counts
    assert re.search(rf"`{re.escape(field_name)}`", text), f"field {field_name!r} is missing from docs/yaml-config.md"
