from __future__ import annotations

import pytest

from relmedner.ingests import YamlIngestsParser

# per-dataset locks: each dataset's entry is asserted independently so adding a third dataset is an
# additive block here rather than a rewritten literal (and a guaranteed merge conflict) across the
# parallel dataset worktrees
EXPECTED: dict[str, tuple[object, ...]] = {
    "anthonyyazdaniml/gliner-biomed-pre-training": (
        "hf",
        (
            ("script", "GlinerBiomedScript", ("entities", "relations")),
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
            "disi-unibo-nlp/Pile-NER-biomed-IOB",
            None,
            "train",
            None,
            ("tokens", "ner_tags"),
        ),
    ),
    # the one local-source entry: no subset/split/match_on/columns_out, just the avro path, so its
    # tuple is deliberately shorter than the hf ones above
    "~/Desktop/interventions.avro": (
        "local",
        (
            ("script", "CtkpInterventionsScript", ("entities",)),
            "~/Desktop/interventions.avro",
        ),
    ),
    "knowledgator/sentence_rex": (
        "hf",
        (
            ("script", "SentenceRexScript", ("relations",)),
            "knowledgator/sentence_rex",
            None,
            "train",
            None,
            ("sentences", "labels"),
        ),
    ),
}


def tuples_by_dataset() -> dict[str, tuple[object, ...]]:
    return {entry[1][1]: entry for entry in YamlIngestsParser().generate_tuples()}


@pytest.mark.parametrize("dataset", sorted(EXPECTED))
def test_the_declared_tuple_shape_is_locked_per_dataset(dataset: str) -> None:
    assert tuples_by_dataset()[dataset] == EXPECTED[dataset]


def test_every_declared_dataset_is_accounted_for() -> None:
    assert set(tuples_by_dataset()) == set(EXPECTED)
