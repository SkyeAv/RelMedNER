from __future__ import annotations

import pytest

from relmedner.ingests import YamlIngestsParser

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
    "~/Desktop/interventions.avro": (
        "local",
        (
            ("script", "CtkpInterventionsScript", ("entities",)),
            1.0,
            "~/Desktop/interventions.avro",
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
}


def entry_key(payload: tuple[object, ...]) -> str:
    """one key per DECLARED INGEST, not per repo: the discriminator after the repo id (subset for
    "hf", file for "hf_json") joins the key whenever one is declared, because two ingests read
    different subsets of aps/super_glue and a bare repo id would collide in the EXPECTED literal --
    the loser would vanish from both locks and the failure would be silent"""
    dataset = str(payload[2])
    discriminator = payload[3] if len(payload) > 3 else None
    return f"{dataset}:{discriminator}" if discriminator is not None else dataset


def tuples_by_ingest() -> dict[str, tuple[object, ...]]:
    # the repo id moved to payload position 2 when the weight field joined DatasetBase
    return {entry_key(payload): (source, payload) for source, payload in YamlIngestsParser().generate_tuples()}


@pytest.mark.parametrize("ingest", sorted(EXPECTED))
def test_the_declared_tuple_shape_is_locked_per_ingest(ingest: str) -> None:
    assert tuples_by_ingest()[ingest] == EXPECTED[ingest]


def test_every_declared_ingest_is_accounted_for() -> None:
    assert set(tuples_by_ingest()) == set(EXPECTED)
