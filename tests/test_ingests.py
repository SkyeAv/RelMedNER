from __future__ import annotations

import pathlib
import re

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
# the single health-community allowlist match_on lock, shared by all seven tensorshield reddit
# entries (values match the corpus EXACTLY: apply_match is exact membership, so casing must be
# the Reddit canonical form or the row is silently dropped)
EXPECTED_REDDIT_MATCH: tuple[object, ...] = (
    (
        "communityName",
        (
            "r/AskDocs",
            "r/medical",
            "r/medicine",
            "r/Health",
            "r/diabetes",
            "r/ADHD",
            "r/autism",
            "r/Anxiety",
            "r/depression",
            "r/SuicideWatch",
            "r/cancer",
            "r/Celiac",
            "r/ibs",
            "r/IBD",
            "r/CrohnsDisease",
            "r/UlcerativeColitis",
            "r/eczema",
            "r/Psoriasis",
            "r/acne",
            "r/migraine",
            "r/ChronicPain",
            "r/Menopause",
            "r/endometriosis",
            "r/PCOS",
            "r/infertility",
            "r/birthcontrol",
            "r/lupus",
            "r/MultipleSclerosis",
            "r/hypothyroidism",
            "r/Hashimotos",
            "r/asthma",
            "r/COPD",
            "r/epilepsy",
            "r/schizophrenia",
            "r/bipolar",
            "r/BPD",
            "r/OCD",
            "r/ptsd",
            "r/EatingDisorders",
            "r/Dentistry",
            "r/pharmacy",
            "r/nursing",
        ),
    ),
)

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
    # the seven tensorshield reddit ingests share one allowlist via EXPECTED_REDDIT_MATCH: the
    # parsed match_on tuple must equal the declaration, so a yaml-side fork fails here
    "tensorshield/reddit_dataset_157": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "tensorshield/reddit_dataset_157",
            None,
            "train",
            (("communityName", EXPECTED_REDDIT_MATCH[0][1]),),
            ("text",),
        ),
    ),
    "tensorshield/reddit_dataset_171": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "tensorshield/reddit_dataset_171",
            None,
            "train",
            (("communityName", EXPECTED_REDDIT_MATCH[0][1]),),
            ("text",),
        ),
    ),
    "tensorshield/reddit_dataset_217": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "tensorshield/reddit_dataset_217",
            None,
            "train",
            (("communityName", EXPECTED_REDDIT_MATCH[0][1]),),
            ("text",),
        ),
    ),
    "tensorshield/reddit_dataset_237": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "tensorshield/reddit_dataset_237",
            None,
            "train",
            (("communityName", EXPECTED_REDDIT_MATCH[0][1]),),
            ("text",),
        ),
    ),
    "tensorshield/reddit_dataset_30": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "tensorshield/reddit_dataset_30",
            None,
            "train",
            (("communityName", EXPECTED_REDDIT_MATCH[0][1]),),
            ("text",),
        ),
    ),
    "tensorshield/reddit_dataset_84": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "tensorshield/reddit_dataset_84",
            None,
            "train",
            (("communityName", EXPECTED_REDDIT_MATCH[0][1]),),
            ("text",),
        ),
    ),
    "tensorshield/reddit_dataset_85": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "tensorshield/reddit_dataset_85",
            None,
            "train",
            (("communityName", EXPECTED_REDDIT_MATCH[0][1]),),
            ("text",),
        ),
    ),
}


def entry_key(payload: tuple[object, ...], split_qualified: bool = False) -> str:
    """one key per DECLARED INGEST, not per repo: the discriminator after the repo id (subset for
    "hf", file for "hf_json") joins the key whenever one is declared, because two ingests read
    different subsets of aps/super_glue and a bare repo id would collide in the EXPECTED literal --
    the loser would vanish from both locks and the failure would be silent"""
    dataset = str(payload[2])
    discriminator = payload[3] if len(payload) > 3 else None
    # only a SCALAR discriminator qualifies the key: for the local sources payload position 3 is
    # columns_out (a tuple), and there the declared path is already unique per file. A repo id
    # declared once PER SPLIT (nvidia/Nemotron-PII) has no subset, so the split qualifies instead
    # or the train and test locks would collide and one would silently vanish
    if split_qualified and not isinstance(discriminator, str) and len(payload) > 4 and isinstance(payload[4], str):
        discriminator = payload[4]
    return f"{dataset}:{discriminator}" if isinstance(discriminator, str) else dataset


def tuples_by_ingest() -> dict[str, tuple[object, ...]]:
    # the repo id moved to payload position 2 when the weight field joined DatasetBase; a repo id
    # appearing on more than one declared ingest gets split-qualified keys (see entry_key)
    entries: list[tuple[str, tuple[object, ...]]] = list(YamlIngestsParser().generate_tuples())
    declared: dict[str, int] = {}
    for _, payload in entries:
        declared[str(payload[2])] = declared.get(str(payload[2]), 0) + 1
    return {entry_key(payload, split_qualified=declared[str(payload[2])] > 1): (source, payload) for source, payload in entries}


@pytest.mark.parametrize("ingest", sorted(EXPECTED))
def test_the_declared_tuple_shape_is_locked_per_ingest(ingest: str) -> None:
    assert tuples_by_ingest()[ingest] == EXPECTED[ingest]


def test_every_declared_ingest_is_accounted_for() -> None:
    assert set(tuples_by_ingest()) == set(EXPECTED)


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
