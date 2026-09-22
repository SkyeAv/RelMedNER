from __future__ import annotations

import importlib.util
import io
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastavro import reader, writer

from relmedner.models import (
    ChoiceField,
    Classification,
    Description,
    Entity,
    FullmapTask,
    Relation,
    RelationField,
    Structure,
    StructureField,
    TrainingExample,
)
from relmedner.scripts import GlinerBiomedScript, SentenceRexScript
from relmedner.scripts.sentence_rex import parse_tagged_sentence
from relmedner.utils import ResolvedMention, ScriptUtils

GLINER_DATA_MODULE: str = "gliner2_training_data"


def load_gliner_data() -> ModuleType:
    """loads gliner2.training.data by path because importing the package pulls torch"""

    if GLINER_DATA_MODULE in sys.modules:
        return sys.modules[GLINER_DATA_MODULE]

    Located: ModuleSpec | None = importlib.util.find_spec("gliner2")
    if Located is None or not Located.submodule_search_locations:
        pytest.skip("gliner2 is not installed")

    Source: Path = Path(next(iter(Located.submodule_search_locations))) / "training" / "data.py"
    Spec: ModuleSpec | None = importlib.util.spec_from_file_location(GLINER_DATA_MODULE, Source)
    if Spec is None or Spec.loader is None:
        pytest.skip("gliner2 training data module is unavailable")

    Module: ModuleType = importlib.util.module_from_spec(Spec)
    sys.modules[GLINER_DATA_MODULE] = Module
    Spec.loader.exec_module(Module)
    return Module


ENTITY_EXAMPLE: TrainingExample = TrainingExample(
    text="Dr. Sarah Johnson prescribed Metformin 500mg daily for diabetes.",
    entities=[
        Entity(label="person", mentions=["Dr. Sarah Johnson"], description="Names of people"),
        Entity(label="medication", mentions=["Metformin"]),
        Entity(label="condition", mentions=["diabetes"]),
    ],
)

CLASSIFICATION_EXAMPLE: TrainingExample = TrainingExample(
    text="This smartphone has an amazing camera but the battery life is poor.",
    classifications=[
        Classification(
            task="product_aspects",
            labels=["camera", "battery", "screen"],
            true_label=["camera", "battery"],
            multi_label=True,
            prompt="Which aspects are discussed?",
            label_descriptions=[Description(key="camera", description="Photo quality")],
        )
    ],
)

STRUCTURE_EXAMPLE: TrainingExample = TrainingExample(
    text="Book a single room at Grand Hotel for 2 nights with breakfast included.",
    structures=[
        Structure(
            name="booking",
            fields=[
                StructureField(name="hotel", value="Grand Hotel", description="Name of the hotel"),
                StructureField(name="nights", value="2"),
                StructureField(name="room_type", value=ChoiceField(value="single", choices=["single", "double", "suite"])),
            ],
        )
    ],
)

RELATION_EXAMPLE: TrainingExample = TrainingExample(
    text="Elon Musk founded SpaceX in 2002. SpaceX is located in Hawthorne.",
    entities=[Entity(label="person", mentions=["Elon Musk"]), Entity(label="organization", mentions=["SpaceX"])],
    relations=[
        Relation(name="founded", fields=[RelationField(name="head", value="Elon Musk"), RelationField(name="tail", value="SpaceX")]),
        Relation(name="located_in", fields=[RelationField(name="head", value="SpaceX"), RelationField(name="tail", value="Hawthorne")]),
    ],
)

EXAMPLES: list[TrainingExample] = [ENTITY_EXAMPLE, CLASSIFICATION_EXAMPLE, STRUCTURE_EXAMPLE, RELATION_EXAMPLE]
EXAMPLE_IDS: list[str] = ["entities", "classifications", "structures", "relations"]


DERIVED_KEYS: frozenset[str] = frozenset({"record_metadata"})


@pytest.mark.parametrize("example", EXAMPLES, ids=EXAMPLE_IDS)
def test_our_output_round_trips_through_the_real_gliner_types(example: TrainingExample) -> None:
    """gliner2 re-emits everything we declare -- it only adds keys it derives itself"""

    GlinerData: ModuleType = load_gliner_data()
    Output: dict[str, Any] = example.to_output()
    Restored: dict[str, Any] = GlinerData.InputExample.from_dict(Output).to_dict()

    assert Restored["input"] == Output["input"]
    assert {key: value for key, value in Restored["output"].items() if key not in DERIVED_KEYS} == Output["output"]


def test_record_metadata_is_the_only_key_gliner_adds_to_our_output() -> None:
    """locks the derived-key allowance so a future gliner2 bump cannot widen it unnoticed"""

    GlinerData: ModuleType = load_gliner_data()
    Added: set[str] = set()
    for example in EXAMPLES:
        Output: dict[str, Any] = example.to_output()
        Added |= set(GlinerData.InputExample.from_dict(Output).to_dict()["output"]) - set(Output["output"])

    assert Added == {"record_metadata"}


def test_the_derived_record_metadata_anchors_on_our_first_declared_field() -> None:
    GlinerData: ModuleType = load_gliner_data()
    Restored: dict[str, Any] = GlinerData.InputExample.from_dict(STRUCTURE_EXAMPLE.to_output()).to_dict()

    assert Restored["output"]["record_metadata"] == {"booking": {"mode": "natural", "anchor": "hotel"}}


@pytest.mark.parametrize("example", EXAMPLES, ids=EXAMPLE_IDS)
def test_gliner_strict_validation_accepts_our_output(example: TrainingExample) -> None:
    GlinerData: ModuleType = load_gliner_data()
    Parsed: Any = GlinerData.InputExample.from_dict(example.to_output())

    assert Parsed.validate() == []


def test_a_dataset_of_every_example_validates_without_raising() -> None:
    GlinerData: ModuleType = load_gliner_data()
    Dataset: Any = GlinerData.TrainingDataset([GlinerData.InputExample.from_dict(example.to_output()) for example in EXAMPLES])
    Report: dict[str, Any] = Dataset.validate()

    assert Report["invalid"] == 0
    assert Report["valid"] == len(EXAMPLES)
    assert Dataset.validate_relation_consistency() == []


def test_an_example_with_a_mention_outside_the_text_is_rejected_by_gliner() -> None:
    GlinerData: ModuleType = load_gliner_data()
    Invalid: TrainingExample = TrainingExample(text="Alice manages the team.", entities=[Entity(label="person", mentions=["Bob"])])
    Parsed: Any = GlinerData.InputExample.from_dict(Invalid.to_output())

    assert Parsed.validate() != []


def test_a_fullmap_mined_row_validates_through_real_gliner(monkeypatch: pytest.MonkeyPatch) -> None:
    """the miner's exact output shape -- entities with fullmap-evidence descriptions plus
    distant relations -- must pass gliner2's strict mention-in-text validation"""
    from tablassert import rs

    def fake_rows(_db: object, distinct: list[str]) -> list[dict[str, object]]:
        table = {
            "aspirin": ("CHEBI:15365", "aspirin", "SmallMolecule"),
            "headach": ("HP:0000001", "headache", "Disease"),
        }
        return [
            {
                "term": term,
                "CURIE": curie,
                "PREFERRED_NAME": name,
                "CATEGORY_NAME": category,
                "TAXON_ID": 0,
                "SOURCE_NAME": "t",
                "SOURCE_VERSION": "t",
            }
            for term in distinct
            for curie, name, category in [table.get(term, (None, None, None))]
            if curie is not None
        ]

    import relmedner.fullmap_mine as mine

    monkeypatch.setattr(mine, "lookup_rows", fake_rows)
    assert rs.normalize_terms(["headache"])[0] == "headach"
    Example: TrainingExample = mine.FullmapMiner.resolve_batch(
        [("Aspirin is associated with headache.", FullmapTask(type="fullmap", outputs=["entities", "relations"]))], db=None
    )[0]

    assert Example.entities and Example.relations
    GlinerData: ModuleType = load_gliner_data()
    Parsed: Any = GlinerData.InputExample.from_dict(Example.to_output())

    assert Parsed.validate() == []


def test_gazetteer_shaped_script_relations_validate_through_real_gliner(monkeypatch: pytest.MonkeyPatch) -> None:
    """US-003's GlinerBiomedScript relation wiring must emit output the real gliner2 types accept"""

    def fake_resolve(spans: list[tuple[str, str]]) -> list[ResolvedMention]:
        return [
            ResolvedMention(mention="Aspirin", category="Drug", origin="fallback"),
            ResolvedMention(mention="migraine", category="Disease", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Tokens: list[str] = ["Aspirin", "is", "used", "to", "treat", "migraine"]
    Example: TrainingExample = GlinerBiomedScript().run((Tokens, [[0, 0, "Drug"], [5, 5, "Condition"]]))

    assert Example.relations == [
        Relation(
            name="treats",
            fields=[RelationField(name="head", value="Aspirin"), RelationField(name="tail", value="migraine")],
            description=ScriptUtils.predicate_description("treats"),
        )
    ]
    GlinerData: ModuleType = load_gliner_data()
    Parsed: Any = GlinerData.InputExample.from_dict(Example.to_output())

    assert Parsed.validate() == []


SENTENCE_REX_ROWS: list[tuple[str, str, str, str]] = [
    (
        "<e1>Pope Pius XII</e1> re - opened the cause on 7 December 1954 , "
        "and Pope John Paul II proclaimed him <e2> Venerable </e2> on 6 July 1985 .",
        "canonization status",
        "canonization_status",
        "Pope Pius XII re - opened the cause on 7 December 1954 , and Pope John Paul II proclaimed him  Venerable  on 6 July 1985 .",
    ),
    (
        'It is sometimes called the " nutmeg family " , after its most famous member , '
        "<e1> Myristica fragrans </e1> , the source of the spices <e2> nutmeg </e2> and mace .",
        "this taxon is source of",
        "this_taxon_is_source_of",
        'It is sometimes called the " nutmeg family " , after its most famous member ,  Myristica fragrans  , '
        "the source of the spices  nutmeg  and mace .",
    ),
]
SENTENCE_REX_ROW_IDS: list[str] = ["pope_canonization_status", "nutmeg_taxon_source"]


def expected_sentence_rex_relation(predicate: str, head: str, tail: str) -> Relation:
    """SentenceRexScript emits exactly one asserted relation per well-formed row; native
    snake_case predicates (both measured labels here) resolve to description None"""
    return Relation(
        name=predicate,
        fields=[RelationField(name="head", value=head), RelationField(name="tail", value=tail)],
        description=ScriptUtils.predicate_description(predicate),
        evidence="asserted",
        negated=False,
    )


@pytest.mark.parametrize("row", SENTENCE_REX_ROWS, ids=SENTENCE_REX_ROW_IDS)
def test_the_sentence_rex_rows_validate_through_real_gliner(row: tuple[str, str, str, str]) -> None:
    """rows 0 and 1 verbatim from the knowledgator/sentence_rex card: SentenceRexScript strips ONLY
    the four tag literals (KD-4), so the doubled spaces around ' Venerable ' and ' Myristica
    fragrans ' survive (gliner2 whitespace-tokenizes them inert) while head/tail surfaces stay
    verbatim -- the 43,044/43,044 measured well-formed rows only pass gliner2 because
    InputExample.validate() requires every relation field value to occur in the emitted text"""
    Sentence, Label, Predicate, TagStripped = row
    Example: TrainingExample = SentenceRexScript().run((Sentence, Label))

    assert Example.text == TagStripped
    assert Example.populated() == frozenset({"relations"})
    assert Example.relations == [expected_sentence_rex_relation(Predicate, *parse_tagged_sentence(Sentence))]

    GlinerData: ModuleType = load_gliner_data()
    Output: dict[str, Any] = Example.to_output()
    assert Output["input"] == Example.text
    assert Output["output"]["relations"] == [{Predicate: {field.name: field.value for field in Example.relations[0].fields}}]
    assert GlinerData.InputExample.from_dict(Output).validate() == []


def test_the_sentence_rex_guard_rejects_a_relation_tail_absent_from_the_text() -> None:
    """negative parity for REQ-SCRIPT-3: gliner2 hard-rejects a relation whose tail does not occur
    in the text, which is exactly why the script must not transform anything but the four tag
    literals -- any normalization that let a surface drift from the tagged row would turn every
    emitted example invalid (the 43,044-row train split validates only because surfaces stay verbatim)"""
    GlinerData: ModuleType = load_gliner_data()
    Sentence, Label, Predicate, _ = SENTENCE_REX_ROWS[0]
    Good: TrainingExample = SentenceRexScript().run((Sentence, Label))
    assert GlinerData.InputExample.from_dict(Good.to_output()).validate() == []

    Corrupted: TrainingExample = TrainingExample(
        text=Good.text,
        relations=[
            Relation(
                name=Predicate,
                fields=[RelationField(name="head", value="Pope Pius XII"), RelationField(name="tail", value="John XXIII")],
                description=ScriptUtils.predicate_description(Predicate),
                evidence="asserted",
                negated=False,
            )
        ],
    )
    assert GlinerData.InputExample.from_dict(Corrupted.to_output()).validate() != []


def test_the_sentence_rex_relations_carry_asserted_evidence_and_exact_head_tail_fields() -> None:
    """provenance contract: every emitted relation is evidence='asserted' (gold-tagged spans, never
    distant or negated) with fields exactly {head, tail}; the gliner2 projection drops evidence/negated
    (they ride the avro records instead), so assert them on the model and the field names on both"""
    for Sentence, Label, _Predicate, _ in SENTENCE_REX_ROWS:
        Example: TrainingExample = SentenceRexScript().run((Sentence, Label))
        Relation: Any = Example.relations[0]

        assert Relation.evidence == "asserted" and Relation.negated is False
        assert [field.name for field in Relation.fields] == ["head", "tail"]
        Output: dict[str, Any] = Example.to_output()
        assert list(next(iter(Output["output"]["relations"][0].values()))) == ["head", "tail"]


@pytest.mark.parametrize("example", EXAMPLES, ids=EXAMPLE_IDS)
def test_examples_round_trip_through_avro(example: TrainingExample) -> None:
    Schema: dict[str, Any] = TrainingExample.avro_schema_to_python()
    Buffer: io.BytesIO = io.BytesIO()
    writer(Buffer, Schema, [example.asdict()])
    Buffer.seek(0)
    Restored: list[dict[str, Any]] = list(reader(Buffer))

    assert len(Restored) == 1
    assert TrainingExample(**Restored[0]) == example


def test_weight_rides_avro_but_stays_out_of_the_gliner_projection() -> None:
    """the mixing weight is avro provenance: stock gliner2 has no per-example weight channel
    (InputExample.from_dict reads input/output only), so to_output() must not emit it -- the
    actual consumption is weighted duplication at the avro->JSONL export step"""
    Weighted: TrainingExample = TrainingExample(
        text="Alice manages the team.",
        weight=2.5,
        entities=[Entity(label="person", mentions=["Alice"])],
    )
    Schema: dict[str, Any] = TrainingExample.avro_schema_to_python()
    Buffer: io.BytesIO = io.BytesIO()
    writer(Buffer, Schema, [Weighted.asdict()])
    Buffer.seek(0)
    Restored: TrainingExample = TrainingExample(**next(iter(reader(Buffer))))

    assert Restored == Weighted and Restored.weight == 2.5

    Output: dict[str, Any] = Weighted.to_output()
    assert set(Output) == {"input", "output"}
    GlinerData: ModuleType = load_gliner_data()
    assert GlinerData.InputExample.from_dict(Output).validate() == []


def test_the_structure_union_survives_avro_for_every_arm() -> None:
    Example: TrainingExample = TrainingExample(
        text="iPhone 15 costs $999 in blue, black and white.",
        structures=[
            Structure(
                name="product",
                fields=[
                    StructureField(name="name", value="iPhone 15"),
                    StructureField(name="colors", value=["blue", "black", "white"]),
                    StructureField(name="tier", value=ChoiceField(value="pro", choices=["base", "pro"])),
                ],
            )
        ],
    )
    Schema: dict[str, Any] = TrainingExample.avro_schema_to_python()
    Buffer: io.BytesIO = io.BytesIO()
    writer(Buffer, Schema, [Example.asdict()])
    Buffer.seek(0)
    Restored: TrainingExample = TrainingExample(**next(iter(reader(Buffer))))

    assert Restored == Example
    assert [type(field.value) for field in Restored.structures[0].fields] == [str, list, ChoiceField]


def test_the_generated_schema_matches_its_snapshot() -> None:
    Schema: dict[str, Any] = TrainingExample.avro_schema_to_python()
    Fields: list[str] = [field["name"] for field in Schema["fields"]]

    assert Schema["type"] == "record"
    assert Schema["name"] == "TrainingExample"
    assert Schema["namespace"] == "relmedner.ingests"
    assert Fields == ["text", "weight", "entities", "classifications", "structures", "relations"]


def test_relation_descriptions_are_emitted_and_accepted_by_the_real_gliner_types() -> None:
    """relations carry biolink slot descriptions; gliner2's processor consumes
    relation_descriptions as label prompts (processor.py reads schema["relation_descriptions"])
    even though InputExample does not serialize the key yet -- an upstream gap mirrored in
    DERIVED_KEYS-style allowances, so assert emission + acceptance, not round-trip"""
    Described: TrainingExample = TrainingExample(
        text="dexamethasone treats COPD",
        entities=[Entity(label="Drug", mentions=["dexamethasone"]), Entity(label="Disease", mentions=["COPD"])],
        relations=[
            Relation(
                name="treats",
                fields=[RelationField(name="head", value="dexamethasone"), RelationField(name="tail", value="COPD")],
                description=ScriptUtils.predicate_description("treats"),
            )
        ],
    )
    Output: dict[str, Any] = Described.to_output()
    assert set(Output["output"]["relation_descriptions"]) == {"treats"}

    GlinerData: ModuleType = load_gliner_data()
    assert GlinerData.InputExample.from_dict(Output).validate() == []
