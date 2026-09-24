from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from relmedner.enums import DedupMode, OutputShapes, ProcessingTypes
from relmedner.models import (
    ChoiceField,
    Classification,
    Dataset,
    DatasetBase,
    Description,
    Entity,
    HuggingFaceDataset,
    HuggingFaceJsonDataset,
    HuggingFaceParquetDataset,
    LocalAvroDataset,
    LocalDelimitedDataset,
    Relation,
    RelationField,
    RunConfig,
    ScriptTask,
    StrictBase,
    Structure,
    StructureField,
    TrainingExample,
    YamlIngests,
)
from relmedner.pipeline import dispatch_row, matches_declared_outputs
from relmedner.streams import StreamedRow
from relmedner.utils import ScriptUtils


def every_strict_subclass(base: type[StrictBase] = StrictBase) -> list[type[StrictBase]]:
    Subclasses: list[type[StrictBase]] = []
    for Subclass in base.__subclasses__():
        Subclasses.append(Subclass)
        Subclasses.extend(every_strict_subclass(Subclass))
    return Subclasses


@pytest.mark.parametrize("model", every_strict_subclass(), ids=lambda model: model.__name__)
def test_every_strict_model_generates_an_avro_schema(model: type[StrictBase]) -> None:
    Schema: dict[str, Any] = model.avro_schema_to_python()

    assert Schema["type"] == "record"
    assert Schema["name"] == model.__name__


def test_the_structure_value_union_carries_all_three_arms() -> None:
    Schema: dict[str, Any] = StructureField.avro_schema_to_python()
    Value: Any = next(field["type"] for field in Schema["fields"] if field["name"] == "value")
    Arms: list[Any] = [arm if isinstance(arm, str) else arm["type"] for arm in Value]

    assert Arms == ["string", "array", "record"]


def test_populated_reports_only_the_fields_that_carry_tasks() -> None:
    Example: TrainingExample = TrainingExample(
        text="Alice manages engineering team.",
        entities=[Entity(label="person", mentions=["Alice"])],
        relations=[
            Relation(
                name="manages",
                fields=[RelationField(name="head", value="Alice"), RelationField(name="tail", value="engineering team")],
            )
        ],
    )

    assert Example.populated() == frozenset({"entities", "relations"})


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        (("entities",), True),
        (("relations",), False),
        (("entities", "relations"), True),
        (("classifications",), False),
        (("entities", "classifications", "structures", "relations"), True),
    ],
)
def test_matches_declared_outputs_keeps_any_nonempty_subset_of_the_declared_shapes(outputs: tuple[str, ...], expected: bool) -> None:
    """permitted-shapes contract: one script emits different shapes per row family, so a row ships when
    it produced something and everything it produced was declared"""
    Example: TrainingExample = TrainingExample(text="Alice", entities=[Entity(label="person", mentions=["Alice"])])

    assert matches_declared_outputs((outputs, Example)) is expected


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        (("entities",), True),
        (("entities", "relations"), True),
        (("relations",), True),
        (("classifications",), False),
    ],
)
def test_matches_declared_outputs_keeps_extra_populated_shapes(outputs: tuple[str, ...], expected: bool) -> None:
    """subset semantics: a declared [entities] ingest keeps rows that also extracted relations"""
    Example: TrainingExample = TrainingExample(
        text="Aspirin treats headache",
        entities=[Entity(label="Drug", mentions=["Aspirin"])],
        relations=[Relation(name="treats", fields=[RelationField(name="head", value="Aspirin"), RelationField(name="tail", value="headache")])],
    )

    assert matches_declared_outputs((outputs, Example)) is expected


def test_an_example_with_no_tasks_never_matches_a_declaration() -> None:
    assert matches_declared_outputs((("entities",), TrainingExample(text="nothing here"))) is False


def test_to_output_emits_descriptions_only_when_they_are_declared() -> None:
    Bare: dict[str, Any] = TrainingExample(text="Alice", entities=[Entity(label="person", mentions=["Alice"])]).to_output()
    Described: dict[str, Any] = TrainingExample(
        text="Alice", entities=[Entity(label="person", mentions=["Alice"], description="Names of people")]
    ).to_output()

    assert "entity_descriptions" not in Bare["output"]
    assert Described["output"]["entity_descriptions"] == {"person": "Names of people"}


def test_to_output_emits_relation_descriptions_only_when_they_are_declared() -> None:
    """relation_descriptions mirrors entity_descriptions: emitted only when a relation carries one"""
    Bare: TrainingExample = TrainingExample(
        text="Aspirin treats headache",
        relations=[Relation(name="treats", fields=[RelationField(name="head", value="Aspirin"), RelationField(name="tail", value="headache")])],
    )
    Described: TrainingExample = TrainingExample(
        text="Aspirin treats headache",
        relations=[
            Relation(
                name="treats",
                fields=[RelationField(name="head", value="Aspirin"), RelationField(name="tail", value="headache")],
                description="holds between an intervention and a condition it ameliorates",
            )
        ],
    )

    assert "relation_descriptions" not in Bare.to_output()["output"]
    assert Described.to_output()["output"]["relation_descriptions"] == {"treats": "holds between an intervention and a condition it ameliorates"}


def test_predicate_description_reads_biolink_slots_for_every_declared_predicate() -> None:
    """the biolink slot yaml covers all 23 gazetteer predicates; the lookup normalizes underscores"""
    from relmedner.gazetteer import PREDICATE_TRIGGERS

    missing: list[str] = [predicate for predicate in PREDICATE_TRIGGERS if ScriptUtils.predicate_description(predicate) is None]
    assert not missing, missing
    assert ScriptUtils.predicate_description("superclass_of") is not None
    assert ScriptUtils.predicate_description("not_a_biolink_predicate") is None


def test_choice_fields_round_trip_into_the_gliner_choice_shape() -> None:
    Example: TrainingExample = TrainingExample(
        text="Book a single room.",
        structures=[
            Structure(
                name="booking",
                fields=[StructureField(name="room_type", value=ChoiceField(value="single", choices=["single", "double"]))],
            )
        ],
    )

    assert Example.to_output()["output"]["json_structures"] == [{"booking": {"room_type": {"value": "single", "choices": ["single", "double"]}}}]


def test_multi_label_and_label_descriptions_survive_projection() -> None:
    Example: TrainingExample = TrainingExample(
        text="Great camera, poor battery.",
        classifications=[
            Classification(
                task="aspects",
                labels=["camera", "battery"],
                true_label=["camera", "battery"],
                multi_label=True,
                label_descriptions=[Description(key="camera", description="Photo quality")],
            )
        ],
    )
    Projected: dict[str, Any] = Example.to_output()["output"]["classifications"][0]

    assert Projected["multi_label"] is True
    assert Projected["true_label"] == ["camera", "battery"]
    assert Projected["label_descriptions"] == {"camera": "Photo quality"}


def test_relations_carry_provenance_defaults() -> None:
    Relation_: Relation = Relation(
        name="treats",
        fields=[RelationField(name="head", value="Aspirin"), RelationField(name="tail", value="headache")],
    )

    assert Relation_.negated is False and Relation_.evidence == "asserted"


# ---------------------------------------------------------------- mixing weights --


def test_weight_defaults_to_neutral_and_rejects_negative() -> None:
    """1.0 keeps every existing producer honest (scripts and the miner build weightless examples);
    a NEGATIVE weight is a model error, but 0 is now the documented soft drop (US-011): the
    record still reaches avro provenance and duplicates zero times at export"""
    assert TrainingExample(text="Alice").weight == 1.0
    assert TrainingExample(text="Alice", weight=0.0).weight == 0.0
    with pytest.raises(ValidationError):
        TrainingExample(text="Alice", weight=-1.5)
    assert (
        HuggingFaceDataset(
            task=ScriptTask(type="script", name="GlinerBiomedScript", outputs=["entities"]),
            weight=0.0,
            source="hf",
            dataset="a/b",
            columns_out=["text"],
        ).weight
        == 0.0
    )


def _dataset(source: str, weight: float) -> HuggingFaceDataset:
    return HuggingFaceDataset(
        task=ScriptTask(type="script", name="GlinerBiomedScript", outputs=["entities"]),
        weight=weight,
        source="hf",
        dataset=source,
        columns_out=["text"],
    )


def test_weights_by_source_maps_every_declared_source() -> None:
    Ingests: YamlIngests = YamlIngests(datasets=[_dataset("a/b", 1.0), _dataset("c/d", 2.5)])

    assert Ingests.weights_by_source() == {"a/b": 1.0, "c/d": 2.5}


def test_weights_by_source_rejects_conflicting_duplicate_sources() -> None:
    """rows key on the source name, so two entries sharing a source with different weights would
    make the stamp ambiguous -- fail loudly instead of silently picking one"""
    Ingests: YamlIngests = YamlIngests(datasets=[_dataset("a/b", 1.0), _dataset("a/b", 2.0)])

    with pytest.raises(ValueError, match="conflicting weights"):
        Ingests.weights_by_source()


def test_weights_by_source_accepts_duplicate_sources_with_equal_weights() -> None:
    Ingests: YamlIngests = YamlIngests(datasets=[_dataset("a/b", 1.5), _dataset("a/b", 1.5)])

    assert Ingests.weights_by_source() == {"a/b": 1.5}


def test_dispatch_row_stamps_the_declared_source_weight() -> None:
    """the source key rides the streamed row until dispatch; the stamped example keeps its
    content and the declared weight becomes avro provenance"""
    Row: StreamedRow = ("some/source", (("script", "GlinerBiomedScript", ("entities",)), ([], [])))
    Outputs, Example = dispatch_row(Row, weights={"some/source": 2.5}, edge_trusts={})

    assert Outputs == ("entities",)
    assert Example.weight == 2.5
    assert Example.text == ""


def test_dispatch_row_keeps_the_neutral_weight_when_none_is_declared() -> None:
    Row: StreamedRow = ("some/source", (("script", "GlinerBiomedScript", ("entities",)), ([], [])))
    _Outputs, Example = dispatch_row(Row, weights={"some/source": 1.0}, edge_trusts={})

    assert Example.weight == 1.0


def test_x_defaults_namespace_accepted_and_ignored() -> None:
    """compose-spec x- convention (https://compose-spec.github.io/compose-spec/11-extension.html):
    reusable fragments live in a top-level x-defaults map OUTSIDE the validated datasets list;
    anchors are resolved by CSafeLoader before pydantic sees anything, so the value only has to
    round-trip untouched while dataset tuples stay byte-identical"""
    Declaration: dict[str, Any] = {
        "datasets": [
            {
                "source": "hf",
                "dataset": "a/b",
                "task": {"type": "script", "name": "GlinerBiomedScript", "outputs": ["entities"]},
                "columns_out": ["text"],
            }
        ]
    }
    Baseline: YamlIngests = YamlIngests.model_validate(Declaration)
    Extended: YamlIngests = YamlIngests.model_validate({"x-defaults": {"hf-train": {"weight": 1.0, "split": "train"}}, **Declaration})

    assert Extended.x_defaults == {"hf-train": {"weight": 1.0, "split": "train"}}
    assert Baseline.x_defaults is None
    assert Extended.generate_tuples() == Baseline.generate_tuples()


def test_unknown_top_level_key_still_rejected() -> None:
    """extra="forbid" guards the positional tuple locks in test_ingests.py; x-defaults is the only
    top-level key admitted beyond datasets"""
    with pytest.raises(ValidationError):
        YamlIngests.model_validate({"bogus": 1, "datasets": []})


def test_x_defaults_declared_after_datasets() -> None:
    """model_fields declaration order is the positional contract frozen by DatasetBase.to_tuple()
    and the EXPECTED locks; appending (never inserting or renaming) is the only safe model change,
    so a future reorder fails loudly here"""
    # "x_trust" joined after gazetteer (US-011): appended, never inserted -- it must stay LAST
    assert tuple(YamlIngests.model_fields) == ("datasets", "x_defaults", "gazetteer", "x_trust")


@pytest.mark.parametrize(
    "model",
    (HuggingFaceDataset, HuggingFaceJsonDataset, HuggingFaceParquetDataset, LocalAvroDataset, LocalDelimitedDataset),
    ids=lambda model: model.__name__,
)
def test_dataset_tuple_fields_freeze_the_packing_order(model: type[DatasetBase]) -> None:
    """tuple_fields is the explicit packing order DatasetBase.to_tuple() iterates and the streams
    unpack positionally; pinning it equal to model_fields minus NON_PAYLOAD_FIELDS means any
    reorder, rename, or unlisted append (e.g. US-008 filters, reserved in NON_PAYLOAD_FIELDS)
    fails CI here instead of silently shifting the EXPECTED tuple locks in test_ingests.py"""
    assert model.tuple_fields == tuple(name for name in model.model_fields if name not in DatasetBase.NON_PAYLOAD_FIELDS)


def test_relation_provenance_survives_avro_but_stays_out_of_the_gliner_projection() -> None:
    """gliner2's Relation(**fields) swallows extra keys (dropping head/tail) and validates
    every string value as a mention -- so evidence/negated ride in avro records only"""
    import io

    from fastavro import reader, writer

    Mined: Relation = Relation(
        name="treats",
        fields=[RelationField(name="head", value="Aspirin"), RelationField(name="tail", value="headache")],
        negated=False,
        evidence="distant",
    )
    Example: TrainingExample = TrainingExample(text="Aspirin treats headache.", relations=[Mined])

    Buffer: io.BytesIO = io.BytesIO()
    writer(Buffer, Example.avro_schema_to_python(), [Example.asdict()])
    Buffer.seek(0)
    Restored: TrainingExample = TrainingExample(**next(iter(reader(Buffer))))

    assert Restored == Example and Restored.relations[0].evidence == "distant"
    assert Restored.to_output()["output"]["relations"] == [{"treats": {"head": "Aspirin", "tail": "headache"}}]


def test_huggingface_json_dataset_pins_the_positional_payload_contract() -> None:
    """registry.build_stream splats to_tuple positionally into HuggingFaceJsonDataStream.__init__, so the
    model field order minus source IS the stream constructor order"""
    Entry: HuggingFaceJsonDataset = HuggingFaceJsonDataset(
        task=ScriptTask(type=ProcessingTypes.SCRIPT, name="PubmedAbstractsScript", outputs=[OutputShapes.ENTITIES]),
        source="hf_json",
        dataset="knowledgator/PubMedAbstractsNER",
        file="train.json",
        split="train",
        match_on=None,
        columns_out=["tokenized_text", "ner"],
    )

    # "filters" joined DatasetBase after this lock was written, and it sits in
    # NON_PAYLOAD_FIELDS precisely so an appended field cannot shift a payload position:
    # model_fields carries it, tuple_fields (the packing order asserted below) does not.
    # "trust" and "trust_edges" joined after weight (US-011), before the keyword-only "filters"
    # slot -- like filters they sit in NON_PAYLOAD_FIELDS, so tuple_fields (asserted below)
    # carries none of them and no payload position shifted
    assert list(HuggingFaceJsonDataset.model_fields) == [
        "task",
        "weight",
        "trust",
        "trust_edges",
        "filters",
        "source",
        "dataset",
        "file",
        "split",
        "match_on",
        "columns_out",
    ]
    assert HuggingFaceJsonDataset.tuple_fields == ("task", "weight", "dataset", "file", "split", "match_on", "columns_out")
    assert Entry.to_tuple() == (
        "hf_json",
        (
            ("script", "PubmedAbstractsScript", ("entities",)),
            1.0,
            "knowledgator/PubMedAbstractsNER",
            "train.json",
            "train",
            None,
            ("tokenized_text", "ner"),
        ),
    )


def test_the_dataset_discriminated_union_accepts_the_hf_json_variant() -> None:
    """the annotated Dataset union is what ingests.yaml validation dispatches on; a missing arm would
    reject the new source at yaml parse time"""
    Parsed: HuggingFaceJsonDataset = TypeAdapter(Dataset).validate_python(
        {
            "task": {"type": "script", "name": "PubmedAbstractsScript", "outputs": ["entities"]},
            "source": "hf_json",
            "dataset": "knowledgator/PubMedAbstractsNER",
            "file": "train.json",
            "split": "train",
            "match_on": None,
            "columns_out": ["tokenized_text", "ner"],
        }
    )

    assert isinstance(Parsed, HuggingFaceJsonDataset)


def test_huggingface_json_dataset_rejects_an_empty_file_name() -> None:
    """file rides into an hf://datasets/{dataset}/{file} URL; an empty name would only fail deep inside
    the hub client, so the model gate fails loudly instead"""
    with pytest.raises(ValidationError):
        HuggingFaceJsonDataset(
            task=ScriptTask(type=ProcessingTypes.SCRIPT, name="PubmedAbstractsScript", outputs=[OutputShapes.ENTITIES]),
            source="hf_json",
            dataset="knowledgator/PubMedAbstractsNER",
            file="",
            split="train",
            match_on=None,
            columns_out=["tokenized_text"],
        )


def test_huggingface_parquet_dataset_pins_the_positional_payload_contract() -> None:
    """registry.build_stream splats to_tuple positionally into HuggingFaceParquetDataStream.__init__, so the
    model field order minus source IS the stream constructor order; the packing order mirrors HuggingFaceJsonDataset
    exactly so both builder-backed sources stay interchangeable"""
    Entry: HuggingFaceParquetDataset = HuggingFaceParquetDataset(
        task=ScriptTask(type=ProcessingTypes.SCRIPT, name="GlinerBiomedScript", outputs=[OutputShapes.ENTITIES, OutputShapes.RELATIONS]),
        source="hf_parquet",
        dataset="bigbio/ehr_rel",
        file="ehr_rel_bigbio_pairs/train/0000.parquet",
        split="train",
        match_on=None,
        columns_out=["text", "relations"],
    )

    # "filters" sits in NON_PAYLOAD_FIELDS, so an appended field cannot shift a payload position:
    # model_fields carries it, tuple_fields (the packing order asserted below) does not.
    assert list(HuggingFaceParquetDataset.model_fields) == [
        "task",
        "weight",
        "trust",
        "trust_edges",
        "filters",
        "source",
        "dataset",
        "file",
        "split",
        "match_on",
        "columns_out",
    ]
    assert HuggingFaceParquetDataset.tuple_fields == ("task", "weight", "dataset", "file", "split", "match_on", "columns_out")
    assert Entry.to_tuple() == (
        "hf_parquet",
        (
            ("script", "GlinerBiomedScript", ("entities", "relations")),
            1.0,
            "bigbio/ehr_rel",
            "ehr_rel_bigbio_pairs/train/0000.parquet",
            "train",
            None,
            ("text", "relations"),
        ),
    )
    assert Entry.row_key == "bigbio/ehr_rel"


def test_the_dataset_discriminated_union_accepts_the_hf_parquet_variant() -> None:
    """the annotated Dataset union is what ingests.yaml validation dispatches on; a missing arm would
    reject the new source at yaml parse time"""
    Parsed: HuggingFaceParquetDataset = TypeAdapter(Dataset).validate_python(
        {
            "task": {"type": "script", "name": "GlinerBiomedScript", "outputs": ["entities"]},
            "source": "hf_parquet",
            "dataset": "bigbio/ehr_rel",
            "file": "ehr_rel_bigbio_pairs/train/0000.parquet",
            "split": "train",
            "match_on": None,
            "columns_out": ["text"],
        }
    )

    assert isinstance(Parsed, HuggingFaceParquetDataset)


def test_huggingface_parquet_dataset_rejects_an_empty_file_name() -> None:
    """file rides into the hf://datasets/{dataset}@refs/convert/parquet/{file} URL; an empty name
    would only fail deep inside the hub client, so the model gate fails loudly instead"""
    with pytest.raises(ValidationError):
        HuggingFaceParquetDataset(
            task=ScriptTask(type=ProcessingTypes.SCRIPT, name="GlinerBiomedScript", outputs=[OutputShapes.ENTITIES]),
            source="hf_parquet",
            dataset="bigbio/ehr_rel",
            file="",
            split="train",
            match_on=None,
            columns_out=["text"],
        )


# ---------------------------------------------------------------- dedup mode (US-004) --


def test_run_config_defaults_to_near_dedup() -> None:
    """REQ-INT-1: dedup is ON by default -- repeats must not reach training data unless it is
    explicitly disabled. use_enum_values stores the StrEnum's str value, which still compares
    equal to the member, so the default reads back as DedupMode.NEAR"""
    assert RunConfig().dedup_mode == DedupMode.NEAR


def test_run_config_from_flags_threads_dedup_mode() -> None:
    """REQ-INT-1: the CLI flag lands on the config; the model default follows when no flag
    is given, and the pre-existing positional order (test_run, output) is unchanged"""
    assert RunConfig.from_flags(True).dedup_mode == DedupMode.NEAR
    assert RunConfig.from_flags(True, dedup_mode=DedupMode.OFF).dedup_mode == DedupMode.OFF
    Config: RunConfig = RunConfig.from_flags(False, "custom.avro", DedupMode.EXACT)
    assert Config.dedup_mode == DedupMode.EXACT
    assert Config.sample_limit is None and Config.output == "custom.avro"


def test_run_config_rejects_unknown_dedup_mode_loudly() -> None:
    """REQ-INT-1: an unknown mode fails fast at model validation, never silently mid-run"""
    with pytest.raises(ValidationError):
        RunConfig(dedup_mode="bogus")


def test_slot_descriptions_load_through_libyaml_and_match_the_pure_python_loader() -> None:
    """the biolink slot table is parsed with CSafeLoader for speed; this pins that the result is
    the same flattened table the pure-python safe_load produced, so the speedup can never change
    a predicate description"""
    from importlib.resources import files

    import yaml

    from relmedner.utils import ScriptUtils

    schema = files("biolink_model").joinpath("schema/biolink_model.yaml").read_text(encoding="utf-8")
    assert yaml.load(schema, Loader=yaml.CSafeLoader) == yaml.safe_load(schema)
    assert ScriptUtils._load_slot_descriptions()["treats"]
