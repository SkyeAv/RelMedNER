from __future__ import annotations

from typing import Any

import pytest

from relmedner.families import validate_category
from relmedner.models import Classification, Entity, TrainingExample
from relmedner.scripts import CtkpInterventionsScript  # registry must stay populated alongside the new script
from relmedner.scripts.super_glue_record import SuperGlueRecordScript
from relmedner.types import Script
from relmedner.utils import ScriptUtils

SCRIPT: SuperGlueRecordScript = SuperGlueRecordScript()

PASSAGE: str = (
    "Violence has flared across southern Afghanistan in recent weeks. Taliban militants attacked a police convoy, an Afghan official said Monday."
)
QUERY: str = "The attack came ahead of Afghanistan's @placeholder elections."


def offsets(text: str, passage: str = PASSAGE) -> tuple[int, int]:
    """char-exact [start, end) for the first occurrence of text in passage (mirrors hub rows, verified)"""
    start: int = passage.index(text)
    return (start, start + len(text))


def spans(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "text": ["Taliban militants", "Afghan official"],
        "start": [offsets("Taliban militants")[0], offsets("Afghan official")[0]],
        "end": [offsets("Taliban militants")[1], offsets("Afghan official")[1]],
    }
    return {**base, **overrides}


def row(**overrides: Any) -> tuple[Any, ...]:
    """one ReCoRD row as the columns_out projection delivers it to ScriptValues"""
    base: dict[str, Any] = {
        "passage": PASSAGE,
        "query": QUERY,
        "entities": ["Taliban militants", "Afghan official", "Nuristan"],
        "entity_spans": spans(),
        "answers": ["Taliban militants"],
    }
    merged: dict[str, Any] = {**base, **overrides}
    return (merged["passage"], merged["query"], merged["entities"], merged["entity_spans"], merged["answers"])


def test_the_script_self_registers_under_its_declared_name() -> None:
    """importing the module must install the instance in the shared registry (US-002 wires the
    ingest around Script.dispatch, which resolves by this exact NAME key)"""
    assert isinstance(Script.REGISTRY["SuperGlueRecordScript"], SuperGlueRecordScript)


def test_the_declared_category_is_a_biolink_class() -> None:
    """NamedThing must stay a real tablassert Categories member; the import-time guard only
    protects THIS class's constant, so the contract is re-asserted here for drift"""
    assert SuperGlueRecordScript.NAME == "SuperGlueRecordScript"
    assert ScriptUtils.is_biolink_category(SuperGlueRecordScript.CATEGORY)


def test_the_import_time_category_guard_rejects_a_non_biolink_class() -> None:
    """a typo'd CATEGORY (NotAClass) must fail loudly at import, not silently train garbage labels;
    families.validate_category is the shared guard the module calls on its own constant"""
    with pytest.raises(ValueError, match="NotAClass"):
        validate_category("NotAClass", "BrokenRecordScript")


def test_gold_spans_group_under_one_namedthing_entity() -> None:
    """ReCoRD annotates no entity types, so all gold spans collapse to one NamedThing label; the
    biolink class definition rides along as the gliner2 label prompt"""
    Example: TrainingExample = SCRIPT.run(row())

    assert [entity.label for entity in Example.entities] == ["NamedThing"]
    assert Example.entities[0].mentions == ["Taliban militants", "Afghan official"]
    assert Example.entities[0].description is not None


def test_the_passage_ships_verbatim_including_highlight_markers() -> None:
    """@highlight lines carry document structure the downstream trainer consumes; rewriting or
    stripping them would desync every char offset in the span table"""
    marked: str = "Afghanistan holds elections.\n@highlight\nTaliban militants attacked a convoy."
    Marked: tuple[int, int] = offsets("Afghanistan holds elections.", marked)
    MarkedTable: dict[str, Any] = {"text": ["Afghanistan holds elections."], "start": [Marked[0]], "end": [Marked[1]]}
    Example: TrainingExample = SCRIPT.run(row(passage=marked, entity_spans=MarkedTable))

    assert Example.text == marked
    assert Example.entities[0].mentions == ["Afghanistan holds elections."]


def test_repeated_surfaces_dedupe_preserving_first_occurrence_order() -> None:
    """gold annotators repeat surfaces across a passage; duplicates would double-count the cloze
    label set and skew multi_label, so only the first occurrence survives"""
    First: tuple[int, int] = offsets("Taliban militants")
    Second: tuple[int, int] = (PASSAGE.index("Afghan official"), PASSAGE.index("Afghan official") + len("Afghan official"))
    Table: dict[str, Any] = {
        "text": ["Taliban militants", "Taliban militants", "Afghan official"],
        "start": [First[0], First[0], Second[0]],
        "end": [First[1], First[1], Second[1]],
    }
    Example: TrainingExample = SCRIPT.run(row(entity_spans=Table))

    assert Example.entities[0].mentions == ["Taliban militants", "Afghan official"]
    assert Example.classifications[0].labels == ["Taliban militants", "Afghan official"]


def test_a_ragged_span_table_yields_zero_spans() -> None:
    """parallel lists must stay parallel; a ragged table has no positional meaning, so trusting
    any prefix of it would invent surfaces that were never gold"""
    Table: dict[str, Any] = {"text": ["Taliban militants", "Afghan official"], "start": [0], "end": [0, 1]}
    Example: TrainingExample = SCRIPT.run(row(entity_spans=Table))

    assert Example.entities == []
    assert Example.classifications == []


def test_a_span_table_that_is_not_a_dict_yields_zero_spans() -> None:
    """older ReCoRD revisions delivered a list of structs; only the dict-of-lists shape is gold
    here, and anything else ships entity-less for the pipeline filter to drop"""
    for Malformed in ([{"text": ["x"], "start": [0], "end": [1]}], None, "spans", 7):
        Example: TrainingExample = SCRIPT.run(row(entity_spans=Malformed))

        assert Example.entities == []
        assert Example.classifications == []


def test_a_span_table_missing_a_key_yields_zero_spans() -> None:
    """start without end cannot be sliced; partial tables must not crash the stream worker"""
    Example: TrainingExample = SCRIPT.run(row(entity_spans={"text": ["Taliban militants"], "start": [42]}))

    assert Example.entities == []


def test_out_of_bounds_and_reversed_spans_are_dropped() -> None:
    """offsets beyond the passage or with start >= end point at nothing; they must drop individually
    while sibling valid spans still ship (per-span filtering, not per-row)"""
    End: int = len(PASSAGE)
    Table: dict[str, Any] = {
        "text": ["Taliban militants", "beyond", "backwards", "empty"],
        "start": [offsets("Taliban militants")[0], End + 5, 30, 30],
        "end": [offsets("Taliban militants")[1], End + 20, 20, 30],
    }
    Example: TrainingExample = SCRIPT.run(row(entity_spans=Table))

    assert Example.entities[0].mentions == ["Taliban militants"]


def test_a_span_whose_slice_does_not_match_its_text_is_dropped() -> None:
    """the slice check is the corruption guard: a shifted offset pair (e.g. after upstream text
    normalization) must never become a fabricated mention"""
    Text: str = "Taliban militants"
    Start: int = offsets(Text)[0]
    Table: dict[str, Any] = {
        "text": [Text],
        "start": [Start],
        "end": [Start + len(Text) + 1],  # slice carries one extra char, so it can never equal text
    }
    Example: TrainingExample = SCRIPT.run(row(entity_spans=Table))

    assert Example.entities == []


def test_bool_typed_and_non_int_offsets_are_rejected() -> None:
    """bool is an int subclass in python; a True offset is schema corruption, not span 1, and
    mirrors the bool rejection in ScriptUtils.mention_spans"""
    Text: str = "Taliban militants"
    Table: dict[str, Any] = {
        "text": [Text, Text, Text, Text],
        "start": [True, offsets(Text)[0], "0", offsets(Text)[0]],
        "end": [offsets(Text)[1], False, offsets(Text)[1], 3.5],
    }
    Example: TrainingExample = SCRIPT.run(row(entity_spans=Table))

    assert Example.entities == []


def test_empty_answers_produce_no_classification_but_entities_still_ship() -> None:
    """test-split rows have empty answers; the entity half must still train, and the pipeline's
    declared-outputs filter decides what the entity-less alternative would have dropped"""
    Example: TrainingExample = SCRIPT.run(row(answers=[]))

    assert [entity.label for entity in Example.entities] == ["NamedThing"]
    assert Example.classifications == []


def test_answers_outside_the_surviving_label_set_are_filtered() -> None:
    """answers are candidate surfaces and must land in labels; case-mismatched stragglers drop
    rather than corrupting the label set (train gold only on surfaces the model can actually emit)"""
    Example: TrainingExample = SCRIPT.run(row(answers=["taliban militants", "Nuristan"]))

    assert Example.classifications == []


def test_a_non_list_answers_value_yields_no_classification() -> None:
    """hub-side schema drift can deliver answers as a bare string; without the isinstance(list)
    guard the true-label comprehension would iterate its characters and corrupt the cloze task,
    so the guard must turn drift into no classification instead of crashing the stream worker"""
    Example: TrainingExample = SCRIPT.run(row(answers="Taliban militants"))

    assert Example.classifications == []
    assert [entity.label for entity in Example.entities] == ["NamedThing"]


def test_a_non_string_answer_element_is_filtered_from_true_label() -> None:
    """a drifted non-string element inside the answers list must be filtered by the
    isinstance(str) guard; a dropped guard would flow it past the true_label: list[str]
    contract into Classification.true_label"""
    Example: TrainingExample = SCRIPT.run(row(answers=[42]))

    assert Example.classifications == []
    assert [entity.label for entity in Example.entities] == ["NamedThing"]


def test_a_non_string_query_ships_the_classification_with_prompt_none() -> None:
    """a drifted non-string query must not crash the row: the classification still ships (gold
    answers and labels are intact) and the isinstance(str) guard degrades prompt to None instead
    of silently passing a non-str instruction downstream"""
    Example: TrainingExample = SCRIPT.run(row(query=42))

    assert len(Example.classifications) == 1
    assert Example.classifications[0].prompt is None
    assert Example.classifications[0].true_label == ["Taliban militants"]


def test_a_multi_answer_row_marks_the_classification_multi_label() -> None:
    """ReCoRD allows several valid fills for one cloze; multi_label must reflect that so the
    trainer scores every gold fill instead of demanding exactly one"""
    Example: TrainingExample = SCRIPT.run(row(answers=["Taliban militants", "Afghan official"]))

    assert Example.classifications == [
        Classification(
            task="cloze entity resolution",
            labels=["Taliban militants", "Afghan official"],
            true_label=["Taliban militants", "Afghan official"],
            multi_label=True,
            prompt=QUERY,
        )
    ]


def test_the_classification_carries_the_query_as_its_prompt() -> None:
    """gliner2 consumes prompt as the cloze instruction; losing it would reduce the task to an
    unlabeled span-choice question"""
    Example: TrainingExample = SCRIPT.run(row())

    assert Example.classifications[0].prompt == QUERY
    assert Example.classifications[0].task == "cloze entity resolution"
    assert Example.classifications[0].true_label == ["Taliban militants"]


def test_the_entities_column_is_never_a_span_source() -> None:
    """candidate strings without a verified slice (Nuristan) are cloze distractors, not gold
    mentions; only span-table surfaces may become mentions"""
    Example: TrainingExample = SCRIPT.run(row())

    assert "Nuristan" not in str(Example.entities)


def test_an_empty_passage_produces_an_empty_example() -> None:
    """an empty passage cannot host slices and would break downstream tokenization; ship the
    canonical empty example and let the declared-outputs filter drop it"""
    Example: TrainingExample = SCRIPT.run(row(passage="", entity_spans=spans(), answers=[]))

    assert Example.text == ""
    assert Example.entities == []
    assert Example.classifications == []
    assert Example.populated() == frozenset()


def test_dispatch_routes_through_the_registry_and_keeps_declared_outputs() -> None:
    """US-002's ingest invokes Script.dispatch with the declared outputs tuple; the script must
    answer through that path, not only via a direct instance call"""
    Outputs, Example = Script.dispatch("SuperGlueRecordScript", (("entities", "classifications"), row()))

    assert Outputs == ("entities", "classifications")
    assert isinstance(Example, TrainingExample)
    assert [entity.label for entity in Example.entities] == ["NamedThing"]
    assert Example.classifications[0].task == "cloze entity resolution"


def test_a_realistic_record_row_yields_both_declared_shapes() -> None:
    """end-to-end shape check on a real-world news passage: one NamedThing entity group plus one
    cloze classification, exactly the two shapes US-002 declares in ingests.yaml"""
    Example: TrainingExample = SCRIPT.run(row())

    assert Example.text == PASSAGE
    assert Example.entities == [
        Entity(
            label="NamedThing",
            mentions=["Taliban militants", "Afghan official"],
            description=ScriptUtils.biolink_category_description("NamedThing"),
        )
    ]
    assert Example.populated() == frozenset({"entities", "classifications"})


def test_the_existing_registry_is_unaffected_by_the_new_script() -> None:
    """all edits are additive: the CtkpInterventionsScript entry must survive the new import"""
    assert isinstance(Script.REGISTRY["CtkpInterventionsScript"], CtkpInterventionsScript)
