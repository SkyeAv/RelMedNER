from __future__ import annotations

from typing import Any

from relmedner.models import Classification, TrainingExample
from relmedner.scripts import SuperGlueMultiRCScript
from relmedner.types import Script

SCRIPT: SuperGlueMultiRCScript = SuperGlueMultiRCScript()


def row(
    paragraph: Any = "the patient took aspirin daily",
    question: Any = "what did the patient take?",
    answer: Any = "aspirin",
    label: Any = 1,
) -> tuple[Any, ...]:
    """one MultiRC row in the columns_out order: paragraph, question, answer, label"""
    return (paragraph, question, answer, label)


def test_the_script_self_registers_under_its_declared_name() -> None:
    assert isinstance(Script.REGISTRY["SuperGlueMultiRCScript"], SuperGlueMultiRCScript)


def test_the_text_joins_paragraph_question_and_answer_exactly() -> None:
    Example: TrainingExample = SCRIPT.run(row())

    assert Example.text == "the patient took aspirin daily Question: what did the patient take? Answer: aspirin"


def test_the_text_contains_the_paragraph_the_question_and_the_answer() -> None:
    Example: TrainingExample = SCRIPT.run(row())

    assert "the patient took aspirin daily" in Example.text
    assert "what did the patient take?" in Example.text
    assert "aspirin" in Example.text


def test_an_int_label_of_one_maps_to_true() -> None:
    Example: TrainingExample = SCRIPT.run(row(label=1))

    assert Example.classifications == [Classification(task="answer verification", labels=["False", "True"], true_label=["True"], multi_label=False)]


def test_an_int_label_of_zero_maps_to_false() -> None:
    Example: TrainingExample = SCRIPT.run(row(label=0))

    assert Example.classifications == [Classification(task="answer verification", labels=["False", "True"], true_label=["False"], multi_label=False)]


def test_the_classlabel_string_forms_map_like_their_ints() -> None:
    """streaming may decode the ClassLabel as its string form; neither decode form is assumed"""
    Verified: TrainingExample = SCRIPT.run(row(label="True"))
    Rejected: TrainingExample = SCRIPT.run(row(label="False"))

    assert Verified.classifications[0].true_label == ["True"]
    assert Rejected.classifications[0].true_label == ["False"]


def test_a_negative_label_ships_text_only() -> None:
    """-1 marks unlabeled rows in MultiRC; the row degrades to text, never to a guessed label"""
    Example: TrainingExample = SCRIPT.run(row(label=-1))

    assert Example.classifications == []
    assert Example.populated() == frozenset()


def test_a_none_label_ships_text_only() -> None:
    Example: TrainingExample = SCRIPT.run(row(label=None))

    assert Example.classifications == []


def test_bool_labels_never_masquerade_as_ints() -> None:
    """isinstance(True, int) is True in Python, so bools are guarded before the int check"""
    Verified: TrainingExample = SCRIPT.run(row(label=True))
    Rejected: TrainingExample = SCRIPT.run(row(label=False))

    assert Verified.classifications == []
    assert Rejected.classifications == []


def test_a_float_label_ships_text_only() -> None:
    Example: TrainingExample = SCRIPT.run(row(label=1.0))

    assert Example.classifications == []


def test_an_out_of_range_int_label_ships_text_only() -> None:
    Example: TrainingExample = SCRIPT.run(row(label=2))

    assert Example.classifications == []


def test_an_unknown_string_label_ships_text_only() -> None:
    """only the exact ClassLabel strings map; free-text verdicts are never coerced into labels"""
    Example: TrainingExample = SCRIPT.run(row(label="correct"))

    assert Example.classifications == []


def test_none_fields_coerce_to_empty_strings_and_composition_still_holds() -> None:
    """a missing column surfaces as None through the projection; the composition must survive it"""
    Example: TrainingExample = SCRIPT.run(row(paragraph=None, answer=None))

    assert Example.text == " Question: what did the patient take? Answer: "


def test_an_all_empty_row_produces_an_empty_classification_free_example() -> None:
    """no text means nothing to teach, so even a mapped label on an empty row is dropped"""
    Example: TrainingExample = SCRIPT.run(row(paragraph="", question="", answer=""))

    assert Example.text == ""
    assert Example.classifications == []
    assert Example.populated() == frozenset()


def test_a_none_paragraph_with_a_label_still_classifies_the_remaining_text() -> None:
    Example: TrainingExample = SCRIPT.run(row(paragraph=None, label=0))

    assert Example.text == " Question: what did the patient take? Answer: aspirin"
    assert Example.classifications[0].true_label == ["False"]


def test_whitespace_inside_fields_passes_through_verbatim() -> None:
    """the passage keeps its own newlines and tabs; only the ends are stripped by the coercion"""
    Paragraph: str = "line one\nline two\twith a tab\n"
    Example: TrainingExample = SCRIPT.run(row(paragraph=Paragraph))

    assert Example.text == "line one\nline two\twith a tab Question: what did the patient take? Answer: aspirin"


def test_dispatch_routes_through_the_registry_and_keeps_declared_outputs() -> None:
    Outputs, Example = Script.dispatch("SuperGlueMultiRCScript", (("classifications",), row(label=1)))

    assert Outputs == ("classifications",)
    assert Example.classifications[0].true_label == ["True"]
