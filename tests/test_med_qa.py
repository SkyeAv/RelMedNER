from __future__ import annotations

from typing import Any

import pytest

from relmedner.models import TrainingExample
from relmedner.scripts import MedQaScript
from relmedner.types import Script

SCRIPT: MedQaScript = MedQaScript()

# real row 1 of GBaker/MedQA-USMLE-4-options phrases_no_exclude_train.jsonl, copied verbatim
# from the measured census (laptop 2026-09-24); the negative tests mutate this row, never
# invent one. `answer` and `meta_info` exist on the real row but are NOT declared in
# columns_out, so they are absent here on purpose.
ROW: dict[str, Any] = {
    "question": (
        "A 3-month-old baby died suddenly at night while asleep. His mother noticed that he had died "
        "only after she awoke in the morning. No cause of death was determined based on the autopsy. "
        "Which of the following precautions could have prevented the death of the baby?"
    ),
    "options": {
        "A": "Placing the infant in a supine position on a firm mattress while sleeping",
        "B": "Keeping the infant covered and maintaining a high room temperature",
        "C": "Application of a device to maintain the sleeping position",
        "D": "Avoiding pacifier use during sleep",
    },
    "answer_idx": "A",
}

RENDERED: list[str] = [
    "A. Placing the infant in a supine position on a firm mattress while sleeping",
    "B. Keeping the infant covered and maintaining a high room temperature",
    "C. Application of a device to maintain the sleeping position",
    "D. Avoiding pacifier use during sleep",
]


def run_row(row: dict[str, Any] = ROW) -> TrainingExample:
    return SCRIPT.run((row["question"], row["options"], row["answer_idx"]))


def test_run_emits_one_classification_with_the_full_option_vocabulary() -> None:
    """the gold label is one of the four option letters over a FIXED vocabulary: a per-row
    vocabulary would make the classification task unlearnable across rows"""
    example = run_row()
    assert len(example.classifications) == 1
    classification = example.classifications[0]
    assert classification.task == "medical multiple-choice question answering"
    assert classification.labels == ["A", "B", "C", "D"]
    assert classification.true_label == ["A"]
    assert classification.multi_label is False


def test_text_is_the_question_then_the_options_in_fixed_order() -> None:
    """rendering the options is what makes the letter label learnable; the order must be the
    vocabulary order, not dict iteration order, or the label stops meaning a position"""
    example = run_row()
    assert example.text.startswith(ROW["question"])
    positions = [example.text.index(option) for option in RENDERED]
    assert positions == sorted(positions)
    for option in RENDERED:
        assert option in example.text


def test_every_option_appears_exactly_once_so_the_label_is_not_leaked() -> None:
    """the correct option text sits in the prompt like every distractor: once. A duplicated or
    appended answer string would hand the model the label instead of the task"""
    example = run_row()
    for letter, text in ROW["options"].items():
        assert example.text.count(f"{letter}. {text}") == 1


def test_unmappable_answer_idx_ships_text_only() -> None:
    """skip-don't-coerce: a label outside A-D is never guessed into the vocabulary. bools are
    listed because isinstance(True, int) is True in Python, so an unguarded bool would otherwise
    masquerade as an index"""
    for bad in ("E", "", "a1", 1, 1.5, True, False, None, ["A"]):
        example = SCRIPT.run((ROW["question"], ROW["options"], bad))
        assert example.text
        assert example.classifications == []


def test_answer_idx_case_and_padding_are_normalized() -> None:
    """normalization is not a guess: ' a ' and 'A' name the same option letter"""
    for good in ("a", " A ", "A"):
        example = SCRIPT.run((ROW["question"], ROW["options"], good))
        assert example.classifications[0].true_label == ["A"]


@pytest.mark.parametrize(
    "options",
    [
        None,
        "A. x B. y",
        ["x", "y", "z", "w"],
        {"A": "x", "B": "y", "C": "z"},
        {"A": "x", "B": "y", "C": "z", "D": "w", "E": "v"},
        {"A": "x", "B": "y", "C": "z", "D": ""},
        {"A": "x", "B": "y", "C": "z", "D": "   "},
        {"A": "x", "B": "y", "C": None, "D": "w"},
        {"A": "x", "B": 7, "C": "z", "D": "w"},
    ],
)
def test_an_unusable_option_set_ships_text_only(options: Any) -> None:
    """without all four rendered options the letter label is unlearnable, and inventing option
    text would be a guess; the example still ships its question text. The five-key case is in
    the list because a superset of the vocabulary means the declared letter mapping no longer
    describes the row"""
    example = SCRIPT.run((ROW["question"], options, "A"))
    assert example.text == ROW["question"]
    assert example.classifications == []


def test_empty_question_and_options_ship_empty() -> None:
    """the pipeline's empty-record path drops TrainingExample(text=""); shipping a stub instead
    would silently train on blank rows"""
    for values in (("", {}, "A"), (None, None, None), ("   ", {"A": " ", "B": "", "C": "", "D": ""}, "B")):
        example = SCRIPT.run(values)
        assert example.text == ""
        assert example.classifications == []


def test_empty_question_with_usable_options_ships_the_options_alone() -> None:
    """a missing stem is still a learnable four-option item; fabricating a stem would be a guess"""
    example = SCRIPT.run(("", ROW["options"], "C"))
    assert example.text == " ".join(RENDERED)
    assert example.classifications[0].true_label == ["C"]


def test_a_batch_of_real_rows_yields_one_labeled_example_each() -> None:
    """nonzero yield over the declared split's real shape: rows 0-4 verbatim, including the
    unicode (degree signs, curly quotes) the corpus actually carries"""
    rows: list[tuple[Any, ...]] = [
        (
            "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. "
            "Which of the following is the best treatment for this patient?",
            {"A": "Ampicillin", "B": "Ceftriaxone", "C": "Doxycycline", "D": "Nitrofurantoin"},
            "D",
        ),
        (ROW["question"], ROW["options"], "A"),
        (
            "A mother brings her 3-week-old infant to the pediatrician's office. "
            "Which of the following embryologic errors could account for this presentation?",
            {
                "A": "Abnormal migration of ventral pancreatic bud",
                "B": "Complete failure of proximal duodenum to recanalize",
                "C": "Abnormal hypertrophy of the pylorus",
                "D": "Failure of lateral body folds to move ventrally and fuse in the midline",
            },
            "A",
        ),
        (
            "A pulmonary autopsy specimen from a 58-year-old woman who died of acute hypoxic respiratory "
            "failure was examined. Which of the following is the most likely pathogenesis?",
            {
                "A": "Thromboembolism",
                "B": "Pulmonary ischemia",
                "C": "Pulmonary hypertension",
                "D": "Pulmonary passive congestion",
            },
            "A",
        ),
        (
            "A 20-year-old woman presents with menorrhagia for the past several years. "
            "Which of the following is the most likely cause of this patient\u2019s symptoms?",
            {"A": "Hemophilia A", "B": "Lupus anticoagulant", "C": "Protein C deficiency", "D": "Von Willebrand disease"},
            "D",
        ),
    ]
    examples = [SCRIPT.run(values) for values in rows]
    assert len(examples) == 5
    assert all(len(example.classifications) == 1 for example in examples)
    assert [example.classifications[0].true_label for example in examples] == [["D"], ["A"], ["A"], ["A"], ["D"]]
    assert all(example.text for example in examples)


def test_the_script_reads_exactly_three_declared_columns() -> None:
    """columns_out is [question, options, answer_idx]: the row's `answer` (the correct option
    text) must never reach the script, or the label could leak into the example text. A fourth
    value is a declaration drift bug, so it fails loudly instead of being ignored"""
    with pytest.raises(ValueError):
        SCRIPT.run((ROW["question"], ROW["options"], ROW["answer_idx"], "Placing the infant"))


def test_the_script_self_registers_under_its_declared_name() -> None:
    """the yaml task.name resolves through Script.REGISTRY, which __init_subclass__ populates via
    Script.dispatch, which resolves by this exact NAME key"""
    assert isinstance(Script.REGISTRY["MedQaScript"], MedQaScript)
