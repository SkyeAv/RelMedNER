from __future__ import annotations

from typing import Any

from relmedner.models import TrainingExample
from relmedner.scripts import PubmedQaScript
from relmedner.types import Script

SCRIPT: PubmedQaScript = PubmedQaScript()

# real row 0 of the bigbio/pubmed_qa pubmed_qa_labeled_fold0_source train parquet
# (refs/convert/parquet), copied verbatim from the measured probe output (laptop 2026-09-24);
# the negative tests mutate this, never invented rows
ROW: dict[str, Any] = {
    "QUESTION": "Does histologic chorioamnionitis correspond to clinical chorioamnionitis?",
    "CONTEXTS": [
        "To evaluate the degree to which histologic chorioamnionitis, a frequent finding in placentas "
        "submitted for histopathologic evaluation, correlates with clinical indicators of infection in "
        "the mother.",
        "A retrospective review was performed on 52 cases with a histologic diagnosis of acute "
        "chorioamnionitis from 2,051 deliveries at University Hospital, Newark, from January 2003 to "
        "July 2003. Third-trimester placentas without histologic chorioamnionitis (n = 52) served as "
        "controls. Cases and controls were selected sequentially. Maternal medical records were reviewed "
        "for indicators of maternal infection.",
        "Histologic chorioamnionitis was significantly associated with the usage of antibiotics (p = "
        "0.0095) and a higher mean white blood cell count (p = 0.018). The presence of 1 or more "
        "clinical indicators was significantly associated with the presence of histologic "
        "chorioamnionitis (p = 0.019).",
    ],
    "final_decision": "yes",
}


def run_row(row: dict[str, Any] = ROW) -> TrainingExample:
    return SCRIPT.run((row["QUESTION"], row["CONTEXTS"], row["final_decision"]))


def test_run_emits_one_classification_with_the_full_vocabulary() -> None:
    example = run_row()
    assert example.text.startswith("Does histologic chorioamnionitis")
    # question first, then every context sentence
    for context in ROW["CONTEXTS"]:
        assert context in example.text
    assert len(example.classifications) == 1
    classification = example.classifications[0]
    assert classification.task == "biomedical question answering"
    # one consistent vocabulary across the gold folds and the silver artificial split
    assert classification.labels == ["no", "yes", "maybe"]
    assert classification.true_label == ["yes"]
    assert classification.multi_label is False


def test_text_composition_question_then_contexts() -> None:
    example = run_row()
    q_end = example.text.index(ROW["CONTEXTS"][0])
    assert example.text[:q_end].strip() == ROW["QUESTION"]


def test_empty_question_and_contexts_ship_empty() -> None:
    example = SCRIPT.run(("", [], "yes"))
    assert example.text == ""
    assert example.classifications == []
    example = SCRIPT.run((None, None, "yes"))
    assert example.text == ""
    assert example.classifications == []


def test_non_string_contexts_drop_out_of_the_join() -> None:
    example = SCRIPT.run(("Q?", ["real context", 42, None, True], "no"))
    assert example.text == "Q? real context"
    assert example.classifications[0].true_label == ["no"]


def test_unmappable_labels_ship_text_only() -> None:
    for bad in ("maybe later", "", 1, 1.5, True, False, None):
        example = SCRIPT.run((ROW["QUESTION"], ROW["CONTEXTS"], bad))
        assert example.text
        assert example.classifications == []


def test_label_case_is_normalized() -> None:
    example = SCRIPT.run(("Q?", ["C."], "YES"))
    assert example.classifications[0].true_label == ["yes"]


def test_the_script_self_registers_under_its_declared_name() -> None:
    """the yaml task.name resolves through Script.REGISTRY, which __init_subclass__ populates via
    Script.dispatch, which resolves by this exact NAME key"""
    assert isinstance(Script.REGISTRY["PubmedQaScript"], PubmedQaScript)
