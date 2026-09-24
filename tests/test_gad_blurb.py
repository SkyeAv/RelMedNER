from __future__ import annotations

from typing import Any

from relmedner.models import Classification, TrainingExample
from relmedner.scripts import GadBlurbScript
from relmedner.types import Script

SCRIPT: GadBlurbScript = GadBlurbScript()

SAMPLE_TEXT: str = (
    "this study proposes that A/A genotype at position -607 in @GENE$ gene can be used as a new "
    "genetic maker in Thai population for predicting @DISEASE$ development."
)


def row(text: Any = SAMPLE_TEXT, labels: Any = "1") -> tuple[Any, ...]:
    """one gad_blurb_bigbio_text row in the declared columns_out order: text, labels.
    Fixtures are copied from the measured probe output, not invented. The default label is the
    scalar string form (a mutable list default trips ruff B006; the decode tests pass the
    measured list shape explicitly)"""
    return (text, labels)


def classification(true_label: str) -> Classification:
    return Classification(
        task="gene-disease association",
        labels=["associated", "not associated"],
        true_label=[true_label],
        multi_label=False,
    )


def test_the_script_self_registers_under_its_declared_name() -> None:
    assert isinstance(Script.REGISTRY["GadBlurbScript"], GadBlurbScript)


def test_the_text_passes_through_unmodified_placeholders_included() -> None:
    """the corpus's @GENE$ / @DISEASE$ placeholders ship as-is (user decision 2026-09-23): the
    signal is the association judgment, and no surface rewriting may break text containment"""
    Example: TrainingExample = SCRIPT.run(row())

    assert Example.text == SAMPLE_TEXT
    assert "@GENE$" in Example.text and "@DISEASE$" in Example.text


def test_a_one_element_string_list_is_the_measured_shape_and_maps() -> None:
    """the parquet conversion carries labels as a one-element list of strings; this is the shape
    the live hub rows actually arrive in"""
    Associated: TrainingExample = SCRIPT.run(row(labels=["1"]))
    NotAssociated: TrainingExample = SCRIPT.run(row(labels=["0"]))

    assert Associated.classifications == [classification("associated")]
    assert NotAssociated.classifications == [classification("not associated")]


def test_an_int_label_maps_like_its_string_form() -> None:
    """neither decode form is assumed: the _source variant of the same corpus carries plain ints"""
    assert SCRIPT.run(row(labels=1)).classifications == [classification("associated")]
    assert SCRIPT.run(row(labels=0)).classifications == [classification("not associated")]


def test_a_string_label_maps_like_its_int_form() -> None:
    assert SCRIPT.run(row(labels="1")).classifications == [classification("associated")]
    assert SCRIPT.run(row(labels="0")).classifications == [classification("not associated")]


def test_bool_labels_never_masquerade_as_ints() -> None:
    """isinstance(True, int) is True in Python, so bools are rejected before the int check"""
    assert SCRIPT.run(row(labels=True)).classifications == []
    assert SCRIPT.run(row(labels=False)).classifications == []
    assert SCRIPT.run(row(labels=[True])).classifications == []


def test_a_none_label_ships_text_only() -> None:
    Example: TrainingExample = SCRIPT.run(row(labels=None))

    assert Example.classifications == []
    assert Example.text == SAMPLE_TEXT
    assert Example.populated() == frozenset()


def test_a_float_label_ships_text_only() -> None:
    assert SCRIPT.run(row(labels=1.0)).classifications == []


def test_an_empty_label_list_ships_text_only() -> None:
    assert SCRIPT.run(row(labels=[])).classifications == []


def test_a_multi_element_label_list_ships_text_only() -> None:
    """two labels is not a decodable judgment; guessing the first would invent data"""
    assert SCRIPT.run(row(labels=["1", "0"])).classifications == []


def test_an_out_of_vocabulary_label_ships_text_only() -> None:
    """only "0"/"1" (and 0/1) exist in the measured census; "yes"/"2" are never guessed"""
    assert SCRIPT.run(row(labels="yes")).classifications == []
    assert SCRIPT.run(row(labels=2)).classifications == []


def test_an_empty_text_row_ships_empty_and_the_output_filter_drops_it() -> None:
    for text in ("", "   ", None, 42):
        Example: TrainingExample = SCRIPT.run(row(text=text))
        assert Example.text == ""
        assert Example.populated() == frozenset()


def test_a_realistic_batch_yields_nonzero_and_exact_shapes() -> None:
    """the silent-zero-yield bug is the failure mode this repo has shipped once; a handful of
    real-shaped rows (two labeled, one text-only via a bad label) must produce two
    classifications with the exact declared vocabulary"""
    Rows: list[tuple[Any, ...]] = [
        row(),
        row(
            text=(
                "Common polymorphisms in the genes @GENE$ and LOC387715 are independently "
                "related to @DISEASE$ progression after adjustment for other known AMD risk factors."
            ),
            labels=["1"],
        ),
        row(labels=["unknown"]),
    ]
    Examples: list[TrainingExample] = [SCRIPT.run(values) for values in Rows]

    assert len(Examples) == 3
    assert [bool(example.classifications) for example in Examples] == [True, True, False]
    for example in Examples[:2]:
        assert example.classifications[0].labels == ["associated", "not associated"]
        assert example.classifications[0].true_label in (["associated"], ["not associated"])
        assert example.classifications[0].multi_label is False
        assert example.text  # every shipped example carries text
