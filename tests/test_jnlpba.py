from __future__ import annotations

from typing import Any

import pytest

from relmedner.families import validate_label_map
from relmedner.models import TrainingExample
from relmedner.scripts import JnlpbaScript, PileNerBiomedScript
from relmedner.types import Script
from relmedner.utils import ScriptUtils

SCRIPT: JnlpbaScript = JnlpbaScript()

ROW_IL2: dict[str, Any] = {
    "id": "1",
    "tokens": [
        "IL-2",
        "gene",
        "expression",
        "and",
        "NF-kappa",
        "B",
        "activation",
        "through",
        "CD28",
        "requires",
        "reactive",
        "oxygen",
        "production",
        "by",
        "5-lipoxygenase",
        ".",
    ],
    "ner_tags": [7, 8, 0, 0, 15, 16, 0, 0, 15, 0, 0, 0, 0, 0, 15, 0],
}
ROW_SECOND: dict[str, Any] = {
    "id": "2",
    "tokens": [
        "Activation",
        "of",
        "the",
        "CD28",
        "surface",
        "receptor",
        "provides",
        "a",
        "major",
        "costimulatory",
        "signal",
        "for",
        "T",
        "cell",
        "activation",
        "resulting",
        "in",
        "enhanced",
        "production",
        "of",
        "interleukin-2",
        "(",
        "IL-2",
        ")",
        "and",
        "cell",
        "proliferation",
        ".",
    ],
    "ner_tags": [0, 0, 0, 15, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 15, 0, 0, 0, 0, 0],
}


def row_il2(**overrides: Any) -> tuple[str, list[str], list[str]]:
    """(row with field overrides) -> the (tokens, ner_tags) values tuple run() unpacks; the
    override kwargs keep every negative test one mutation away from a real gold row"""
    tokens = overrides.get("tokens", ROW_IL2["tokens"])
    tags = overrides.get("ner_tags", ROW_IL2["ner_tags"])
    return tokens, tags


def surfaces_of(example: TrainingExample) -> list[str]:
    """every emitted mention surface, what the containment and count assertions read"""
    return [mention for entity in example.entities for mention in entity.mentions]


# ---------------------------------------------------------------------------
# import-time guards
# ---------------------------------------------------------------------------


def test_the_script_self_registers_under_its_declared_name() -> None:
    """the yaml task.name resolves through Script.REGISTRY, which __init_subclass__ populates via
    Script.dispatch, which resolves by this exact NAME key"""
    assert isinstance(Script.REGISTRY["JnlpbaScript"], JnlpbaScript)
    assert isinstance(Script.REGISTRY["PileNerBiomedScript"], PileNerBiomedScript)


def test_every_mapped_label_is_a_biolink_class() -> None:
    """the map is the contract the closed ClassLabel vocabulary measured; the import-time guard
    is re-asserted here so a drift can never ship"""
    for raw_label, category in JnlpbaScript.LABEL_MAP.items():
        assert ScriptUtils.is_biolink_category(category), f"map {raw_label!r} -> {category!r} is not a biolink class"


def test_the_import_time_label_guard_rejects_a_non_biolink_class() -> None:
    """a drifted LABEL_MAP value must fail at import rather than mislabel training data"""
    with pytest.raises(ValueError, match="not a biolink class"):
        validate_label_map({"gene": "NotACategory"}, "JnlpbaScript")


# ---------------------------------------------------------------------------
# tag-decoding drop rules (the parquet ints -> IOB strings layer)
# ---------------------------------------------------------------------------


def test_an_out_of_vocabulary_tag_index_decodes_as_background() -> None:
    """an int outside the fixed ClassLabel vocabulary is not a span and never a guess: it decodes
    as O so the IOB decoder treats it as background; the orphaned I- sibling still promotes to a
    single-token span, so the row still ships its remaining gold"""
    tags: list[Any] = list(ROW_IL2["ner_tags"])
    tags[0] = 99
    example: TrainingExample = SCRIPT.run(row_il2(ner_tags=tags))
    surfaces: list[str] = surfaces_of(example)
    assert "IL-2" not in surfaces
    assert "gene" in surfaces


def test_bool_tags_never_index_the_vocabulary_and_string_tags_round_trip() -> None:
    """bool is an int subclass and must not index the vocabulary (every True would decode as
    B-GENE); an all-bool row therefore ships text with zero spans, while a str-int row decodes
    through the same table and matches the int row's surfaces exactly (future schema revision)"""
    tags_bool: list[Any] = [True] * len(ROW_IL2["ner_tags"])
    example_bool: TrainingExample = SCRIPT.run(row_il2(ner_tags=tags_bool))
    assert surfaces_of(example_bool) == []
    assert example_bool.text == ScriptUtils.join_tokens(ROW_IL2["tokens"])
    tags_str: list[Any] = [str(tag) for tag in ROW_IL2["ner_tags"]]
    example_str: TrainingExample = SCRIPT.run(row_il2(ner_tags=tags_str))
    assert surfaces_of(example_str) == surfaces_of(SCRIPT.run(row_il2()))


def test_a_ragged_tokens_tags_pair_ships_text_with_zero_spans() -> None:
    """a dropped token must never shift every later offset: a ragged pair yields the rejoined text
    with zero entities rather than misaligned spans"""
    example: TrainingExample = SCRIPT.run(row_il2(tokens=ROW_IL2["tokens"][:-1]))
    assert example.text == ScriptUtils.join_tokens(ROW_IL2["tokens"][:-1])
    assert example.entities == []
    assert example.relations == []


def test_an_empty_token_list_ships_an_empty_example() -> None:
    """an all-empty row degrades to the empty TrainingExample the declared-outputs filter drops,
    never an exception and never a fabricated span"""
    example: TrainingExample = SCRIPT.run(([], []))
    assert example.text == ""
    assert example.entities == []


def test_non_list_columns_ship_text_with_zero_spans() -> None:
    """a malformed column (None, a quoted scalar) yields [] via the skip-don't-coerce helpers and
    the row collapses to its text"""
    example: TrainingExample = SCRIPT.run((None, None))
    assert example.text == ""
    assert example.entities == []


# ---------------------------------------------------------------------------
# the real gold row (shape the negative tests mutate)
# ---------------------------------------------------------------------------


def test_the_gold_row_decodes_all_four_gold_spans() -> None:
    """tags 7,8 (B-DNA/I-DNA), 15,16 (B-PROTEIN/I-PROTEIN), 15 (B-PROTEIN), 15 (B-PROTEIN) on the
    real row decode to four mentions under exactly three label-map categories; the mapped labels
    never surface as raw PascalCase tails"""
    example: TrainingExample = SCRIPT.run(row_il2())
    # distinct surfaces: a fullmap multi-class fan-out lists one surface under several label
    # groups (docs/secondary-labels.md), so the decode contract counts unique surfaces
    assert sorted(set(surfaces_of(example))) == ["5-lipoxygenase", "CD28", "IL-2 gene", "NF-kappa B"]
    categories: set[str] = {entity.label for entity in example.entities}
    # "IL-2 gene" also carries its measured tail-rule secondary label
    # (SECONDARY_TAIL_LABELS["gene"] -> GeneMention, docs/secondary-labels.md), which is a
    # morphological class, not a raw PascalCase tail of the corpus vocabulary
    assert categories == {"Gene", "GeneMention", "Protein"}
    assert example.text == ScriptUtils.join_tokens(ROW_IL2["tokens"])


def test_the_second_gold_row_keeps_multi_token_io_spans_together() -> None:
    """the second real row's 15,16,16 run must emit one joined mention surface, not three
    single-token ones (the IOB I-extension path)"""
    example: TrainingExample = SCRIPT.run((ROW_SECOND["tokens"], ROW_SECOND["ner_tags"]))
    surfaces: list[str] = surfaces_of(example)
    assert surfaces, "the second real row must still emit at least one mention"
    assert "CD28 surface receptor" in surfaces
