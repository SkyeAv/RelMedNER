from __future__ import annotations

from typing import Any

import pytest

from relmedner.families import validate_label_map
from relmedner.models import TrainingExample
from relmedner.scripts import JnlpbaScript, NcbiDiseaseScript
from relmedner.types import Script
from relmedner.utils import ScriptUtils

SCRIPT: NcbiDiseaseScript = NcbiDiseaseScript()

# real rows 0-2 of the ncbi/ncbi_disease train parquet (refs/convert/parquet), copied verbatim
# from the measured probe output (wenceslaus 2026-09-24); the negative tests mutate these, never
# invented rows
ROW_APC: dict[str, Any] = {
    "id": "0",
    "tokens": [
        "Identification",
        "of",
        "APC2",
        ",",
        "a",
        "homologue",
        "of",
        "the",
        "adenomatous",
        "polyposis",
        "coli",
        "tumour",
        "suppressor",
        ".",
    ],
    "ner_tags": [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0],
}
ROW_WIDE: dict[str, Any] = {
    "id": "1",
    "tokens": [
        "The",
        "adenomatous",
        "polyposis",
        "coli",
        "(",
        "APC",
        ")",
        "tumour",
        "-",
        "suppressor",
        "protein",
        "controls",
        "the",
        "Wnt",
        "signalling",
        "pathway",
        "by",
        "forming",
        "a",
        "complex",
        "with",
        "glycogen",
        "synthase",
        "kinase",
        "3beta",
        "(",
        "GSK",
        "-",
        "3beta",
        ")",
        ",",
        "axin",
        "/",
        "conductin",
        "and",
        "betacatenin",
        ".",
    ],
    "ner_tags": [0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}
ROW_EMPTY: dict[str, Any] = {
    "id": "2",
    "tokens": ["Complex", "formation", "induces", "the", "rapid", "degradation", "of", "betacatenin", "."],
    "ner_tags": [0, 0, 0, 0, 0, 0, 0, 0, 0],
}


def row_apc(**overrides: Any) -> tuple[list[str], list[Any]]:
    """(row with field overrides) -> the (tokens, ner_tags) values tuple run() unpacks; the
    override kwargs keep every negative test one mutation away from a real gold row"""
    tokens = overrides.get("tokens", ROW_APC["tokens"])
    tags = overrides.get("ner_tags", ROW_APC["ner_tags"])
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
    assert isinstance(Script.REGISTRY["NcbiDiseaseScript"], NcbiDiseaseScript)
    assert isinstance(Script.REGISTRY["JnlpbaScript"], JnlpbaScript)


def test_every_mapped_label_is_a_biolink_class() -> None:
    """the map is the contract the closed ClassLabel vocabulary measured; the import-time guard
    is re-asserted here so a drift can never ship"""
    for raw_label, category in NcbiDiseaseScript.LABEL_MAP.items():
        assert ScriptUtils.is_biolink_category(category), f"map {raw_label!r} -> {category!r} is not a biolink class"


def test_the_import_time_label_guard_rejects_a_non_biolink_class() -> None:
    """a drifted LABEL_MAP value must fail at import rather than mislabel training data"""
    with pytest.raises(ValueError, match="not a biolink class"):
        validate_label_map({"disease": "NotACategory"}, "NcbiDiseaseScript")


# ---------------------------------------------------------------------------
# tag-decoding drop rules (the parquet ints -> IOB strings layer)
# ---------------------------------------------------------------------------


def test_an_out_of_vocabulary_tag_index_decodes_as_background() -> None:
    """an int outside the fixed 3-tag ClassLabel vocabulary is not a span and never a guess: it
    decodes as O, and the orphaned I- run promotes to a single multi-token span (the iob_spans
    contract), so the row ships the remaining gold tokens as one mention rather than fabricating
    a span over the killed B- token"""
    tags: list[Any] = list(ROW_APC["ner_tags"])
    tags[8] = 99
    example: TrainingExample = SCRIPT.run(row_apc(ner_tags=tags))
    assert surfaces_of(example) == ["polyposis coli tumour"]
    assert "adenomatous polyposis coli tumour" not in surfaces_of(example)
    assert example.text == ScriptUtils.join_tokens(ROW_APC["tokens"])


def test_bool_tags_never_index_the_vocabulary_and_string_tags_round_trip() -> None:
    """bool is an int subclass and must not index the vocabulary (every True would decode as
    B-DISEASE); an all-bool row therefore ships text with zero spans, while a str-int row decodes
    through the same table and matches the int row's surfaces exactly (future schema revision)"""
    tags_bool: list[Any] = [True] * len(ROW_APC["ner_tags"])
    example_bool: TrainingExample = SCRIPT.run(row_apc(ner_tags=tags_bool))
    assert surfaces_of(example_bool) == []
    assert example_bool.text == ScriptUtils.join_tokens(ROW_APC["tokens"])
    tags_str: list[Any] = [str(tag) for tag in ROW_APC["ner_tags"]]
    example_str: TrainingExample = SCRIPT.run(row_apc(ner_tags=tags_str))
    assert surfaces_of(example_str) == surfaces_of(SCRIPT.run(row_apc()))


def test_a_ragged_tokens_tags_pair_ships_text_with_zero_spans() -> None:
    """a dropped token must never shift every later offset: a ragged pair yields the rejoined text
    with zero entities rather than misaligned spans"""
    example: TrainingExample = SCRIPT.run(row_apc(tokens=ROW_APC["tokens"][:-1]))
    assert example.text == ScriptUtils.join_tokens(ROW_APC["tokens"][:-1])
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
# the real gold rows (shape the negative tests mutate)
# ---------------------------------------------------------------------------


def test_the_gold_row_decodes_the_single_gold_span() -> None:
    """tags 1,2,2,2 (B-DISEASE/I-DISEASE) on the real row decode to exactly one Disease mention
    spanning all four tagged tokens (the corpus annotates the full disease phrase including
    'tumour'); the mapped label never surfaces as a raw PascalCase tail"""
    example: TrainingExample = SCRIPT.run(row_apc())
    assert surfaces_of(example) == ["adenomatous polyposis coli tumour"]
    categories: set[str] = {entity.label for entity in example.entities}
    assert categories == {"Disease"}
    assert example.text == ScriptUtils.join_tokens(ROW_APC["tokens"])


def test_the_second_gold_row_keeps_the_wide_io_span_together() -> None:
    """the second real row's 1,2,2,2,2,2,2 run must emit ONE joined mention surface spanning the
    parenthesized short form AND 'tumour' (the corpus's full-mention convention, measured span
    1-7), not seven single-token ones (the IOB I-extension path)"""
    example: TrainingExample = SCRIPT.run((ROW_WIDE["tokens"], ROW_WIDE["ner_tags"]))
    surfaces: list[str] = surfaces_of(example)
    assert surfaces == ["adenomatous polyposis coli ( APC ) tumour"]
    assert "adenomatous polyposis coli ( APC ) tumour" in example.text


def test_every_emitted_surface_occurs_in_the_emitted_text() -> None:
    """gliner2's InputExample.validate() rejects a mention surface that is not a substring of the
    emitted text; assert the invariant directly over the gold rows and their joined surfaces"""
    for row in (ROW_APC, ROW_WIDE):
        example: TrainingExample = SCRIPT.run((row["tokens"], row["ner_tags"]))
        for surface in surfaces_of(example):
            assert surface in example.text


def test_a_handful_of_real_rows_yields_at_least_one_entity() -> None:
    """nonzero-yield guard against the silent-zero-yield bug the repo shipped once: the first
    three real train rows (one all-background) must together emit at least two entities across
    the two annotated rows, and the all-background row alone ships text only"""
    examples: list[TrainingExample] = [
        SCRIPT.run((ROW_APC["tokens"], ROW_APC["ner_tags"])),
        SCRIPT.run((ROW_WIDE["tokens"], ROW_WIDE["ner_tags"])),
        SCRIPT.run((ROW_EMPTY["tokens"], ROW_EMPTY["ner_tags"])),
    ]
    total_entities: int = sum(len(example.entities) for example in examples)
    assert total_entities >= 2
    assert examples[2].entities == []
    assert examples[2].text == ScriptUtils.join_tokens(ROW_EMPTY["tokens"])


def test_a_resolution_fan_out_never_crashes_the_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """the fullmap resolution chain can return MORE resolutions than spans (the multi-class
    fan-out adds a secondary class per mention), which a strict positional zip of spans with
    resolutions turned into a pipeline-crashing ValueError on real hub rows (smoke, wenceslaus
    2026-09-24). pair_spans re-pairs by span_index: every fan-out item extends its span, and
    group_entities dedups the mention surfaces."""
    import relmedner.scripts.ncbi_disease as mod

    real_resolve = ScriptUtils.resolve_mentions
    monkeypatch.setattr(
        mod.ScriptUtils,
        "resolve_mentions",
        lambda mentions, label_map=None: real_resolve(mentions, label_map=label_map) * 2 if mentions else [],
    )
    example: TrainingExample = SCRIPT.run((ROW_APC["tokens"], ROW_APC["ner_tags"]))
    assert example.entities, "the gold span must still emit under doubled resolutions"
    for surface in surfaces_of(example):
        assert surface in example.text
    labels = {entity.label for entity in example.entities}
    assert labels <= {"Disease"}
