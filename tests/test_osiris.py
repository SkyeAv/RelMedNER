from __future__ import annotations

from typing import Any

import pytest

from relmedner.families import validate_label_map
from relmedner.models import TrainingExample
from relmedner.scripts import OsirisScript
from relmedner.types import Script
from relmedner.utils import ScriptUtils

SCRIPT: OsirisScript = OsirisScript()

# real row 95 (the shortest document, 494 chars, 3 gold mentions: 1 gene + 2 variants) of the
# bigbio/osiris osiris_bigbio_kb train parquet (refs/convert/parquet), copied verbatim from the
# measured probe output (laptop 2026-09-24); the negative tests mutate this, never invented rows
ROW: dict[str, Any] = {
    "id": "95",
    "document_id": "15138193",
    "passages": [
        {
            "id": "9a7fc193-e296-4778-baf1-cc37922b3f1f",
            "type": "title",
            "text": ["Toll-like receptor 2 Arg677Trp  polymorphism is associated with susceptibility to tuberculosis in Tunisian patients."],
            "offsets": [[0, 116]],
        },
        {
            "id": "e82af0c8-a9cb-46df-9a6f-7f3a2c90bd9e",
            "type": "abstract",
            "text": [
                "Toll-like receptor 2 (TLR2)  is critical in the immune response to mycobacteria. Herein, "
                "we report that the frequency of a human TLR2  Arg677Trp  polymorphism  (C2029T nucleotide "
                "substitution)  in tuberculosis patients in Tunisia is significantly higher than in healthy "
                "controls (P < 0.0001). This finding suggests that this polymorphism could be a risk factor "
                "for tuberculosis."
            ],
            "offsets": [[117, 495]],
        },
    ],
    "entities": [
        {
            "id": "95e51448-d8a5-4a3b-94b9-0520cd37e3d3",
            "type": "gene",
            "text": ["Toll-like receptor 2 (TLR2)"],
            "offsets": [[117, 144]],
            "normalized": [{"db_name": "NCBI Gene", "db_id": "7097"}],
        },
        {
            "id": "3d7f9ea0-1b39-4dda-b987-c5b71f6321e7",
            "type": "variant",
            "text": ["Arg677Trp"],
            "offsets": [[21, 30]],
            "normalized": [{"db_name": "HGVS-like", "db_id": "R677W"}],
        },
        {
            "id": "bd81d045-0fc7-45e0-9d79-6c4205983c8a",
            "type": "variant",
            "text": ["(C2029T nucleotide substitution)"],
            "offsets": [[277, 309]],
            "normalized": [{"db_name": "HGVS-like", "db_id": "C2029T"}],
        },
    ],
}


def run_row(row: dict[str, Any] = ROW) -> TrainingExample:
    return SCRIPT.run((row["passages"], row["entities"]))


def test_label_map_targets_are_biolink_categories() -> None:
    validate_label_map(OsirisScript.LABEL_MAP, OsirisScript.NAME)
    for category in OsirisScript.LABEL_MAP.values():
        assert ScriptUtils.is_biolink_category(category)


def test_label_map_covers_the_measured_vocabulary() -> None:
    # full-split census (laptop 2026-09-24): the corpus carries exactly `gene` + `variant`
    assert set(OsirisScript.LABEL_MAP) == {"gene", "variant"}


def test_passage_text_places_passages_at_their_offsets() -> None:
    text = OsirisScript.passage_text(ROW["passages"])
    assert len(text) == 495
    # the inter-passage gap (1 char, measured on 105/105 rows) is gap-filled, and both
    # title-relative and abstract-relative entity offsets stay addressable
    assert text[21:30] == "Arg677Trp"
    assert text[117:144] == "Toll-like receptor 2 (TLR2)"


def test_passage_text_skips_malformed_passages() -> None:
    assert OsirisScript.passage_text([]) == ""
    assert OsirisScript.passage_text([None, {"text": ["x"]}]) == ""


def test_run_emits_the_gold_spans_under_their_biolink_classes() -> None:
    example = run_row()
    assert example.text
    surfaces = [(mention, entity.label) for entity in example.entities for mention in entity.mentions]
    # every emitted surface is a substring of the emitted text (token re-join guarantee)
    for mention, _label in surfaces:
        assert mention in example.text
    labels = {label for _mention, label in surfaces}
    assert labels <= {"Gene", "SequenceVariant"}
    assert "Arg677Trp" in [mention for mention, _label in surfaces]


def test_run_is_trust_gold_no_fullmap_remap() -> None:
    # the gold `variant` label ships as SequenceVariant verbatim; surfaces like "patients"
    # never enter because the corpus does not annotate them, but the trust-gold stance is
    # structural: every emitted label is exactly a LABEL_MAP value, never a fullmap remapping
    example = run_row()
    for entity in example.entities:
        assert entity.label in set(OsirisScript.LABEL_MAP.values())


def test_char_spans_skips_malformed_entities() -> None:
    spans = OsirisScript.char_spans(
        [
            None,
            {"text": "not-a-list", "offsets": [[0, 5]], "type": "gene"},
            {"text": ["x"], "offsets": [[0]], "type": "gene"},
            {"text": ["x", "y"], "offsets": [[0, 1]], "type": "gene"},  # length mismatch drops
            {"text": [True], "offsets": [[0, 4]], "type": "gene"},
            {"text": ["ok"], "offsets": [[1, 3]], "type": "variant"},
        ]
    )
    assert spans == [(1, 3, "variant")]


def test_run_malformed_entities_yield_text_only_or_empty() -> None:
    example = SCRIPT.run((ROW["passages"], [None, {"type": "gene"}]))
    assert example.text
    assert example.entities == []
    empty = SCRIPT.run(([], ROW["entities"]))
    assert empty.text == ""
    assert empty.entities == []


def test_the_script_self_registers_under_its_declared_name() -> None:
    """the yaml task.name resolves through Script.REGISTRY, which __init_subclass__ populates via
    Script.dispatch, which resolves by this exact NAME key"""
    assert isinstance(Script.REGISTRY["OsirisScript"], OsirisScript)


@pytest.mark.parametrize("bad_type", ["protein", "disease", ""])
def test_unmapped_entity_types_drop_not_coerce(bad_type: str) -> None:
    entity = {"text": ["Arg677Trp"], "offsets": [[21, 30]], "type": bad_type}
    example = SCRIPT.run((ROW["passages"], [entity]))
    assert all(entity.label in {"Gene", "SequenceVariant"} for entity in example.entities)
    if bad_type:
        surfaces = [m for e in example.entities for m in e.mentions]
        assert "Arg677Trp" not in surfaces
