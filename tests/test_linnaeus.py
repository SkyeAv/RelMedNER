from __future__ import annotations

from typing import Any

import pytest

from relmedner.families import validate_label_map
from relmedner.models import TrainingExample
from relmedner.scripts import JnlpbaScript, LinnaeusScript
from relmedner.types import Script
from relmedner.utils import ScriptUtils

SCRIPT: LinnaeusScript = LinnaeusScript()

# real row 94 (the shortest document, 3,857 chars, 9 gold species mentions) of the
# bigbio/linnaeus linnaeus_bigbio_kb train parquet (refs/convert/parquet), copied verbatim from
# the measured probe output (wenceslaus 2026-09-24); the negative tests mutate this, never
# invented rows
ROW: dict[str, Any] = {
    "id": "94",
    "document_id": "pmcA1621059",
    "passages": [
        {
            "id": "pmcA1621059__text",
            "type": "Article",
            "text": [
                "Case report: rapidly fatal bowel ischaemia on clozapine treatment\n"
                "Abstract\n"
                "Background\n"
                "There have been previous reported deaths due to clozapine-induced constipation. In "
                "all these cases patients have experienced prior abdominal symptoms over a period of "
                "weeks or months.\n"
                "\n"
                "Case presentation\n"
                "We report the sudden death due to constipation of a healthy young male patient on "
                "clozapine without any known history of prior abdominal symptoms.\n"
                "\n"
                "Conclusion\n"
                "Psychiatrists need to be alert to the medical emergencies which can occur in the "
                "context of clozapine treatment and also need to make other clinicians who may have "
                "contact with their patients aware of these.\n"
                "\n"
                "\n"
                "\n"
                "Background\n"
                "There have been six previously published cases of death secondary to "
                "clozapine-induced constipation [1-3]. Of these, two patients died from faecal "
                "peritonitis, two from aspiration of faeculent vomitus as a result of bowel "
                "obstruction and two from bowel necrosis. In all these cases there had been prior "
                "complaints of constipation and/or other abdominal symptoms for weeks to months "
                "before the fatal event. Here we describe a case of constipation, presumably "
                "clozapine-induced, where death from bowel ischaemia occured within 2 days from the "
                "first complaint of constipation and without any prior reported abdominal symptoms "
                "which might have provided a warning to the clinicians involved.\n"
                "\n"
                "Case presentation\n"
                "A 20-year-old male with a year long history of schizophrenia which had been "
                "unresponsive to trials of two atypical antipsychotic drugs was commenced on "
                "clozapine. The dose was titrated over the next year to 900 mg daily. Due to "
                "persisting negative symptoms amisulpiride 400 mg twice daily was added with good "
                "response after one month. The patient was reviewed regularly over the next year, "
                "continued to improve and did not report any side effects to members of the "
                "multidisciplinary mental health team working to support him in the community. He "
                "appeared to be fit and healthy. Although he usually lived in supported "
                "accommodation he was staying temporarily with his family and from their account he "
                "complained of having constipation for 2 days before presenting to his GP with "
                "severe abdominal pain. He was prescribed medication and returned home but his "
                "condition deteriorated further and a few hours later an ambulance was called. He "
                "collapsed and died before reaching hospital. Post mortem examination revealed that "
                "he had impacted faeces which had pressed against the bowel wall causing ischaemia. "
                "This had led to infarction of this part of the bowel.\n"
                "\n"
                "Conclusion\n"
                "This case demonstrates that death can occur over a very short time course from "
                "constipation, in this case presumably induced by clozapine. Death from constipation "
                "and subsequent bowel infarction is relatively common in elderly patients and "
                "infarction causes a far more rapid and dangerous deterioration than does intestinal "
                "obstruction. In the present case this meant that this patient did not have any "
                "contact with psychiatric services between the onset of his symptoms and his rapid "
                "demise, in spite of regular follow-up. Although the risk of neutropenia is "
                "relatively well-known, it should be borne in mind that clozapine is reported to be "
                "associated with a number of other syndromes which may be rapidly fatal including "
                "not only constipation and obstruction but also cardiovascular collapse, seizures "
                "and ketoacidosis. Psychiatrists working with such patients should not only "
                "themselves be vigilant regarding such complications but should take steps to see "
                "that other clinicians to whom the patient may present are also aware of them.\n"
                "\n"
                "Competing interests\n"
                "The author(s) declare that they have no competing interests.\n"
                "\n"
                "Authors' contributions\n"
                "Both authors were equally involved in the preparation of this manuscript.\n"
                "\n"
                "Pre-publication history\n"
                "The pre-publication history for this paper can be accessed here:\n"
                "\n"
                "\n"
                "\n"
            ],
            "offsets": [[0, 3857]],
        }
    ],
    "entities": [
        {
            "id": "pmcA1621059__T0",
            "type": "species",
            "text": ["patients"],
            "offsets": [[185, 193]],
            "normalized": [{"db_name": "ncbi", "db_id": "9606"}],
        },
        {
            "id": "pmcA1621059__T1",
            "type": "species",
            "text": ["patient"],
            "offsets": [[360, 367]],
            "normalized": [{"db_name": "ncbi", "db_id": "9606"}],
        },
        {
            "id": "pmcA1621059__T2",
            "type": "species",
            "text": ["patients"],
            "offsets": [[631, 639]],
            "normalized": [{"db_name": "ncbi", "db_id": "9606"}],
        },
        {
            "id": "pmcA1621059__T3",
            "type": "species",
            "text": ["patients"],
            "offsets": [[791, 799]],
            "normalized": [{"db_name": "ncbi", "db_id": "9606"}],
        },
        {
            "id": "pmcA1621059__T4",
            "type": "species",
            "text": ["patient"],
            "offsets": [[1715, 1722]],
            "normalized": [{"db_name": "ncbi", "db_id": "9606"}],
        },
        {
            "id": "pmcA1621059__T5",
            "type": "species",
            "text": ["patients"],
            "offsets": [[2772, 2780]],
            "normalized": [{"db_name": "ncbi", "db_id": "9606"}],
        },
        {
            "id": "pmcA1621059__T6",
            "type": "species",
            "text": ["patient"],
            "offsets": [[2923, 2930]],
            "normalized": [{"db_name": "ncbi", "db_id": "9606"}],
        },
        {
            "id": "pmcA1621059__T7",
            "type": "species",
            "text": ["patients"],
            "offsets": [[3400, 3408]],
            "normalized": [{"db_name": "ncbi", "db_id": "9606"}],
        },
        {
            "id": "pmcA1621059__T8",
            "type": "species",
            "text": ["patient"],
            "offsets": [[3540, 3547]],
            "normalized": [{"db_name": "ncbi", "db_id": "9606"}],
        },
    ],
    "events": [],
    "coreferences": [],
    "relations": [],
}


def row(**overrides: Any) -> tuple[list[Any], list[Any]]:
    """(row with field overrides) -> the (passages, entities) values tuple run() unpacks; the
    override kwargs keep every negative test one mutation away from a real gold row"""
    passages = overrides.get("passages", ROW["passages"])
    entities = overrides.get("entities", ROW["entities"])
    return passages, entities


def surfaces_of(example: TrainingExample) -> list[str]:
    """every emitted mention surface, what the containment and count assertions read"""
    return [mention for entity in example.entities for mention in entity.mentions]


# ---------------------------------------------------------------------------
# import-time guards
# ---------------------------------------------------------------------------


def test_the_script_self_registers_under_its_declared_name() -> None:
    """the yaml task.name resolves through Script.REGISTRY, which __init_subclass__ populates via
    Script.dispatch, which resolves by this exact NAME key"""
    assert isinstance(Script.REGISTRY["LinnaeusScript"], LinnaeusScript)
    assert isinstance(Script.REGISTRY["JnlpbaScript"], JnlpbaScript)


def test_every_mapped_label_is_a_biolink_class() -> None:
    """the map is the contract the closed corpus vocabulary measured (a single `species` type
    over all 4,259 spans); the import-time guard is re-asserted here so a drift can never ship"""
    for raw_label, category in LinnaeusScript.LABEL_MAP.items():
        assert ScriptUtils.is_biolink_category(category), f"map {raw_label!r} -> {category!r} is not a biolink class"


def test_the_import_time_label_guard_rejects_a_non_biolink_class() -> None:
    """a drifted LABEL_MAP value must fail at import rather than mislabel training data"""
    with pytest.raises(ValueError, match="not a biolink class"):
        validate_label_map({"species": "NotACategory"}, "LinnaeusScript")


# ---------------------------------------------------------------------------
# drop rules (one negative test per rule, each asserting the DROPPED shape)
# ---------------------------------------------------------------------------


def first_two() -> list[Any]:
    """the real row's first two gold entities: one `patients` (T0) and one `patient` (T1); every
    drop test mutates T0 and pins T1 as the well-formed sibling that ships, because
    group_entities dedups surfaces per label and a full-row count could not see one dropped
    mention among its four duplicates"""
    return [dict(ROW["entities"][0]), dict(ROW["entities"][1])]


def test_a_malformed_entity_dict_is_skipped_and_the_sibling_ships() -> None:
    """a non-dict entity entry is not a span and never a guess: char_spans skips it, so the row
    still ships its well-formed gold sibling"""
    example: TrainingExample = SCRIPT.run(row(entities=["not-a-dict", ROW["entities"][1]]))
    assert surfaces_of(example) == ["patient"]


def test_a_text_offsets_length_mismatch_drops_the_whole_entity() -> None:
    """two text parts over one offset (or the reverse) makes the offset-to-surface alignment
    untrustworthy, so the whole entity drops instead of emitting a half-guessed span"""
    entities = first_two()
    entities[0] = {**entities[0], "text": ["patients", "extra"]}
    example: TrainingExample = SCRIPT.run(row(entities=entities))
    assert surfaces_of(example) == ["patient"]


def test_an_out_of_bounds_char_span_is_dropped_by_the_bridge() -> None:
    """a span past the document end is out of bounds for every tokenizer position: the char
    bridge drops it (skip-don't-coerce), never snapping it back into the text"""
    entities = first_two()
    entities[0] = {**entities[0], "offsets": [[3800, 3900]]}
    example: TrainingExample = SCRIPT.run(row(entities=entities))
    assert surfaces_of(example) == ["patient"]


def test_a_degenerate_zero_length_char_span_is_dropped_by_the_bridge() -> None:
    """[185, 185) selects no characters: the bridge drops the degenerate span rather than
    emitting an empty surface"""
    entities = first_two()
    entities[0] = {**entities[0], "offsets": [[185, 185]]}
    example: TrainingExample = SCRIPT.run(row(entities=entities))
    assert surfaces_of(example) == ["patient"]
    assert "" not in surfaces_of(example)


def test_an_unmapped_entity_type_drops_instead_of_guessing() -> None:
    """trust-gold keeps the corpus's gold `species` label, and a type with no honest biolink
    target in LABEL_MAP ships nothing rather than a guessed or PascalCased label
    (skip-don't-coerce)"""
    entities = first_two()
    entities[0] = {**entities[0], "type": "strain"}
    example: TrainingExample = SCRIPT.run(row(entities=entities))
    assert surfaces_of(example) == ["patient"]
    assert {entity.label for entity in example.entities} == {"OrganismTaxon"}


def test_a_non_list_passages_column_yields_the_empty_example() -> None:
    """a malformed passages column (None, a quoted scalar) yields [] via the skip-don't-coerce
    helpers and the row collapses to the empty example the declared-outputs filter drops"""
    example: TrainingExample = SCRIPT.run((None, ROW["entities"]))
    assert example.text == ""
    assert example.entities == []


# ---------------------------------------------------------------------------
# the real gold row (shape the negative tests mutate)
# ---------------------------------------------------------------------------


def test_the_gold_row_decodes_all_nine_species_mentions() -> None:
    """the nine end-exclusive char spans of the real row all decode: group_entities emits one
    OrganismTaxon entity whose deduped surfaces are the row's two gold surface forms, and every
    one of the nine gold surfaces occurs in the emitted text; the mapped label never surfaces as
    a raw PascalCase tail"""
    example: TrainingExample = SCRIPT.run(row())
    assert len(example.entities) == 1
    assert sorted(surfaces_of(example)) == ["patient", "patients"]
    categories: set[str] = {entity.label for entity in example.entities}
    assert categories == {"OrganismTaxon"}
    for entity in ROW["entities"]:
        assert entity["text"][0] in example.text


def test_every_emitted_surface_occurs_in_the_emitted_text() -> None:
    """gliner2's InputExample.validate() rejects a mention surface that is not a substring of the
    emitted text; assert the invariant directly over every emitted surface of the real row"""
    example: TrainingExample = SCRIPT.run(row())
    for surface in surfaces_of(example):
        assert surface in example.text


def test_the_real_row_yields_entities() -> None:
    """nonzero-yield guard against the silent-zero-yield bug the repo shipped once: the real row
    must emit its gold species entity (one OrganismTaxon group over two surface forms), not zero,
    and the declared outputs are entities only"""
    example: TrainingExample = SCRIPT.run(row())
    assert len(example.entities) == 1
    assert surfaces_of(example) == ["patients", "patient"]
    assert example.relations == []
