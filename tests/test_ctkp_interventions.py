from __future__ import annotations

from typing import Any

from relmedner.models import TrainingExample
from relmedner.scripts import CtkpInterventionsScript
from relmedner.types import Script
from relmedner.utils import ScriptUtils

SCRIPT: CtkpInterventionsScript = CtkpInterventionsScript()


def record(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "nct_id": "NCT01203189",
        "intervention_type": "DRUG",
        "name": "ketoconazole 2% shampoo",
        "description": "Subjects will wash their hair twice weekly with ketoconazole 2% shampoo.",
        "matches": [],
        "other_names": [],
        "synonym_curies": [],
        "unmapped": True,
    }
    return {**base, **overrides}


def match(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "curie": "RXCUI:106336",
        "category": "Drug",
        "preferred_name": "ketoconazole 20 MG/ML Medicated Shampoo",
        "source": "text",
        "matched_text": "ketoconazole 2% shampoo",
        "unmapped": False,
    }
    return {**base, **overrides}


def test_the_script_self_registers_under_its_declared_name() -> None:
    assert isinstance(Script.REGISTRY["CtkpInterventionsScript"], CtkpInterventionsScript)


def test_text_joins_the_name_and_description() -> None:
    Example: TrainingExample = SCRIPT.run((record(matches=[match()]),))

    assert Example.text.startswith("ketoconazole 2% shampoo. Subjects will wash")


def test_text_is_the_bare_name_when_the_description_is_null() -> None:
    Example: TrainingExample = SCRIPT.run((record(description=None, matches=[match()]),))

    assert Example.text == "ketoconazole 2% shampoo"


def test_a_kp_match_becomes_an_entity_carrying_its_gold_curie() -> None:
    Example: TrainingExample = SCRIPT.run((record(matches=[match()]),))

    assert [entity.label for entity in Example.entities] == ["Drug"]
    assert Example.entities[0].mentions == ["ketoconazole 2% shampoo"]


def test_the_same_surface_under_two_categories_ships_multiclass() -> None:
    """multi-class contract: when the KP ever emits one matched_text under two distinct biolink
    categories (snapshot 20260920 never does -- measured, see docs/ctkp-interventions.md), the
    surface must ship under BOTH labels with each label keeping its own gold curie evidence"""
    Record: dict[str, Any] = record(
        name="semaglutide",
        description="semaglutide is given subcutaneously.",
        matches=[
            match(curie="CHEBI:176885", category="SmallMolecule", preferred_name="semaglutide", matched_text="semaglutide"),
            match(curie="MESH:C000654876", category="ChemicalEntity", preferred_name="semaglutide", matched_text="semaglutide"),
        ],
    )
    Example: TrainingExample = SCRIPT.run((Record,))

    # 'semaglutide' carries the -tide suffix, so the surface ALSO earns the PeptideDrug
    # secondary label on top of both KP categories (docs/secondary-labels.md)
    by_label = {entity.label: entity.mentions for entity in Example.entities}
    assert by_label == {
        "SmallMolecule": ["semaglutide"],
        "ChemicalEntity": ["semaglutide"],
        "PeptideDrug": ["semaglutide"],
    }


def test_multiple_matches_group_under_their_own_labels() -> None:
    Record: dict[str, Any] = record(
        name="Nab-paclitaxel plus Gemcitabine",
        description="Gemcitabine is given intravenously.",
        matches=[
            match(curie="CHEBI:175901", category="SmallMolecule", preferred_name="Gemcitabine", matched_text="gemcitabine"),
            match(curie="MESH:C520255", category="ChemicalEntity", preferred_name="nab-paclitaxel", matched_text="nab-paclitaxel"),
        ],
    )
    Example: TrainingExample = SCRIPT.run((Record,))

    assert sorted(entity.label for entity in Example.entities) == ["ChemicalEntity", "SmallMolecule"]


def test_the_biolink_prefix_is_stripped_off_a_kp_category() -> None:
    """the KP emits both 'Procedure' and 'biolink:Procedure'; they must collapse to one label"""
    Record: dict[str, Any] = record(
        intervention_type="PROCEDURE",
        name="acupuncture",
        description="Needles are inserted at acupuncture points.",
        matches=[match(category="biolink:Procedure", curie="MESH:D015670", matched_text="acupuncture")],
    )
    Example: TrainingExample = SCRIPT.run((Record,))

    assert [entity.label for entity in Example.entities] == ["Procedure"]


def test_a_match_absent_from_the_text_is_dropped() -> None:
    """gliner2 discards spans it cannot find in the text, so the KP's span never ships; the record
    still falls back to its AACT type, keyed on the name, which IS in the text"""
    Example: TrainingExample = SCRIPT.run((record(matches=[match(matched_text="rifampin")]),))

    assert [entity.mentions for entity in Example.entities] == [["ketoconazole 2% shampoo"]]
    assert "rifampin" not in str(Example.entities)


def test_a_match_absent_from_the_text_ships_nothing_when_the_type_is_untypeable() -> None:
    Example: TrainingExample = SCRIPT.run((record(intervention_type="OTHER", matches=[match(matched_text="rifampin")]),))

    assert Example.entities == []


def test_a_match_whose_category_is_not_a_biolink_class_is_dropped() -> None:
    """the bogus category never becomes a label; the row degrades to the AACT-type fallback"""
    Example: TrainingExample = SCRIPT.run((record(matches=[match(category="NotARealBiolinkClass")]),))

    assert [entity.label for entity in Example.entities] == ["Drug"]
    assert "NotARealBiolinkClass" not in str(Example.entities)


def test_an_unmapped_record_falls_back_to_its_aact_intervention_type() -> None:
    Record: dict[str, Any] = record(
        intervention_type="DEVICE",
        name="relton-hall frame",
        description="relton-hall frame viscoelastic polymer pads",
        matches=[],
    )
    Example: TrainingExample = SCRIPT.run((Record,))

    assert [entity.label for entity in Example.entities] == ["Device"]
    assert Example.entities[0].mentions == ["relton-hall frame"]


def test_an_unmapped_match_with_no_category_still_falls_back_on_the_type() -> None:
    """the unmapped half of the KP carries matched_text but a null category"""
    Record: dict[str, Any] = record(matches=[match(curie=None, category=None, preferred_name=None, unmapped=True)])
    Example: TrainingExample = SCRIPT.run((Record,))

    assert [entity.label for entity in Example.entities] == ["Drug"]


def test_an_untypeable_record_ships_text_with_no_entities() -> None:
    """OTHER has no defensible biolink class; the pipeline's declared-outputs filter drops the row"""
    Example: TrainingExample = SCRIPT.run((record(intervention_type="OTHER", matches=[]),))

    assert Example.entities == []
    assert Example.populated() == frozenset()


def test_an_empty_record_produces_an_empty_example() -> None:
    Example: TrainingExample = SCRIPT.run((record(name="", description=None, matches=[]),))

    assert Example.text == ""
    assert Example.entities == []


def test_every_label_map_target_is_a_biolink_category() -> None:
    assert all(ScriptUtils.is_biolink_category(category) for category in CtkpInterventionsScript.LABEL_MAP.values())


def test_dispatch_routes_through_the_registry_and_keeps_declared_outputs() -> None:
    Outputs, Example = Script.dispatch("CtkpInterventionsScript", (("entities",), (record(matches=[match()]),)))

    assert Outputs == ("entities",)
    assert [entity.label for entity in Example.entities] == ["Drug"]
