from __future__ import annotations

import pytest

from relmedner.utils import PredicateRangeGate, ResolutionGate, ResolvedMention, ScriptUtils


@pytest.mark.parametrize(
    ("curie", "label", "expected"),
    [
        ("FB:FBgn0005052", "illness", True),  # flu -> Drosophila gene: the classic mismatch
        ("ZFIN:ZDB-GENE-070117-2198", "person", True),  # men -> zebrafish gene
        ("MGI:96703", "gene/protein", False),  # genomic labels keep model-organism hits
        ("NCBIGene:581", "protein", False),  # non-model-organism prefixes never gate
        ("CHEBI:18243", "entity type", False),  # no model-organism prefix
        (None, "illness", False),  # no curie, nothing to gate
    ],
)
def test_is_model_organism_mismatch(curie: str | None, label: str, expected: bool) -> None:
    assert ResolutionGate.is_model_organism_mismatch(curie, label) is expected


@pytest.mark.parametrize(
    ("label", "category", "expected"),
    [
        ("organization", "SmallMolecule", False),  # FDA -> FADH2, the probe's flagship mismatch
        ("organization", "Agent", True),
        ("person", "SmallMolecule", False),  # baby -> pubchem compound
        ("person", "Human", True),
        ("attribute", "PopulationOfIndividualOrganisms", False),  # physical -> UMLS Physics
        ("abbreviation", "Gene", False),  # DNC -> fly gene SLC25A19: info label vs gene hit
        ("organism", "Gene", False),  # dogs -> fly gene dog
        ("disease", "Disease", True),
        ("body part", "GrossAnatomicalStructure", True),
        ("substance", "Protein", True),  # polyclonal antibodies must survive
        ("entity type", "SmallMolecule", True),  # OPEN: dopamine stays
        ("trait", "PhenotypicFeature", True),  # OPEN: eye color stays
        ("system", "Publication", True),  # OPEN: electronic medical record stays
        ("totally-unmapped-label", "Gene", True),  # no bucket, no opinion
        ("medical condition", "NotARealBiolinkClass", True),  # raw categories carry no ancestors
    ],
)
def test_is_label_compatible(label: str, category: str, expected: bool) -> None:
    assert ResolutionGate.is_label_compatible(label, category) is expected


@pytest.mark.parametrize(
    ("label", "category", "expected"),
    [
        ("first name", "Gene", False),  # surnames fullmap-hit as every entity kind; gene hits must reject
        ("first name", "Human", True),
        ("last name", "Protein", False),
        ("middle name", "SmallMolecule", False),
        ("user name", "ChemicalEntity", False),
        ("surname", "IndividualOrganism", True),
        ("street address", "Gene", False),
        ("state", "GeographicLocation", True),
        ("county", "Gene", False),
        ("postcode", "GeographicLocation", True),
        ("coordinate", "GeographicLocation", True),
        ("religious belief", "SmallMolecule", False),
        ("religious belief", "Attribute", True),  # SocioeconomicAttribute survives through its Attribute ancestor
        ("education level", "ChemicalEntity", False),
        ("education level", "SocioeconomicAttribute", True),
        # Agent already sits in the person bucket's allowed set and the gate change is strictly additive,
        # so an Agent fullmap hit under employment_status stays accepted (the bucket cannot reject it)
        ("employment status", "Agent", True),
        ("employment status", "SocioeconomicAttribute", True),
        ("blood type", "Disease", False),
        ("blood type", "ClinicalMeasurement", True),
    ],
)
def test_is_label_compatible_pii_buckets(label: str, category: str, expected: bool) -> None:
    """Nemotron-PII labels gate fullmap hits exactly like biomed labels do: person-name labels only
    survive Human/IndividualOrganism ancestors, socioeconomic labels survive Attribute descendants,
    place labels survive GeographicLocation, and blood type only measurement-grade hits; every
    rejection here is what pushes the mention onto the fallback/raw tail instead of a wrong category"""
    assert ResolutionGate.is_label_compatible(label, category) is expected


@pytest.mark.parametrize(
    ("mention", "label", "expected"),
    [
        ("HVA", "entity type", True),  # 4-hydroxypentanoate, a fullmap collision
        ("KRT7", "gene/protein", False),  # genomic labels keep acronyms
        ("aspirin", "entity type", False),  # lowercase mentions never gate
        ("CP", "medical condition", False),  # bucket labels route through the bucket check instead
    ],
)
def test_is_acronym_over_open_label(mention: str, label: str, expected: bool) -> None:
    assert ResolutionGate.is_acronym_over_open_label(mention, label) is expected


def test_accept_composes_all_three_checks() -> None:
    assert ResolutionGate.accept("FDA", "organization", "CHEBI:17877", "SmallMolecule") is False
    assert ResolutionGate.accept("dogs", "organism", "FB:FBgn0016793", "ChemicalEntity") is False
    assert ResolutionGate.accept("Bax", "protein", "NCBIGene:581", "Gene") is True
    assert ResolutionGate.accept("eye color", "trait", "EFO:0003949", "PhenotypicFeature") is True


@pytest.mark.skipif(not ScriptUtils.fullmap_available(), reason="fullmap database is not mounted")
def test_fullmap_person_name_label_never_keeps_a_non_human_hit() -> None:
    """end-to-end gate proof: the common surname 'Boyce' fullmap-hits as non-Human entities too, so
    under the first_name label any such hit must reject and the mention falls through to the Human
    fallback -- no resolved mention may surface origin='fullmap' with a category outside Human"""
    # the dataset vocabulary is inlined ({"first_name": "Human"}, the NemotronPiiScript entry) so
    # this gate layer stays self-contained below the ingest layer that declares the full map
    Resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions([("Boyce", "first_name")], label_map={"first_name": "Human"})
    assert {item.category for item in Resolved if item.origin == "fullmap"} <= {"Human"}


def test_ancestors_walk_the_biolink_mro_and_empty_for_raw_labels() -> None:
    assert "Disease" in ResolutionGate.ancestors("Disease")
    assert "DiseaseOrPhenotypicFeature" in ResolutionGate.ancestors("Disease")
    assert "Entity" in ResolutionGate.ancestors("Disease")
    assert ResolutionGate.ancestors("MedicalCondition") == frozenset()


@pytest.mark.parametrize(
    ("predicate", "head", "tail", "expected"),
    [
        ("expressed_in", "Gene", "GrossAnatomicalStructure", True),
        ("expressed_in", "ChemicalEntity", "CellLine", False),  # chemicals do not express
        ("treats", "SmallMolecule", "Disease", True),
        ("treats", "Disease", "Disease", False),  # a disease does not treat
        ("treats", "GeographicLocation", "Disease", False),
        ("participates_in", "Gene", "SmallMolecule", False),
        ("participates_in", "Gene", "BiologicalProcess", True),
        ("in_taxon", "Gene", "OrganismTaxon", True),
        ("in_taxon", "SmallMolecule", "SmallMolecule", False),
        ("in_taxon", "Measurement", "Measurement", True),  # raw labels impose no constraint
        ("biomarker_for", "Disease", "Disease", False),  # measured probe reject
        ("occurs_in", "ChemicalEntity", "SmallMolecule", False),  # measured probe reject
        ("caused_by", "PhenotypicFeature", "Disease", True),  # the edge the raw-label fix rescued
    ],
)
def test_predicate_range_gate(predicate: str, head: str, tail: str, expected: bool) -> None:
    assert PredicateRangeGate.accept(predicate, head, tail) is expected


def test_predicate_range_gate_raw_labels_impose_no_constraint() -> None:
    """PascalCased raw labels have empty ancestor sets; without the no-opinion rule they would
    fail every constrained range and kill correct edges like caused_by into a raw tail"""
    assert PredicateRangeGate.accept("caused_by", "Disease", "MedicalCondition") is True
    assert PredicateRangeGate.accept("treats", "Drug", "Chemical") is True


def test_predicate_range_gate_covers_every_declared_predicate() -> None:
    from relmedner.gazetteer import PREDICATE_TRIGGERS

    missing: list[str] = [predicate for predicate in PREDICATE_TRIGGERS if predicate not in PredicateRangeGate.DOMRANGE]
    assert not missing, missing


def test_predicate_range_gate_unknown_predicate_carries_no_opinion() -> None:
    assert PredicateRangeGate.accept("not_a_predicate", "Gene", "Disease") is True
