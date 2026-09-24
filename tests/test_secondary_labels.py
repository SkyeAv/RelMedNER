"""secondary label rules (docs/secondary-labels.md): morphology -> extra non-biolink labels.

every rule here was measured against MedMentions ST21pv gold before landing; these tests pin
the measured decisions (kept rules fire, dropped rules and stoplisted tokens never fire) and
the group_entities expansion contract.
"""

from __future__ import annotations

from relmedner.utils import ResolvedMention, ScriptUtils, secondary_labels


def test_drug_inn_stems_fire_their_secondary_label() -> None:
    assert secondary_labels("pembrolizumab") == ("MonoclonalAntibodyDrug",)
    assert secondary_labels("trametinib") == ("KinaseInhibitorDrug",)
    assert secondary_labels("ramipril") == ("AceInhibitorDrug",)
    assert secondary_labels("valsartan") == ("AngiotensinAntagonist",)
    assert secondary_labels("timolol") == ("BetaBlockerDrug",)
    assert secondary_labels("simvastatin") == ("StatinDrug",)
    assert secondary_labels("celecoxib") == ("Cox2InhibitorDrug",)
    assert secondary_labels("sitagliptin") == ("GliptinDrug",)
    assert secondary_labels("lidocaine") == ("LocalAnestheticDrug",)
    assert secondary_labels("amlodipine") == ("CalciumChannelBlocker",)
    assert secondary_labels("omeprazole") == ("ProtonPumpInhibitor",)
    assert secondary_labels("ranitidine") == ("H2AntagonistDrug",)
    assert secondary_labels("ciprofloxacin") == ("QuinoloneAntibiotic",)
    assert secondary_labels("vancomycin") == ("AntibioticMycin",)
    assert secondary_labels("amoxicillin") == ("PenicillinAntibiotic",)
    assert secondary_labels("doxycycline") == ("TetracyclineAntibiotic",)
    assert secondary_labels("teriparatide") == ("PeptideDrug",)


def test_disease_morphology_rules_fire() -> None:
    assert secondary_labels("advanced melanoma") == ("NeoplasticProcess",)
    assert secondary_labels("rheumatoid arthritis") == ("InflammatoryDisease",)
    assert secondary_labels("anemia") == ("BloodCellDisease",)
    assert secondary_labels("diabetic nephropathy") == ("OrganDisease",)
    assert secondary_labels("cholecystectomy") == ("SurgicalRemoval",)
    assert secondary_labels("colonoscopy") == ("Endoscopy",)
    assert secondary_labels("arthroplasty") == ("SurgicalRepair",)
    assert secondary_labels("chemotherapy") == ("TherapyProcedure",)
    assert secondary_labels("lymphocytes") == ("CellType",)
    assert secondary_labels("T lymphocytes") == ("CellType",)


def test_tail_rules_fire_on_the_exact_last_word() -> None:
    assert secondary_labels("insulin") == ("InsulinDrug",)
    assert secondary_labels("Neutral Protamine Hagedorn insulin") == ("InsulinDrug",)
    assert secondary_labels("HBV vaccine") == ("Vaccine",)
    assert secondary_labels("liver biopsy") == ("BiopsyAssay",)
    assert secondary_labels("low density lipoprotein receptor") == ("ReceptorProtein",)
    assert secondary_labels("EZH2 gene") == ("GeneMention",)
    assert secondary_labels("vascular endothelial growth factor") == ("GrowthFactorProtein",)


def test_measured_false_positive_producers_never_fire() -> None:
    assert secondary_labels("pulmonary disease") == ()  # -ase dropped: 'disease' stoplisted
    assert secondary_labels("Embase") == ()  # -ase dropped
    assert secondary_labels("myostatin") == ()  # -statin stoplisted (it is a gene)
    assert secondary_labels("insulin resistance") == ()  # disease span, insulin is not the last word
    assert secondary_labels("hepatitis C virus") == ()  # -itis is not the last token
    assert secondary_labels("posttonsillectomy pain") == ()  # -ectomy is not the last token
    assert secondary_labels("melanoma cells") == ()  # 'cells' carries no morphological cue
    assert secondary_labels("Interferon gamma-1b") == ()  # drug-stem tail rules are last-word only (measured end-anchored)
    assert secondary_labels("glucose") == ()  # -ose rule never landed
    assert secondary_labels("diagram") == ()  # -gram rule never landed


def test_local_anesthetic_cocaine_is_a_kept_hit() -> None:
    """'cocaine' is 7 chars, above the floor, and IS a -caine drug in the gold corpus"""
    assert secondary_labels("cocaine") == ("LocalAnestheticDrug",)
    assert secondary_labels("interferon") == ("InterferonDrug",)


def test_case_insensitivity_and_short_tokens() -> None:
    assert secondary_labels("PEMBROLIZUMAB") == ("MonoclonalAntibodyDrug",)
    assert secondary_labels("asa") == ()
    assert secondary_labels("nib") == ()


def test_secondary_labels_never_collide_with_biolink_categories() -> None:
    """a secondary label equal to a biolink class would silently merge with primary labels and
    inherit biolink descriptions; the rule tables must stay outside the biolink vocabulary"""
    emitted = set(__import__("relmedner.constants", fromlist=["x"]).SECONDARY_SUFFIX_LABELS.values()) | set(
        __import__("relmedner.constants", fromlist=["x"]).SECONDARY_TAIL_LABELS.values()
    )
    assert all(not ScriptUtils.is_biolink_category(label) for label in emitted)


def test_group_entities_expands_with_the_secondary_label() -> None:
    resolved = [ResolvedMention(mention="pembrolizumab", category="Protein", curie="CHEBI:1", preferred_name="Pembrolizumab", origin="fullmap")]
    entities = ScriptUtils.group_entities(resolved)

    by_label = {entity.label: entity for entity in entities}
    assert set(by_label) == {"Protein", "MonoclonalAntibodyDrug"}
    assert by_label["MonoclonalAntibodyDrug"].mentions == ["pembrolizumab"]
    # a non-biolink label has no biolink definition by design, so the description is the
    # evidence string alone (the primary label carries definition + evidence)
    assert by_label["MonoclonalAntibodyDrug"].description == "[fullmap: CHEBI:1 | Pembrolizumab]"


def test_group_entities_secondary_without_curie_ships_undescribed() -> None:
    resolved = [ResolvedMention(mention="melanoma", category="Disease", origin="raw")]
    entities = ScriptUtils.group_entities(resolved)

    by_label = {entity.label: entity for entity in entities}
    assert set(by_label) == {"Disease", "NeoplasticProcess"}
    assert by_label["NeoplasticProcess"].description is None


def test_group_entities_dedupes_when_primary_equals_secondary() -> None:
    """defense in depth: if a rule table ever grew a label colliding with a primary category,
    the mention must still ship once per label, not twice"""
    resolved = [
        ResolvedMention(mention="pembrolizumab", category="MonoclonalAntibodyDrug", origin="raw"),
        ResolvedMention(mention="pembrolizumab", category="MonoclonalAntibodyDrug", origin="raw"),
    ]
    entities = ScriptUtils.group_entities(resolved)

    assert len(entities) == 1
    assert entities[0].mentions == ["pembrolizumab"]
