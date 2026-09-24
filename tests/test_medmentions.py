from __future__ import annotations

from typing import Any

import pytest

from relmedner.families import validate_label_map
from relmedner.models import TrainingExample
from relmedner.scripts import Bc5CdrScript, MedMentionsScript
from relmedner.types import Script
from relmedner.utils import ScriptUtils

SCRIPT: MedMentionsScript = MedMentionsScript()

ROW_MEDMENTIONS: dict[str, Any] = {
    "pmid": "27773526",
    "title": "An S116R Phosphorylation Site Mutation in Human Fibroblast Growth Factor-1 Differentially Affects "
    "Mitogenic and Glucose-Lowering Activities",
    "abstract": "Fibroblast growth factor-1 (FGF-1), a potent human mitogen and insulin sensitizer, signals "
    "through both tyrosine kinase receptor -mediated autocrine / paracrine pathways as well as a "
    "nuclear intracrine pathway. Phosphorylation of FGF-1 at serine 116 (S116) has been proposed to "
    "regulate intracrine signaling. Position S116 is located within a ∼17 amino acid C-terminal "
    "loop that contains a rich set of functional determinants including heparin ∖ heparan sulfate "
    "affinity, thiol reactivity, nuclear localization, pharmacokinetics, functional half-life, "
    "nuclear ligand affinity, stability, and structural dynamics. Mutational targeting of specific "
    "functionality in this region without perturbing other functional determinants is a design "
    "challenge. S116R is a non-phosphorylatable variant present in bovine FGF-1 and other members "
    "of the human FGF family. We show that the S116R mutation in human FGF-1 is accommodated with "
    "no perturbation of biophysical or structural properties, and is therefore an attractive "
    "mutation with which to elucidate the functional role of phosphorylation. Characterization of "
    "S116R shows reduction in NIH 3T3 fibroblast mitogenic stimulation, increase in fibroblast "
    "growth factor receptor-1c activation, and prolonged duration of glucose lowering in ob/ob "
    "hyperglycemic mice. A novel FGF-1 / fibroblast growth factor receptor-1c dimerization "
    "interaction combined with non-phosphorylatable intracrine signaling is hypothesized to be "
    "responsible for these observed functional effects.",
    "entities": [
        {"type": "T082", "cui": "UMLS:C1519254", "offset": 3, "length": 26, "text": "S116R Phosphorylation Site"},
        {"type": "T038", "cui": "UMLS:C0026882", "offset": 30, "length": 8, "text": "Mutation"},
        {"type": "T204", "cui": "UMLS:C0086418", "offset": 42, "length": 5, "text": "Human"},
        {"type": "T103", "cui": "UMLS:C0079349", "offset": 48, "length": 26, "text": "Fibroblast Growth Factor-1"},
        {"type": "T038", "cui": "UMLS:C1752930", "offset": 98, "length": 9, "text": "Mitogenic"},
        {"type": "T033", "cui": "UMLS:C0860801", "offset": 112, "length": 27, "text": "Glucose-Lowering Activities"},
        {"type": "T103", "cui": "UMLS:C0079349", "offset": 140, "length": 26, "text": "Fibroblast growth factor-1"},
        {"type": "T103", "cui": "UMLS:C0079349", "offset": 168, "length": 5, "text": "FGF-1"},
        {"type": "T204", "cui": "UMLS:C0086418", "offset": 185, "length": 5, "text": "human"},
        {"type": "T103", "cui": "UMLS:C0018284", "offset": 191, "length": 7, "text": "mitogen"},
        {"type": "T103", "cui": "UMLS:C0021641", "offset": 203, "length": 7, "text": "insulin"},
        {"type": "T038", "cui": "UMLS:C0037080", "offset": 223, "length": 7, "text": "signals"},
        {"type": "T103", "cui": "UMLS:C0206364", "offset": 244, "length": 24, "text": "tyrosine kinase receptor"},
        {"type": "T038", "cui": "UMLS:C0525010", "offset": 279, "length": 9, "text": "autocrine"},
        {"type": "T082", "cui": "UMLS:C0521447", "offset": 323, "length": 7, "text": "nuclear"},
        {"type": "T038", "cui": "UMLS:C1154545", "offset": 331, "length": 18, "text": "intracrine pathway"},
        {"type": "T038", "cui": "UMLS:C0031715", "offset": 351, "length": 15, "text": "Phosphorylation"},
        {"type": "T103", "cui": "UMLS:C0079349", "offset": 370, "length": 5, "text": "FGF-1"},
        {"type": "T103", "cui": "UMLS:C0036720", "offset": 379, "length": 10, "text": "serine 116"},
        {"type": "T103", "cui": "UMLS:C0036720", "offset": 391, "length": 4, "text": "S116"},
        {"type": "T038", "cui": "UMLS:C3158094", "offset": 418, "length": 8, "text": "regulate"},
        {"type": "T038", "cui": "UMLS:C1154545", "offset": 427, "length": 20, "text": "intracrine signaling"},
        {"type": "T103", "cui": "UMLS:C0036720", "offset": 458, "length": 4, "text": "S116"},
        {"type": "T082", "cui": "UMLS:C1707271", "offset": 487, "length": 26, "text": "amino acid C-terminal loop"},
        {"type": "T103", "cui": "UMLS:C0574031", "offset": 542, "length": 23, "text": "functional determinants"},
        {"type": "T103", "cui": "UMLS:C0019134", "offset": 576, "length": 7, "text": "heparin"},
        {"type": "T103", "cui": "UMLS:C0019143", "offset": 586, "length": 15, "text": "heparan sulfate"},
        {"type": "T103", "cui": "UMLS:C0038734", "offset": 612, "length": 5, "text": "thiol"},
        {"type": "T082", "cui": "UMLS:C0525021", "offset": 630, "length": 20, "text": "nuclear localization"},
        {"type": "T082", "cui": "UMLS:C0521447", "offset": 692, "length": 7, "text": "nuclear"},
        {"type": "T103", "cui": "UMLS:C0023688", "offset": 700, "length": 6, "text": "ligand"},
        {"type": "T038", "cui": "UMLS:C0596957", "offset": 732, "length": 19, "text": "structural dynamics"},
        {"type": "T038", "cui": "UMLS:C0026882", "offset": 753, "length": 10, "text": "Mutational"},
        {"type": "T103", "cui": "UMLS:C0574031", "offset": 840, "length": 23, "text": "functional determinants"},
        {"type": "T103", "cui": "UMLS:C0036720", "offset": 887, "length": 5, "text": "S116R"},
        {"type": "T033", "cui": "UMLS:C0243095", "offset": 898, "length": 20, "text": "non-phosphorylatable"},
        {"type": "T204", "cui": "UMLS:C3667982", "offset": 938, "length": 6, "text": "bovine"},
        {"type": "T103", "cui": "UMLS:C0079349", "offset": 945, "length": 5, "text": "FGF-1"},
        {"type": "T204", "cui": "UMLS:C0086418", "offset": 976, "length": 5, "text": "human"},
        {"type": "T103", "cui": "UMLS:C0016026", "offset": 982, "length": 10, "text": "FGF family"},
        {"type": "T038", "cui": "UMLS:C0026882", "offset": 1011, "length": 14, "text": "S116R mutation"},
        {"type": "T204", "cui": "UMLS:C0086418", "offset": 1029, "length": 5, "text": "human"},
        {"type": "T103", "cui": "UMLS:C0079349", "offset": 1035, "length": 5, "text": "FGF-1"},
        {"type": "T082", "cui": "UMLS:C0026383", "offset": 1096, "length": 10, "text": "structural"},
        {"type": "T038", "cui": "UMLS:C0026882", "offset": 1150, "length": 8, "text": "mutation"},
        {"type": "T038", "cui": "UMLS:C0031715", "offset": 1206, "length": 15, "text": "phosphorylation"},
        {"type": "T103", "cui": "UMLS:C0036720", "offset": 1243, "length": 5, "text": "S116R"},
        {"type": "T017", "cui": "UMLS:C1257739", "offset": 1268, "length": 7, "text": "NIH 3T3"},
        {"type": "T017", "cui": "UMLS:C0016030", "offset": 1276, "length": 10, "text": "fibroblast"},
        {"type": "T038", "cui": "UMLS:C1752930", "offset": 1287, "length": 21, "text": "mitogenic stimulation"},
        {"type": "T103", "cui": "UMLS:C0170936", "offset": 1322, "length": 36, "text": "fibroblast growth factor receptor-1c"},
        {"type": "T038", "cui": "UMLS:C1514758", "offset": 1359, "length": 10, "text": "activation"},
        {"type": "T033", "cui": "UMLS:C0860801", "offset": 1397, "length": 16, "text": "glucose lowering"},
        {"type": "T038", "cui": "UMLS:C0028754", "offset": 1417, "length": 5, "text": "ob/ob"},
        {"type": "T038", "cui": "UMLS:C0020456", "offset": 1423, "length": 13, "text": "hyperglycemic"},
        {"type": "T204", "cui": "UMLS:C0025929", "offset": 1437, "length": 4, "text": "mice"},
        {"type": "T103", "cui": "UMLS:C0079349", "offset": 1451, "length": 5, "text": "FGF-1"},
        {"type": "T103", "cui": "UMLS:C0170936", "offset": 1459, "length": 36, "text": "fibroblast growth factor receptor-1c"},
        {"type": "T033", "cui": "UMLS:C0243095", "offset": 1535, "length": 20, "text": "non-phosphorylatable"},
        {"type": "T038", "cui": "UMLS:C1154545", "offset": 1556, "length": 20, "text": "intracrine signaling"},
    ],
}


def record(**overrides: Any) -> dict[str, Any]:
    """(real record with field overrides) -> the 1-tuple run() unpacks; document-level kwargs
    patch the record fields, entity-level kwargs (and entities_override) flow through entities(),
    so a negative test is always one mutation away from real gold"""
    base = {key: value for key, value in ROW_MEDMENTIONS.items() if key not in ("entities", "title", "abstract")}
    base["entities"] = entities(**overrides)
    base["title"] = overrides.get("title", ROW_MEDMENTIONS["title"])
    base["abstract"] = overrides.get("abstract", ROW_MEDMENTIONS["abstract"])
    return base


def entities(**overrides: Any) -> list[dict[str, Any]]:
    """(all real entities) with every annotation dict merged with the same overrides; pass
    entities_override to replace the list wholesale"""
    overrides_copy = dict(overrides)
    override_entities = overrides_copy.pop("entities_override", None)
    if override_entities is not None:
        return override_entities
    return [{**entity, **overrides_copy} for entity in ROW_MEDMENTIONS["entities"]]


def text_of(rec: dict[str, Any]) -> str:
    return f"{rec['title']}\n{rec['abstract']}"


def surfaces_of(example: TrainingExample) -> list[str]:
    """every emitted mention surface, what the containment assertions read"""
    return [mention for entity in example.entities for mention in entity.mentions]


# ---------------------------------------------------------------------------
# import-time guards
# ---------------------------------------------------------------------------


def test_the_script_self_registers_under_its_declared_name() -> None:
    """the yaml task.name resolves through Script.REGISTRY, which __init_subclass__ populates via
    Script.dispatch, which resolves by this exact NAME key"""
    assert isinstance(Script.REGISTRY["MedMentionsScript"], MedMentionsScript)
    assert isinstance(Script.REGISTRY["Bc5CdrScript"], Bc5CdrScript)


def test_the_label_map_covers_the_complete_measured_type_vocabulary() -> None:
    """the census measured exactly these 21 semantic types over all 203,282 annotations; a type
    missing here would silently drop its gold spans"""
    assert len(MedMentionsScript.LABEL_MAP) == 21
    assert all(code.startswith("T") for code in MedMentionsScript.LABEL_MAP)


def test_every_label_map_target_is_a_biolink_class() -> None:
    """the import-time validate_label_map call already guards this; re-asserted here for drift"""
    assert all(ScriptUtils.is_biolink_category(category) for category in MedMentionsScript.LABEL_MAP.values())


def test_the_import_time_label_guard_rejects_a_non_biolink_class() -> None:
    """a drifted LABEL_MAP value must fail at import rather than mislabel training data"""
    with pytest.raises(ValueError, match="not a biolink class"):
        validate_label_map({"T103": "NotACategory"}, "MedMentionsScript")


# ---------------------------------------------------------------------------
# the real gold record (shape the negative tests mutate)
# ---------------------------------------------------------------------------


def test_the_real_record_ships_every_gold_span() -> None:
    """the converter's census slice-matched 203,282 of 203,282 annotations, so the real record's
    60 gold spans must ALL survive the span contract; group_entities then collapses repeated
    same-surface annotations within a label group (the bc5cdr grouping contract), so the emitted
    surfaces equal the record's UNIQUE mapped surfaces, and every one is char-exact in text"""
    text = text_of(ROW_MEDMENTIONS)
    example: TrainingExample = SCRIPT.run((record(),))
    surfaces = surfaces_of(example)
    expected = {entity["text"] for entity in ROW_MEDMENTIONS["entities"]}
    # a surface may ride under several label groups (secondary morphological labels), so the
    # gold-span contract asserts the distinct surface set, not slot counts
    assert set(surfaces) == expected
    assert all(surface in text for surface in surfaces)


def test_the_real_record_groups_by_mapped_biolink_class() -> None:
    """the record's measured types (T017/T033/T038/T082/T103/T204) group under exactly their six
    mapped biolink classes"""
    example: TrainingExample = SCRIPT.run((record(),))
    labels = {entity.label for entity in example.entities}
    assert labels == {
        "AnatomicalEntity",
        "ClinicalFinding",
        "BiologicalProcess",
        "GeographicLocation",
        "ChemicalEntity",
        "OrganismTaxon",
        # measured secondary morphological labels on gold surfaces: 'insulin' (-in tail rule)
        # and 'tyrosine kinase receptor' (-receptor suffix rule); see docs/secondary-labels.md
        "InsulinDrug",
        "ReceptorProtein",
    }


def test_medmentions_ships_no_relations() -> None:
    """the corpus annotates entities only; relations stay empty on the gold record and on the
    empty example alike"""
    example: TrainingExample = SCRIPT.run((record(),))
    assert example.relations == []


# ---------------------------------------------------------------------------
# trust-gold curie handling (the bc5cdr MESH pattern over UMLS CUIs)
# ---------------------------------------------------------------------------


def test_a_pre_namespaced_cui_rides_as_the_curie_without_fullmap_resolution() -> None:
    """gold annotations are trusted as-is: the annotation's own UMLS id becomes the curie and the
    origin is fullmap only because a curie exists, never because anything was re-resolved"""
    example: TrainingExample = SCRIPT.run((record(),))
    chemical = next(entity for entity in example.entities if entity.label == "ChemicalEntity")
    assert chemical.mentions
    resolved = SCRIPT.mentions_of(record(), text_of(ROW_MEDMENTIONS))
    umls = [mention for mention in resolved if mention.curie and mention.curie.startswith("UMLS:")]
    assert umls, "the real record's census-CUI'd spans must resolve"
    assert all(mention.origin == "fullmap" for mention in umls)


def test_a_non_namespaced_cui_sentinel_stays_raw() -> None:
    """a cui without the UMLS: namespace never becomes a curie, but the span itself still ships
    under its biolink label with raw origin (skip-don't-coerce on identity)"""
    rec = record(cui="C0123456")
    resolved = SCRIPT.mentions_of(rec, text_of(rec))
    assert resolved
    assert all(mention.curie is None and mention.origin == "raw" for mention in resolved)
    example: TrainingExample = SCRIPT.run((rec,))
    assert surfaces_of(example)


# ---------------------------------------------------------------------------
# span-contract drop rules (every negative test one mutation from the real gold record)
# ---------------------------------------------------------------------------


def test_an_unknown_semantic_type_drops() -> None:
    """a type outside the measured 21-type vocabulary is not a span and never a guess"""
    rec = record(type="T999")
    example: TrainingExample = SCRIPT.run((rec,))
    assert surfaces_of(example) == []
    assert example.text == text_of(rec)


def test_a_slice_mismatch_drops() -> None:
    """the converter's measured slice-match rate is 100 percent, so a shifted span is a converter
    bug, not corpus noise: the offending span drops, its gold siblings survive"""
    rec = record(offset=12345)
    example: TrainingExample = SCRIPT.run((rec,))
    assert surfaces_of(example) == []


def test_out_of_bounds_offsets_drop() -> None:
    rec = record(offset=10**9)
    example: TrainingExample = SCRIPT.run((rec,))
    assert surfaces_of(example) == []


def test_negative_and_non_int_offsets_drop() -> None:
    """bool is an int subclass and must not pass the offset gate; negatives are never spans"""
    for bad_offset in (-1, True, "7"):
        rec = record(offset=bad_offset)
        example: TrainingExample = SCRIPT.run((rec,))
        assert surfaces_of(example) == [], f"offset {bad_offset!r} must drop"


def test_empty_surfaces_and_non_string_cuis_drop() -> None:
    rec = record(text="", cui=None)
    example: TrainingExample = SCRIPT.run((rec,))
    assert surfaces_of(example) == []


def test_a_document_whose_annotations_all_drop_still_ships_text() -> None:
    """the declared-outputs filter reads the example shape, so an all-dropped record must keep
    its text (skip-don't-coerce: no exception, no fabricated span)"""
    rec = record(type="T999")
    example: TrainingExample = SCRIPT.run((rec,))
    assert example.text == text_of(rec)
    assert example.entities == []


def test_an_empty_entity_list_ships_text_with_zero_spans() -> None:
    rec = record(entities_override=[])
    example: TrainingExample = SCRIPT.run((rec,))
    assert example.text == text_of(rec)
    assert example.entities == []


# ---------------------------------------------------------------------------
# record-level degradation (skip-don't-coerce)
# ---------------------------------------------------------------------------


def test_a_non_dict_record_degrades_to_the_empty_example() -> None:
    example: TrainingExample = SCRIPT.run(("junk",))
    assert example.text == ""
    assert example.entities == []
    assert example.relations == []


def test_a_blank_text_record_degrades_to_the_empty_example() -> None:
    example: TrainingExample = SCRIPT.run((record(title="", abstract=""),))
    assert example.text == ""


def test_a_non_list_entities_table_yields_zero_spans() -> None:
    rec = record(entities_override="junk")
    example: TrainingExample = SCRIPT.run((rec,))
    assert example.text == text_of(rec)
    assert example.entities == []


def test_non_dict_entity_entries_drop() -> None:
    rec = record(entities_override=["junk", None, ROW_MEDMENTIONS["entities"][0]])
    example: TrainingExample = SCRIPT.run((rec,))
    assert len(surfaces_of(example)) == 1
