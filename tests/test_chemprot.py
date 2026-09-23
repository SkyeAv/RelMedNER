from __future__ import annotations

from typing import Any

import pytest

from relmedner.families import validate_label_map
from relmedner.models import TrainingExample
from relmedner.scripts import ChemprotScript, KnowledgatorBiomedScript  # registry must stay populated alongside the new script
from relmedner.scripts.chemprot import _validate_predicate_map
from relmedner.types import Script
from relmedner.utils import ScriptUtils

SCRIPT: ChemprotScript = ChemprotScript()

# verbatim bigbio/chemprot train rows captured by the wenceslaus full-split census
# (grep FIXTURE_ROW /home/sgoetz/chemprot-census.log); every negative test below mutates one of
# these real rows so no test trains on an invented row shape
ROW_16357751: dict[str, Any] = {
    "pmid": "16357751",
    "text": "Selective costimulation modulators: a novel approach for the treatment of rheumatoid arthritis.\n"
    "T cells have a central role in the orchestration of the immune pathways that contribute to the inflammation and joint "
    "destruction characteristic of rheumatoid arthritis (RA). The requirement for a dual signal for T-cell activation and the "
    "construction of a fusion protein that prevents engagement of the costimulatory molecules required for this activation has led to "
    "a new approach to RA therapy. This approach is mechanistically distinct from other currently used therapies; it targets events "
    "early rather than late in the immune cascade, and it results in immunomodulation rather than complete immunosuppression. The "
    "fusion protein abatacept is a selective costimulation modulator that avidly binds to the CD80/CD86 ligands on an "
    "antigen-presenting cell, resulting in the inability of these ligands to engage the CD28 receptor on the T cell. Abatacept "
    "dose-dependently reduces T-cell proliferation, serum concentrations of acute-phase reactants, and other markers of inflammation, "
    "including the production of rheumatoid factor by B cells. Recent studies have provided consistent evidence that treatment with "
    "abatacept results in a rapid onset of efficacy that is maintained over the course of treatment in patients with inadequate "
    "response to methotrexate and anti-tumor necrosis factor therapies. This efficacy includes patient-centered outcomes and "
    "radiographic measurement of disease progression. Abatacept has also demonstrated a very favorable safety profile to date. This "
    "article reviews the rationale for this therapeutic approach and highlights some of the recent studies that demonstrate the "
    "benefits obtained by using abatacept. This clinical experience indicates that abatacept is a significant addition to the "
    "therapeutic armamentarium for the management of patients with RA.",
    "entities": {
        "id": ["T1", "T2", "T3", "T4", "T5"],
        "type": ["CHEMICAL", "GENE-N", "GENE-Y", "GENE-Y", "GENE-N"],
        "text": ["methotrexate", "tumor necrosis factor", "CD80", "CD86", "CD28 receptor"],
        "offsets": [[1342, 1354], [1364, 1385], [805, 809], [810, 814], [912, 925]],
    },
    "relations": {"type": [], "arg1": [], "arg2": []},
}

ROW_14967461: dict[str, Any] = {
    "pmid": "14967461",
    "text": "Emerging role of epidermal growth factor receptor inhibition in therapy for advanced malignancy: focus on NSCLC.\n"
    "Combination chemotherapy regimens have emerged as the standard approach in advanced non-small-cell lung cancer. Meta-analyses "
    "have demonstrated a 2-month increase in median survival after platinum-based therapy vs. best supportive care, and an absolute "
    "10% improvement in the 1-year survival rate. Just as importantly, cytotoxic therapy has produced benefits in symptom control and "
    "quality of life. Newer agents, including the taxanes, vinorelbine, gemcitabine, and irinotecan, have expanded our therapeutic "
    "options in the treatment of advanced non-small-cell lung cancer. Despite their contributions, we have reached a therapeutic "
    "plateau, with response rates seldom exceeding 30-40% in cooperative group studies and 1-year survival rates stable between 30% "
    "and 40%. It is doubtful that substituting one agent for another in various combinations will lead to any further improvement in "
    "these rates. The thrust of current research has focused on targeted therapy, and epidermal growth factor receptor inhibition is "
    "one of the most promising clinical strategies. Epidermal growth factor receptor inhibitors currently under investigation include "
    "the small molecules gefitinib (Iressa, ZD1839) and erlotinib (Tarceva, OSI-774), as well as monoclonal antibodies such as "
    "cetuximab (IMC-225, Erbitux). Agents that have only begun to undergo clinical evaluation include CI-1033, an irreversible "
    "pan-erbB tyrosine kinase inhibitor, and PKI166 and GW572016, both examples of dual kinase inhibitors (inhibiting epidermal "
    "growth factor receptor and Her2). Preclinical models have demonstrated synergy for all these agents in combination with either "
    "chemotherapy or radiotherapy, leading to great enthusiasm regarding their ultimate contribution to lung cancer therapy. However, "
    "serious clinical challenges persist. These include the identification of the optimal dose(s); the proper integration of these "
    "agents into popular, established cytotoxic regimens; and the selection of the optimal setting(s) in which to test these "
    "compounds. Both gefitinib and erlotinib have shown clinical activity in pretreated, advanced non-small-cell lung cancer, but "
    "placebo-controlled randomized Phase III studies evaluating gefitinib in combination with standard cytotoxic therapy, to our "
    "chagrin, have failed to demonstrate a survival advantage compared with chemotherapy alone.",
    "entities": {
        "id": [
            "T1",
            "T2",
            "T3",
            "T4",
            "T5",
            "T6",
            "T7",
            "T8",
            "T9",
            "T10",
            "T11",
            "T12",
            "T13",
            "T14",
            "T15",
            "T16",
            "T17",
            "T18",
            "T19",
            "T20",
            "T21",
            "T22",
            "T23",
            "T24",
            "T25",
            "T26",
            "T27",
            "T28",
            "T29",
        ],
        "type": [
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "CHEMICAL",
            "GENE-Y",
            "GENE-Y",
            "GENE-N",
            "GENE-N",
            "GENE-Y",
            "GENE-Y",
            "GENE-Y",
            "GENE-Y",
        ],
        "text": [
            "gefitinib",
            "Iressa",
            "ZD1839",
            "erlotinib",
            "Tarceva",
            "OSI-774",
            "cetuximab",
            "IMC-225",
            "Erbitux",
            "CI-1033",
            "tyrosine",
            "PKI166",
            "GW572016",
            "platinum",
            "gefitinib",
            "erlotinib",
            "gefitinib",
            "taxanes",
            "vinorelbine",
            "gemcitabine",
            "irinotecan",
            "Epidermal growth factor receptor",
            "erbB",
            "tyrosine kinase",
            "kinase",
            "epidermal growth factor receptor",
            "Her2",
            "epidermal growth factor receptor",
            "epidermal growth factor receptor",
        ],
        "offsets": [
            [1277, 1286],
            [1288, 1294],
            [1296, 1302],
            [1308, 1317],
            [1319, 1326],
            [1328, 1335],
            [1379, 1388],
            [1390, 1397],
            [1399, 1406],
            [1476, 1483],
            [1510, 1518],
            [1541, 1547],
            [1552, 1560],
            [301, 309],
            [2142, 2151],
            [2156, 2165],
            [2310, 2319],
            [540, 547],
            [549, 560],
            [562, 573],
            [579, 589],
            [1175, 1207],
            [1505, 1509],
            [1510, 1525],
            [1584, 1590],
            [1614, 1646],
            [1651, 1655],
            [1081, 1113],
            [17, 49],
        ],
    },
    "relations": {
        "type": [
            "CPR:4",
            "CPR:4",
            "CPR:4",
            "CPR:4",
            "CPR:4",
            "CPR:4",
            "CPR:4",
            "CPR:4",
            "CPR:4",
            "CPR:4",
            "CPR:4",
            "CPR:4",
            "CPR:4",
            "CPR:4",
            "CPR:4",
            "CPR:4",
            "CPR:4",
        ],
        "arg1": ["T10", "T10", "T12", "T12", "T12", "T13", "T13", "T13", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"],
        "arg2": ["T23", "T24", "T25", "T26", "T27", "T25", "T26", "T27", "T22", "T22", "T22", "T22", "T22", "T22", "T22", "T22", "T22"],
    },
}


@pytest.fixture(autouse=True)
def no_fullmap(monkeypatch: pytest.MonkeyPatch) -> None:
    """label-map tests pin the deterministic fallback path: fullmap resolution is environment-
    dependent (mounted DB vs not), so it is stubbed to a miss and the LABEL_MAP fallback tier
    is what every label assertion in this file exercises"""
    monkeypatch.setattr(ScriptUtils, "_fullmap_best", classmethod(lambda cls, normalized: {}))


def row_16357751(**overrides: Any) -> tuple[Any, ...]:
    """one chemprot row as columns_out [text, entities, relations] delivers it; overrides mutate
    a copy of the verbatim census row (the 5-entity, 0-relation abatacept abstract)"""
    merged: dict[str, Any] = {**ROW_16357751, **overrides}
    return (merged["text"], merged["entities"], merged["relations"])


def row_14967461(**overrides: Any) -> tuple[Any, ...]:
    """the 29-entity, 17-relation EGFR-inhibition abstract; relation drop-rule tests mutate its
    gold CPR:4 relations one defect at a time"""
    merged: dict[str, Any] = {**ROW_14967461, **overrides}
    return (merged["text"], merged["entities"], merged["relations"])


def entities_with(**column_overrides: Any) -> dict[str, Any]:
    """copy the verbatim entities table with one parallel column replaced (raggedness, malformed
    offsets, drifted labels) while the sibling columns stay gold"""
    merged: dict[str, Any] = {**ROW_16357751["entities"], **column_overrides}
    return merged


def relations_with(**column_overrides: Any) -> dict[str, Any]:
    """copy the verbatim 17-relation table with one parallel column replaced"""
    merged: dict[str, Any] = {**ROW_14967461["relations"], **column_overrides}
    return merged


# ---------------------------------------------------------------------------
# import-time guards (the module-scope validation must fail loudly on drift)
# ---------------------------------------------------------------------------


def test_the_script_self_registers_under_its_declared_name() -> None:
    """importing the module must install the instance in the shared registry (the ingest wires
    Script.dispatch, which resolves by this exact NAME key)"""
    assert isinstance(Script.REGISTRY["ChemprotScript"], ChemprotScript)


def test_every_mapped_label_and_predicate_is_accounted_for() -> None:
    """the import-time guards only protect the module's own constants, so the contract is
    re-asserted here for drift: label-map values stay biolink classes and every predicate-map
    value is either a tablassert Predicates member or a documented native snake_case omission"""
    for raw_label, category in ChemprotScript.LABEL_MAP.items():
        assert ScriptUtils.is_biolink_category(category), f"fallback {raw_label!r} -> {category!r} is not a biolink class"
    members: frozenset[str] = ScriptUtils.biolink_predicates()
    for label, predicate in ChemprotScript.PREDICATE_MAP.items():
        assert predicate in members or predicate in ChemprotScript.NATIVE_PREDICATES, f"{label!r} -> {predicate!r} is unguarded"


def test_the_import_time_label_guard_rejects_a_non_biolink_class() -> None:
    """a typo'd label-map value (NotAClass) must fail loudly at import, not silently train garbage
    labels; families.validate_label_map is the shared guard the module calls on its own constant"""
    with pytest.raises(ValueError, match="NotAClass"):
        validate_label_map({"drift": "NotAClass"}, "BrokenChemprotScript")


def test_the_import_time_predicate_guard_rejects_an_unmapped_predicate() -> None:
    """a typo'd predicate-map value must fail loudly at import rather than training relations under
    a garbage name; the module-scope guard re-runs over the (monkeypatched) map here"""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ChemprotScript, "PREDICATE_MAP", {"CPR:1": "not_a_predicate"})
    with pytest.raises(ValueError, match="not_a_predicate"):
        _validate_predicate_map()
    monkeypatch.undo()


# ---------------------------------------------------------------------------
# row-level drop rules: text
# ---------------------------------------------------------------------------


def test_a_non_string_text_value_produces_the_empty_example() -> None:
    """hub-side schema drift can deliver a non-str text column; the isinstance(str) guard must turn
    it into the canonical empty example for the declared-outputs filter, never a crash mid-stream"""
    for Drifted in (None, 42, ["text"]):
        Example: TrainingExample = SCRIPT.run((Drifted, ROW_16357751["entities"], ROW_16357751["relations"]))

        assert Example.text == ""
        assert Example.populated() == frozenset()


def test_an_empty_text_produces_the_empty_example() -> None:
    """an empty text cannot host token spans or relations, so the row ships the canonical empty
    example and downstream filtering drops it"""
    Example: TrainingExample = SCRIPT.run(("", ROW_16357751["entities"], ROW_14967461["relations"]))

    assert Example.text == ""
    assert Example.populated() == frozenset()


# ---------------------------------------------------------------------------
# entity-table drop rules (skip-don't-coerce transpose)
# ---------------------------------------------------------------------------


def test_a_malformed_entity_entry_drops_only_itself() -> None:
    """offsets with the wrong arity are a per-entry defect, not a per-row one: the CD80 span drops
    while its four well-formed siblings still ship (measured malformed rate: 0, guard anyway)"""
    Offsets: list[Any] = list(ROW_16357751["entities"]["offsets"])
    Offsets[2] = [805]  # T3 CD80 loses its end offset
    Example: TrainingExample = SCRIPT.run(row_16357751(entities=entities_with(offsets=Offsets)))

    Surfaces: list[str] = [mention for entity in Example.entities for mention in entity.mentions]
    assert len(Surfaces) == 4
    assert "CD80" not in Surfaces
    assert "CD86" in Surfaces


def test_bool_and_non_int_offsets_drop_the_entry() -> None:
    """bool is an int subclass in python; a True offset is schema corruption, not span 1, and a
    str offset can never slice tokens, so both entries drop while the surviving three ship"""
    Offsets: list[Any] = list(ROW_16357751["entities"]["offsets"])
    Offsets[2] = [True, 809]  # T3 CD80
    Offsets[3] = [810, "814"]  # T4 CD86
    Example: TrainingExample = SCRIPT.run(row_16357751(entities=entities_with(offsets=Offsets)))

    Surfaces: list[str] = [mention for entity in Example.entities for mention in entity.mentions]
    assert len(Surfaces) == 3
    assert "CD80" not in Surfaces and "CD86" not in Surfaces
    assert "methotrexate" in Surfaces


def test_a_non_string_entity_id_or_type_drops_the_entry() -> None:
    """the transpose requires str ids and types to build the id-keyed struct map; a drifted int
    type for T2 drops that span only and never becomes a coercion or a crash"""
    Types: list[Any] = list(ROW_16357751["entities"]["type"])
    Types[1] = 3  # T2 tumor necrosis factor
    Example: TrainingExample = SCRIPT.run(row_16357751(entities=entities_with(type=Types)))

    Surfaces: list[str] = [mention for entity in Example.entities for mention in entity.mentions]
    assert len(Surfaces) == 4
    assert "tumor necrosis factor" not in Surfaces
    assert "CD28 receptor" in Surfaces


def test_a_ragged_entities_table_yields_zero_entities() -> None:
    """parallel lists must stay parallel; a ragged table has no positional meaning, so trusting any
    prefix would invent surfaces that were never gold. The row still ships its text (0 measured,
    guard anyway)"""
    Types: list[str] = list(ROW_16357751["entities"]["type"])[:-1]  # drop the last element
    Example: TrainingExample = SCRIPT.run(row_16357751(entities=entities_with(type=Types)))

    assert Example.text != ""
    assert Example.entities == []
    assert Example.relations == []
    assert Example.populated() == frozenset()


def test_a_non_dict_entities_value_yields_a_text_only_row() -> None:
    """the hub auto-conversion delivers entities as a dict of parallel lists; any other shape yields
    zero entities without a crash and the text-only row falls to the declared-outputs filter"""
    for Malformed in (None, "entities", 7, ["T1"]):
        Example: TrainingExample = SCRIPT.run(row_16357751(entities=Malformed))

        assert Example.text != ""
        assert Example.entities == []
        assert Example.populated() == frozenset()


def test_an_entities_table_missing_a_required_key_yields_zero_entities() -> None:
    """id/type/text/offsets are all required to rebuild a span; a partial table must not crash the
    stream worker and ships no entities"""
    Partial: dict[str, Any] = {"id": ROW_16357751["entities"]["id"], "type": ROW_16357751["entities"]["type"]}
    Example: TrainingExample = SCRIPT.run(row_16357751(entities=Partial))

    assert Example.text != ""
    assert Example.entities == []


def test_out_of_bounds_and_reversed_offsets_drop_per_span() -> None:
    """the bridge bounds-checks raw spans before any snapping: an end past the text extent and an
    inverted start/end point at nothing, so each drops individually while the three in-bounds
    siblings ship (the per-span, not per-row, filtering contract)"""
    Offsets: list[Any] = list(ROW_16357751["entities"]["offsets"])
    Offsets[2] = [len(ROW_16357751["text"]) + 50, len(ROW_16357751["text"]) + 54]  # T3 CD80 out of bounds
    Offsets[3] = [814, 810]  # T4 CD86 reversed
    Example: TrainingExample = SCRIPT.run(row_16357751(entities=entities_with(offsets=Offsets)))

    Surfaces: list[str] = [mention for entity in Example.entities for mention in entity.mentions]
    assert len(Surfaces) == 3
    assert "CD80" not in Surfaces and "CD86" not in Surfaces
    assert "methotrexate" in Surfaces


def test_zero_surviving_spans_ship_text_only() -> None:
    """when every span fails the bridge the row degrades to its rejoined text with no shapes; the
    declared-outputs filter drops it downstream (measured rate: 0, guard anyway)"""
    Offsets: list[Any] = [[len(ROW_16357751["text"]) + 10, len(ROW_16357751["text"]) + 14] for _ in ROW_16357751["entities"]["id"]]
    Example: TrainingExample = SCRIPT.run(row_16357751(entities=entities_with(offsets=Offsets)))

    assert Example.text != ""
    assert Example.entities == []
    assert Example.relations == []
    assert Example.populated() == frozenset()


# ---------------------------------------------------------------------------
# relation-table drop rules (skip-don't-coerce transpose)
# ---------------------------------------------------------------------------


def test_a_non_dict_relations_value_keeps_the_entities() -> None:
    """relations drift must never cost the row its entities: a non-dict table means zero relations
    and the five gold entities still ship"""
    Example: TrainingExample = SCRIPT.run(row_14967461(relations="relations"))

    assert len(Example.relations) == 0
    assert sum(len(entity.mentions) for entity in Example.entities) > 0
    assert Example.populated() == frozenset({"entities"})


def test_a_ragged_relations_table_keeps_the_entities() -> None:
    """parallel-list discipline applies to relations too: a ragged table has no positional meaning,
    so all relations drop rather than trusting a prefix, and the entities still ship"""
    Arg2: list[str] = list(ROW_14967461["relations"]["arg2"])[:-1]
    Example: TrainingExample = SCRIPT.run(row_14967461(relations=relations_with(arg2=Arg2)))

    assert Example.relations == []
    assert sum(len(entity.mentions) for entity in Example.entities) > 0


def test_empty_relations_ship_the_entities_only() -> None:
    """602 measured rows carry entities with zero relations under the permitted-shapes contract
    (producing fewer shapes than declared is fine); the verbatim 16357751 row is one of them and
    its five entities must still train"""
    Example: TrainingExample = SCRIPT.run(row_16357751())

    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "ChemicalEntity": ["methotrexate"],
        "GeneFamily": ["anti-tumor necrosis factor", "CD28 receptor"],
        "Gene": ["CD80", "CD86"],
    }
    assert Example.relations == []
    assert Example.populated() == frozenset({"entities"})


# ---------------------------------------------------------------------------
# gold-predicate drop rules (each defect mutates one verbatim CPR:4 relation)
# ---------------------------------------------------------------------------


def test_a_dangling_arg_id_drops_only_that_relation() -> None:
    """a relation whose arg id has no surviving span references nothing trainable; it drops while
    the 16 well-formed siblings still ship (0 measured dangling args, guard anyway)"""
    Arg1: list[Any] = list(ROW_14967461["relations"]["arg1"])
    Arg1[0] = "T99"
    Example: TrainingExample = SCRIPT.run(row_14967461(relations=relations_with(arg1=Arg1)))

    assert len(Example.relations) == 16


def test_a_self_loop_relation_drops() -> None:
    """the corpus's direction is always chemical -> gene, so a relation whose head and tail resolve
    to the same surface (here via the duplicated PKI166 span T10) carries no signal and drops
    under the sentence_rex case-insensitive self-loop rule; the 16 siblings ship"""
    Arg2: list[Any] = list(ROW_14967461["relations"]["arg2"])
    Arg2[0] = "T10"  # arg1[0] is also T10 (CI-1033), so both fields resolve to the same surface
    Example: TrainingExample = SCRIPT.run(row_14967461(relations=relations_with(arg2=Arg2)))

    assert len(Example.relations) == 16
    for relation in Example.relations:
        assert relation.fields[0].value.lower() != relation.fields[1].value.lower()


def test_a_non_string_relation_field_drops_the_relation() -> None:
    """a drifted non-str type or arg cannot key PREDICATE_MAP or the surface map; the relation
    drops instead of crashing the stream worker, siblings ship"""
    Types: list[Any] = list(ROW_14967461["relations"]["type"])
    Types[0] = 4
    Example: TrainingExample = SCRIPT.run(row_14967461(relations=relations_with(type=Types)))

    assert len(Example.relations) == 16


def test_a_cpr0_relation_drops_but_the_row_entities_still_ship() -> None:
    """CPR:0 is the loader's own Undefined flag (3 measured relations); the relation drops because
    an undefined label must never train, while the row's 29 gold entities still ship"""
    Types: list[Any] = list(ROW_14967461["relations"]["type"])
    Types[0] = "CPR:0"
    Example: TrainingExample = SCRIPT.run(row_14967461(relations=relations_with(type=Types)))

    assert len(Example.relations) == 16
    assert {relation.name for relation in Example.relations} == {"decreases_amount_or_activity_of"}
    assert len(ChemprotScript.entity_structs(ROW_14967461["entities"])) == 29
    assert sum(len(entity.mentions) for entity in Example.entities) > 0


def test_a_cpr10_relation_drops_because_the_pipeline_never_asserts_negations() -> None:
    """CPR:10 is the corpus's asserted no-interaction label (683 measured relations); models.py
    pins negated=False on every emitted relation, so shipping CPR:10 would break that landed
    invariant. The relation drops, the entities still ship"""
    Types: list[Any] = list(ROW_14967461["relations"]["type"])
    Types[0] = "CPR:10"
    Example: TrainingExample = SCRIPT.run(row_14967461(relations=relations_with(type=Types)))

    assert len(Example.relations) == 16
    assert all(relation.negated is False for relation in Example.relations)


def test_an_unknown_cpr_label_drops_the_relation() -> None:
    """labels outside the measured CPR:1..CPR:10 vocabulary are defensive-drift territory; the
    skip-don't-coerce rule drops them rather than guessing a predicate (never train garbage)"""
    for Unknown in ("CPR:11", "CPR:", "INTERACTS_WITH"):
        Types: list[Any] = list(ROW_14967461["relations"]["type"])
        Types[0] = Unknown
        Example: TrainingExample = SCRIPT.run(row_14967461(relations=relations_with(type=Types)))

        assert len(Example.relations) == 16, Unknown


# ---------------------------------------------------------------------------
# predicate-map edges
# ---------------------------------------------------------------------------


def test_cpr9_maps_to_a_biolink_member_predicate() -> None:
    """CPR:9 Substrate is one of the 7 labels with an honest biolink slot; the mapped member keeps
    its slot definition as the relation description (evidence the predicate is a real member)"""
    Types: list[Any] = list(ROW_14967461["relations"]["type"])
    Types[0] = "CPR:9"
    Example: TrainingExample = SCRIPT.run(row_14967461(relations=relations_with(type=Types)))

    Relation0 = Example.relations[0]  # relation 0 survives the map (position 0 of the emit order)
    assert Relation0.name == "is_substrate_of"
    assert Relation0.description is not None
    assert Relation0.evidence == "asserted"
    assert Relation0.negated is False


def test_cpr7_and_cpr8_stay_native_snake_case_predicates() -> None:
    """biolink has no slot for CPR:7 Modulator or CPR:8 Cofactor, and regulates/has_catalyst would
    overcommit a direction or level the corpus leaves unspecified; both ship as documented native
    snake_case omissions with no slot description (the README carries the omission)"""
    Types: list[Any] = list(ROW_14967461["relations"]["type"])
    Types[0] = "CPR:7"
    Types[1] = "CPR:8"
    Example: TrainingExample = SCRIPT.run(row_14967461(relations=relations_with(type=Types)))

    assert Example.relations[0].name == "modulator"
    assert Example.relations[0].description is None
    assert Example.relations[0].fields[1].value == "pan-erbB"
    assert Example.relations[1].name == "cofactor"
    assert Example.relations[1].description is None


# ---------------------------------------------------------------------------
# label-map edges (the autouse no_fullmap fixture pins the fallback tier)
# ---------------------------------------------------------------------------


def test_chemical_resolves_through_the_lowercased_label_map() -> None:
    """the corpus spells labels UPPERCASE (CHEMICAL) while the map keys are lowercase; the shared
    lookup probes the lowercased label first, so the verbatim row groups under ChemicalEntity"""
    Example: TrainingExample = SCRIPT.run(row_16357751())

    assert {entity.label: entity.mentions for entity in Example.entities}["ChemicalEntity"] == ["methotrexate"]


def test_a_whitespace_variant_resolves_through_the_normalizing_fallback() -> None:
    """the lookup's second probe runs normalize_iob_label, so a trailing-space drift variant of a
    mapped label ('chemical ') still reaches ChemicalEntity without a dedicated map entry"""
    Types: list[Any] = list(ROW_16357751["entities"]["type"])
    Types[0] = "chemical "
    Example: TrainingExample = SCRIPT.run(row_16357751(entities=entities_with(type=Types)))

    assert {entity.label: entity.mentions for entity in Example.entities}["ChemicalEntity"] == ["methotrexate"]


def test_an_unmapped_label_ships_pascalcased_raw() -> None:
    """labels outside the measured 3-class census have no honest biolink target and must not be
    guessed; the raw-tail rule surfaces them biolink-cased (DRIFTTYPE -> Drifttype) so zero-shot
    vocabulary stays trainable under a uniform naming scheme"""
    Types: list[Any] = list(ROW_16357751["entities"]["type"])
    Types[0] = "DRIFTTYPE"
    Example: TrainingExample = SCRIPT.run(row_16357751(entities=entities_with(type=Types)))

    ByLabel = {entity.label: entity.mentions for entity in Example.entities}
    assert ByLabel["Drifttype"] == ["methotrexate"]
    assert "ChemicalEntity" not in ByLabel


# ---------------------------------------------------------------------------
# nonzero yield over the verbatim rows (the silent-zero-yield guard)
# ---------------------------------------------------------------------------


def test_the_verbatim_cpr4_row_yields_29_entities_and_17_decreases_relations() -> None:
    """the census-measured nonzero-yield anchor: all 29 gold entity structs survive the transpose
    and bridge, group to 24 mentions (5 surfaces repeat within one label and dedupe on grouping),
    and all 17 gold CPR:4 antagonist relations emit as decreases_amount_or_activity_of with
    head=arg1 (chemical) and tail=arg2 (gene) surfaces that are substrings of the emitted text"""
    Example: TrainingExample = SCRIPT.run(row_14967461())

    assert len(ChemprotScript.entity_structs(ROW_14967461["entities"])) == 29
    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "ChemicalEntity": [
            "gefitinib",
            "Iressa",
            "ZD1839",
            "erlotinib",
            "Tarceva",
            "OSI-774",
            "cetuximab",
            "IMC-225",
            "Erbitux",
            "CI-1033",
            "tyrosine",
            "PKI166",
            "GW572016",
            "platinum-based",
            "taxanes",
            "vinorelbine",
            "gemcitabine",
            "irinotecan",
        ],
        "Gene": [
            "Epidermal growth factor receptor",
            "pan-erbB",
            "epidermal growth factor receptor",
            "Her2",
        ],
        "GeneFamily": ["tyrosine kinase", "kinase"],
    }
    assert sum(len(entity.mentions) for entity in Example.entities) == 24
    assert len(Example.relations) == 17
    assert {relation.name for relation in Example.relations} == {"decreases_amount_or_activity_of"}
    for relation in Example.relations:
        assert relation.fields[0].value in Example.text  # gliner2 rule: head surface occurs in text
        assert relation.fields[1].value in Example.text  # gliner2 rule: tail surface occurs in text
    assert Example.populated() == frozenset({"entities", "relations"})


def test_both_verbatim_rows_ship_nonzero_examples_with_every_surface_in_text() -> None:
    """silent zero yield is the worst failure mode for a scripted ingest (the stream looks healthy
    while training on nothing); both census rows must emit populated examples, and every emitted
    mention plus every relation head/tail must occur in the emitted text (the gliner2 substring
    invariant the trainer relies on)"""
    for Values in (row_16357751(), row_14967461()):
        Example: TrainingExample = SCRIPT.run(Values)

        assert Example.text != ""
        assert Example.entities != []
        for entity in Example.entities:
            for mention in entity.mentions:
                assert mention in Example.text
        for relation in Example.relations:
            assert relation.fields[0].value in Example.text
            assert relation.fields[1].value in Example.text


# ---------------------------------------------------------------------------
# dispatch + registry
# ---------------------------------------------------------------------------


def test_dispatch_routes_through_the_registry_and_keeps_declared_outputs() -> None:
    """the ingest invokes Script.dispatch with the declared outputs tuple; the script must answer
    through that path, not only via a direct instance call"""
    Outputs, Example = Script.dispatch("ChemprotScript", (("entities", "relations"), row_14967461()))

    assert Outputs == ("entities", "relations")
    assert isinstance(Example, TrainingExample)
    assert len(Example.relations) == 17
    assert Example.populated() == frozenset({"entities", "relations"})


def test_the_existing_registry_is_unaffected_by_the_new_script() -> None:
    """all edits are additive: the KnowledgatorBiomedScript entry must survive the new import"""
    assert isinstance(Script.REGISTRY["KnowledgatorBiomedScript"], KnowledgatorBiomedScript)
