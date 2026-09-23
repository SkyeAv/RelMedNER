from __future__ import annotations

from typing import Any

import pytest

from relmedner.families import validate_label_map
from relmedner.models import TrainingExample
from relmedner.scripts import ChemprotScript, DrugprotScript
from relmedner.scripts.drugprot import _validate_predicate_map
from relmedner.types import Script
from relmedner.utils import ScriptUtils

SCRIPT: DrugprotScript = DrugprotScript()

# verbatim OpenMed/drugprot-parquet train rows captured by the wenceslaus probe
# (wenceslaus ~/drugprot-rows.json, the first two streaming rows); every negative test below
# mutates one of these real rows so no test trains on an invented row shape
ROW_17512723: dict[str, Any] = {
    "pmid": "17512723",
    "title": "RDH12, a retinol dehydrogenase causing Leber's congenital amaurosis, is also involved in steroid metabolism.",
    "abstract": "Three retinol dehydrogenases (RDHs) were tested for steroid converting abilities: human and murine RDH "
    "12 and human RDH13. RDH12 is involved in retinal degeneration in Leber's congenital amaurosis (LCA). "
    "We show that murine Rdh12 and human RDH13 do not reveal activity towards the checked steroids, but "
    "that human type 12 RDH reduces dihydrotestosterone to androstanediol, and is thus also involved in "
    "steroid metabolism. Furthermore, we analyzed both expression and subcellular localization of these "
    "enzymes.",
    "text": "RDH12, a retinol dehydrogenase causing Leber's congenital amaurosis, is also involved in steroid "
    "metabolism. Three retinol dehydrogenases (RDHs) were tested for steroid converting abilities: human and "
    "murine RDH 12 and human RDH13. RDH12 is involved in retinal degeneration in Leber's congenital amaurosis "
    "(LCA). We show that murine Rdh12 and human RDH13 do not reveal activity towards the checked steroids, but "
    "that human type 12 RDH reduces dihydrotestosterone to androstanediol, and is thus also involved in steroid "
    "metabolism. Furthermore, we analyzed both expression and subcellular localization of these enzymes.",
    "entities": [
        {"id": "T1", "type": "CHEMICAL", "text": "androstanediol", "start": 466, "end": 480},
        {"id": "T2", "type": "CHEMICAL", "text": "retinol", "start": 115, "end": 122},
        {"id": "T3", "type": "CHEMICAL", "text": "retinol", "start": 9, "end": 16},
        {"id": "T4", "type": "GENE-Y", "text": "human RDH13", "start": 219, "end": 230},
        {"id": "T5", "type": "GENE-Y", "text": "RDH12", "start": 232, "end": 237},
        {"id": "T6", "type": "GENE-Y", "text": "murine Rdh12", "start": 326, "end": 338},
        {"id": "T7", "type": "GENE-Y", "text": "human RDH13", "start": 343, "end": 354},
        {"id": "T8", "type": "GENE-N", "text": "RDHs", "start": 139, "end": 143},
        {"id": "T9", "type": "GENE-Y", "text": "human type 12 RDH", "start": 417, "end": 434},
        {"id": "T10", "type": "GENE-N", "text": "retinol dehydrogenases", "start": 115, "end": 137},
        {"id": "T11", "type": "GENE-N", "text": "human and murine RDH 12", "start": 191, "end": 214},
        {"id": "T12", "type": "GENE-Y", "text": "RDH12", "start": 0, "end": 5},
        {"id": "T13", "type": "GENE-N", "text": "retinol dehydrogenase", "start": 9, "end": 30},
    ],
    "relations": [{"type": "PRODUCT-OF", "arg1": "T1", "arg2": "T9"}],
}


ROW_23557993: dict[str, Any] = {
    "pmid": "23557993",
    "title": "A diarylheptanoid phytoestrogen from Curcuma comosa, 1,7-diphenyl-4,6-heptadien-3-ol, accelerates human "
    "osteoblast proliferation and differentiation.",
    "abstract": "Curcuma comosa Roxb. is ginger-family plant used to relieve menopausal symptoms. Previous work showed "
    "that C. comosa extracts protect mice from ovariectomy-induced osteopenia with minimal effects on "
    "reproductive organs, and identified the diarylheptanoid (3R)-1,7-diphenyl-(4E,6E)-4,6-heptadien-3-ol "
    "(DPHD) as the major active component of C. comosa rhizomes. At 1-10μM, DPHD increased differentiation "
    "in transformed mouse osteoblasts, but the effect of DPHD on normal bone cells was unknown. We examined "
    "the concentration dependency and mechanism of action of DPHD relative to 17β-estradiol in "
    "nontransformed human osteoblasts (h-OB). The h-OB were 10-100 fold more sensitive to DPHD than "
    "transformed osteoblasts: DPHD increased h-OB proliferation at 10nM and, at 100nM, activated MAP kinase "
    "signaling within 30min. In long-term differentiation assays, responses of h-OB to DPHD were "
    "significant at 10nM, and optimal response in most cases was at 100nM. At 7-21 days, DPHD accelerated "
    "osteoblast differentiation, indicated by alkaline phosphatase activity and osteoblast-specific mRNA "
    "production. Effects of DPHD were eliminated by the estrogen receptor antagonist ICI182780. During "
    "differentiation, DPHD promoted early expression of osteoblast transcription factors, RUNX2 and "
    "osterix. Subsequently, DPHD accelerated production of bone structural genes, including COL1A1 and "
    "osteocalcin comparably to 17β-estradiol. In h-OB, DPHD increased the osteoprotegerin to RANKL ratio "
    "and supported mineralization more efficiently than 10nM 17β-estradiol. We conclude that DPHD promotes "
    "human osteoblast function in vitro effectively at nanomolar concentrations, making it a promising "
    "compound to protect bone in menopausal women.",
    "text": "A diarylheptanoid phytoestrogen from Curcuma comosa, 1,7-diphenyl-4,6-heptadien-3-ol, accelerates human "
    "osteoblast proliferation and differentiation. Curcuma comosa Roxb. is ginger-family plant used to relieve "
    "menopausal symptoms. Previous work showed that C. comosa extracts protect mice from ovariectomy-induced "
    "osteopenia with minimal effects on reproductive organs, and identified the diarylheptanoid "
    "(3R)-1,7-diphenyl-(4E,6E)-4,6-heptadien-3-ol (DPHD) as the major active component of C. comosa rhizomes. "
    "At 1-10μM, DPHD increased differentiation in transformed mouse osteoblasts, but the effect of DPHD on "
    "normal bone cells was unknown. We examined the concentration dependency and mechanism of action of DPHD "
    "relative to 17β-estradiol in nontransformed human osteoblasts (h-OB). The h-OB were 10-100 fold more "
    "sensitive to DPHD than transformed osteoblasts: DPHD increased h-OB proliferation at 10nM and, at 100nM, "
    "activated MAP kinase signaling within 30min. In long-term differentiation assays, responses of h-OB to "
    "DPHD were significant at 10nM, and optimal response in most cases was at 100nM. At 7-21 days, DPHD "
    "accelerated osteoblast differentiation, indicated by alkaline phosphatase activity and osteoblast-specific "
    "mRNA production. Effects of DPHD were eliminated by the estrogen receptor antagonist ICI182780. During "
    "differentiation, DPHD promoted early expression of osteoblast transcription factors, RUNX2 and osterix. "
    "Subsequently, DPHD accelerated production of bone structural genes, including COL1A1 and osteocalcin "
    "comparably to 17β-estradiol. In h-OB, DPHD increased the osteoprotegerin to RANKL ratio and supported "
    "mineralization more efficiently than 10nM 17β-estradiol. We conclude that DPHD promotes human osteoblast "
    "function in vitro effectively at nanomolar concentrations, making it a promising compound to protect bone "
    "in menopausal women.",
    "entities": [
        {"id": "T1", "type": "CHEMICAL", "text": "DPHD", "start": 1259, "end": 1263},
        {"id": "T2", "type": "CHEMICAL", "text": "estrogen", "start": 1287, "end": 1295},
        {"id": "T3", "type": "CHEMICAL", "text": "ICI182780", "start": 1316, "end": 1325},
        {"id": "T4", "type": "CHEMICAL", "text": "DPHD", "start": 1351, "end": 1355},
        {"id": "T5", "type": "CHEMICAL", "text": "DPHD", "start": 1452, "end": 1456},
        {"id": "T6", "type": "CHEMICAL", "text": "17β-estradiol", "start": 1553, "end": 1566},
        {"id": "T7", "type": "CHEMICAL", "text": "DPHD", "start": 1577, "end": 1581},
        {"id": "T8", "type": "CHEMICAL", "text": "17β-estradiol", "start": 1683, "end": 1696},
        {"id": "T9", "type": "CHEMICAL", "text": "DPHD", "start": 1715, "end": 1719},
        {"id": "T10", "type": "CHEMICAL", "text": "diarylheptanoid", "start": 389, "end": 404},
        {"id": "T11", "type": "CHEMICAL", "text": "(3R)-1,7-diphenyl-(4E,6E)-4,6-heptadien-3-ol", "start": 405, "end": 449},
        {"id": "T12", "type": "CHEMICAL", "text": "DPHD", "start": 451, "end": 455},
        {"id": "T13", "type": "CHEMICAL", "text": "DPHD", "start": 521, "end": 525},
        {"id": "T14", "type": "CHEMICAL", "text": "DPHD", "start": 604, "end": 608},
        {"id": "T15", "type": "CHEMICAL", "text": "DPHD", "start": 711, "end": 715},
        {"id": "T16", "type": "CHEMICAL", "text": "17β-estradiol", "start": 728, "end": 741},
        {"id": "T17", "type": "CHEMICAL", "text": "DPHD", "start": 830, "end": 834},
        {"id": "T18", "type": "CHEMICAL", "text": "DPHD", "start": 865, "end": 869},
        {"id": "T19", "type": "CHEMICAL", "text": "DPHD", "start": 1025, "end": 1029},
        {"id": "T20", "type": "CHEMICAL", "text": "DPHD", "start": 1119, "end": 1123},
        {"id": "T21", "type": "CHEMICAL", "text": "diarylheptanoid", "start": 2, "end": 17},
        {"id": "T22", "type": "CHEMICAL", "text": "1,7-diphenyl-4,6-heptadien-3-ol", "start": 53, "end": 84},
        {"id": "T23", "type": "GENE-N", "text": "alkaline phosphatase", "start": 1177, "end": 1197},
        {"id": "T24", "type": "GENE-Y", "text": "estrogen receptor", "start": 1287, "end": 1304},
        {"id": "T25", "type": "GENE-Y", "text": "RUNX2", "start": 1419, "end": 1424},
        {"id": "T26", "type": "GENE-Y", "text": "osterix", "start": 1429, "end": 1436},
        {"id": "T27", "type": "GENE-Y", "text": "COL1A1", "start": 1516, "end": 1522},
        {"id": "T28", "type": "GENE-Y", "text": "osteocalcin", "start": 1527, "end": 1538},
        {"id": "T29", "type": "GENE-Y", "text": "osteoprotegerin", "start": 1596, "end": 1611},
        {"id": "T30", "type": "GENE-Y", "text": "RANKL", "start": 1615, "end": 1620},
        {"id": "T31", "type": "GENE-N", "text": "MAP kinase", "start": 932, "end": 942},
    ],
    "relations": [
        {"type": "ANTAGONIST", "arg1": "T3", "arg2": "T24"},
        {"type": "ACTIVATOR", "arg1": "T18", "arg2": "T31"},
        {"type": "ACTIVATOR", "arg1": "T1", "arg2": "T24"},
        {"type": "INDIRECT-UPREGULATOR", "arg1": "T4", "arg2": "T25"},
        {"type": "INDIRECT-UPREGULATOR", "arg1": "T4", "arg2": "T26"},
        {"type": "INDIRECT-UPREGULATOR", "arg1": "T5", "arg2": "T27"},
        {"type": "INDIRECT-UPREGULATOR", "arg1": "T5", "arg2": "T28"},
    ],
}


@pytest.fixture(autouse=True)
def no_fullmap(monkeypatch: pytest.MonkeyPatch) -> None:
    """label-map tests pin the deterministic fallback path: fullmap resolution is environment-
    dependent (mounted DB vs not), so it is stubbed to a miss and the LABEL_MAP fallback tier
    is what every label assertion in this file exercises"""
    monkeypatch.setattr(ScriptUtils, "_fullmap_best", classmethod(lambda cls, normalized: {}))


def row_17512723(**overrides: Any) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """(row with field overrides) -> the (text, entities, relations) values tuple run() unpacks;
    the override kwargs keep every negative test one mutation away from a real gold row"""
    text = overrides.get("text", ROW_17512723["text"])
    entities = overrides.get("entities", ROW_17512723["entities"])
    relations = overrides.get("relations", ROW_17512723["relations"])
    return text, entities, relations


def entities_with(**field_sets: Any) -> list[dict[str, Any]]:
    """rebuild ROW_17512723's entity list with one field replaced across every entry, so a
    malformed-value test can poison exactly the field it names"""
    key: str = next(iter(field_sets))
    values: Any = field_sets[key]
    return [{**entry, key: values[index % len(values)]} for index, entry in enumerate(ROW_17512723["entities"])]


def surfaces_of(example: TrainingExample) -> list[str]:
    """every emitted mention surface, what the containment and count assertions read"""
    return [mention for entity in example.entities for mention in entity.mentions]


# ---------------------------------------------------------------------------
# import-time guards (the module-scope validation must fail loudly on drift)
# ---------------------------------------------------------------------------


def test_the_script_self_registers_under_its_declared_name() -> None:
    """the yaml task.name resolves through Script.REGISTRY, which __init_subclass__ populates via
    Script.dispatch, which resolves by this exact NAME key"""
    assert isinstance(Script.REGISTRY["DrugprotScript"], DrugprotScript)


def test_every_mapped_label_and_predicate_is_accounted_for() -> None:
    """the map is the contract the census measured; the import-time guards are re-asserted here
    for drift: label-map values stay biolink classes and every predicate-map value is either a
    tablassert Predicates member or a documented native snake_case omission"""
    for raw_label, category in DrugprotScript.LABEL_MAP.items():
        assert ScriptUtils.is_biolink_category(category), f"fallback {raw_label!r} -> {category!r} is not a biolink class"
    members: frozenset[str] = ScriptUtils.biolink_predicates()
    for label, predicate in DrugprotScript.PREDICATE_MAP.items():
        assert predicate in members or predicate in DrugprotScript.NATIVE_PREDICATES, f"{label!r} -> {predicate!r} is unguarded"


def test_the_import_time_label_guard_rejects_a_non_biolink_class() -> None:
    """a drifted LABEL_MAP value must fail at import rather than mislabel training data, so the
    guard this pins is the last line of defense before a wrong class ships"""
    with pytest.raises(ValueError, match="not a biolink class"):
        validate_label_map({"chemical": "NotACategory"}, "DrugprotScript")


def test_the_import_time_predicate_guard_rejects_an_unmapped_predicate() -> None:
    """a typo'd or invented predicate must fail at import rather than train a garbage relation"""
    original = dict(DrugprotScript.PREDICATE_MAP)
    DrugprotScript.PREDICATE_MAP["ACTIVATOR"] = "not_a_predicate"
    try:
        with pytest.raises(ValueError, match="documented native predicate"):
            _validate_predicate_map()
    finally:
        DrugprotScript.PREDICATE_MAP.clear()
        DrugprotScript.PREDICATE_MAP.update(original)


# ---------------------------------------------------------------------------
# text-column drop rules
# ---------------------------------------------------------------------------


def test_a_non_string_text_value_produces_the_empty_example() -> None:
    """a drifted text column must not raise (skip-don't-coerce); the declared-outputs filter
    drops the empty example downstream"""
    example: TrainingExample = SCRIPT.run((12345, ROW_17512723["entities"], ROW_17512723["relations"]))
    assert example.text == ""
    assert example.populated() == frozenset()


def test_an_empty_text_produces_the_empty_example() -> None:
    """an empty text cannot anchor any gold offset, so the row ships nothing rather than guessing
    a re-derivation"""
    example: TrainingExample = SCRIPT.run(("", ROW_17512723["entities"], ROW_23557993["relations"]))
    assert example.text == ""
    assert example.populated() == frozenset()


# ---------------------------------------------------------------------------
# entity-table drop rules (each poisons one field of one entry; siblings survive)
# ---------------------------------------------------------------------------


def test_a_malformed_entity_entry_drops_only_itself() -> None:
    """a non-dict entry among gold entities drops exactly that entity: the well-formed siblings
    still ship (skip-don't-coerce measured at 0 defective entries; the guard is defensive)"""
    poisoned: list[Any] = [*ROW_17512723["entities"][:2], "not-a-dict", *ROW_17512723["entities"][2:]]
    example: TrainingExample = SCRIPT.run(row_17512723(entities=poisoned))
    surfaces: list[str] = surfaces_of(example)
    assert len(surfaces) == 10
    assert "androstanediol" in surfaces


def test_bool_and_non_int_offsets_drop_the_entry() -> None:
    """bool is an int subclass, so True would otherwise coerce to offset 1 and fabricate a span;
    both the bool and the float poison their entry while the gold siblings survive"""
    starts: list[Any] = [entry["start"] for entry in ROW_17512723["entities"]]
    starts[0] = True
    starts[1] = 46.5
    example: TrainingExample = SCRIPT.run(row_17512723(entities=entities_with(start=starts)))
    surfaces: list[str] = surfaces_of(example)
    assert len(surfaces) == 9
    assert "androstanediol" not in surfaces
    assert "retinol" in surfaces


def test_a_non_string_entity_id_or_type_drops_the_entry() -> None:
    """the id keys the relation join and the type keys the label map: a non-string in either slot
    drops the entry rather than coercing a key that cannot be trusted"""
    types: list[Any] = [entry["type"] for entry in ROW_17512723["entities"]]
    types[0] = None
    example: TrainingExample = SCRIPT.run(row_17512723(entities=entities_with(type=types)))
    surfaces: list[str] = surfaces_of(example)
    assert len(surfaces) == 9
    assert "androstanediol" not in surfaces


def test_a_non_list_entities_value_yields_a_text_only_row() -> None:
    """a drifted entity column degrades to text-only; the declared-outputs filter drops the row
    because no declared shape was produced"""
    example: TrainingExample = SCRIPT.run(row_17512723(entities="not-a-list"))
    assert example.text != ""
    assert example.entities == []
    assert example.populated() == frozenset()


# ---------------------------------------------------------------------------
# relation-table drop rules
# ---------------------------------------------------------------------------


def test_a_relation_self_loop_drops_only_the_relation() -> None:
    """a self-loop would fabricate head == tail (0 measured; defensive): the relation drops, the
    row's entities still ship"""
    looped: list[dict[str, Any]] = [{"type": "INHIBITOR", "arg1": "T1", "arg2": "T1"}]
    example: TrainingExample = SCRIPT.run(row_17512723(relations=looped))
    assert example.relations == []
    assert len(surfaces_of(example)) == 10


def test_a_dangling_relation_arg_drops_the_relation() -> None:
    """an arg id absent from the well-formed entity table cannot anchor a surface (0 measured;
    defensive): the relation drops, never guesses a tail"""
    dangling: list[dict[str, Any]] = [{"type": "INHIBITOR", "arg1": "T1", "arg2": "T99"}]
    example: TrainingExample = SCRIPT.run(row_17512723(relations=dangling))
    assert example.relations == []
    assert len(surfaces_of(example)) == 10


def test_an_unmapped_relation_type_drops_the_relation() -> None:
    """an unknown gold label must not train a guessed predicate (the drift guard): the relation
    drops and the row's entities still ship"""
    unknown: list[dict[str, Any]] = [{"type": "FUTURE-TYPE", "arg1": "T1", "arg2": "T9"}]
    example: TrainingExample = SCRIPT.run(row_17512723(relations=unknown))
    assert example.relations == []
    assert len(surfaces_of(example)) == 10


def test_a_relation_whose_arg_did_not_survive_the_bridge_drops() -> None:
    """an out-of-bounds entity span dies in the char-offset bridge (0 measured; defensive), and a
    relation anchored on it must not outlive its span: the relation drops, the surviving entities
    still ship"""
    broken: list[dict[str, Any]] = [
        {**ROW_17512723["entities"][0], "start": 999_999, "end": 1_000_000},
        *ROW_17512723["entities"][1:],
    ]
    example: TrainingExample = SCRIPT.run(row_17512723(entities=broken))
    surfaces: list[str] = surfaces_of(example)
    assert len(surfaces) == 9
    assert "androstanediol" not in surfaces
    assert example.relations == []


def test_a_relations_free_row_still_ships_entities() -> None:
    """1,275 of 4,250 measured rows carry entities with zero relations and ship entities-only
    under the permitted-shapes contract, with no code path special-casing them"""
    example: TrainingExample = SCRIPT.run(row_17512723(relations=[]))
    assert len(surfaces_of(example)) == 10
    assert example.relations == []
    assert example.populated() == frozenset({"entities"})


# ---------------------------------------------------------------------------
# the gold rows (nonzero-yield + containment over verbatim data)
# ---------------------------------------------------------------------------


def test_both_verbatim_rows_ship_nonzero_examples_with_every_surface_in_text() -> None:
    """the nonzero-yield proof over verbatim gold rows: every emitted entity and relation surface
    is a substring of the emitted text (gliner2's InputExample.validate() rule), and every gold
    relation ships under its mapped predicate"""
    for row in (ROW_17512723, ROW_23557993):
        example: TrainingExample = SCRIPT.run((row["text"], row["entities"], row["relations"]))
        surfaces: list[str] = surfaces_of(example)
        # the emitted text must contain every emitted mention plus every relation head/tail (the
        # gliner2 substring invariant); the bridge's mid-token snapping may rejoin a gold surface
        # with separated punctuation, so surfaces are asserted against text, not against gold
        # surface equality
        assert all(surface in example.text for surface in surfaces)
        assert len(example.relations) == len(row["relations"])
        for relation in example.relations:
            values: list[str] = [field.value for field in relation.fields]
            assert all(value in example.text for value in values)
    names: set[str] = {
        relation.name for relation in SCRIPT.run((ROW_23557993["text"], ROW_23557993["entities"], ROW_23557993["relations"])).relations
    }
    assert names <= {mapped for mapped in DrugprotScript.PREDICATE_MAP.values()}


def test_dispatch_routes_through_the_registry_and_keeps_declared_outputs() -> None:
    """the ingest invokes Script.dispatch with the declared outputs tuple; the script must answer
    through that path, not only via a direct instance call"""
    outputs, example = Script.dispatch("DrugprotScript", (("entities", "relations"), row_17512723()))

    assert outputs == ("entities", "relations")
    assert isinstance(example, TrainingExample)
    assert len(example.relations) == 1
    assert example.populated() == frozenset({"entities", "relations"})


def test_the_existing_registry_is_unaffected_by_the_new_script() -> None:
    """all edits are additive: the ChemprotScript entry must survive the new import"""
    assert isinstance(Script.REGISTRY["ChemprotScript"], ChemprotScript)
