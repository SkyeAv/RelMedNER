from __future__ import annotations

import json
from typing import Any

from relmedner.models import Relation, RelationField, TrainingExample
from relmedner.scripts import JsonExtractionScript  # registry must stay populated (US-001 wiring)
from relmedner.scripts.json_extraction import (
    HENRIQUE_SOURCE,
    JIRAYA_SOURCE,
    OWKIN_SOURCE,
    PROFESSOR_BOB_SOURCE,
    ROBOROVSKI_SOURCE,
    SANDEEPPANEM_SOURCE,
    flatten_json,
)
from relmedner.scripts.json_extraction import (
    JsonExtractionScript as JsonExtractionScriptClass,
)
from relmedner.types import Script
from relmedner.utils import ScriptUtils

SCRIPT: JsonExtractionScript = JsonExtractionScript()


def run_row(text: Any, json_value: Any, source: Any = OWKIN_SOURCE) -> TrainingExample:
    """one row as the columns_out projection (text, json, source) delivers it to ScriptValues"""
    return SCRIPT.run((text, json_value, source))


# --- fixtures copied from real hub shapes (datasets-server first rows of agentlans/json-extraction;
# values stitched so the surviving siblings are verbatim-contained, the dropped ones exercise one
# guard each). US-001 docstring carries the full-split census these shapes come from.

OWKIN_TEXT: str = (
    "Randomized trial of the effect of an integrative medicine approach to the management of asthma "
    "in adults on disease-related quality of life and pulmonary function. Participants were adults "
    "aged 18 to 80 years with asthma. The intervention consisted of six group sessions on the use of "
    "nutritional manipulation, yoga techniques, and journaling. Participants also received nutritional "
    "supplements: fish oil, vitamin C, and a standardized hops extract. Primary outcome measures were "
    "the Asthma Quality of Life Questionnaire (AQLQ) and standard pulmonary function tests (PFTs)."
)
OWKIN_DOC: dict[str, Any] = {
    "conditions": ["asthma", "Chronic Obstructive Pulmonary Disease"],
    "interventions": [
        {"name": "fish oil", "type": "Dietary Supplement"},
        {"name": "vitamin C", "type": "Device"},
        {"name": "Integrative Medicine", "type": "Behavioral"},
    ],
}

PROFESSOR_BOB_TEXT: str = (
    "Ontario is one of the thirteen provinces and territories of Canada. Located in Central Canada, "
    "it is Canada's most populous province, and is home to the nation's capital city, Ottawa, and the "
    "nation's most populous city, Toronto, which is Ontario's provincial capital. Wayne Gretzky is a "
    "member of the Hockey Hall of Fame in Toronto."
)
PROFESSOR_BOB_DOC: list[dict[str, str]] = [
    {"subject": "Ontario", "predicate": "member of", "object": "Canada"},
    {"subject": "Wayne Gretzky", "predicate": "member of", "object": "Hockey Hall of Fame"},
    {"subject": "Ontario", "predicate": "instance of", "object": "province of Canada"},
    {"subject": "Ontario", "predicate": "no relation", "object": "Manitoba"},
    {"subject": "Toronto", "predicate": "member of", "object": "Ontario"},
    {"subject": "Ontario", "predicate": "member of", "object": "Northern Canada"},
]

ROBOROVSKI_TEXT: str = (
    "Komorida was born in Kumamoto Prefecture on July 10, 1981. After graduating from high school, "
    "he joined the J1 League club Avispa Fukuoka in 2000. John Moses Browning was an American "
    "firearm designer who developed many varieties of military and civilian firearms."
)
ROBOROVSKI_DOC: dict[str, Any] = {
    "name": "Komorida",
    "birthDate": "1981-07-10",
    "birthPlace": "Kumamoto Prefecture",
    "results": [
        {"name": "John Moses Browning", "entity": "Person", "occupation": "firearm designer"},
        {"name": "Kumamoto Prefecture"},
    ],
}

SANDEEPPANEM_TEXT: str = (
    "FRONT DESK CLERK (FEE BASIS, JOHN D DINGELL VA MEDICAL CENTER)\n"
    "Experience\n"
    "Company Name January 2004 to April 2006 Income Tax Preparer, Jackson Hewitt\n"
    "City , State\n"
    "Prepared income tax returns for 180 clients per tax season via data entry."
)
SANDEEPPANEM_DOC: dict[str, Any] = {
    "current_title": "Front Desk Clerk",
    "previous_titles": ["Income Tax Preparer"],
    "current_company": "Company Name",
    "previous_companies": ["Company Name", "John D Dingell Va Medical Center", "Jackson Hewitt"],
    "years_experience": None,
    "location": None,
    "leadership_experience": True,
    "summary": "Customer-service focused professional with experience in front desk operations.",
}

JIRAYA_TEXT: str = (
    "Showing 26-50 of 1630 results. Software Engineer II Software Development and Engineering "
    "BENTONVILLE, AR 11/11/24 Full Stack Software Engineer Software Development and Engineering "
    "SUNNYVALE, CA 06/13/24 Principal, Software Engineer"
)
JIRAYA_DOC: list[dict[str, str]] = [
    {
        "Job Title": "Software Engineer II",
        "Job Location": "BENTONVILLE, AR",
        "Job ID": "WD1996646",
        "Job Link": "https://careers.walmart.com/us/jobs/WD1996646-software-engineer-ii",
    },
    {
        "Job Title": "Full Stack Software Engineer",
        "Job Location": "SUNNYVALE, CA",
        "Job ID": "WD1757555",
        "Job Link": "https://careers.walmart.com/us/jobs/WD1757555-full-stack-software-engineer",
    },
]

HENRIQUE_TEXT: str = (
    "Title: Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models\n"
    "Authors: Yinjie Wang, Ling Yang, Bowen Li, Ye Tian, Ke Shen, Mengdi Wang\n"
    "Published: 2025-09-08 17:58:06+00:00\n"
    "arXiv ID: http://arxiv.org/abs/2509.06949v1\n"
    "\n"
    "Abstract:\n"
    "We propose TraceRL, a trajectory-aware reinforcement learning framework for diffusion language "
    "models that incorporates preferred inference trajectory into post-training."
)
HENRIQUE_DOC: dict[str, Any] = {
    "title": "Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models",
    "publication_date": "2025-09-08 17:58:06+00:00",
    "arxiv_id": "http://arxiv.org/abs/2509.06949v1",
    "authors": ["Bowen Li", "Ke Shen"],
    "repository_url": "https://github.com/Gen-Verse/dLLM-RL",
    "model_name": ["TraDo-8B-Instruct", "TraDo-4B-Instruct"],
}


def enc(document: Any) -> str:
    """the hub json column is the encoded JSON document as one string"""
    return json.dumps(document)


# a handful of realistic rows per source for the nonzero-yield and containment sweep
REALISTIC_ROWS: dict[str, list[tuple[str, str, str]]] = {
    OWKIN_SOURCE: [
        (OWKIN_TEXT, enc(OWKIN_DOC), OWKIN_SOURCE),
        (
            OWKIN_TEXT,
            enc(
                {
                    "conditions": ["asthma"],
                    "interventions": [
                        {"name": "fish oil", "type": "Drug"},
                        {"name": "vitamin C", "type": "Dietary Supplement"},
                    ],
                }
            ),
            OWKIN_SOURCE,
        ),
    ],
    PROFESSOR_BOB_SOURCE: [
        (PROFESSOR_BOB_TEXT, enc(PROFESSOR_BOB_DOC), PROFESSOR_BOB_SOURCE),
        (
            "Politics\nOntario has numerous political parties which run for election. The 2018 "
            "provincial election resulted in a Progressive Conservative majority government under "
            "party leader Doug Ford, who was sworn in as Premier on June 29.",
            enc(
                [
                    {"subject": "Ontario Liberal Party", "predicate": "no relation", "object": "Doug Ford"},
                    {"subject": "Ontario", "predicate": "head of government", "object": "Doug Ford"},
                ]
            ),
            PROFESSOR_BOB_SOURCE,
        ),
    ],
    ROBOROVSKI_SOURCE: [
        (ROBOROVSKI_TEXT, enc(ROBOROVSKI_DOC), ROBOROVSKI_SOURCE),
        (
            ROBOROVSKI_TEXT,
            enc(
                {
                    "entity": "Person",
                    "name": "John Moses Browning",
                    "data": {"occupation": "firearm designer", "notable_inventions": ["M1911 pistol"]},
                }
            ),
            ROBOROVSKI_SOURCE,
        ),
    ],
    SANDEEPPANEM_SOURCE: [
        (SANDEEPPANEM_TEXT, enc(SANDEEPPANEM_DOC), SANDEEPPANEM_SOURCE),
        (
            "EXTENSION METHODOLOGIST\nProfile\nSelf-motivated, honest, competent, innovative.\n"
            "Experience\nCompany Name City , State Extension Methodologist\n"
            "working for the government of the republic of Zambia in the ministry of Agriculture.",
            enc(
                {
                    "current_title": "Extension Methodologist",
                    "previous_titles": ["Research Assistant"],
                    "current_company": "Ministry of Agriculture and Livestock",
                    "previous_companies": ["Plan Zambia"],
                    "years_experience": 13,
                    "location": "Petauke, Zambia",
                    "leadership_experience": True,
                    "summary": "Extension Methodologist with experience in agricultural extension coordination.",
                }
            ),
            SANDEEPPANEM_SOURCE,
        ),
    ],
    JIRAYA_SOURCE: [
        (JIRAYA_TEXT, enc(JIRAYA_DOC), JIRAYA_SOURCE),
        (
            "Pharmacy Manager, Store # 05829 Pharmacy DANVILLE, VA 10/23/24 Pharmacy Manager- "
            "$30,000 Sign on bonus Pharmacy DEVINE, TX 08/19/24 Pharmacy Manager - $25,000 "
            "Sign-On Bonus Pharmacy HERRIN, IL 11/06/24",
            enc(
                [
                    {
                        "Job Title": "(USA) Pharmacy Manager, Store # 05829",
                        "Job Location": "DANVILLE, VA",
                        "Job ID": "WD2025323",
                        "Job Link": "https://careers.walmart.com/us/jobs/WD2025323-usa-pharmacy-manager-store-05829",
                    },
                    {
                        "Job Title": "Pharmacy Manager- $30,000 Sign on bonus",
                        "Job Location": "DEVINE, TX",
                        "Job ID": "WD1962097",
                        "Job Link": "https://careers.walmart.com/us/jobs/WD1962097-pharmacy-manager-30-000-sign-on-bonus",
                    },
                    {
                        "Job Title": "(USA) Pharmacy Manager - $25,000 Sign-On Bonus",
                        "Job Location": "HERRIN, IL",
                        "Job ID": "WD2035847",
                        "Job Link": "https://careers.walmart.com/us/jobs/WD2035847-usa-pharmacy-manager-25-000-sign-on-bonus",
                    },
                ]
            ),
            JIRAYA_SOURCE,
        ),
    ],
    HENRIQUE_SOURCE: [
        (HENRIQUE_TEXT, enc(HENRIQUE_DOC), HENRIQUE_SOURCE),
        (
            "Title: LLaDA-VLA: Vision Language Diffusion Action Models\n"
            "Authors: Yuqing Wen, Hebei Li, Kefan Gu\n"
            "Published: 2025-09-08 17:45:40+00:00\n"
            "arXiv ID: http://arxiv.org/abs/2509.06932v1\n"
            "\n"
            "Abstract:\n"
            "We present LLaDA-VLA, the first Vision-Language-Diffusion-Action model built upon "
            "pretrained d-VLMs for robotic manipulation.",
            enc(
                {
                    "title": "LLaDA-VLA: Vision Language Diffusion Action Models",
                    "publication_datetime": "2025-09-08 17:45:40+00:00",
                    "arxiv_id": "http://arxiv.org/abs/2509.06932v1",
                    "abstract": "We present LLaDA-VLA, the first Vision-Language-Diffusion-Action model.",
                    "name": ["University of Science and Technology of China"],
                    "role_or_type": ["University", "Company"],
                }
            ),
            HENRIQUE_SOURCE,
        ),
    ],
}


def entities_by_label(Example: TrainingExample) -> dict[str, list[str]]:
    return {entity.label: entity.mentions for entity in Example.entities}


# --- registry and import-time guards


def test_the_script_self_registers_under_its_declared_name() -> None:
    """US-001 wired the ingest around Script.dispatch, which resolves by this exact NAME key;
    importing the module must install the instance in the shared registry"""
    assert isinstance(Script.REGISTRY["JsonExtractionScript"], JsonExtractionScriptClass)


def test_the_import_time_predicate_guard_accepts_the_census_landed_map() -> None:
    """PREDICATE_MAP is validated at import; re-assert the contract here for drift, and pin the
    census fact the single entry rests on (member of is a biolink member, nothing else is)"""
    assert ScriptUtils.resolve_predicate("member of") == ("member_of", True)
    for member in JsonExtractionScriptClass.PREDICATE_MAP.values():
        assert member in ScriptUtils.biolink_predicates()


# --- row guards: each malformed shape ships the canonical empty example, and its well-formed
# sibling in the same test still ships (skip-don't-coerce, per-leaf and per-row)


def test_a_malformed_json_string_yields_the_empty_example() -> None:
    """0 parse failures were measured corpus-wide, but a drifted builder could still ship broken
    strings; the row drops whole rather than crashing the stream worker"""
    for broken in ('{"conditions": ["asthma",}', "{unterminated", "not json at all", ""):
        Example: TrainingExample = run_row(OWKIN_TEXT, broken)

        assert Example.text == ""
        assert Example.populated() == frozenset()

    Sibling: TrainingExample = run_row(OWKIN_TEXT, enc(OWKIN_DOC))
    assert Sibling.structures[0].fields


def test_a_non_dict_or_list_decoded_json_yields_the_empty_example() -> None:
    """the column can decode to a bare scalar ("42", "\"asthma\"", "true", "null"); those are not
    documents and ship the empty example instead of a structure over nothing"""
    for scalar in ("42", '"asthma"', "true", "null"):
        Example: TrainingExample = run_row(OWKIN_TEXT, scalar)

        assert Example.text == ""
        assert Example.populated() == frozenset()

    Sibling: TrainingExample = run_row(OWKIN_TEXT, enc(OWKIN_DOC))
    assert Sibling.structures[0].fields


def test_blank_or_non_string_text_yields_the_empty_example() -> None:
    """gliner2 cannot train on an empty document and downstream tokenization would break; the
    canonical empty example lets the declared-outputs filter drop the row"""
    for blank in ("", None, 42):
        Example: TrainingExample = run_row(blank, enc(OWKIN_DOC))

        assert Example.text == ""
        assert Example.populated() == frozenset()

    Sibling: TrainingExample = run_row(OWKIN_TEXT, enc(OWKIN_DOC))
    assert Sibling.text == OWKIN_TEXT


def test_a_wrong_arity_values_tuple_yields_the_empty_example() -> None:
    """the declared columns_out projection is exactly (text, json, source); a drifted arity must
    fail loud at the guard, not unpack into a wrong-positions example"""
    for values in (
        (OWKIN_TEXT, enc(OWKIN_DOC)),
        (OWKIN_TEXT, enc(OWKIN_DOC), OWKIN_SOURCE, "extra"),
        (),
    ):
        Example: TrainingExample = SCRIPT.run(values)

        assert Example.text == ""
        assert Example.populated() == frozenset()

    Sibling: TrainingExample = run_row(OWKIN_TEXT, enc(OWKIN_DOC))
    assert Sibling.structures[0].fields


# --- entity drop rules: the leaf drops, never the row; a well-formed sibling ships in the same test


def test_an_unlocatable_surface_drops_from_entities() -> None:
    """containment is the hallucination guard: a mapped value the text never contains (here a
    stitched NORTH DAKOTA job location) must not become a mention, while the verbatim sibling
    still ships -- gliner2 rejects any mention outside the text"""
    Hallucinated: list[dict[str, str]] = [
        JIRAYA_DOC[0],
        JIRAYA_DOC[1],
        {
            "Job Title": "Pharmacy Manager- $30,000 Sign on bonus",
            "Job Location": "NORTH DAKOTA",
            "Job ID": "WD1962097",
            "Job Link": "https://careers.walmart.com/us/jobs/WD1962097-pharmacy-manager-30-000-sign-on-bonus",
        },
    ]
    Example: TrainingExample = run_row(JIRAYA_TEXT, enc(Hallucinated), JIRAYA_SOURCE)

    assert entities_by_label(Example) == {"GeographicLocation": ["BENTONVILLE, AR", "SUNNYVALE, CA"]}


def test_a_company_name_placeholder_drops_from_entities() -> None:
    """5,650/13,992 (40.4%) measured sandeeppanem company leaves start with the mojibake
    placeholder prefix; they stay structure fields but never become Agent mentions, while a real
    company verbatim in the resume text still ships"""
    Example: TrainingExample = run_row(SANDEEPPANEM_TEXT, enc(SANDEEPPANEM_DOC), SANDEEPPANEM_SOURCE)

    assert entities_by_label(Example) == {"Agent": ["Jackson Hewitt"]}
    flattened = dict((field.name, field.value) for field in Example.structures[0].fields)
    assert flattened["current_company"] == "Company Name"  # placeholder survives as structure only


def test_a_non_person_marker_name_drops_from_roborovski_entities() -> None:
    """the Person sibling marker qualifies only 33 name leaves corpus-wide; a name leaf whose
    containing object carries no Person marker (top-level "name", a results entry without
    "entity") must not become an Agent mention, while the marked sibling ships"""
    Example: TrainingExample = run_row(ROBOROVSKI_TEXT, enc(ROBOROVSKI_DOC), ROBOROVSKI_SOURCE)

    assert entities_by_label(Example) == {"Agent": ["John Moses Browning"]}


def test_an_owkin_device_intervention_drops_while_its_drug_sibling_survives() -> None:
    """the type guard, not containment, carries this decision: "vitamin C" occurs verbatim in the
    text but its intervention type is Device, so it stays unmapped; the "fish oil" Drug sibling
    maps to ChemicalEntity. Mislabeling a device as a chemical would corrupt the entity family"""
    Doc: dict[str, Any] = {
        "conditions": [],
        "interventions": [
            {"name": "vitamin C", "type": "Device"},
            {"name": "fish oil", "type": "Drug"},
        ],
    }
    Example: TrainingExample = run_row(OWKIN_TEXT, enc(Doc))

    assert entities_by_label(Example) == {"ChemicalEntity": ["fish oil"]}


# --- relation drop rules (ProfessorBob only): the triple drops, never the row


def test_a_member_of_row_emits_one_relation_with_head_tail_and_description() -> None:
    """the census-landed PREDICATE_MAP's single entry: one Relation named member_of with the
    biolink head/tail field convention, the slot definition as the gliner2 label prompt, and
    asserted evidence provenance"""
    Triples: list[dict[str, str]] = [{"subject": "Wayne Gretzky", "predicate": "member of", "object": "Hockey Hall of Fame"}]
    Example: TrainingExample = run_row(PROFESSOR_BOB_TEXT, enc(Triples), PROFESSOR_BOB_SOURCE)

    assert Example.relations == [
        Relation(
            name="member_of",
            fields=[
                RelationField(name="head", value="Wayne Gretzky"),
                RelationField(name="tail", value="Hockey Hall of Fame"),
            ],
            description=ScriptUtils.predicate_description("member_of"),
            evidence="asserted",
        )
    ]
    assert Example.relations[0].description is not None


def test_a_non_biolink_predicate_emits_no_relation() -> None:
    """ "instance of" (1,598 measured triples) is outside the Predicates vocabulary; the map guard
    drops it even though both endpoints occur verbatim in the text, while the member of sibling
    still ships"""
    Triples: list[dict[str, str]] = [
        {"subject": "Ontario", "predicate": "instance of", "object": "Canada"},
        {"subject": "Wayne Gretzky", "predicate": "member of", "object": "Hockey Hall of Fame"},
    ]
    Example: TrainingExample = run_row(PROFESSOR_BOB_TEXT, enc(Triples), PROFESSOR_BOB_SOURCE)

    assert [(relation.name, [(field.name, field.value) for field in relation.fields]) for relation in Example.relations] == [
        ("member_of", [("head", "Wayne Gretzky"), ("tail", "Hockey Hall of Fame")])
    ]


def test_a_no_relation_predicate_drops_the_relation() -> None:
    """ "no relation" is the corpus's top predicate (14,333 triples) and a labeling-negative
    artifact; both endpoints can be verbatim in the text and the triple must still emit nothing,
    while the member of sibling ships"""
    Triples: list[dict[str, str]] = [
        {"subject": "Ontario", "predicate": "no relation", "object": "Canada"},
        {"subject": "Wayne Gretzky", "predicate": "member of", "object": "Hockey Hall of Fame"},
    ]
    Example: TrainingExample = run_row(PROFESSOR_BOB_TEXT, enc(Triples), PROFESSOR_BOB_SOURCE)

    assert [[field.value for field in relation.fields] for relation in Example.relations] == [["Wayne Gretzky", "Hockey Hall of Fame"]]


def test_a_self_loop_relation_drops() -> None:
    """head == tail is schema noise, not an edge; the self-loop drops while the well-formed
    sibling in the same row still ships"""
    Triples: list[dict[str, str]] = [
        {"subject": "Ontario", "predicate": "member of", "object": "Ontario"},
        {"subject": "Wayne Gretzky", "predicate": "member of", "object": "Hockey Hall of Fame"},
    ]
    Example: TrainingExample = run_row(PROFESSOR_BOB_TEXT, enc(Triples), PROFESSOR_BOB_SOURCE)

    assert [[field.value for field in relation.fields] for relation in Example.relations] == [["Wayne Gretzky", "Hockey Hall of Fame"]]


def test_a_relation_endpoint_outside_the_text_drops() -> None:
    """endpoint containment is the relation-side hallucination guard: "Northern Canada" never
    occurs in the text, so the triple drops even though its predicate maps, while the verbatim
    sibling ships"""
    Triples: list[dict[str, str]] = [
        {"subject": "Ontario", "predicate": "member of", "object": "Northern Canada"},
        {"subject": "Wayne Gretzky", "predicate": "member of", "object": "Hockey Hall of Fame"},
    ]
    Example: TrainingExample = run_row(PROFESSOR_BOB_TEXT, enc(Triples), PROFESSOR_BOB_SOURCE)

    assert [[field.value for field in relation.fields] for relation in Example.relations] == [["Wayne Gretzky", "Hockey Hall of Fame"]]


# --- flatten edge cases: the path convention is the measured census convention


def test_flatten_dot_joins_nested_dict_paths() -> None:
    """roborovski's measured shape (result.properties.name) sets the dot-join convention the
    LABEL_MAPS keys are written against"""
    assert flatten_json({"result": {"properties": {"name": "Stalemate"}}}) == [("result.properties.name", "Stalemate")]


def test_flatten_list_of_dicts_takes_the_bracket_path() -> None:
    """owkin's measured shape (interventions[].name): every list-of-dict child takes parent[]
    and recurses per item, preserving document order"""
    assert flatten_json(
        {
            "interventions": [
                {"name": "fish oil", "type": "Dietary Supplement"},
                {"name": "vitamin C", "type": "Drug"},
            ]
        }
    ) == [
        ("interventions[].name", "fish oil"),
        ("interventions[].type", "Dietary Supplement"),
        ("interventions[].name", "vitamin C"),
        ("interventions[].type", "Drug"),
    ]


def test_flatten_a_root_list_takes_the_empty_prefix_bracket_path() -> None:
    """Jiraya's measured shape: a decoded root list flattens to [].Job Location, the exact key
    the (JIRAYA_SOURCE, "[].Job Location") map entry matches"""
    assert flatten_json([{"Job Location": "BENTONVILLE, AR"}]) == [("[].Job Location", "BENTONVILLE, AR")]


def test_flatten_collapses_scalar_lists_into_one_terminal_list_field() -> None:
    """sandeeppanem industries[] and owkin conditions[] ride as one terminal parent[] field
    holding the whole list, so a mapped scalar list fans out per member at mention time"""
    assert flatten_json({"alternateNames": ["Dar", "Dar es Salaam"]}) == [("alternateNames[]", ["Dar", "Dar es Salaam"])]


def test_flatten_drops_nulls_blank_strings_and_empty_containers() -> None:
    """null and "" leaves and empty {} / [] containers produce no field at all; they stay
    invisible rather than becoming empty-string structure noise"""
    assert flatten_json({"a": None, "b": "", "c": "  ", "d": {}, "e": [], "f": "x"}) == [("f", "x")]


def test_flatten_stringifies_bools_ints_and_floats() -> None:
    """sandeeppanem leadership_experience (bool) and years_experience (int, often null) and
    HenriqueGodoy's metric values (float) must reach structure fields as strings, never as
    mixed-type values that would break the avro record"""
    assert flatten_json({"leadership_experience": True, "years_experience": 13, "value": 6.1}) == [
        ("leadership_experience", "true"),
        ("years_experience", "13"),
        ("value", "6.1"),
    ]


def test_flatten_keeps_scalar_members_of_a_mixed_list_in_a_terminal_field() -> None:
    """a container anywhere in the list puts every item behind the [] convention; scalar members
    of such a mixed list ride along as one terminal field at the same path"""
    assert flatten_json({"items": ["note", {"name": "x"}]}) == [("items[].name", "x"), ("items[]", ["note"])]


# --- nonzero yield over realistic rows + the gliner2 containment assertion


def test_every_source_yields_over_a_handful_of_realistic_rows() -> None:
    """the silent-zero-yield bug is the failure mode this repo has already shipped once: every
    source config must produce populated structures over real-shaped rows, entities where its
    map applies (HenriqueGodoy is deliberately structures-only), and ProfessorBob relations off
    the census-landed predicate"""
    assert set(REALISTIC_ROWS) == {
        OWKIN_SOURCE,
        PROFESSOR_BOB_SOURCE,
        ROBOROVSKI_SOURCE,
        SANDEEPPANEM_SOURCE,
        JIRAYA_SOURCE,
        HENRIQUE_SOURCE,
    }
    entity_sources: set[str] = {OWKIN_SOURCE, ROBOROVSKI_SOURCE, SANDEEPPANEM_SOURCE, JIRAYA_SOURCE}
    saw_entities: set[str] = set()
    saw_relations: bool = False
    for source, rows in REALISTIC_ROWS.items():
        assert len(rows) >= 2, f"{source} needs a handful of rows, not one"
        for text, json_string, row_source in rows:
            Example: TrainingExample = run_row(text, json_string, row_source)
            assert Example.structures and Example.structures[0].fields, f"{row_source} shipped no structure fields"
            assert Example.structures[0].name == row_source
            if Example.entities:
                saw_entities.add(row_source)
            if Example.relations:
                saw_relations = True
            if row_source == HENRIQUE_SOURCE:
                assert Example.entities == [], "HenriqueGodoy declares structures only; no entity map exists"
    assert saw_entities == entity_sources
    assert saw_relations


def test_every_emitted_mention_and_relation_endpoint_occurs_in_the_emitted_text() -> None:
    """gliner2's InputExample.validate() rejects any surface outside the text, so this is a
    hard training-data invariant: sweep every realistic row and re-assert containment for all
    emitted entity mentions and relation head/tail values"""
    for rows in REALISTIC_ROWS.values():
        for text, json_string, row_source in rows:
            Example: TrainingExample = run_row(text, json_string, row_source)
            for entity in Example.entities:
                assert entity.description is not None
                for mention in entity.mentions:
                    assert mention in Example.text, f"{row_source}: mention {mention!r} outside the text"
            for relation in Example.relations:
                for field in relation.fields:
                    assert field.value in Example.text, f"{row_source}: relation endpoint {field.value!r} outside the text"


def test_a_realistic_owkin_row_yields_the_declared_shapes() -> None:
    """end-to-end shape check on the owkin fixture: structures plus the two mapped entity labels
    (Disease off conditions[], ChemicalEntity off the chem-qualified interventions[].name),
    exactly the (structures, entities) shapes the ingest declares"""
    Example: TrainingExample = run_row(OWKIN_TEXT, enc(OWKIN_DOC))

    assert Example.text == OWKIN_TEXT
    assert Example.populated() == frozenset({"structures", "entities"})
    assert entities_by_label(Example) == {"Disease": ["asthma"], "ChemicalEntity": ["fish oil"]}
    flattened = dict((field.name, field.value) for field in Example.structures[0].fields)
    assert flattened["conditions[]"] == ["asthma", "Chronic Obstructive Pulmonary Disease"]


def test_dispatch_routes_through_the_registry_and_keeps_declared_outputs() -> None:
    """the ingest invokes Script.dispatch with the declared outputs tuple; the script must
    answer through that path, not only via a direct instance call"""
    Outputs, Example = Script.dispatch("JsonExtractionScript", (("structures", "entities"), (OWKIN_TEXT, enc(OWKIN_DOC), OWKIN_SOURCE)))

    assert Outputs == ("structures", "entities")
    assert isinstance(Example, TrainingExample)
    assert Example.structures[0].fields
    assert Example.entities
