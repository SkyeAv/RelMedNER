from __future__ import annotations

from typing import Any

from relmedner.models import TrainingExample
from relmedner.scripts import CtkpInterventionsScript  # registry must stay populated alongside the new script
from relmedner.scripts.medical_entity_json import MedicalEntityJsonScript
from relmedner.types import Script
from relmedner.utils import ScriptUtils

SCRIPT: MedicalEntityJsonScript = MedicalEntityJsonScript()

# real rows of Pennlaine/Medical-Entity-JSON-Extraction, split test, captured verbatim on
# wenceslaus 2026-09-22 (the laptop never streams a corpus). ANDREW_REYNOLDS carries verbatim and
# casefold-only surfaces plus a hapax label (PersonalMotivation), EMILY_TURNER carries three hapax
# labels (CareerDedication, ImplementedSolution, TargetGroup), and TONYA_TROSTEL carries a measured
# no-substring paraphrase: Age "34" against a passage that only says "would be 35 years old in a
# month". Full-split census behind the class docstring: 259 entries, 102 verbatim (39.4%), 130
# casefold-only (50.2%), 27 paraphrases (10.4%) drop; 50 distinct labels, 30 hapaxes.

ROW_ANDREW_REYNOLDS: str = (
    "[INST]Dr. Andrew Reynolds, a 39-year-old pediatric psychiatrist, developed an interest in child and "
    "adolescent psychiatry during his psychiatric residency. He now advocates for mental health awareness"
    " and supports initiatives for adolescent depression prevention. What is adolescent depression preven"
    "tion? Extract the name, age, specialty, personal motivation for specialization, advocacy cause, and "
    'focus area and return in Json format.[/INST]```json\n{\n    "Question": "Dr. Andrew Reynolds, a 3'
    "9-year-old pediatric psychiatrist, developed an interest in child and adolescent psychiatry during h"
    "is psychiatric residency. He now advocates for mental health awareness and supports initiatives for "
    'adolescent depression prevention. What is adolescent depression prevention?",\n    "Answer": "Ad'
    "olescent depression prevention is a focus area that aims to reduce the prevalence of depression amon"
    'g adolescents.",\n    "Entities": [\n        {\n            "Name": "Dr. Andrew Reynolds"\n  '
    '      },\n        {\n            "Age": "39"\n        },\n        {\n            "Specialty": '
    '"Pediatric Psychiatrist"\n        },\n        {\n            "Personal Motivation": "Interest i'
    'n Child and Adolescent Psychiatry"\n        },\n        {\n            "Advocacy Cause": "Mental'
    ' Health Awareness"\n        },\n        {\n            "Focus Area": "Adolescent Depression Prev'
    'ention"\n        }\n    ]\n}'
)

ROW_EMILY_TURNER: str = (
    "[INST]Emily Turner, a 32-year-old nurse practitioner, has dedicated her career to improving healthca"
    "re access in rural communities. She has implemented telehealth programs to support patients with chr"
    "onic illnesses. What are telehealth programs? Extract the name, age, profession, career dedication, "
    'implemented solution, and target group and return in Json format.[/INST]```json\n{\n    "Question"'
    ': "Emily Turner, a 32-year-old nurse practitioner, has dedicated her career to improving healthcare'
    " access in rural communities. She has implemented telehealth programs to support patients with chron"
    'ic illnesses. What are telehealth programs?",\n    "Answer": "Telehealth programs are healthcare'
    ' services provided remotely via telecommunications technology.",\n    "Entities": [\n        {\n '
    '           "Name": "Emily Turner"\n        },\n        {\n            "Age": "32"\n        }'
    ',\n        {\n            "Profession": "Nurse Practitioner"\n        },\n        {\n           '
    ' "Career Dedication": "Improving Healthcare Access in Rural Communities"\n        },\n        {\n'
    '            "Implemented Solution": "Telehealth Programs"\n        },\n        {\n            "'
    'Target Group": "Patients with Chronic Illnesses"\n        }\n    ]\n}'
)

ROW_MICHAEL_JOHNSON: str = (
    "[INST]Michael Johnson, a 45-year-old cardiologist, is renowned for his research on heart disease pre"
    "vention. He runs community workshops educating people about healthy lifestyles. What are healthy lif"
    "estyles? Extract the name, age, profession, research focus, community activity, and topic of educati"
    'on and return in Json format.[/INST]```json\n{\n    "Question": "Michael Johnson, a 45-year-old c'
    "ardiologist, is renowned for his research on heart disease prevention. He runs community workshops e"
    'ducating people about healthy lifestyles. What are healthy lifestyles?",\n    "Answer": "Healthy'
    ' lifestyles involve behaviors and habits that promote physical, mental, and social well-being.",\n '
    '   "Entities": [\n        {\n            "Name": "Michael Johnson"\n        },\n        {\n   '
    '         "Age": "45"\n        },\n        {\n            "Profession": "Cardiologist"\n     '
    '   },\n        {\n            "Research Focus": "Heart Disease Prevention"\n        },\n        '
    '{\n            "Community Activity": "Community Workshops"\n        },\n        {\n            "'
    'Topic of Education": "Healthy Lifestyles"\n        }\n    ]\n}'
)

ROW_SARAH_LOPEZ: str = (
    "[INST]Sarah Lopez, a 28-year-old occupational therapist, specializes in helping children with develo"
    "pmental delays. She incorporates play-based therapies to enhance motor and cognitive skills. What ar"
    "e play-based therapies? Extract the name, age, profession, specialization, therapeutic approach, and"
    ' skills targeted and return in Json format.[/INST]```json\n{\n    "Question": "Sarah Lopez, a 28-'
    "year-old occupational therapist, specializes in helping children with developmental delays. She inco"
    'rporates play-based therapies to enhance motor and cognitive skills. What are play-based therapies?"'
    ',\n    "Answer": "Play-based therapies use play activities to promote developmental skills in chi'
    'ldren.",\n    "Entities": [\n        {\n            "Name": "Sarah Lopez"\n        },\n      '
    '  {\n            "Age": "28"\n        },\n        {\n            "Profession": "Occupational '
    'Therapist"\n        },\n        {\n            "Specialization": "Helping Children with Developm'
    'ental Delays"\n        },\n        {\n            "Therapeutic Approach": "Play-based Therapies"'
    '\n        },\n        {\n            "Skills Targeted": "Motor and Cognitive Skills"\n        }\n'
    "    ]\n}"
)

ROW_TONYA_TROSTEL: str = (
    "[INST]Tonya Trostel, who would be 35 years old in a month, had a traumatic childbirth experience inv"
    "olving placenta accreta and became an advocate for blood donations on her daughter's birthday that h"
    "appened on last Friday, 24, October, 2024. What is placenta accreta? Extract the name, age, childbir"
    "th complication, advocacy cause, and date of advocacy event and return in Json format.[/INST]```json"
    '\n{\n    "Question": "Tonya Trostel, who would be 35 years old in a month, had a traumatic childb'
    "irth experience involving placenta accreta and became an advocate for blood donations on her daughte"
    'r\'s birthday that happened on last Friday, 24, October, 2024. What is placenta accreta?",\n    "An'
    'swer": "Placenta accreta is a serious pregnancy condition where the placenta grows too deeply into'
    ' the uterine wall.",\n    "Entities": [\n        {\n            "Name": "Tonya Trostel"\n    '
    '    },\n        {\n            "Age": "34"\n        },\n        {\n            "Childbirth Comp'
    'lication": "Placenta Accreta"\n        },\n        {\n            "Advocacy Cause": "Blood Don'
    'ations"\n        },\n        {\n            "Date of Advocacy Event": "24, October, 2024"\n    '
    "    }\n    ]\n}"
)

ENT_NOT_LIST_SIBLING: str = '[\n        {\n            "Name": "Marie Curie"\n        }\n    ]'

ENT_MALFORMED: str = (
    '[\n        {\n            "Name": "Marie Curie",\n            "Age": "36"\n        },\n     '
    '   "stray",\n        {},\n        {\n            "Profession": "Physicist"\n        }\n    ]'
)

ENT_PARAPHRASE: str = (
    '[\n        {\n            "Name": "Marie Curie"\n        },\n        {\n            "Condition"'
    ': "a wasting illness of the bones"\n        }\n    ]'
)

SYNTH_PASSAGE: str = "Marie Curie, a 36-year-old physicist, studied radioactivity in Paris."


def synth_row(entities_json: str, passage: str = SYNTH_PASSAGE) -> str:
    """one synthetic row in the measured shape: [INST]passage question instruction[/INST] fence,
    JSON object, closing fence; entities_json is the literal Entities payload (valid or broken)"""
    return (
        f"[INST]{passage} What is radioactivity? Extract the name, age, and profession and "
        f'return in Json format.[/INST]```json\n{{\n    "Question": "{passage}",\n    '
        f'"Answer": "A physicist.",\n    "Entities": {entities_json}\n}}\n```'
    )


def row(text: Any) -> tuple[Any, ...]:
    """one columns_out projection ([text]) as ScriptValues delivers it"""
    return (text,)


def test_the_script_self_registers_under_its_declared_name() -> None:
    """importing the module must install the instance in the shared registry (the ingest pipeline
    resolves Script.dispatch by this exact NAME key, and ingests.yaml declares MedicalEntityJsonScript)"""
    assert isinstance(Script.REGISTRY["MedicalEntityJsonScript"], MedicalEntityJsonScript)


def test_the_declared_name_matches_the_ingests_yaml_entry() -> None:
    """NAME is the join key between ingests.yaml and Script.REGISTRY; a drifted name fails
    parse_ingests loudly, and this lock catches the drift at the unit level too"""
    assert MedicalEntityJsonScript.NAME == "MedicalEntityJsonScript"


def test_a_well_formed_real_row_ships_entities_under_raw_pascal_labels() -> None:
    """trust-gold stance: the corpus's own attribute slots ship as raw PascalCase tails (no
    fullmap, no LABEL_MAP), in first-occurrence order, with the passage as the emitted text"""
    Example: TrainingExample = SCRIPT.run(row(ROW_ANDREW_REYNOLDS))

    assert [entity.label for entity in Example.entities] == [
        "Name",
        "Age",
        "Specialty",
        "PersonalMotivation",
        "AdvocacyCause",
        "FocusArea",
    ]
    assert Example.text == ScriptUtils.join_tokens(ROW_ANDREW_REYNOLDS.split("[INST]", 1)[1].split("[/INST]")[0].split())
    assert Example.populated() == frozenset({"entities"})


def test_the_emitted_text_is_the_tokenized_rejoined_inst_inner_content() -> None:
    """gliner2 re-tokenizes the emitted text downstream; the re-join normalizes whitespace without
    ever rewriting characters, so every gold surface stays findable"""
    Example: TrainingExample = SCRIPT.run(row(ROW_EMILY_TURNER))

    assert Example.text == ScriptUtils.join_tokens(ROW_EMILY_TURNER.split("[INST]", 1)[1].split("[/INST]")[0].split())
    assert "dedicated her career to improving healthcare access" in Example.text


def test_a_verbatim_surface_ships_in_passage_casing() -> None:
    """Name surfaces copy the passage exactly ("Dr. Andrew Reynolds"): no casing rewrite, no
    token surgery; the mention is a literal slice of the emitted text"""
    Example: TrainingExample = SCRIPT.run(row(ROW_ANDREW_REYNOLDS))

    assert Example.entities[0].label == "Name"
    assert Example.entities[0].mentions == ["Dr. Andrew Reynolds"]
    assert Example.entities[0].mentions[0] in Example.text


def test_a_casefold_only_surface_ships_in_passage_casing_not_the_json_title_case() -> None:
    """half the corpus (130/259 = 50.2%) Title-Cases surfaces the passage lowercases; the emitted
    mention must carry the PASSAGE casing (the slice of the emitted text), never the JSON casing,
    or the gliner2 containment check would sanitize the mention away"""
    Example: TrainingExample = SCRIPT.run(row(ROW_ANDREW_REYNOLDS))

    by_label: dict[str, list[str]] = {entity.label: entity.mentions for entity in Example.entities}
    assert by_label["Specialty"] == ["pediatric psychiatrist"]
    assert "Pediatric Psychiatrist" not in by_label["Specialty"]
    assert by_label["PersonalMotivation"] == ["interest in child and adolescent psychiatry"]
    assert by_label["Age"] == ["39"]  # located inside the hyphenated "39-year-old" token


def test_a_no_substring_paraphrase_drops_while_its_siblings_ship() -> None:
    """27/259 entries (10.4%, measured) name a surface the passage never contains even casefolded
    (here Age "34" vs a passage saying "would be 35 years old in a month"); skip-don't-coerce:
    the mention drops, the row's locatable siblings still ship"""
    Example: TrainingExample = SCRIPT.run(row(ROW_TONYA_TROSTEL))

    mentions: list[str] = [mention for entity in Example.entities for mention in entity.mentions]
    assert "34" not in mentions
    assert "Tonya Trostel" in mentions
    assert "Name" in [entity.label for entity in Example.entities]
    assert "Age" not in [entity.label for entity in Example.entities]


def test_an_inst_non_match_ships_text_only_and_the_sibling_row_still_ships() -> None:
    """no [INST] pair means no passage to anchor surfaces in; the row ships text-only and the
    declared-outputs filter drops it (0 such rows in the measured split, still defended)"""
    Broken: TrainingExample = SCRIPT.run(row("plain biography text with no instruction markers"))
    Sibling: TrainingExample = SCRIPT.run(row(ROW_ANDREW_REYNOLDS))

    assert Broken.entities == []
    assert Broken.populated() == frozenset()
    assert Sibling.entities != []


def test_unparseable_json_ships_the_passage_text_only_and_the_sibling_row_still_ships() -> None:
    """a corrupt JSON body must not crash the stream worker and must not invent entities; the
    [INST] half of the row is still trustworthy, so the passage ships as text (0 such rows in the
    measured split, still defended)"""
    Corrupt: str = ROW_ANDREW_REYNOLDS.replace("```json\n{", "```json\n{corrupt", 1)
    Broken: TrainingExample = SCRIPT.run(row(Corrupt))
    Sibling: TrainingExample = SCRIPT.run(row(ROW_ANDREW_REYNOLDS))

    assert Broken.entities == []
    assert Broken.text == ScriptUtils.join_tokens(ROW_ANDREW_REYNOLDS.split("[INST]", 1)[1].split("[/INST]")[0].split())
    assert Sibling.entities != []


def test_entities_not_a_list_ships_text_only_and_the_sibling_row_still_ships() -> None:
    """schema drift can turn Entities into a bare string; without the isinstance(list) guard the
    iteration would walk the string's characters and emit one entity per char, so the guard turns
    drift into a text-only row the filter drops"""
    Broken: TrainingExample = SCRIPT.run(row(synth_row('"Name"')))
    Sibling: TrainingExample = SCRIPT.run(row(synth_row(ENT_NOT_LIST_SIBLING)))

    assert Broken.entities == []
    assert Sibling.entities[0].label == "Name"


def test_entity_entries_that_are_not_single_key_dicts_drop_and_siblings_ship() -> None:
    """the measured split has 0 malformed entries, but the contract is per-entry skip-don't-coerce:
    a multi-key dict, a bare string, or an empty dict carry no single (label, surface) pair, so
    they drop individually while the well-formed sibling entry still ships"""
    Example: TrainingExample = SCRIPT.run(row(synth_row(ENT_MALFORMED)))

    assert [entity.label for entity in Example.entities] == ["Profession"]
    assert Example.entities[0].mentions == ["physicist"]
    flat: list[str] = [mention for entity in Example.entities for mention in entity.mentions]
    assert "Marie Curie" not in flat


def test_a_surface_unlocatable_even_case_insensitively_drops_and_its_sibling_ships() -> None:
    """the hallucination guard: a surface with no case-insensitive occurrence in the passage is a
    paraphrase, not a mention; it drops (the measured 27/259 = 10.4%) while the locatable sibling
    in the same row still ships"""
    Example: TrainingExample = SCRIPT.run(row(synth_row(ENT_PARAPHRASE)))

    assert [entity.label for entity in Example.entities] == ["Name"]
    assert Example.entities[0].mentions == ["Marie Curie"]


def test_a_present_closing_fence_is_tolerated() -> None:
    """measured on the full split: every row's fenced block ends at the JSON object with NO
    closing fence, and raw_decode stops at the object's end, so the real fixtures above already
    exercise the fence-less path; this test pins the other half: a fence present (a drifted
    revision, or a builder that adds one) must parse to the identical payload"""
    Fenced: str = ROW_ANDREW_REYNOLDS + "\n```"
    Example: TrainingExample = SCRIPT.run(row(Fenced))

    assert [entity.label for entity in Example.entities] == [
        "Name",
        "Age",
        "Specialty",
        "PersonalMotivation",
        "AdvocacyCause",
        "FocusArea",
    ]


def test_a_non_string_text_value_never_raises() -> None:
    """hub streaming hands back None or stringified values depending on cache state; run() must
    degrade to the empty example instead of crashing the stream worker"""
    for Drifted in (None, 42, ["not", "a", "string"]):
        Example: TrainingExample = SCRIPT.run(row(Drifted))

        assert Example.text == ""
        assert Example.entities == []


def test_nonzero_yield_over_a_handful_of_realistic_rows() -> None:
    """end-to-end yield on real rows: every realistic row emits at least one entity, and five
    unmodified rows ship 28 of their 29 entries (only TONYA_TROSTEL hits the measured paraphrase
    drop rule, its Age "34"; its other four entries ship)"""
    Rows: list[str] = [
        ROW_ANDREW_REYNOLDS,
        ROW_EMILY_TURNER,
        ROW_MICHAEL_JOHNSON,
        ROW_SARAH_LOPEZ,
        ROW_TONYA_TROSTEL,
    ]
    Examples: list[TrainingExample] = [SCRIPT.run(row(each)) for each in Rows]

    assert all(example.entities for example in Examples)
    assert sum(len(entity.mentions) for example in Examples for entity in example.entities) == 28


def test_every_emitted_mention_occurs_in_the_emitted_text() -> None:
    """the gliner2 validator rule over every entity of every emitted example: a mention that is
    not a literal substring of the text would be sanitized downstream, so this is the suite-level
    containment lock"""
    Rows: list[str] = [
        ROW_ANDREW_REYNOLDS,
        ROW_EMILY_TURNER,
        ROW_MICHAEL_JOHNSON,
        ROW_SARAH_LOPEZ,
        ROW_TONYA_TROSTEL,
    ]
    for each in Rows:
        Example: TrainingExample = SCRIPT.run(row(each))
        for entity in Example.entities:
            for mention in entity.mentions:
                assert mention in Example.text


def test_label_census_edges_name_age_and_a_hapax_label_ship() -> None:
    """Name and Age are the two census heads (50/50 rows each) and must ship on a well-formed row;
    PersonalMotivation is a measured hapax (30 of 50 labels occur once) and must ship as a raw
    PascalCase tail, proving hapax labels are not filtered anywhere in the chain"""
    Example: TrainingExample = SCRIPT.run(row(ROW_ANDREW_REYNOLDS))

    assert Example.entities[0].label == "Name"
    assert Example.entities[1].label == "Age"
    assert "PersonalMotivation" in [entity.label for entity in Example.entities]
    Emily: TrainingExample = SCRIPT.run(row(ROW_EMILY_TURNER))
    emily_labels: set[str] = {entity.label for entity in Emily.entities}
    assert {"CareerDedication", "ImplementedSolution", "TargetGroup"} <= emily_labels


def test_dispatch_routes_through_the_registry_and_keeps_declared_outputs() -> None:
    """the ingest pipeline invokes Script.dispatch with the declared outputs tuple; the script
    must answer through that path, not only via a direct instance call"""
    Outputs, Example = Script.dispatch("MedicalEntityJsonScript", (("entities",), row(ROW_ANDREW_REYNOLDS)))

    assert Outputs == ("entities",)
    assert isinstance(Example, TrainingExample)
    assert [entity.label for entity in Example.entities][0] == "Name"


def test_the_existing_registry_is_unaffected_by_the_new_script() -> None:
    """all edits are additive: the CtkpInterventionsScript entry must survive the new import"""
    assert isinstance(Script.REGISTRY["CtkpInterventionsScript"], CtkpInterventionsScript)
