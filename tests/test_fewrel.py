from __future__ import annotations

from typing import Any

from relmedner.models import TrainingExample
from relmedner.scripts import CtkpInterventionsScript  # registry must stay populated alongside the new script
from relmedner.scripts.fewrel import FewRelScript, entity_runs, resolve_pid_label, strip_participant_suffix
from relmedner.types import Script
from relmedner.utils import ScriptUtils

SCRIPT: FewRelScript = FewRelScript()

# real train_wiki P931 row shape from the wenceslaus probe: gold head "TJQ" (index 16) and
# multi-token tail "Tanjung Pandan" (indices 13-14), converter-resolved snake_case label
TRAIN_TOKENS: list[str] = [
    "Merpati",
    "flight",
    "106",
    "departed",
    "Jakarta",
    "(",
    "CGK",
    ")",
    "on",
    "a",
    "domestic",
    "flight",
    "to",
    "Tanjung",
    "Pandan",
    "(",
    "TJQ",
    ")",
    ".",
]
TRAIN_TEXT: str = ScriptUtils.join_tokens(TRAIN_TOKENS)

# real train_wiki multi-run shape from the probe: the head entity carries two surviving
# contiguous runs, so it must emit one mention per run
MULTIRUN_TOKENS: list[str] = [
    "Route",
    "41",
    "ends",
    "just",
    "north",
    "of",
    "the",
    "Richmond",
    "-",
    "Pittsfield",
    "line",
    "at",
    "U.S.",
    "Route",
    "20",
    ",",
    "just",
    "east",
    "of",
    "the",
    "Hancock",
    "Shaker",
    "Village",
    "in",
    "Hancock",
    "...",
]
MULTIRUN_TEXT: str = ScriptUtils.join_tokens(MULTIRUN_TOKENS)

# val_semeval self-loop shape: head and tail first surviving runs differ only in case, so the
# case-insensitive comparison must drop the relation while both gold spans still ship
SELFLOOP_TOKENS: list[str] = [
    "the",
    "system",
    "reports",
    "that",
    "the",
    "configuration",
    "of",
    "antenna",
    "Elements",
    "was",
    "checked",
    "by",
    "elements",
    "today",
    ".",
]

# pubmed_unsupervised shape from the probe: gold spans with an empty relation label
PUBMED_TOKENS: list[str] = [
    "the",
    "contribution",
    "of",
    "brush",
    "border",
    "cytoskeletal",
    "proteins",
    "(",
    "actin",
    ",",
    "myosin",
    ")",
    "in",
    "intestinal",
    "brush",
    "border",
    "was",
    "examined",
    ".",
]


def train_record(**overrides: Any) -> tuple[Any, ...]:
    """one whole avro record as LocalAvroDataStream delivers it: the converter contract is
    exactly {"tokens", "label", "h_indices", "t_indices"} (no h_text/t_text, no P-id map)"""
    base: dict[str, Any] = {
        "tokens": TRAIN_TOKENS,
        "label": "place_served_by_transport_hub",
        "h_indices": [[16]],
        "t_indices": [[13, 14]],
    }
    return ({**base, **overrides},)


def test_the_script_self_registers_under_its_declared_name() -> None:
    """importing the module must install the instance in the shared registry (Script.dispatch
    resolves by this exact NAME key, so a wrong NAME silently breaks ingest routing)"""
    assert FewRelScript.NAME == "FewRelScript"
    assert isinstance(Script.REGISTRY["FewRelScript"], FewRelScript)


def test_the_declared_category_is_a_biolink_class() -> None:
    """NamedThing must stay a real tablassert Categories member; the import-time guard only
    protects THIS class's constant, so the contract is re-asserted here for drift"""
    assert ScriptUtils.is_biolink_category(FewRelScript.CATEGORY)


def test_the_existing_registry_is_unaffected_by_the_new_script() -> None:
    """all edits are additive: the CtkpInterventionsScript entry must survive the new import"""
    assert isinstance(Script.REGISTRY["CtkpInterventionsScript"], CtkpInterventionsScript)


def test_a_wellformed_row_ships_text_entities_and_snake_case_relation() -> None:
    """the probe's real P931 row is the contract anchor: text is the re-joined token stream, both
    gold spans ship as NamedThing mentions, and the converter-resolved snake_case label survives
    normalization untouched"""
    Example: TrainingExample = SCRIPT.run(train_record())

    assert Example.text == TRAIN_TEXT
    assert [entity.label for entity in Example.entities] == ["NamedThing"]
    assert Example.entities[0].mentions == ["TJQ", "Tanjung Pandan"]
    assert len(Example.relations) == 1
    Relation = Example.relations[0]
    assert Relation.name == "place_served_by_transport_hub"
    assert [(field.name, field.value) for field in Relation.fields] == [("head", "TJQ"), ("tail", "Tanjung Pandan")]
    assert Relation.evidence == "asserted"
    assert Relation.negated is False


def test_a_row_with_semeval_participant_suffix_resolves_to_the_bare_predicate() -> None:
    """val_semeval labels carry "(e1,e2)"/"(e2,e1)" slot suffixes on 17 labels; the suffix is
    slot metadata and must never leak into the emitted predicate name"""
    Suffix = "Component-Of(e1,e2)"
    Example: TrainingExample = SCRIPT.run(train_record(label=Suffix))

    assert Example.relations[0].name == "component_of"


def test_strip_participant_suffix_strips_both_slots_and_optional_space() -> None:
    """the converter reuses this helper at build time, so both slot directions and the spaced
    variant must strip to the same bare name; non-suffixed labels return unchanged"""
    assert strip_participant_suffix("Component-Of(e1,e2)") == "Component-Of"
    assert strip_participant_suffix("Product-Producer(e2,e1)") == "Product-Producer"
    assert strip_participant_suffix("Message-Topic (e1,e2)") == "Message-Topic"
    assert strip_participant_suffix("place_served_by_transport_hub") == "place_served_by_transport_hub"


def test_resolve_pid_label_resolves_through_pid2name_and_drops_a_missing_pid() -> None:
    """the converter's build-time chain: a bare P<digits> id resolves through pid2name[pid][0];
    a P-id with no entry yields "" (skip-don't-coerce, never a guessed name), non-P-id labels
    pass through, and suffix stripping happens before the P-id check"""
    pid2name: dict[str, Any] = {"P931": ["place served by transport hub", "location served"], "P17": "Germany"}
    assert resolve_pid_label("P931", pid2name) == "place served by transport hub"
    assert resolve_pid_label("P999", pid2name) == ""
    assert resolve_pid_label("P17", pid2name) == ""
    assert resolve_pid_label("component_of", pid2name) == "component_of"
    assert resolve_pid_label("Component-Of(e1,e2)", pid2name) == "Component-Of"


def test_self_loop_drops_the_relation_but_still_ships_entities() -> None:
    """9 measured rows (all val_semeval) mark head and tail with the same surface differing only
    in case; training a self-relation would teach the model to link a span to itself, so the
    relation drops while both gold spans still train as entities"""
    SelfLoop = ({"tokens": SELFLOOP_TOKENS, "label": "Participant-Property(e1,e2)", "h_indices": [[12]], "t_indices": [[8]]},)
    Sibling = ({"tokens": SELFLOOP_TOKENS, "label": "Participant-Property(e1,e2)", "h_indices": [[12]], "t_indices": [[5]]},)

    Dropped: TrainingExample = SCRIPT.run(SelfLoop)
    assert Dropped.relations == []
    assert [entity.label for entity in Dropped.entities] == ["NamedThing"]
    assert Dropped.entities[0].mentions == ["elements", "Elements"]

    Survivor: TrainingExample = SCRIPT.run(Sibling)
    assert [field.value for field in Survivor.relations[0].fields] == ["elements", "configuration"]


def test_an_unresolvable_pid_ships_entities_only_and_the_resolved_sibling_ships() -> None:
    """the container carries the raw label and the streaming run has no pid2name mapping, so a
    P-id that still reaches the script unresolved drops its relation rather than guessing a
    name; the well-formed resolved sibling on the same row must still ship its relation"""
    Unresolved: TrainingExample = SCRIPT.run(train_record(label="P931"))
    assert Unresolved.relations == []
    assert [entity.label for entity in Unresolved.entities] == ["NamedThing"]
    assert Unresolved.entities[0].mentions == ["TJQ", "Tanjung Pandan"]

    Resolved: TrainingExample = SCRIPT.run(train_record())
    assert Resolved.relations[0].name == "place_served_by_transport_hub"


def test_out_of_bounds_index_runs_drop_alone_while_sibling_runs_survive() -> None:
    """an index outside 0 <= i < len(tokens) points at nothing and must drop that run alone;
    per-run filtering (not per-row) keeps the valid sibling run and the relation trainable"""
    for bad_run in ([[99]], [[-1]]):
        Example: TrainingExample = SCRIPT.run(train_record(h_indices=[[16], bad_run[0]]))
        assert Example.entities[0].mentions == ["TJQ", "Tanjung Pandan"]
        assert [field.value for field in Example.relations[0].fields] == ["TJQ", "Tanjung Pandan"]


def test_empty_tokens_yield_the_empty_text_only_example() -> None:
    """an empty token stream cannot host index runs and would break downstream tokenization;
    ship the canonical empty example the declared-outputs filter drops, and handle a drifted
    missing tokens field the same way instead of raising"""
    for Empty in (train_record(tokens=[]), train_record(tokens=None)):
        Example: TrainingExample = SCRIPT.run(Empty)
        assert Example.text == ""
        assert Example.entities == []
        assert Example.relations == []
        assert Example.populated() == frozenset()

    Sibling: TrainingExample = SCRIPT.run(train_record())
    assert Sibling.text == TRAIN_TEXT


def test_malformed_indices_shapes_drop_the_run_without_raising() -> None:
    """indices may arrive as any drifted shape; only list-of-list-of-int survives. A dropped
    head leaves the tail sibling mention shipping (per-run skip, not a crash), matching the
    skip-don't-coerce invariant for a stream worker"""
    Malformed: list[Any] = [
        "16",  # not a list at all
        [],  # right container, no runs
        [[]],  # empty run
        [["16"]],  # str element, not an int
        [[True]],  # bool is an int subclass but schema corruption
        [[16, 18]],  # non-contiguous gap would fabricate a surface
    ]
    for bad_head in Malformed:
        Example: TrainingExample = SCRIPT.run(train_record(h_indices=bad_head))
        assert [entity.label for entity in Example.entities] == ["NamedThing"]
        assert Example.entities[0].mentions == ["Tanjung Pandan"]
        assert Example.relations == []

    Sibling: TrainingExample = SCRIPT.run(train_record())
    assert Sibling.entities[0].mentions == ["TJQ", "Tanjung Pandan"]


def test_entity_runs_surface_one_mention_per_contiguous_run() -> None:
    """multi-run gold spans are measured reality (1,624 2-run, 68 3-run, 5 4-run of 89,600
    train_wiki mentions); each contiguous run must emit its own mention, not be squashed"""
    assert entity_runs(MULTIRUN_TOKENS, [[9], [25]]) == ["Pittsfield", "..."]
    assert entity_runs(TRAIN_TOKENS, [[13, 14]]) == ["Tanjung Pandan"]
    assert entity_runs(TRAIN_TOKENS, "16") == []
    assert entity_runs(TRAIN_TOKENS, None) == []


def test_a_multirun_row_emits_one_mention_per_run() -> None:
    """end-to-end over the probe's real multi-run row: both head runs survive as mentions under
    one NamedThing label and the first run fills the relation's head field"""
    MultiRun = ({"tokens": MULTIRUN_TOKENS, "label": "place_served_by_transport_hub", "h_indices": [[9], [25]], "t_indices": [[20, 21, 22, 23, 24]]},)
    Example: TrainingExample = SCRIPT.run(MultiRun)

    assert Example.text == MULTIRUN_TEXT
    assert Example.entities[0].mentions == ["Pittsfield", "...", "Hancock Shaker Village in Hancock"]
    assert len(Example.relations) == 1
    assert Example.relations[0].fields[0].value == "Pittsfield"


def test_a_row_with_an_empty_label_ships_entities_only() -> None:
    """pubmed_unsupervised rows (2,500 measured) carry gold spans but no relation label; the
    entity half must still train and the empty predicate must never emit an empty relation"""
    Unlabeled: TrainingExample = SCRIPT.run(train_record(tokens=PUBMED_TOKENS, label="", h_indices=[[3, 4, 5, 6]], t_indices=[[8], [10]]))
    assert Unlabeled.relations == []
    assert Unlabeled.entities[0].mentions == ["brush border cytoskeletal proteins", "actin", "myosin"]
    assert Unlabeled.populated() == frozenset({"entities"})

    Sibling: TrainingExample = SCRIPT.run(train_record())
    assert len(Sibling.relations) == 1


def test_realistic_multirun_fixtures_all_yield_nonempty_examples() -> None:
    """nonzero-yield guard against the silent zero-yield bug class: every realistic split shape
    (train_wiki, val_semeval suffix, val_pubmed native snake_case, multi-run) must produce a
    populated example, and the mixed batch must not lose rows"""
    Fixtures: list[tuple[Any, ...]] = [
        train_record(),
        train_record(label="Component-Of(e1,e2)"),
        train_record(tokens=PUBMED_TOKENS, label="was_examined", h_indices=[[3, 4, 5, 6]], t_indices=[[8], [10]]),
        ({"tokens": MULTIRUN_TOKENS, "label": "place_served_by_transport_hub", "h_indices": [[9], [25]], "t_indices": [[20, 21, 22, 23, 24]]},),
    ]
    Yielded: list[TrainingExample] = [SCRIPT.run(record) for record in Fixtures]

    assert len(Yielded) == len(Fixtures)
    assert all(example.populated() for example in Yielded)
    assert sum(len(example.relations) for example in Yielded) == 4


def test_every_emitted_surface_is_a_verbatim_substring_of_the_text() -> None:
    """InputExample.validate() requires every entity mention and every relation field value to
    occur in the text; surfaces are token slices of the same list join_tokens consumes, and this
    asserts it over EVERY emitted surface of the realistic fixtures, not just one"""
    Fixtures: list[tuple[Any, ...]] = [
        train_record(),
        train_record(label="Component-Of(e1,e2)"),
        train_record(tokens=PUBMED_TOKENS, label="was_examined", h_indices=[[3, 4, 5, 6]], t_indices=[[8], [10]]),
        ({"tokens": MULTIRUN_TOKENS, "label": "place_served_by_transport_hub", "h_indices": [[9], [25]], "t_indices": [[20, 21, 22, 23, 24]]},),
        train_record(tokens=SELFLOOP_TOKENS, label="Participant-Property(e1,e2)", h_indices=[[12]], t_indices=[[5]]),
    ]
    for record in Fixtures:
        Example: TrainingExample = SCRIPT.run(record)
        surfaces: list[str] = [
            *(mention for entity in Example.entities for mention in entity.mentions),
            *(field.value for relation in Example.relations for field in relation.fields),
        ]
        assert surfaces, "fixture must emit at least one surface"
        assert all(surface in Example.text for surface in surfaces)


def test_dispatch_routes_through_the_registry() -> None:
    """ingests invoke Script.dispatch by the declared NAME key; the script must answer through
    that path, not only via a direct instance call"""
    _outputs, Example = Script.dispatch("FewRelScript", ((), train_record()))

    assert isinstance(Example, TrainingExample)
    assert Example.relations[0].name == "place_served_by_transport_hub"
