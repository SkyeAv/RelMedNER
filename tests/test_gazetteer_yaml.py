from __future__ import annotations

import logging
import pickle

import pytest
from pydantic import ValidationError

from relmedner import gazetteer
from relmedner.gazetteer import (
    BUILTIN_PREDICATE_TRIGGERS,
    configure_gazetteer,
    extract_relations,
    report_zero_emission_triggers,
)
from relmedner.models import GazetteerPredicate, GazetteerSpec, YamlIngests
from relmedner.pipeline import ResolveMinedBatches


@pytest.fixture(autouse=True)
def restore_builtin_gazetteer_tables():
    """every test here reconfigures the module tables; reset to the builtins on teardown so the
    pinned-vocabulary locks in test_gazetteer.py see the shipped table no matter the test order"""
    yield
    configure_gazetteer(None)


def test_no_section_leaves_the_tables_byte_identical_to_the_builtins() -> None:
    """gazetteer: absent (or configure(None)) must reproduce today's builtin table exactly"""
    snapshot = dict(gazetteer.PREDICATE_TRIGGERS)
    configure_gazetteer(None)
    assert gazetteer.PREDICATE_TRIGGERS == snapshot == dict(BUILTIN_PREDICATE_TRIGGERS)


def test_a_yaml_declared_predicate_fires_in_extract_relations(caplog: pytest.LogCaptureFixture) -> None:
    """the additive merge must make a YAML-only phrase scan and emit like a builtin one"""
    configure_gazetteer(GazetteerSpec.model_validate({"predicates": [{"name": "treats", "triggers": [["cures"]]}]}))
    Tokens: list[str] = ["Aspirin", "cures", "migraine"]
    Spans: list[tuple[int, int, str]] = [(0, 0, "ChemicalEntity"), (2, 2, "Disease")]
    assert [relation.name for relation in extract_relations(Tokens, Spans)] == ["treats"]
    # a fired declared predicate must not be reported as zero-emission
    with caplog.at_level(logging.WARNING, logger="relmedner.gazetteer"):
        report_zero_emission_triggers()
    assert not any("treats" in record.message for record in caplog.records)


def test_an_unknown_predicate_is_a_validation_error() -> None:
    """a typo'd predicate would emit KGX edges that fail Biolink validation downstream"""
    with pytest.raises(ValidationError, match="Predicates member"):
        GazetteerPredicate.model_validate({"name": "heals", "triggers": [["heals"]]})


def test_an_empty_triggers_list_is_a_validation_error() -> None:
    """a predicate with no phrases can never fire and only masks an authoring bug"""
    with pytest.raises(ValidationError, match="at least 1 item"):
        GazetteerPredicate.model_validate({"name": "treats", "triggers": []})


def test_an_uppercase_token_is_a_validation_error() -> None:
    """uppercase table entries can never match the lowercased scan input, so they are bugs"""
    with pytest.raises(ValidationError, match="uppercase token"):
        GazetteerPredicate.model_validate({"name": "treats", "triggers": [["Cures"]]})


def test_a_phrase_claimed_by_yaml_and_a_builtin_predicate_is_a_validation_error() -> None:
    """("caused", "by") ships under builtin caused_by; YAML treats may not claim it"""
    Spec = GazetteerSpec.model_validate({"predicates": [{"name": "treats", "triggers": [["caused", "by"]]}]})
    with pytest.raises(ValidationError, match="claimed by both"):
        configure_gazetteer(Spec)


def test_a_phrase_claimed_by_two_yaml_predicates_is_a_validation_error() -> None:
    """ownership conflicts inside the YAML section itself fail at parse time, before configure"""
    with pytest.raises(ValidationError, match="claimed by both"):
        GazetteerSpec.model_validate(
            {
                "predicates": [
                    {"name": "treats", "triggers": [["cures"]]},
                    {"name": "causes", "triggers": [["cures"]]},
                ]
            }
        )


def test_an_unknown_key_in_the_section_is_a_validation_error() -> None:
    """extra=forbid: a typo'd section key fails loudly instead of being silently ignored"""
    with pytest.raises(ValidationError, match="Extra inputs"):
        YamlIngests.model_validate(
            {
                "datasets": [],
                "gazetteer": {"predicates": [{"name": "treats", "triggers": [["cures"]]}], "bogus_key": 1},
            }
        )


def test_configure_is_idempotent_across_two_parses() -> None:
    """two parses of the same YAML rebuild the same table; the second configure must not append
    the YAML phrases a second time"""
    Payload = {
        "datasets": [],
        "gazetteer": {"predicates": [{"name": "treats", "triggers": [["cures"]]}, {"name": "precedes", "triggers": [["antedates"]]}]},
    }
    First: YamlIngests = YamlIngests.model_validate(Payload)
    configure_gazetteer(First.gazetteer)
    FirstTable = dict(gazetteer.PREDICATE_TRIGGERS)
    Second: YamlIngests = YamlIngests.model_validate(Payload)
    configure_gazetteer(Second.gazetteer)
    assert gazetteer.PREDICATE_TRIGGERS == FirstTable
    assert gazetteer.PREDICATE_TRIGGERS["treats"][-1] == ("cures",)
    assert gazetteer.PREDICATE_TRIGGERS["precedes"] == (("followed", "by"), ("prior", "to"), ("preceded", "by"), ("antedates",))


def test_a_zero_emission_declared_key_logs_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    """the US-009 zero-yield lesson applied to the gazetteer: a declared predicate that never
    fires must warn exactly once, never silently ship an empty relation arm"""
    configure_gazetteer(GazetteerSpec.model_validate({"predicates": [{"name": "precedes", "triggers": [["antedates"]]}]}))
    Logger = logging.getLogger("relmedner.gazetteer")
    with caplog.at_level(logging.WARNING, logger=Logger.name):
        report_zero_emission_triggers()
        report_zero_emission_triggers()
    Warnings = [record for record in caplog.records if record.levelno == logging.WARNING and "precedes" in record.message]
    assert len(Warnings) == 1


@pytest.mark.parametrize(
    "declared",
    [
        {"qualifiers": [{"slot": "anatomical_context"}]},
        {"negation_cues": [["did", "not"]]},
    ],
)
def test_qualifier_and_negation_sections_fail_loudly_until_pr_22_lands(declared: dict) -> None:
    """the qualifier/negation scanner lives on the add-qualifiers branch (PR #22, unmerged):
    declaring these sections now must raise a structured substrate error at configure time,
    never be silently accepted and ignored"""
    Spec = GazetteerSpec.model_validate(declared)
    with pytest.raises(NotImplementedError, match=r"#22"):
        configure_gazetteer(Spec)


WORKER_TOKENS: list[str] = ["Aspirin", "cures", "migraine"]
WORKER_SPANS: list[tuple[int, int, str]] = [(0, 0, "ChemicalEntity"), (2, 2, "Disease")]


def test_the_mining_dofn_reapplies_the_spec_on_a_simulated_fresh_worker() -> None:
    """BLOCKER regression: Beam pickles the DoFn instance (spec rides as data) but Flink
    sdkworkers re-import relmedner.gazetteer fresh, seeing only builtins. Simulate the worker
    by reverting the module table to the builtins AFTER driver-side construction + pickling,
    then unpickle + setup and assert the YAML-declared predicate fires in extract_relations"""
    Spec = GazetteerSpec.model_validate({"predicates": [{"name": "treats", "triggers": [["cures"]]}]})
    Fn = ResolveMinedBatches(Spec, {})
    configure_gazetteer(None)  # fresh sdkworker import state: builtin table only
    WorkerFn = pickle.loads(pickle.dumps(Fn))
    WorkerFn.setup()
    assert [relation.name for relation in extract_relations(WORKER_TOKENS, WORKER_SPANS)] == ["treats"]


def test_the_mining_dofn_with_none_spec_keeps_the_builtins_on_the_worker() -> None:
    """a spec-less run must not leak the driver's configure across the wire either: setup()
    with gazetteer=None rebuilds the builtin table exactly, so a yaml-only phrase stays dead"""
    Spec = GazetteerSpec.model_validate({"predicates": [{"name": "treats", "triggers": [["cures"]]}]})
    configure_gazetteer(Spec)  # a different driver run configured this process
    WorkerFn = pickle.loads(pickle.dumps(ResolveMinedBatches(None, {})))
    WorkerFn.setup()
    assert gazetteer.PREDICATE_TRIGGERS == dict(BUILTIN_PREDICATE_TRIGGERS)
    assert extract_relations(WORKER_TOKENS, WORKER_SPANS) == []
