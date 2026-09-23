import inspect

import pytest
from cyclopts.exceptions import CoercionError

from relmedner.cli import APP, build_dataset
from relmedner.constants import TEST_ROW_LIMIT
from relmedner.enums import DedupMode
from relmedner.models import RunConfig


@pytest.mark.parametrize("flag", ["--test-run", "-t"])
def test_build_dataset_test_run_flag_parses(flag: str) -> None:
    Command, Bound, _ = APP.parse_args(["build-dataset", flag], exit_on_error=False)
    assert Command.__name__ == "build_dataset"
    assert Bound.arguments.get("test_run", False) is True


def test_build_dataset_default_no_test_run() -> None:
    Command, Bound, _ = APP.parse_args(["build-dataset"], exit_on_error=False)
    assert Command.__name__ == "build_dataset"
    assert Bound.arguments.get("test_run", False) is False


def test_run_config_from_test_run_flag_samples_five() -> None:
    Config: RunConfig = RunConfig.from_flags(True)
    assert Config.sample_limit == TEST_ROW_LIMIT == 5


def test_run_config_from_full_run_flag_is_unlimited() -> None:
    Config: RunConfig = RunConfig.from_flags(False)
    assert Config.sample_limit is None


# ---------------------------------------------------------------- dedup mode (US-004) --


@pytest.mark.parametrize("value", ["off", "exact", "near"])
def test_build_dataset_dedup_mode_flag_parses(value: str) -> None:
    """REQ-INT-3: --dedup-mode coerces each accepted value to its DedupMode member (cyclopts
    derives the allowed choices from the enum type itself)"""
    Command, Bound, _ = APP.parse_args(["build-dataset", "--dedup-mode", value], exit_on_error=False)
    assert Command.__name__ == "build_dataset"
    assert Bound.arguments.get("dedup_mode") == DedupMode(value)


def test_build_dataset_default_dedup_mode_is_near() -> None:
    """REQ-INT-3: the flag defaults to near. cyclopts omits unbound defaults from
    Bound.arguments (verified against cyclopts 4.25.2), so the signature default is the
    source of truth for the parse-without-flag case"""
    assert inspect.signature(build_dataset).parameters["dedup_mode"].default is DedupMode.NEAR
    _Command, Bound, _ = APP.parse_args(["build-dataset"], exit_on_error=False)
    assert Bound.arguments.get("dedup_mode", DedupMode.NEAR) == DedupMode.NEAR


def test_build_dataset_rejects_invalid_dedup_mode() -> None:
    """REQ-INT-3: a garbage mode fails at argument parsing with a clean error, before any
    pipeline work; exit_on_error=False surfaces the same CoercionError the CLI renders"""
    with pytest.raises(CoercionError, match="Invalid value"):
        APP.parse_args(["build-dataset", "--dedup-mode", "garbage"], exit_on_error=False)


# ---------------------------------------------------------------- validate-trust (US-011) --


def test_validate_trust_command_registers_with_defaults() -> None:
    Command, Bound, _ = APP.parse_args(["validate-trust"], exit_on_error=False)
    assert Command.__name__ == "validate_trust_command"
    # all driver defaults are None at the CLI: an absent flag defers to the x-trust yaml
    # section (resolve_trust_settings owns the precedence, tested in test_trust.py)
    assert Bound.arguments.get("backend") is None
    assert Bound.arguments.get("report") is None
    assert Bound.arguments.get("sample_size") is None


def test_validate_trust_source_and_sample_size_parse() -> None:
    _Command, Bound, _ = APP.parse_args(
        ["validate-trust", "--source", "knowledgator/PubMedAbstractsNER", "--sample-size", "5", "-t"],
        exit_on_error=False,
    )
    assert Bound.arguments.get("source") == "knowledgator/PubMedAbstractsNER"
    assert Bound.arguments.get("sample_size") == 5
    assert Bound.arguments.get("test_run") is True
