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
