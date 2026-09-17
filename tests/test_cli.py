import pytest

from relmedner.cli import APP
from relmedner.constants import TEST_ROW_LIMIT
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
