from itertools import count

import pytest

from relmedner.cli import APP
from relmedner.pipeline import limit_rows


@pytest.mark.parametrize("flag", ["--test-run", "-t"])
def test_build_dataset_test_run_flag_parses(flag: str) -> None:
    Command, Bound, _ = APP.parse_args(["build-dataset", flag], exit_on_error=False)
    assert Command.__name__ == "build_dataset"
    assert Bound.arguments.get("test_run", False) is True


def test_build_dataset_default_no_test_run() -> None:
    Command, Bound, _ = APP.parse_args(["build-dataset"], exit_on_error=False)
    assert Command.__name__ == "build_dataset"
    assert Bound.arguments.get("test_run", False) is False


def test_limit_rows_test_run_yields_ten() -> None:
    Limited: list[int] = list(limit_rows(iter(count()), test_run=True))
    assert Limited == list(range(10))


def test_limit_rows_full_run_yields_all() -> None:
    Rows: list[int] = list(limit_rows(iter(range(20)), test_run=False))
    assert Rows == list(range(20))
