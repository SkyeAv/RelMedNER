"""FULLMAP_DIR env override: the beam sdkworker container must be able to point the
constant at its own mount without touching the baked default. The envless default is
asserted as captured (not as a literal), because the remote sync routine rewrites the
source literal to the host's Desktop path before the suite runs."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

import relmedner.constants  # noqa: F401  (the reload target must exist in sys.modules)

DEFAULT_WITHOUT_ENV: Path = relmedner.constants.FULLMAP_DIR


def test_fullmap_dir_default_is_stable_without_the_env() -> None:
    Constants = importlib.reload(sys.modules["relmedner.constants"])
    assert Constants.FULLMAP_DIR == DEFAULT_WITHOUT_ENV
    assert Constants.FULLMAP_DIR.is_absolute()


def test_fullmap_dir_env_override_points_at_the_worker_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RELMEDNER_FULLMAP_DIR", "/opt/fullmap")
    Constants = importlib.reload(sys.modules["relmedner.constants"])
    assert Constants.FULLMAP_DIR == Path("/opt/fullmap")


def test_fullmap_dir_empty_env_falls_back_to_the_envless_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RELMEDNER_FULLMAP_DIR", "")
    Constants = importlib.reload(sys.modules["relmedner.constants"])
    assert Constants.FULLMAP_DIR == DEFAULT_WITHOUT_ENV


@pytest.fixture(autouse=True)
def _restore_constants(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("RELMEDNER_FULLMAP_DIR", raising=False)
    yield
    importlib.reload(sys.modules["relmedner.constants"])
