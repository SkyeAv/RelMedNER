"""FULLMAP_DIR env override: the beam sdkworker container must be able to point the
constant at its own mount without touching the baked default. The envless default is
asserted as captured (not as a literal), because the remote sync routine rewrites the
source literal to the host's Desktop path before the suite runs."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

import relmedner.constants  # noqa: F401  (the reload target must exist in sys.modules)

# the capture itself must be envless: the remote gate exports RELMEDNER_FULLMAP_DIR pointing at
# $HOME/Desktop/fullmap, and $HOME resolves to /users/sgoetz there while the synced source literal
# reads /home/sgoetz -- the same tree under two names, so a polluting env would make every reload
# comparison a string mismatch between two valid spellings of one path
os.environ.pop("RELMEDNER_FULLMAP_DIR", None)
# another test module may have imported constants while the gate's override was set, so the
# capture must force a reload under the popped env instead of reading the cached attribute
importlib.reload(sys.modules.setdefault("relmedner.constants", relmedner.constants))

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
