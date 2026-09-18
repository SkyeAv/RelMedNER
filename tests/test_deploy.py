from __future__ import annotations

from collections.abc import Sequence

import pytest

from relmedner import deploy
from relmedner.deploy import image_id, tunnel_spec


class FakeCompleted:
    def __init__(self, returncode: int, stdout: bytes) -> None:
        self.returncode: int = returncode
        self.stdout: bytes = stdout


def test_image_id_returns_the_content_id_for_a_present_image(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Sequence[str]] = []

    def fake_run(cmd, capture_output=False, check=False, env=None):  # noqa: ANN001, ANN202, FBT002
        seen.append(cmd)
        return FakeCompleted(0, b"sha256:abc\n")

    monkeypatch.setattr(deploy, "run", fake_run)

    assert image_id(["docker"], "img:1") == "sha256:abc"
    assert seen == [["docker", "image", "inspect", "--format", "{{.Id}}", "img:1"]]


def test_image_id_is_empty_when_the_image_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deploy, "run", lambda *a, **k: FakeCompleted(1, b""))

    assert image_id(["ssh", "host", "docker"], "img:1") == ""


def test_tunnel_spec_forwards_the_jobmanager_port_to_the_remote_data_port() -> None:
    assert tunnel_spec(16126, "10.2.9.11", "sgoetz", "10.0.0.5") == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-N",
        "-L",
        "10.0.0.5:16126:127.0.0.1:16125",
        "sgoetz@10.2.9.11",
    ]
