from __future__ import annotations

from collections.abc import Sequence

import pytest

from relmedner import deploy
from relmedner.constants import BLOB_SERVER_PORT, JOBMANAGER_RPC_PORT, TASKMANAGER_DATA_PORT
from relmedner.deploy import data_port, image_id, tunnel_plan, tunnel_spec
from relmedner.models import WorkerNode


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


def test_data_port_stems_from_the_canonical_port_by_worker_index() -> None:
    assert data_port(0) == TASKMANAGER_DATA_PORT
    assert data_port(1) == TASKMANAGER_DATA_PORT + 1


def test_tunnel_spec_advertises_target_ports_on_loopback() -> None:
    assert tunnel_spec([16123, 16124], "10.2.9.11", "sgoetz") == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-N",
        "-L",
        f"127.0.0.1:{JOBMANAGER_RPC_PORT}:127.0.0.1:{JOBMANAGER_RPC_PORT}",
        "-L",
        f"127.0.0.1:{BLOB_SERVER_PORT}:127.0.0.1:{BLOB_SERVER_PORT}",
        "sgoetz@10.2.9.11",
    ]


def test_tunnel_plan_wires_every_taskmanager_to_the_head_and_its_peers() -> None:
    workers: list[WorkerNode] = [
        WorkerNode(host="10.2.9.11", slots=64, memory="112g", fullmap="/f", outputs="/o"),
        WorkerNode(host="10.2.9.19", slots=16, memory="40g", fullmap="/f", outputs="/o"),
    ]

    plan: dict[str, dict[str, list[int]]] = tunnel_plan(workers, "10.2.9.11")

    assert plan == {
        # head needs only the peer's data port
        "10.2.9.11": {"10.2.9.19": [data_port(1)]},
        # the remote needs the head's rpc + blob ports and the head's data port
        "10.2.9.19": {"10.2.9.11": [JOBMANAGER_RPC_PORT, BLOB_SERVER_PORT, data_port(0)]},
    }


def test_tunnel_plan_single_node_needs_no_tunnels() -> None:
    workers: list[WorkerNode] = [WorkerNode(host="10.2.9.11", slots=64, memory="112g", fullmap="/f", outputs="/o")]

    assert tunnel_plan(workers, "10.2.9.11") == {}


def test_compose_command_falls_back_to_the_nix_profile_standalone(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Sequence[str]] = []

    def fake_run(cmd, capture_output=False, check=False, env=None):  # noqa: ANN001, ANN202, FBT002
        calls.append(cmd)
        if cmd[-2:] == ["compose", "version"]:
            return FakeCompleted(1, b"")
        return FakeCompleted(0, b"/users/sgoetz/.nix-profile/bin/docker-compose\n")

    monkeypatch.setattr(deploy, "run", fake_run)
    Worker: WorkerNode = WorkerNode(host="10.2.9.19", slots=16, memory="40g", fullmap="/f", outputs="/o")

    command: list[str] = deploy.compose_command("sgoetz", Worker)

    # the probe must travel as one argv element — ssh would otherwise re-parse the remote line and
    # silently run the bare `command` builtin
    assert "command -v docker-compose" in calls[1][-1]
    assert "sh" not in calls[1]
    assert command[-3:] == ["/users/sgoetz/.nix-profile/bin/docker-compose", "-p", "relmedner"]
    assert command[:3] == ["ssh", "-o", "BatchMode=yes"]
    assert "sgoetz@10.2.9.19" in command
    assert "sgoetz@10.2.9.19" in calls[0]


def test_compose_command_fails_loud_on_an_empty_standalone_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, capture_output=False, check=False, env=None):  # noqa: ANN001, ANN202, FBT002
        if cmd[-2:] == ["compose", "version"]:
            return FakeCompleted(1, b"")
        return FakeCompleted(0, b"")

    monkeypatch.setattr(deploy, "run", fake_run)
    Worker: WorkerNode = WorkerNode(host="10.2.9.19", slots=16, memory="40g", fullmap="/f", outputs="/o")

    with pytest.raises(SystemExit, match="no docker compose plugin"):
        deploy.compose_command("sgoetz", Worker)
