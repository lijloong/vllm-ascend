# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-DP guard on the cloud Worker's runtime connection builder.

The method is extracted from the real source so the guard under test is the
production code, while heavyweight vLLM/torch imports are replaced by
package placeholders. This keeps the test runnable on a CPU-only host.
"""

from __future__ import annotations

import ast
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/worker/lwd_cloud_model/lwd_cloud_worker.py"
HOST_PACKAGE = "vllm_ascend.distributed.lwd_data_plane.host"
CHANNELS_MODULE = f"{HOST_PACKAGE}.lwd_host_channels"


@dataclass(frozen=True, order=True)
class _ConnectionId:
    edge_id: int
    cloud_id: int
    dp_idx: int


class _HostChannels:
    def __init__(self, has_listener: bool) -> None:
        self._UCX_LISTENERS = {0: object()} if has_listener else {}
        self.built: list[_ConnectionId] = []

    def lwd_add_runtime_connection(self, conn: SimpleNamespace) -> bool:
        self.built.append(conn.connection_id)
        return True


def _extract_worker() -> tuple[type, dict]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LwdCloudWorker")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "lwd_build_runtime_connection"
    )
    smoke = ast.ClassDef(
        name="WorkerSmoke",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                smoke,
            ],
            type_ignores=[],
        )
    )
    namespace: dict = {"Any": object}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["WorkerSmoke"], namespace


WorkerSmoke, _NAMESPACE = _extract_worker()
OWN_DP = _ConnectionId(1, 0, 0)
OTHER_DP = _ConnectionId(1, 0, 1)


def _install_channels(monkeypatch: pytest.MonkeyPatch, channels: _HostChannels) -> None:
    """Satisfy the method's runtime import without loading the UCX stack."""
    for name in (
        "vllm_ascend",
        "vllm_ascend.distributed",
        "vllm_ascend.distributed.lwd_data_plane",
        HOST_PACKAGE,
    ):
        placeholder = types.ModuleType(name)
        placeholder.__path__ = []
        monkeypatch.setitem(sys.modules, name, placeholder)
    monkeypatch.setitem(sys.modules, CHANNELS_MODULE, channels)
    monkeypatch.setattr(
        sys.modules[HOST_PACKAGE],
        "lwd_host_channels",
        channels,
        raising=False,
    )


def _worker(dp_idx: int) -> object:
    worker = WorkerSmoke()
    worker._lwd_dp_idx = dp_idx
    worker._lwd_connections = ()
    worker._lwd_early_recv_connections = ()
    return worker


def test_other_dp_connection_is_skipped_without_state_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another DP rank's connection is 'not my job': no state extension,
    no serve-pair build, and a None result distinct from a refusal."""
    channels = _HostChannels(has_listener=True)
    _install_channels(monkeypatch, channels)
    worker = _worker(dp_idx=0)

    result = worker.lwd_build_runtime_connection(SimpleNamespace(connection_id=OTHER_DP))

    assert result is None
    assert worker._lwd_connections == ()
    assert worker._lwd_early_recv_connections == ()
    assert channels.built == []


def test_own_dp_connection_extends_state_on_every_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The owning DP extends state even on a non-endpoint Worker (no
    listener); it returns None rather than building a serve pair."""
    channels = _HostChannels(has_listener=False)
    _install_channels(monkeypatch, channels)
    worker = _worker(dp_idx=0)

    result = worker.lwd_build_runtime_connection(SimpleNamespace(connection_id=OWN_DP))

    assert result is None
    assert worker._lwd_connections == (OWN_DP,)
    assert channels.built == []


def test_own_dp_endpoint_builds_the_serve_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The owning DP with the shared listener still builds normally and
    registers early-recv state — the guard must not reject it."""
    channels = _HostChannels(has_listener=True)
    _install_channels(monkeypatch, channels)
    worker = _worker(dp_idx=0)

    result = worker.lwd_build_runtime_connection(SimpleNamespace(connection_id=OWN_DP))

    assert result is True
    assert worker._lwd_connections == (OWN_DP,)
    assert worker._lwd_early_recv_connections == (OWN_DP,)
    assert channels.built == [OWN_DP]
