# SPDX-License-Identifier: Apache-2.0
"""Device zero baselines and host peer-restart lazy re-baselining.

Device/HCCL FIFOs always start at seqno 0 and hold frames with missing
prefixes. A restarted host peer's seqno stream continues at an unknown value,
so its first arriving frame after reset becomes the new baseline. The plane's
reset also clears old skip holes without failing the coming-back connection.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest

from vllm_ascend.distributed.lwd_data_plane import lwd_data_plane as plane_module
from vllm_ascend.distributed.lwd_data_plane.host.lwd_host_transfer_queue import LwdHostTransferQueue
from vllm_ascend.distributed.lwd_data_plane.host.ucx_io import LwdNpuUcxChannelIo
from vllm_ascend.distributed.lwd_data_plane.lwd_data_plane import LwdDataPlane
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import LwdTransferFuture
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_queue import LwdTransferQueue
from vllm_ascend.distributed.lwd_data_plane.lwd_types import (
    ConnectionId,
    LwdChannelType,
    LwdLinkError,
    LwdTransferRequest,
)

UP = LwdChannelType.UP
CONNECTION = ConnectionId(0, 0, 0)


class _RecordingQueue(LwdTransferQueue):
    """Records executed seqnos; no NPU work, futures never complete."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.executed: list[int] = []

    def _execute(self, req, predecessor, into=None) -> LwdTransferFuture:  # noqa: ANN001
        self.executed.append(req.seqno)
        if into is None:
            return LwdTransferFuture(req, [], None)
        return into


class _RecordingHostQueue(LwdHostTransferQueue):
    """Host queue with the same execution recorder and no UCX work."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.executed: list[int] = []

    def _execute(self, req, predecessor, into=None) -> LwdTransferFuture:  # noqa: ANN001
        self.executed.append(req.seqno)
        if into is None:
            return LwdTransferFuture(req, [], None)
        return into


class _SkipTransport:
    def __init__(self) -> None:
        self.peer_session = 1
        self.discarded: list[int] = []
        self.restart_cb: Callable[[], None] | None = None
        self.fail_reasons: list[str] = []

    def recv_discard(self, seqno: int) -> None:
        self.discarded.append(seqno)

    def on_peer_restart(self, cb: Callable[[], None]) -> None:
        self.restart_cb = cb

    def fail(self, reason: str) -> None:
        self.fail_reasons.append(reason)


def _skip_io() -> tuple[LwdNpuUcxChannelIo, _SkipTransport]:
    io = object.__new__(LwdNpuUcxChannelIo)
    transport = _SkipTransport()
    io._op = "recv"
    io._ch = transport
    io._skip_armed = set()
    io._skip_peer_session = transport.peer_session
    io._skip_lock = threading.Lock()
    return io, transport


def _recv(seqno: int) -> LwdTransferRequest:
    return LwdTransferRequest(channel=UP, op="recv", num_elements=8, seqno=seqno)


@pytest.fixture()
def plane(monkeypatch: pytest.MonkeyPatch) -> LwdDataPlane:
    monkeypatch.setattr(plane_module, "_queue_class", lambda: _RecordingQueue)
    return LwdDataPlane()


def test_device_first_seqno_zero_executes() -> None:
    queue = _RecordingQueue(CONNECTION, UP, "recv")
    queue.submit(_recv(0))
    assert queue.executed == [0]


def test_device_first_seqno_above_zero_waits_for_missing_prefix() -> None:
    """HCCL has no wire tags: accepting the first frame above zero would
    silently discard the prefix-ordering guarantee."""
    queue = _RecordingQueue(CONNECTION, UP, "recv")
    queue.submit(_recv(2))
    assert queue.executed == []
    queue.submit(_recv(0))
    assert queue.executed == [0]
    queue.submit(_recv(1))
    assert queue.executed == [0, 1, 2]


def test_host_reset_fails_stale_holds_and_re_baselines() -> None:
    """A host restart fails old held frames promptly, then accepts an
    arbitrary first seqno from the peer's continuing counter."""
    queue = _RecordingHostQueue(CONNECTION, UP, "recv")
    queue.submit(_recv(0))
    held = queue.submit(_recv(2))
    assert queue.executed == [0]
    queue.reset_peer_baseline()
    with pytest.raises(RuntimeError, match="lwd-comm op failed") as exc_info:
        held.result()
    assert isinstance(exc_info.value.__cause__, LwdLinkError)
    queue.submit(_recv(7))
    assert queue.executed == [0, 7]


def test_reset_clears_the_planes_pending_skips_of_the_old_stream(plane: LwdDataPlane) -> None:
    """Skip holes recorded before the FIFO exists belong to the old stream;
    after a restart they must not replay onto (and wedge) the new one."""
    plane.skip_seqno(UP, seqno=0, op="recv", connection_id=CONNECTION)
    plane.reset_channel_seqno(CONNECTION, UP, "recv")
    plane.submit_recv(_recv(0))
    queue = plane._queues[(CONNECTION, UP, "recv")]
    assert isinstance(queue, _RecordingQueue)
    assert queue.executed == [0]


def test_reset_keeps_the_coming_back_connection_submittable(plane: LwdDataPlane) -> None:
    """A restart is not an abort: submissions on the connection must keep
    working after the reset (unlike abort_connection's failed marking)."""
    plane.submit_recv(_recv(4))
    plane.reset_channel_seqno(CONNECTION, UP, "recv")
    plane.submit_recv(_recv(0))  # must NOT raise "connection has failed"


def test_reset_without_a_queue_logs_miss(plane: LwdDataPlane, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every built IO registers its own reset, so a miss is a wiring bug that
    must surface in the log instead of passing silently."""
    errors: list[str] = []
    monkeypatch.setattr(plane_module.logger, "error", lambda msg, *args: errors.append(msg % args if args else msg))
    plane.reset_channel_seqno(ConnectionId(3, 3, 0), UP, "recv")
    assert any("reset_channel_seqno MISS" in message for message in errors)


def test_peer_restart_reposts_same_skip_without_losing_dedup() -> None:
    """A seqno may be armed again in a new session, while duplicates within
    either session stay deduplicated and a stale restart callback cannot
    erase an arm already made in the new session."""
    io, transport = _skip_io()
    io.on_peer_restart(lambda: None)
    io.add_skipped(4)
    io.add_skipped(4)
    assert transport.discarded == [4]

    transport.peer_session = 2
    assert transport.restart_cb is not None
    transport.restart_cb()
    io.add_skipped(4)
    io.add_skipped(4)
    assert transport.discarded == [4, 4]

    transport.restart_cb()  # delayed duplicate notification from the old session
    io.add_skipped(4)
    assert transport.discarded == [4, 4]


def test_force_teardown_reposts_same_skip() -> None:
    io, transport = _skip_io()
    io.add_skipped(4)
    io.force_teardown("peer restarted")
    io.add_skipped(4)
    assert transport.discarded == [4, 4]
    assert transport.fail_reasons == ["peer restarted"]
