# SPDX-License-Identifier: Apache-2.0
"""Deferred staged-receive cleanup on edge restart."""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm_ascend.distributed.lwd_data_plane.lwd_types import ConnectionId, LwdChannelType
from vllm_ascend.worker.lwd_cloud_model import lwd_cloud_worker
from vllm_ascend.worker.lwd_cloud_model.lwd_cloud_worker import LwdCloudWorker

UP = LwdChannelType.UP
CONNECTION_A = ConnectionId(0, 0, 0)
CONNECTION_B = ConnectionId(1, 0, 0)


def _worker() -> LwdCloudWorker:
    worker = object.__new__(LwdCloudWorker)
    worker._init_lwd_early_recv_state()
    worker._early_recv_futures = {
        (CONNECTION_A, UP, 1): MagicMock(),
        (CONNECTION_B, UP, 10): MagicMock(),
    }
    worker._early_recv_pending = {
        (CONNECTION_A, UP, 1): {"req": "a"},
        (CONNECTION_B, UP, 11): {"req": "b"},
    }
    worker._early_recv_reserved = {(CONNECTION_A, UP, 2)}
    worker._early_recv_last_delivered = {CONNECTION_A: 4, CONNECTION_B: 9}
    worker._early_recv_current_key = (CONNECTION_B, UP, 10)
    worker._lwd_stager = MagicMock()
    return worker


def test_drop_defers_physical_cleanup_until_every_future_drains() -> None:
    worker = _worker()
    dropped = worker._early_recv_futures[(CONNECTION_B, UP, 10)]
    other = worker._early_recv_futures[(CONNECTION_A, UP, 1)]
    dropped.staged_done.return_value = False

    worker.drop_connection_early_recv(CONNECTION_B)
    assert worker._early_recv_dropping == {CONNECTION_B}
    assert set(worker._early_recv_pending) == {(CONNECTION_A, UP, 1)}
    assert worker._early_recv_last_delivered == {CONNECTION_A: 4}
    dropped.release_device_tensor.assert_not_called()
    worker._lwd_stager.abandon_connection.assert_not_called()

    worker._early_recv_drain_retiring()
    dropped.release_device_tensor.assert_not_called()
    worker._lwd_stager.abandon_connection.assert_not_called()
    assert worker._early_recv_retiring[(CONNECTION_B, UP, 10)] is dropped

    dropped.staged_done.return_value = True
    worker._early_recv_drain_retiring()
    worker._early_recv_drain_retiring()
    dropped.release_device_tensor.assert_called_once_with()
    worker._lwd_stager.abandon_connection.assert_called_once_with(CONNECTION_B)
    other.release_device_tensor.assert_not_called()
    assert set(worker._early_recv_futures) == {(CONNECTION_A, UP, 1)}
    assert worker._early_recv_reserved == {(CONNECTION_A, UP, 2)}
    assert not worker._early_recv_dropping
    assert worker._early_recv_failed_connections == set()
    assert worker._early_recv_current_key is None


def test_drop_waits_for_inflight_submission_reservation() -> None:
    worker = _worker()
    dropped = worker._early_recv_futures[(CONNECTION_B, UP, 10)]
    dropped.staged_done.return_value = True
    worker._early_recv_reserved.add((CONNECTION_B, UP, 12))

    worker.drop_connection_early_recv(CONNECTION_B)
    worker._early_recv_drain_retiring()
    dropped.release_device_tensor.assert_not_called()
    worker._lwd_stager.abandon_connection.assert_not_called()
    assert (CONNECTION_B, UP, 12) in worker._early_recv_reserved
    assert worker._early_recv_dropping == {CONNECTION_B}


def test_module_hook_without_an_endpoint_worker_logs_and_continues(monkeypatch) -> None:  # noqa: ANN001
    """The hook fires on the transport IO thread: a missing worker must be
    visible, and must not take the IO thread down."""
    monkeypatch.setattr(lwd_cloud_worker, "_ENDPOINT_WORKER", None)
    errors: list[str] = []
    monkeypatch.setattr(
        lwd_cloud_worker.lwd_logger, "error", lambda msg, *args: errors.append(msg % args if args else msg)
    )
    lwd_cloud_worker.drop_endpoint_early_recv(CONNECTION_B)  # must not raise
    assert any("early-recv drop MISS" in message for message in errors)
