"""Tests for NetworkController internal behaviors without real sockets.

These tests avoid starting PreviewStreamWorker network activity by injecting
fake workers and directly invoking private methods._on_service_removed and
_store_offset. PyQt6 signals are connected to Python callables and executed
synchronously, so no Qt event loop is required here.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
from pc_controller.src.network.network_controller import NetworkController


class _FakeWorker:
    def __init__(self) -> None:
        self.stopped = False
        self.wait_called_with = None

    def stop(self) -> None:
        self.stopped = True

    def wait(self, msecs: int) -> None:
        self.wait_called_with = msecs


@pytest.fixture
def controller() -> NetworkController:
    c = NetworkController()
    try:
        yield c
    finally:
        c.shutdown()


def test_store_offset(controller: NetworkController) -> None:
    controller._store_offset("dev1", 123456789)
    assert controller._clock_offsets_ns["dev1"] == 123456789


def test_on_service_removed_stops_stream_worker_and_emits_signal(
    controller: NetworkController,
) -> None:
    removed_names: list[str] = []
    controller.device_removed.connect(lambda name: removed_names.append(name))

    fake = _FakeWorker()
    name = "dev1"
    controller._stream_workers[name] = fake
    controller._devices[name] = None

    controller._on_service_removed(name + ".")

    assert fake.stopped is True
    assert fake.wait_called_with == 1000
    assert name in removed_names
    assert name not in controller._devices
    assert name not in controller._stream_workers
