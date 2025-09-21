"""DeviceManager: track connected devices and heartbeat timeouts (FR8)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

try:
    from ..config import get as cfg_get
except Exception:  # pragma: no cover

    def cfg_get(key: str, default=None):
        return default


@dataclass
class DeviceInfo:
    device_id: str
    first_seen_ns: int
    last_heartbeat_ns: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeviceManager:
    def __init__(self, heartbeat_timeout_seconds: int | None = None) -> None:
        if heartbeat_timeout_seconds is None:
            heartbeat_timeout_seconds = int(cfg_get("heartbeat_timeout_seconds", 10))
        self._timeout_ns = int(heartbeat_timeout_seconds) * 1_000_000_000
        self._devices: dict[str, DeviceInfo] = {}

    def register(self, device_id: str) -> None:
        now = time.time_ns()
        if device_id not in self._devices:
            self._devices[device_id] = DeviceInfo(
                device_id=device_id,
                first_seen_ns=now,
                last_heartbeat_ns=now,
                status="Online",
            )

    def set_status(self, device_id: str, status: str) -> None:
        """Set a device's status string (e.g., Online, Offline, Recording)."""
        now = time.time_ns()
        info = self._devices.get(device_id)
        if info is None:
            self.register(device_id)
            info = self._devices[device_id]
        info.status = status
        info.last_heartbeat_ns = now

    def remove(self, device_id: str) -> None:
        self._devices.pop(device_id, None)

    def update_heartbeat(self, device_id: str) -> None:
        now = time.time_ns()
        info = self._devices.get(device_id)
        if info is None:
            self.register(device_id)
            info = self._devices[device_id]
        info.last_heartbeat_ns = now
        if info.status == "Offline":
            info.status = "Online"

    def get_status(self, device_id: str) -> str | None:
        info = self._devices.get(device_id)
        return info.status if info else None

    def get_info(self, device_id: str) -> DeviceInfo | None:
        return self._devices.get(device_id)

    def list_devices(self) -> dict[str, DeviceInfo]:
        return dict(self._devices)

    def check_timeouts(self, now_ns: int | None = None) -> None:
        if now_ns is None:
            now_ns = time.time_ns()
        for info in self._devices.values():
            if now_ns - info.last_heartbeat_ns > self._timeout_ns:
                info.status = "Offline"

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_ns / 1_000_000_000.0
