"""HeartbeatManager: Manages device heartbeat monitoring and connection health (FR8)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HeartbeatStatus:
    """Status information for a device's heartbeat."""

    device_id: str
    last_heartbeat_ns: int
    consecutive_misses: int = 0
    is_healthy: bool = True
    reconnection_attempts: int = 0
    last_reconnect_attempt_ns: int = 0

    def update_heartbeat(self) -> None:
        """Update heartbeat timestamp and reset miss counter."""
        self.last_heartbeat_ns = time.time_ns()
        self.consecutive_misses = 0
        self.is_healthy = True

    def mark_missed(self) -> None:
        """Mark a heartbeat as missed."""
        self.consecutive_misses += 1
        if self.consecutive_misses >= 3:  # Configurable threshold
            self.is_healthy = False

    def mark_reconnect_attempt(self) -> None:
        """Mark a reconnection attempt."""
        self.reconnection_attempts += 1
        self.last_reconnect_attempt_ns = time.time_ns()


class HeartbeatManager:
    """Manages heartbeat monitoring for all connected devices."""

    def __init__(
        self,
        heartbeat_interval_s: float = 3.0,
        timeout_multiplier: int = 3,
        max_reconnect_attempts: int = 10,
        reconnect_backoff_s: float = 5.0,
        callback: Callable[[str], None] | None = None,
    ):
        """Initialize the heartbeat manager.

        Args:
            heartbeat_interval_s: Expected interval between heartbeats
            timeout_multiplier: Number of missed intervals before marking unhealthy
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_backoff_s: Delay between reconnection attempts
            callback: Optional callback function for heartbeat events
        """
        self.heartbeat_interval_s = heartbeat_interval_s
        self.timeout_ns = int(heartbeat_interval_s * timeout_multiplier * 1_000_000_000)
        self.max_reconnect_attempts = max_reconnect_attempts
        self.callback = callback
        self.reconnect_backoff_s = reconnect_backoff_s

        self._devices: dict[str, HeartbeatStatus] = {}
        self._device_offline_callbacks: dict[str, Callable[[str], None]] = {}
        self._device_online_callbacks: dict[str, Callable[[str], None]] = {}
        self._reconnect_callbacks: dict[str, Callable[[str], None]] = {}

        self._monitor_task: asyncio.Task | None = None
        self._running = False

    def register_device(self, device_id: str) -> None:
        """Register a device for heartbeat monitoring."""
        if device_id not in self._devices:
            self._devices[device_id] = HeartbeatStatus(
                device_id=device_id, last_heartbeat_ns=time.time_ns()
            )
            logger.info(f"Registered device for heartbeat monitoring: {device_id}")

    def unregister_device(self, device_id: str) -> None:
        """Unregister a device from heartbeat monitoring."""
        if device_id in self._devices:
            del self._devices[device_id]
            logger.info(f"Unregistered device from heartbeat monitoring: {device_id}")

    def record_heartbeat(
        self, device_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Record a heartbeat from a device.

        Args:
            device_id: The device identifier
            metadata: Optional heartbeat metadata (battery, status, etc.)
        """
        if device_id not in self._devices:
            self.register_device(device_id)

        was_healthy = self._devices[device_id].is_healthy
        self._devices[device_id].update_heartbeat()

        if not was_healthy and self._devices[device_id].is_healthy:
            if device_id in self._device_online_callbacks:
                try:
                    self._device_online_callbacks[device_id](device_id)
                except Exception as e:
                    logger.error(
                        f"Error in device online callback for {device_id}: {e}"
                    )

        logger.debug(f"Heartbeat received from {device_id}")

    def get_device_status(self, device_id: str) -> HeartbeatStatus | None:
        """Get the current status of a device."""
        return self._devices.get(device_id)

    def get_healthy_devices(self) -> set[str]:
        """Get set of currently healthy device IDs."""
        return {
            device_id
            for device_id, status in self._devices.items()
            if status.is_healthy
        }

    def get_unhealthy_devices(self) -> set[str]:
        """Get set of currently unhealthy device IDs."""
        return {
            device_id
            for device_id, status in self._devices.items()
            if not status.is_healthy
        }

    def set_device_offline_callback(
        self, device_id: str, callback: Callable[[str], None]
    ) -> None:
        """Set callback for when a device goes offline."""
        self._device_offline_callbacks[device_id] = callback

    def set_device_online_callback(
        self, device_id: str, callback: Callable[[str], None]
    ) -> None:
        """Set callback for when a device comes back online."""
        self._device_online_callbacks[device_id] = callback

    def set_reconnect_callback(
        self, device_id: str, callback: Callable[[str], None]
    ) -> None:
        """Set callback for when a device needs reconnection."""
        self._reconnect_callbacks[device_id] = callback

    async def start_monitoring(self) -> None:
        """Start the heartbeat monitoring loop."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Heartbeat monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop the heartbeat monitoring loop."""
        if not self._running:
            return

        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
        logger.info("Heartbeat monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop that checks for missed heartbeats."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval_s)
                await self._check_heartbeats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat monitor loop: {e}")

    async def _check_heartbeats(self) -> None:
        """Check all devices for missed heartbeats."""
        current_time_ns = time.time_ns()

        for device_id, status in self._devices.items():
            time_since_last = current_time_ns - status.last_heartbeat_ns

            if time_since_last > self.timeout_ns:
                was_healthy = status.is_healthy
                status.mark_missed()

                if was_healthy and not status.is_healthy:
                    logger.warning(
                        f"Device {device_id} went offline "
                        f"(missed {status.consecutive_misses} heartbeats)"
                    )
                    if device_id in self._device_offline_callbacks:
                        try:
                            self._device_offline_callbacks[device_id](device_id)
                        except Exception as e:
                            logger.error(
                                f"Error in device offline callback for {device_id}: {e}"
                            )

                if status.reconnection_attempts < self.max_reconnect_attempts and (
                    current_time_ns - status.last_reconnect_attempt_ns
                ) > int(self.reconnect_backoff_s * 1_000_000_000):

                    status.mark_reconnect_attempt()
                    attempt_num = status.reconnection_attempts
                    logger.info(
                        f"Triggering reconnection for {device_id} (attempt {attempt_num})"
                    )

                    if device_id in self._reconnect_callbacks:
                        try:
                            self._reconnect_callbacks[device_id](device_id)
                        except Exception as e:
                            logger.error(
                                f"Error in reconnect callback for {device_id}: {e}"
                            )

    def create_heartbeat_message(
        self, device_id: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Create a standardized heartbeat message.

        Args:
            device_id: The device identifier
            metadata: Optional metadata to include

        Returns:
            JSON string of heartbeat message
        """
        message = {
            "v": 1,
            "type": "heartbeat",
            "device_id": device_id,
            "timestamp_ns": time.time_ns(),
            "metadata": metadata or {},
        }
        return json.dumps(message)

    def parse_heartbeat_message(self, message: str) -> dict[str, Any] | None:
        """Parse a heartbeat message.

        Args:
            message: JSON string of heartbeat message

        Returns:
            Parsed message dict or None if invalid
        """
        try:
            data = json.loads(message)
            if data.get("type") == "heartbeat" and "device_id" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def get_status_summary(self) -> dict[str, Any]:
        """Get a summary of all device statuses."""
        return {
            "healthy_devices": list(self.get_healthy_devices()),
            "unhealthy_devices": list(self.get_unhealthy_devices()),
            "total_devices": len(self._devices),
            "heartbeat_interval_s": self.heartbeat_interval_s,
            "device_details": {
                device_id: {
                    "is_healthy": status.is_healthy,
                    "consecutive_misses": status.consecutive_misses,
                    "reconnection_attempts": status.reconnection_attempts,
                    "last_heartbeat_age_s": (time.time_ns() - status.last_heartbeat_ns)
                    / 1_000_000_000,
                }
                for device_id, status in self._devices.items()
            },
        }
