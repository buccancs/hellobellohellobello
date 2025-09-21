"""Shimmer sensor management and simulation (FR1).

Provides both real Shimmer sensor integration and a simulated version for testing.
Real integration uses pyshimmer library for connection management and data streaming.
Simulation generates GSR-like sine+noise samples at shimmer_sampling_rate from config.json.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

try:
    from ..config import get as cfg_get
except Exception:  # pragma: no cover

    def cfg_get(key: str, default=None):
        return default


try:
    import pyshimmer

    SHIMMER_AVAILABLE = True
except ImportError:  # pragma: no cover
    SHIMMER_AVAILABLE = False
    pyshimmer = None

logger = logging.getLogger(__name__)


class RealShimmer:
    """Real Shimmer sensor implementation using pyshimmer library."""

    def __init__(
        self, device_port: str | None = None, sample_rate_hz: int | None = None
    ) -> None:
        if not SHIMMER_AVAILABLE:
            raise RuntimeError(
                "pyshimmer library not available. Install with: pip install pyshimmer"
            )

        self._port = device_port or cfg_get("shimmer_port", "COM3")
        self._rate = int(sample_rate_hz or int(cfg_get("shimmer_sampling_rate", 128)))
        if self._rate <= 0:
            self._rate = 128

        self._shimmer: Any | None = None
        self._connected = False
        self._streaming = False
        self._callback: Callable[[int, float], None] | None = None
        self._data_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def connect(self) -> bool:
        """Connect to Shimmer device."""
        try:
            if self._connected:
                return True

            self._shimmer = pyshimmer.ShimmerBluetooth(self._port)

            # Configure for GSR and PPG
            self._shimmer.add_stream_callback(self._on_shimmer_data)

            # Enable GSR and PPG sensors
            enabled_sensors = (
                pyshimmer.SENSOR_GSR
                | pyshimmer.SENSOR_INT_A13
                | pyshimmer.SENSOR_TIMESTAMP
            )

            self._shimmer.set_enabled_sensors(enabled_sensors)
            self._shimmer.set_sampling_rate(self._rate)

            # Configure GSR range and resolution
            self._shimmer.set_gsr_range(4)

            if self._shimmer.connect():
                self._connected = True
                logger.info(f"Connected to Shimmer on {self._port}")
                return True
            else:
                logger.error(f"Failed to connect to Shimmer on {self._port}")
                return False

        except Exception as e:
            logger.error(f"Shimmer connection error: {e}")
            return False

    def start_streaming(self, callback: Callable[[int, float], None]) -> None:
        """Start streaming data from Shimmer device."""
        if not self._connected:
            raise RuntimeError("Shimmer not connected")

        if self._streaming:
            return

        self._callback = callback
        self._stop_event.clear()

        try:
            if self._shimmer.start_streaming():
                self._streaming = True
                logger.info("Started Shimmer streaming")
            else:
                raise RuntimeError("Failed to start Shimmer streaming")
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            raise

    def stop_streaming(self) -> None:
        """Stop streaming data from Shimmer device."""
        if not self._streaming:
            return

        self._stop_event.set()

        try:
            if self._shimmer:
                self._shimmer.stop_streaming()
            self._streaming = False
            self._callback = None
            logger.info("Stopped Shimmer streaming")
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")

    def disconnect(self) -> None:
        """Disconnect from Shimmer device."""
        self.stop_streaming()

        try:
            if self._shimmer and self._connected:
                self._shimmer.disconnect()
                self._connected = False
                logger.info("Disconnected from Shimmer")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
        finally:
            self._shimmer = None

    def _on_shimmer_data(self, shimmer_data) -> None:
        """Callback for processing Shimmer data packets."""
        if not self._callback or self._stop_event.is_set():
            return

        try:
            timestamp_ns = int(shimmer_data.timestamp * 1000000)

            gsr_raw = shimmer_data.get("GSR", 0)
            gsr_microsiemens = self._convert_gsr_to_microsiemens(gsr_raw)

            self._callback(timestamp_ns, gsr_microsiemens)

        except Exception as e:
            logger.error(f"Error processing Shimmer data: {e}")

    def _convert_gsr_to_microsiemens(self, raw_value: int) -> float:
        """Convert raw GSR ADC value to microsiemens using 12-bit resolution.

        Critical implementation note: Uses 12-bit ADC resolution (0-4095 range)
        as specified in project requirements, not 16-bit.
        """
        ADC_RESOLUTION = 4095.0
        GSR_RANGE_KOHM = 4000.0
        V_REF = 3.0  # Reference voltage

        voltage = (raw_value / ADC_RESOLUTION) * V_REF

        if voltage > 0.01:
            resistance_kohm = GSR_RANGE_KOHM * (V_REF - voltage) / voltage
            conductance_us = 1000.0 / resistance_kohm
            return max(0.0, conductance_us)
        else:
            return 0.0


class SimulatedShimmer:
    def __init__(self, sample_rate_hz: int | None = None) -> None:
        self._rate = int(sample_rate_hz or int(cfg_get("shimmer_sampling_rate", 128)))
        if self._rate <= 0:
            self._rate = 128
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._callback: Callable[[int, float], None] | None = None

    def connect(self) -> bool:
        return True

    def start_streaming(self, callback: Callable[[int, float], None]) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._callback = callback
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_streaming(self) -> None:
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None
        self._callback = None

    def disconnect(self) -> None:
        self.stop_streaming()

    def _loop(self) -> None:
        dt = 1.0 / float(self._rate)
        next_t = time.monotonic()
        phase = 0.0
        two_pi = 2.0 * math.pi
        while self._running.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
                continue
            ts_ns = time.monotonic_ns()
            val = 10.0 + 2.0 * math.sin(phase)
            phase = (phase + two_pi * 1.2 * dt) % two_pi
            cb = self._callback
            if cb:
                with contextlib.suppress(Exception):
                    cb(ts_ns, float(val))
            next_t += dt


def create_shimmer_manager(
    use_real: bool | None = None, **kwargs
) -> RealShimmer | SimulatedShimmer:
    """Factory function to create appropriate Shimmer manager.

    Args:
        use_real: If True, creates RealShimmer. If False, creates SimulatedShimmer.
                 If None, auto-detects based on availability and config.
        **kwargs: Arguments passed to the Shimmer constructor.

    Returns:
        Shimmer manager instance (real or simulated).
    """
    if use_real is None:
        # Auto-detect: use real if available and configured
        use_real = SHIMMER_AVAILABLE and cfg_get("use_real_shimmer", False)

    if use_real:
        if not SHIMMER_AVAILABLE:
            logger.warning(
                "Real Shimmer requested but pyshimmer not available. Falling back to simulation."
            )
            return SimulatedShimmer(**kwargs)
        return RealShimmer(**kwargs)
    else:
        return SimulatedShimmer(**kwargs)


class ShimmerManager:
    """High-level Shimmer management with automatic fallback and error handling."""

    def __init__(self, prefer_real: bool = True, **kwargs) -> None:
        self._prefer_real = prefer_real
        self._manager: RealShimmer | SimulatedShimmer | None = None
        self._kwargs = kwargs
        self._is_real = False

    def initialize(self) -> bool:
        """Initialize Shimmer manager with automatic fallback."""
        try:
            if self._prefer_real and SHIMMER_AVAILABLE:
                try:
                    self._manager = RealShimmer(**self._kwargs)
                    if self._manager.connect():
                        self._is_real = True
                        logger.info("Using real Shimmer sensor")
                        return True
                    else:
                        logger.info(
                            "Real Shimmer connection failed, falling back to simulation"
                        )
                        self._manager = None
                except Exception as e:
                    logger.info(
                        f"Real Shimmer initialization failed: {e}, falling back to simulation"
                    )
                    self._manager = None

            self._manager = SimulatedShimmer(**self._kwargs)
            self._is_real = False
            if self._manager.connect():
                logger.info("Using simulated Shimmer sensor")
                return True
            else:
                logger.error("Failed to initialize even simulated Shimmer")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize Shimmer manager: {e}")
            return False

    @property
    def is_real(self) -> bool:
        """Returns True if using real Shimmer hardware."""
        return self._is_real

    @property
    def is_initialized(self) -> bool:
        """Returns True if manager is initialized."""
        return self._manager is not None

    def connect(self) -> bool:
        """Connect to Shimmer device."""
        if not self._manager:
            return self.initialize()
        return self._manager.connect()

    def start_streaming(self, callback: Callable[[int, float], None]) -> None:
        """Start streaming data."""
        if not self._manager:
            raise RuntimeError("Shimmer manager not initialized")
        return self._manager.start_streaming(callback)

    def stop_streaming(self) -> None:
        """Stop streaming data."""
        if self._manager:
            self._manager.stop_streaming()

    def disconnect(self) -> None:
        """Disconnect from device."""
        if self._manager:
            self._manager.disconnect()
            self._manager = None
