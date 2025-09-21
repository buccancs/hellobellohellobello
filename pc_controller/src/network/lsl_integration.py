"""LSL (Lab Streaming Layer) Integration for Real-Time Monitoring.

This module provides LSL outlets for streaming live sensor data for external
monitoring and analysis tools. It's designed for complementary real-time
monitoring only - NOT for primary data recording or synchronization.

Features:
- GSR data streaming outlet
- PPG data streaming outlet
- Thermal frame streaming outlet
- Configurable sample rates and metadata
- Integration with existing sensor interfaces

Note: LSL is used only for live monitoring. Primary data recording and
synchronization still uses the custom protocol between Hub and Spokes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

try:
    import pylsl

    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    pylsl = None

import numpy as np

try:
    from ..config import get as cfg_get
except Exception:

    def cfg_get(key: str, default=None):
        return default


logger = logging.getLogger(__name__)


@dataclass
class LSLStreamConfig:
    """Configuration for an LSL stream outlet."""

    name: str
    type: str
    channel_count: int
    sample_rate: float
    channel_format: str
    source_id: str
    channel_names: list[str]
    channel_units: list[str]


class LSLOutletManager:
    """Manages LSL outlets for real-time sensor data streaming."""

    def __init__(self):
        """Initialize LSL outlet manager."""
        if not LSL_AVAILABLE:
            logger.warning("pylsl not available - LSL streaming disabled")
            return

        self._outlets: dict[str, pylsl.StreamOutlet] = {}
        self._configs: dict[str, LSLStreamConfig] = {}
        self._enabled = cfg_get("lsl_enabled", "false").lower() == "true"

        if not self._enabled:
            logger.info("LSL streaming disabled via configuration")
            return

        logger.info("LSL outlet manager initialized")

    @property
    def available(self) -> bool:
        """Check if LSL is available and enabled."""
        return LSL_AVAILABLE and self._enabled

    def create_gsr_outlet(self, device_id: str, sample_rate: float = 50.0) -> bool:
        """Create LSL outlet for GSR data streaming.

        Args:
            device_id: Unique identifier for the device
            sample_rate: Expected sample rate in Hz

        Returns:
            True if outlet created successfully, False otherwise
        """
        if not self.available:
            return False

        outlet_name = f"GSR_{device_id}"

        try:
            config = LSLStreamConfig(
                name=f"GSR Sensor - {device_id}",
                type="GSR",
                channel_count=2,
                sample_rate=sample_rate,
                channel_format=pylsl.cf_float32,
                source_id=f"gsr_{device_id}",
                channel_names=["GSR", "PPG"],
                channel_units=["microsiemens", "raw"],
            )

            info = pylsl.StreamInfo(
                name=config.name,
                type=config.type,
                channel_count=config.channel_count,
                nominal_srate=config.sample_rate,
                channel_format=config.channel_format,
                source_id=config.source_id,
            )

            channels = info.desc().append_child("channels")
            for _, (name, unit) in enumerate(
                zip(config.channel_names, config.channel_units, strict=True)
            ):
                ch = channels.append_child("channel")
                ch.append_child_value("label", name)
                ch.append_child_value("unit", unit)
                ch.append_child_value("type", config.type)

            info.desc().append_child_value("manufacturer", "Shimmer")
            info.desc().append_child_value("device_id", device_id)

            outlet = pylsl.StreamOutlet(info)
            self._outlets[outlet_name] = outlet
            self._configs[outlet_name] = config

            logger.info(f"Created LSL GSR outlet for device {device_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create GSR outlet for {device_id}: {e}")
            return False

    def create_thermal_outlet(
        self,
        device_id: str,
        width: int = 256,
        height: int = 192,
        sample_rate: float = 10.0,
    ) -> bool:
        """Create LSL outlet for thermal frame streaming.

        Args:
            device_id: Unique identifier for the device
            width: Frame width in pixels
            height: Frame height in pixels
            sample_rate: Expected frame rate in Hz

        Returns:
            True if outlet created successfully, False otherwise
        """
        if not self.available:
            return False

        outlet_name = f"Thermal_{device_id}"

        try:
            channel_count = width * height

            config = LSLStreamConfig(
                name=f"Thermal Camera - {device_id}",
                type="Thermal",
                channel_count=channel_count,
                sample_rate=sample_rate,
                channel_format=pylsl.cf_float32,
                source_id=f"thermal_{device_id}",
                channel_names=[f"pixel_{i}" for i in range(channel_count)],
                channel_units=["celsius"] * channel_count,
            )

            info = pylsl.StreamInfo(
                name=config.name,
                type=config.type,
                channel_count=config.channel_count,
                nominal_srate=config.sample_rate,
                channel_format=config.channel_format,
                source_id=config.source_id,
            )

            info.desc().append_child_value("manufacturer", "Topdon")
            info.desc().append_child_value("model", "TC001")
            info.desc().append_child_value("device_id", device_id)
            info.desc().append_child_value("frame_width", str(width))
            info.desc().append_child_value("frame_height", str(height))

            outlet = pylsl.StreamOutlet(info)
            self._outlets[outlet_name] = outlet
            self._configs[outlet_name] = config

            logger.info(f"Created LSL thermal outlet for device {device_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create thermal outlet for {device_id}: {e}")
            return False

    def stream_gsr_sample(
        self,
        device_id: str,
        gsr_value: float,
        ppg_value: float,
        timestamp: float | None = None,
    ) -> bool:
        """Stream a single GSR sample.

        Args:
            device_id: Device identifier
            gsr_value: GSR value in microsiemens
            ppg_value: Raw PPG value
            timestamp: Optional timestamp (uses current time if None)

        Returns:
            True if sample streamed successfully, False otherwise
        """
        if not self.available:
            return False

        outlet_name = f"GSR_{device_id}"
        if outlet_name not in self._outlets:
            return False

        try:
            outlet = self._outlets[outlet_name]
            sample = [gsr_value, ppg_value]

            if timestamp is not None:
                outlet.push_sample(sample, timestamp)
            else:
                outlet.push_sample(sample)

            return True

        except Exception as e:
            logger.error(f"Failed to stream GSR sample for {device_id}: {e}")
            return False

    def stream_thermal_frame(
        self, device_id: str, frame: np.ndarray, timestamp: float | None = None
    ) -> bool:
        """Stream a thermal frame.

        Args:
            device_id: Device identifier
            frame: Thermal frame as 2D numpy array (width x height)
            timestamp: Optional timestamp (uses current time if None)

        Returns:
            True if frame streamed successfully, False otherwise
        """
        if not self.available:
            return False

        outlet_name = f"Thermal_{device_id}"
        if outlet_name not in self._outlets:
            return False

        try:
            outlet = self._outlets[outlet_name]
            flattened = frame.flatten().astype(np.float32)

            if timestamp is not None:
                outlet.push_sample(flattened.tolist(), timestamp)
            else:
                outlet.push_sample(flattened.tolist())

            return True

        except Exception as e:
            logger.error(f"Failed to stream thermal frame for {device_id}: {e}")
            return False

    def remove_outlet(self, device_id: str, sensor_type: str) -> bool:
        """Remove an LSL outlet.

        Args:
            device_id: Device identifier
            sensor_type: Type of sensor ("GSR" or "Thermal")

        Returns:
            True if outlet removed successfully, False otherwise
        """
        outlet_name = f"{sensor_type}_{device_id}"

        if outlet_name in self._outlets:
            try:
                del self._outlets[outlet_name]
                del self._configs[outlet_name]
                logger.info(f"Removed LSL {sensor_type} outlet for device {device_id}")
                return True
            except Exception as e:
                logger.error(
                    f"Error removing {sensor_type} outlet for {device_id}: {e}"
                )
                return False

        return False

    def get_active_outlets(self) -> list[str]:
        """Get list of active outlet names."""
        return list(self._outlets.keys())

    def shutdown(self):
        """Shutdown all outlets and clean up resources."""
        if not self.available:
            return

        try:
            for outlet_name in list(self._outlets.keys()):
                self.remove_outlet(
                    device_id=outlet_name.split("_", 1)[1],
                    sensor_type=outlet_name.split("_", 1)[0],
                )
            logger.info("LSL outlet manager shut down")
        except Exception as e:
            logger.error(f"Error during LSL shutdown: {e}")


_lsl_manager: LSLOutletManager | None = None


def get_lsl_manager() -> LSLOutletManager:
    """Get global LSL outlet manager instance."""
    global _lsl_manager
    if _lsl_manager is None:
        _lsl_manager = LSLOutletManager()
    return _lsl_manager
