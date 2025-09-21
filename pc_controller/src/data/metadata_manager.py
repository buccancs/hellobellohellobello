"""Session Metadata Manager for standardized session information.

Implements structured metadata.json creation and management for recording sessions,
ensuring consistent schema across all sensor types and devices.

Features:
- Session-level metadata with participant ID, timestamps, configuration
- Device and sensor metadata collection
- CSV schema standardization
- Integration with existing session management
- Privacy-compliant participant ID handling
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

try:
    from ..config import get as cfg_get
except Exception:

    def cfg_get(key: str, default=None):
        return default


@dataclass
class DeviceInfo:
    """Information about a recording device."""

    device_id: str
    device_type: str
    device_name: str
    model: str
    os_version: str
    app_version: str
    ip_address: str
    sensors: list[str]
    time_offset_ns: int | None = None
    sync_quality: str | None = None


@dataclass
class SensorConfig:
    """Configuration for a specific sensor."""

    sensor_type: str
    sampling_rate: float | None = None
    resolution: str | None = None
    settings: dict[str, str | int | float] | None = None


@dataclass
class SessionMetadata:
    """Complete session metadata structure."""

    session_id: str
    participant_id: str
    created_at: str
    duration_seconds: float | None = None

    # Recording configuration
    pc_controller_version: str = "1.0.0"
    protocol_version: str = "1.0"
    time_sync_enabled: bool = True
    tls_enabled: bool = False

    devices: list[DeviceInfo] | None = None
    sensor_configs: list[SensorConfig] | None = None

    data_files: list[str] | None = None
    csv_schema_version: str = "1.0"
    image_formats: dict[str, str] | None = (
        None
    )

    anonymized: bool = True
    face_blurring_enabled: bool = False

    notes: str = ""
    tags: list[str] | None = None

    def __post_init__(self):
        if self.devices is None:
            self.devices = []
        if self.sensor_configs is None:
            self.sensor_configs = []
        if self.data_files is None:
            self.data_files = []
        if self.tags is None:
            self.tags = []
        if self.image_formats is None:
            self.image_formats = {"rgb": "DNG", "thermal": "PNG"}


class SessionMetadataManager:
    """Manages session metadata creation and updates."""

    def __init__(self, base_session_dir: str):
        """Initialize metadata manager.

        Args:
            base_session_dir: Base directory where session folders are stored
        """
        self.base_session_dir = Path(base_session_dir)
        self._sessions: dict[str, SessionMetadata] = {}

    def create_session_metadata(
        self, participant_id: str, session_id: str | None = None
    ) -> SessionMetadata:
        """Create new session metadata.

        Args:
            participant_id: Anonymized participant identifier
            session_id: Optional session ID (auto-generated if None)

        Returns:
            SessionMetadata object
        """
        if session_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"session_{timestamp}"

        metadata = SessionMetadata(
            session_id=session_id,
            participant_id=self._anonymize_participant_id(participant_id),
            created_at=datetime.now().isoformat(),
            pc_controller_version=cfg_get("app_version", "1.0.0"),
            protocol_version=cfg_get("protocol_version", "1.0"),
            time_sync_enabled=cfg_get("time_sync_enabled", "true").lower() == "true",
            tls_enabled=cfg_get("tls_enabled", "false").lower() == "true",
            anonymized=True,
            face_blurring_enabled=cfg_get("face_blurring_enabled", "false").lower()
            == "true",
            image_formats={
                "rgb": "DNG",
                "thermal": "PNG",
            },
        )

        self._sessions[session_id] = metadata
        return metadata

    def add_device(self, session_id: str, device_info: DeviceInfo) -> bool:
        """Add device information to session metadata.

        Args:
            session_id: Session identifier
            device_info: Device information to add

        Returns:
            True if added successfully, False if session not found
        """
        if session_id not in self._sessions:
            return False

        session = self._sessions[session_id]
        if session.devices is None:
            session.devices = []
        session.devices.append(device_info)
        return True

    def add_sensor_config(self, session_id: str, sensor_config: SensorConfig) -> bool:
        """Add sensor configuration to session metadata.

        Args:
            session_id: Session identifier
            sensor_config: Sensor configuration to add

        Returns:
            True if added successfully, False if session not found
        """
        if session_id not in self._sessions:
            return False

        session = self._sessions[session_id]
        if session.sensor_configs is None:
            session.sensor_configs = []
        session.sensor_configs.append(sensor_config)
        return True

    def update_time_sync_info(
        self, session_id: str, device_id: str, offset_ns: int, quality: str
    ) -> bool:
        """Update time synchronization information for a device.

        Args:
            session_id: Session identifier
            device_id: Device identifier
            offset_ns: Time offset in nanoseconds
            quality: Sync quality ("excellent", "good", "poor")

        Returns:
            True if updated successfully, False otherwise
        """
        if session_id not in self._sessions:
            return False

        session = self._sessions[session_id]
        if session.devices is None:
            return False

        for device in session.devices:
            if device.device_id == device_id:
                device.time_offset_ns = offset_ns
                device.sync_quality = quality
                return True

        return False

    def finalize_session(
        self, session_id: str, duration_seconds: float, data_files: list[str]
    ) -> bool:
        """Finalize session metadata with duration and file list.

        Args:
            session_id: Session identifier
            duration_seconds: Total recording duration
            data_files: List of data file paths relative to session directory

        Returns:
            True if finalized successfully, False if session not found
        """
        if session_id not in self._sessions:
            return False

        metadata = self._sessions[session_id]
        metadata.duration_seconds = duration_seconds
        metadata.data_files = data_files

        return True

    def save_metadata(
        self, session_id: str, session_dir: str | Path | None = None
    ) -> bool:
        """Save session metadata to JSON file.

        Args:
            session_id: Session identifier
            session_dir: Optional session directory (defaults to base_session_dir/session_id)

        Returns:
            True if saved successfully, False otherwise
        """
        if session_id not in self._sessions:
            return False

        session_path: Path
        if session_dir is None:
            session_path = self.base_session_dir / session_id
        else:
            session_path = Path(session_dir)

        try:
            session_path.mkdir(parents=True, exist_ok=True)
            metadata_file = session_path / "metadata.json"

            metadata = self._sessions[session_id]

            metadata_dict = asdict(metadata)

            metadata_dict = self._ensure_json_serializable(metadata_dict)

            with open(metadata_file, "w") as f:
                json.dump(metadata_dict, f, indent=2, sort_keys=True)

            return True

        except Exception:
            return False

    def load_metadata(
        self, session_id: str, session_dir: str | Path | None = None
    ) -> SessionMetadata | None:
        """Load session metadata from JSON file.

        Args:
            session_id: Session identifier
            session_dir: Optional session directory (defaults to base_session_dir/session_id)

        Returns:
            SessionMetadata object if loaded successfully, None otherwise
        """
        session_path: Path
        if session_dir is None:
            session_path = self.base_session_dir / session_id
        else:
            session_path = Path(session_dir)

        metadata_file = session_path / "metadata.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file) as f:
                data = json.load(f)

            data["devices"] = [
                DeviceInfo(**device_data) for device_data in data.get("devices", [])
            ]
            data["sensor_configs"] = [
                SensorConfig(**config_data)
                for config_data in data.get("sensor_configs", [])
            ]

            metadata = SessionMetadata(**data)
            self._sessions[session_id] = metadata

            return metadata

        except Exception:
            return None

    def get_csv_schema(self, sensor_type: str) -> dict[str, str]:
        """Get standardized CSV schema for sensor type.

        Args:
            sensor_type: Type of sensor ("rgb", "gsr", "thermal")

        Returns:
            Dictionary with column names and descriptions
        """
        schemas = {
            "gsr": {
                "timestamp_ns": "Nanosecond timestamp (monotonic clock)",
                "gsr_microsiemens": "Galvanic Skin Response in microsiemens (µS)",
                "ppg_raw": "Raw photoplethysmography value (0-4095, 12-bit ADC)",
            },
            "rgb": {
                "timestamp_ns": "Nanosecond timestamp (monotonic clock)",
                "filename": "DNG filename (relative to session directory)",
                "format": "Image format (DNG raw format for Phase 3)",
                "width": "Frame width in pixels",
                "height": "Frame height in pixels",
            },
            "thermal": {
                "timestamp_ns": "Nanosecond timestamp (monotonic clock)",
                "image_filename": "Thermal PNG image filename (relative to session directory)",
                "w": "Frame width (256)",
                "h": "Frame height (192)",
                "min_temp_celsius": "Minimum temperature in frame (Celsius)",
                "max_temp_celsius": "Maximum temperature in frame (Celsius)",
                **{
                    f"v{i}": f"Temperature value at pixel {i}"
                    for i in range(0, 49152, 1000)[
                        :50
                    ]
                },
            },
        }

        return schemas.get(sensor_type, {})

    def create_csv_header(self, sensor_type: str) -> str:
        """Create CSV header string for sensor type.

        Args:
            sensor_type: Type of sensor ("rgb", "gsr", "thermal")

        Returns:
            CSV header string
        """
        schema = self.get_csv_schema(sensor_type)
        return ",".join(schema.keys())

    def _anonymize_participant_id(self, participant_id: str) -> str:
        """Anonymize participant ID for privacy.

        Args:
            participant_id: Original participant ID

        Returns:
            Anonymized participant ID
        """
        import hashlib

        hash_object = hashlib.sha256(participant_id.encode())
        hex_dig = hash_object.hexdigest()
        return f"P_{hex_dig[:8]}"

    def _ensure_json_serializable(self, obj):
        """Ensure object is JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, str | int | float | bool) or obj is None:
            return obj
        else:
            return str(obj)

    def get_session_summary(
        self, session_id: str
    ) -> dict[str, str | int | float] | None:
        """Get summary information for a session.

        Args:
            session_id: Session identifier

        Returns:
            Summary dictionary or None if session not found
        """
        if session_id not in self._sessions:
            return None

        metadata = self._sessions[session_id]

        return {
            "session_id": metadata.session_id,
            "participant_id": metadata.participant_id,
            "created_at": metadata.created_at,
            "duration_seconds": metadata.duration_seconds or 0.0,
            "device_count": len(metadata.devices) if metadata.devices else 0,
            "sensor_count": (
                len(metadata.sensor_configs) if metadata.sensor_configs else 0
            ),
            "file_count": len(metadata.data_files) if metadata.data_files else 0,
            "time_sync_enabled": metadata.time_sync_enabled,
            "tls_enabled": metadata.tls_enabled,
            "anonymized": metadata.anonymized,
        }
