"""Tests for SessionMetadataManager functionality."""

import tempfile
from pathlib import Path

from pc_controller.src.data.metadata_manager import (
    DeviceInfo,
    SensorConfig,
    SessionMetadata,
    SessionMetadataManager,
)


class TestSessionMetadata:
    """Test SessionMetadata dataclass functionality."""

    def test_session_metadata_creation(self):
        """Test creating SessionMetadata with required fields."""
        metadata = SessionMetadata(
            session_id="test_session",
            participant_id="P_12345678",
            created_at="2024-01-01T10:00:00"
        )

        assert metadata.session_id == "test_session"
        assert metadata.participant_id == "P_12345678"
        assert metadata.created_at == "2024-01-01T10:00:00"
        assert len(metadata.devices) == 0
        assert len(metadata.sensor_configs) == 0
        assert len(metadata.data_files) == 0
        assert metadata.anonymized is True


class TestDeviceInfo:
    """Test DeviceInfo dataclass functionality."""

    def test_device_info_creation(self):
        """Test creating DeviceInfo with required fields."""
        device = DeviceInfo(
            device_id="device_001",
            device_type="android",
            device_name="Pixel 6",
            model="Google Pixel 6",
            os_version="Android 12",
            app_version="1.0.0",
            ip_address="192.168.1.100",
            sensors=["rgb", "gsr"]
        )

        assert device.device_id == "device_001"
        assert device.device_type == "android"
        assert "rgb" in device.sensors
        assert device.time_offset_ns is None


class TestSensorConfig:
    """Test SensorConfig dataclass functionality."""

    def test_sensor_config_creation(self):
        """Test creating SensorConfig with settings."""
        config = SensorConfig(
            sensor_type="gsr",
            sampling_rate=50.0,
            settings={"gain": 1, "filter": "low_pass"}
        )

        assert config.sensor_type == "gsr"
        assert config.sampling_rate == 50.0
        assert config.settings["gain"] == 1


class TestSessionMetadataManager:
    """Test SessionMetadataManager functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = SessionMetadataManager(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_session_metadata(self):
        """Test creating session metadata."""
        metadata = self.manager.create_session_metadata("participant_001")

        assert metadata.session_id.startswith("session_")
        assert metadata.participant_id.startswith("P_")
        assert metadata.anonymized is True
        assert metadata in self.manager._sessions.values()

    def test_create_session_metadata_with_id(self):
        """Test creating session metadata with specific ID."""
        metadata = self.manager.create_session_metadata("participant_001", "custom_session")

        assert metadata.session_id == "custom_session"
        assert "custom_session" in self.manager._sessions

    def test_add_device(self):
        """Test adding device information to session."""
        metadata = self.manager.create_session_metadata("participant_001", "test_session")

        device = DeviceInfo(
            device_id="device_001",
            device_type="android",
            device_name="Test Device",
            model="Test Model",
            os_version="Android 12",
            app_version="1.0.0",
            ip_address="192.168.1.100",
            sensors=["rgb", "gsr"]
        )

        result = self.manager.add_device("test_session", device)
        assert result is True
        assert len(metadata.devices) == 1
        assert metadata.devices[0].device_id == "device_001"

    def test_add_device_nonexistent_session(self):
        """Test adding device to non-existent session."""
        device = DeviceInfo(
            device_id="device_001",
            device_type="android",
            device_name="Test Device",
            model="Test Model",
            os_version="Android 12",
            app_version="1.0.0",
            ip_address="192.168.1.100",
            sensors=["rgb"]
        )

        result = self.manager.add_device("nonexistent_session", device)
        assert result is False

    def test_add_sensor_config(self):
        """Test adding sensor configuration to session."""
        metadata = self.manager.create_session_metadata("participant_001", "test_session")

        sensor_config = SensorConfig(
            sensor_type="gsr",
            sampling_rate=50.0,
            settings={"gain": 1}
        )

        result = self.manager.add_sensor_config("test_session", sensor_config)
        assert result is True
        assert len(metadata.sensor_configs) == 1
        assert metadata.sensor_configs[0].sensor_type == "gsr"

    def test_update_time_sync_info(self):
        """Test updating time sync information."""
        metadata = self.manager.create_session_metadata("participant_001", "test_session")

        device = DeviceInfo(
            device_id="device_001",
            device_type="android",
            device_name="Test Device",
            model="Test Model",
            os_version="Android 12",
            app_version="1.0.0",
            ip_address="192.168.1.100",
            sensors=["gsr"]
        )

        self.manager.add_device("test_session", device)

        result = self.manager.update_time_sync_info("test_session", "device_001", 5000000, "good")
        assert result is True
        assert metadata.devices[0].time_offset_ns == 5000000
        assert metadata.devices[0].sync_quality == "good"

    def test_finalize_session(self):
        """Test finalizing session with duration and files."""
        metadata = self.manager.create_session_metadata("participant_001", "test_session")

        data_files = ["gsr.csv", "rgb.csv", "thermal.csv"]
        result = self.manager.finalize_session("test_session", 120.5, data_files)

        assert result is True
        assert metadata.duration_seconds == 120.5
        assert metadata.data_files == data_files

    def test_save_and_load_metadata(self):
        """Test saving and loading metadata to/from JSON."""
        self.manager.create_session_metadata("participant_001", "test_session")

        device = DeviceInfo(
            device_id="device_001",
            device_type="android",
            device_name="Test Device",
            model="Test Model",
            os_version="Android 12",
            app_version="1.0.0",
            ip_address="192.168.1.100",
            sensors=["gsr"]
        )
        self.manager.add_device("test_session", device)

        result = self.manager.save_metadata("test_session")
        assert result is True

        metadata_file = Path(self.temp_dir) / "test_session" / "metadata.json"
        assert metadata_file.exists()

        loaded_metadata = self.manager.load_metadata("test_session")
        assert loaded_metadata is not None
        assert loaded_metadata.session_id == "test_session"
        assert len(loaded_metadata.devices) == 1
        assert loaded_metadata.devices[0].device_id == "device_001"

    def test_get_csv_schema(self):
        """Test getting CSV schema for different sensor types."""
        gsr_schema = self.manager.get_csv_schema("gsr")
        assert "timestamp_ns" in gsr_schema
        assert "gsr_microsiemens" in gsr_schema
        assert "ppg_raw" in gsr_schema

        rgb_schema = self.manager.get_csv_schema("rgb")
        assert "timestamp_ns" in rgb_schema
        assert "filename" in rgb_schema
        assert "width" in rgb_schema

        thermal_schema = self.manager.get_csv_schema("thermal")
        assert "timestamp_ns" in thermal_schema
        assert "w" in thermal_schema
        assert "h" in thermal_schema
        assert "v0" in thermal_schema
        assert "v1000" in thermal_schema

    def test_create_csv_header(self):
        """Test creating CSV header strings."""
        gsr_header = self.manager.create_csv_header("gsr")
        assert gsr_header == "timestamp_ns,gsr_microsiemens,ppg_raw"

        rgb_header = self.manager.create_csv_header("rgb")
        assert rgb_header.startswith("timestamp_ns")
        assert "filename" in rgb_header

    def test_anonymize_participant_id(self):
        """Test participant ID anonymization."""
        anon_id1 = self.manager._anonymize_participant_id("participant_001")
        anon_id2 = self.manager._anonymize_participant_id("participant_001")
        anon_id3 = self.manager._anonymize_participant_id("participant_002")

        assert anon_id1 == anon_id2
        assert anon_id1 != anon_id3
        assert anon_id1.startswith("P_")

    def test_get_session_summary(self):
        """Test getting session summary."""
        self.manager.create_session_metadata("participant_001", "test_session")

        device = DeviceInfo(
            device_id="device_001",
            device_type="android",
            device_name="Test Device",
            model="Test Model",
            os_version="Android 12",
            app_version="1.0.0",
            ip_address="192.168.1.100",
            sensors=["gsr"]
        )
        self.manager.add_device("test_session", device)
        self.manager.finalize_session("test_session", 120.0, ["gsr.csv"])

        summary = self.manager.get_session_summary("test_session")
        assert summary is not None
        assert summary["session_id"] == "test_session"
        assert summary["duration_seconds"] == 120.0
        assert summary["device_count"] == 1
        assert summary["file_count"] == 1
        assert summary["anonymized"] is True

    def test_get_session_summary_nonexistent(self):
        """Test getting summary for non-existent session."""
        summary = self.manager.get_session_summary("nonexistent_session")
        assert summary is None
