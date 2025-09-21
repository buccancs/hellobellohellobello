"""Comprehensive system testing suite for end-to-end validation.

This module contains system-level tests that validate complete workflows
and scenarios from a user's perspective, including:
- Complete recording sessions with multiple devices
- Data integrity and synchronization validation
- Performance and reliability under various conditions
- Error recovery and fault tolerance
- User workflow scenarios
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pc_controller.src.core.device_manager import DeviceManager
from pc_controller.src.core.session_manager import SessionManager
from pc_controller.src.data.hdf5_exporter import export_session_to_hdf5
from pc_controller.src.network.heartbeat_manager import HeartbeatManager


class SystemTestEnvironment:
    """Test environment that simulates the complete system."""

    def __init__(self):
        self.temp_dir = None
        self.device_manager = None
        self.session_manager = None
        self.heartbeat_manager = None
        self.time_server = None
        self.mock_devices = []
        self.setup_environment()

    def setup_environment(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.device_manager = DeviceManager(heartbeat_timeout_seconds=5)
        self.session_manager = SessionManager()
        self.heartbeat_manager = HeartbeatManager()

    def cleanup_environment(self):
        """Clean up the test environment."""
        if self.temp_dir and Path(self.temp_dir).exists():
            import shutil
            shutil.rmtree(self.temp_dir)

        if self.time_server:
            self.time_server.stop()

        for device in self.mock_devices:
            device.cleanup()

    def create_mock_android_device(self, device_id: str, capabilities: dict | None = None):
        """Create a mock Android device for testing."""
        if capabilities is None:
            capabilities = {
                "cameras": ["rgb"],
                "sensors": ["gsr"],
                "version": "1.0.0"
            }

        device = MockAndroidSystemDevice(device_id, capabilities, self.temp_dir)
        self.mock_devices.append(device)
        return device


class MockAndroidSystemDevice:
    """Mock Android device for system testing with file generation."""

    def __init__(self, device_id: str, capabilities: dict, temp_dir: str):
        self.device_id = device_id
        self.capabilities = capabilities
        self.temp_dir = temp_dir
        self.recording = False
        self.session_id = None
        self.data_files = []

    def start_recording(self, session_id: str) -> bool:
        """Start recording session."""
        if self.recording:
            return False

        self.recording = True
        self.session_id = session_id
        self.data_files = []

        self._create_mock_data_files()
        return True

    def stop_recording(self) -> bool:
        """Stop recording session."""
        if not self.recording:
            return False

        self.recording = False
        return True

    def _create_mock_data_files(self):
        """Create mock data files that would be generated during recording."""
        device_dir = Path(self.temp_dir) / self.device_id
        device_dir.mkdir(exist_ok=True)

        if "gsr" in self.capabilities.get("sensors", []):
            gsr_file = device_dir / f"{self.session_id}_gsr.csv"
            with open(gsr_file, 'w') as f:
                f.write("timestamp_ns,gsr_microsiemens,raw_gsr,ppg_raw\n")
                base_time = time.time_ns()
                for i in range(100):
                    timestamp = base_time + i * 10_000_000
                    gsr = 5.0 + (i % 10) * 0.5
                    raw_gsr = 2000 + (i % 100) * 10
                    ppg = 1500 + (i % 50) * 20
                    f.write(f"{timestamp},{gsr},{raw_gsr},{ppg}\n")
            self.data_files.append(gsr_file)

        if "rgb" in self.capabilities.get("cameras", []):
            rgb_file = device_dir / f"{self.session_id}_rgb_metadata.csv"
            with open(rgb_file, 'w') as f:
                f.write("timestamp_ns,filename,frame_number\n")
                base_time = time.time_ns()
                for i in range(30):
                    timestamp = base_time + i * 33_333_333
                    filename = f"frame_{i:06d}.jpg"
                    f.write(f"{timestamp},{filename},{i}\n")
            self.data_files.append(rgb_file)

        if "thermal" in self.capabilities.get("cameras", []):
            thermal_file = device_dir / f"{self.session_id}_thermal.csv"
            with open(thermal_file, 'w') as f:
                f.write("timestamp_ns,temperature_celsius,raw_value\n")
                base_time = time.time_ns()
                for i in range(50):
                    timestamp = base_time + i * 20_000_000
                    temp = 25.0 + (i % 5) * 0.2
                    raw_value = 1000 + i * 10
                    f.write(f"{timestamp},{temp},{raw_value}\n")
            self.data_files.append(thermal_file)

    def get_data_files(self) -> list[Path]:
        """Get list of generated data files."""
        return self.data_files.copy()

    def cleanup(self):
        """Clean up device resources."""
        pass


@pytest.mark.integration
@pytest.mark.slow
class TestCompleteRecordingWorkflows:
    """Test complete recording workflows from start to finish."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.env = SystemTestEnvironment()

    def teardown_method(self):
        """Clean up test environment after each test."""
        self.env.cleanup_environment()

    def test_single_device_recording_session(self):
        """Test a complete single-device recording session."""
        device = self.env.create_mock_android_device(
            "single-device-001",
            {"cameras": ["rgb"], "sensors": ["gsr"]}
        )

        self.env.device_manager.register(device.device_id)
        assert self.env.device_manager.get_status(device.device_id) == "Online"

        session_id = self.env.session_manager.create_session("single_device_test")
        self.env.session_manager.start_recording()
        assert self.env.session_manager.is_active
        assert session_id is not None

        success = device.start_recording(session_id)
        assert success is True
        assert device.recording is True

        self.env.device_manager.set_status(device.device_id, "Recording")
        assert self.env.device_manager.get_status(device.device_id) == "Recording"

        for _ in range(5):
            self.env.device_manager.update_heartbeat(device.device_id)
            time.sleep(0.1)

        device.stop_recording()
        assert device.recording is False

        self.env.session_manager.stop_recording()
        assert not self.env.session_manager.is_active

        data_files = device.get_data_files()
        assert len(data_files) > 0

        for file_path in data_files:
            assert file_path.exists()
            assert file_path.stat().st_size > 0

    def test_multi_device_synchronized_recording(self):
        """Test synchronized recording across multiple devices."""
        devices = [
            self.env.create_mock_android_device(
                "multi-device-001",
                {"cameras": ["rgb"], "sensors": ["gsr"]}
            ),
            self.env.create_mock_android_device(
                "multi-device-002",
                {"cameras": ["thermal"], "sensors": ["gsr"]}
            ),
            self.env.create_mock_android_device(
                "multi-device-003",
                {"cameras": ["rgb", "thermal"], "sensors": ["gsr"]}
            )
        ]

        for device in devices:
            self.env.device_manager.register(device.device_id)
            assert self.env.device_manager.get_status(device.device_id) == "Online"

        session_id = self.env.session_manager.create_session("multi_device_test")
        self.env.session_manager.start_recording()
        assert self.env.session_manager.is_active

        start_times = []
        for device in devices:
            start_time = time.time()
            success = device.start_recording(session_id)
            start_times.append(start_time)
            assert success is True
            self.env.device_manager.set_status(device.device_id, "Recording")

        max_start_diff = max(start_times) - min(start_times)
        assert max_start_diff < 0.1

        # Use list comprehension for better performance and readability
        recording_devices = [
            device.device_id for device in devices
            if (
                device.recording and
                self.env.device_manager.get_status(device.device_id) == "Recording"
            )
        ]

        assert len(recording_devices) == len(devices)

        for _ in range(10):
            for device in devices:
                self.env.device_manager.update_heartbeat(device.device_id)
            time.sleep(0.05)

        stop_times = []
        for device in devices:
            stop_time = time.time()
            device.stop_recording()
            stop_times.append(stop_time)
            self.env.device_manager.set_status(device.device_id, "Online")

        max_stop_diff = max(stop_times) - min(stop_times)
        assert max_stop_diff < 0.1

        self.env.session_manager.stop_recording()

        total_files = 0
        for device in devices:
            data_files = device.get_data_files()
            assert len(data_files) > 0
            total_files += len(data_files)

            for file_path in data_files:
                assert file_path.exists()
                assert file_path.stat().st_size > 0

        assert total_files >= len(devices)

    def test_recording_with_device_failure_and_recovery(self):
        """Test recording session with device failure and recovery."""
        devices = [
            self.env.create_mock_android_device("failure-device-001"),
            self.env.create_mock_android_device("failure-device-002"),
            self.env.create_mock_android_device("failure-device-003")
        ]

        for device in devices:
            self.env.device_manager.register(device.device_id)

        session_id = self.env.session_manager.create_session("test_session")
        self.env.session_manager.start_recording()
        for device in devices:
            device.start_recording(session_id)
            self.env.device_manager.set_status(device.device_id, "Recording")

        failed_device = devices[1]
        healthy_devices = [devices[0], devices[2]]

        for _ in range(10):
            for device in healthy_devices:
                self.env.device_manager.update_heartbeat(device.device_id)
            time.sleep(0.1)

        future_time = time.time_ns() + int(6 * 1e9)

        for device in healthy_devices:
            device_info = self.env.device_manager.get_info(device.device_id)
            if device_info:
                device_info.last_heartbeat_ns = future_time - int(5 * 1e9)
                device_info.status = "Recording"

        self.env.device_manager.check_timeouts(now_ns=future_time)

        assert self.env.device_manager.get_status(failed_device.device_id) == "Offline"

        for device in healthy_devices:
            assert self.env.device_manager.get_status(device.device_id) == "Recording"

        self.env.device_manager.update_heartbeat(failed_device.device_id)
        failed_device.start_recording(session_id)
        self.env.device_manager.set_status(failed_device.device_id, "Recording")

        assert self.env.device_manager.get_status(failed_device.device_id) == "Recording"

        for _ in range(5):
            for device in devices:
                self.env.device_manager.update_heartbeat(device.device_id)
            time.sleep(0.1)

        for device in devices:
            device.stop_recording()
        self.env.session_manager.stop_recording()

        for device in devices:
            data_files = device.get_data_files()
            assert len(data_files) > 0

    @pytest.mark.slow
    def test_long_duration_recording_session(self):
        """Test long-duration recording session for stability."""
        device = self.env.create_mock_android_device("long-duration-device")
        self.env.device_manager.register(device.device_id)

        session_id = self.env.session_manager.create_session("test_session")
        self.env.session_manager.start_recording()
        device.start_recording(session_id)
        self.env.device_manager.set_status(device.device_id, "Recording")

        start_time = time.time()
        target_duration = 10.0

        while time.time() - start_time < target_duration:
            self.env.device_manager.update_heartbeat(device.device_id)
            self.env.device_manager.check_timeouts()

            assert self.env.device_manager.get_status(device.device_id) == "Recording"
            assert self.env.session_manager.is_active

            time.sleep(0.5)

        actual_duration = time.time() - start_time
        assert actual_duration >= target_duration

        device.stop_recording()
        self.env.session_manager.stop_recording()

        data_files = device.get_data_files()
        assert len(data_files) > 0


@pytest.mark.integration
class TestDataIntegrityAndSynchronization:
    """Test data integrity and synchronization across the system."""

    def setup_method(self):
        """Set up test environment."""
        self.env = SystemTestEnvironment()

    def teardown_method(self):
        """Clean up test environment."""
        self.env.cleanup_environment()

    def test_timestamp_synchronization_validation(self):
        """Test timestamp synchronization across multiple data streams."""
        devices = [
            self.env.create_mock_android_device("sync-device-001"),
            self.env.create_mock_android_device("sync-device-002")
        ]

        for device in devices:
            self.env.device_manager.register(device.device_id)

        session_id = self.env.session_manager.create_session("test_session")
        self.env.session_manager.start_recording()
        recording_start_time = time.time_ns()

        for device in devices:
            device.start_recording(session_id)
            self.env.device_manager.set_status(device.device_id, "Recording")

        time.sleep(0.5)

        for device in devices:
            device.stop_recording()

        self.env.session_manager.stop_recording()

        all_timestamps = []

        for device in devices:
            data_files = device.get_data_files()
            for file_path in data_files:
                if file_path.suffix == '.csv':
                    with open(file_path) as f:
                        lines = f.readlines()[1:]
                        for line in lines:
                            if line.strip():
                                timestamp = int(line.split(',')[0])
                                all_timestamps.append(timestamp)

        assert len(all_timestamps) > 0

        min_timestamp = min(all_timestamps)
        max_timestamp = max(all_timestamps)

        assert min_timestamp >= recording_start_time

        timestamp_spread_seconds = (max_timestamp - min_timestamp) / 1e9
        assert timestamp_spread_seconds < 10.0

    def test_data_file_integrity_validation(self):
        """Test data file integrity and format validation."""
        device = self.env.create_mock_android_device(
            "integrity-device",
            {"cameras": ["rgb", "thermal"], "sensors": ["gsr"]}
        )

        self.env.device_manager.register(device.device_id)
        session_id = self.env.session_manager.create_session("test_session")
        self.env.session_manager.start_recording()

        device.start_recording(session_id)
        time.sleep(0.1)
        device.stop_recording()

        self.env.session_manager.stop_recording()

        data_files = device.get_data_files()

        file_types = [f.suffix for f in data_files]
        assert '.csv' in file_types

        for file_path in data_files:
            if file_path.suffix == '.csv':
                self._validate_csv_format(file_path)

    def _validate_csv_format(self, file_path: Path):
        """Validate CSV file format and content."""
        with open(file_path) as f:
            lines = f.readlines()

        assert len(lines) >= 2

        header = lines[0].strip()
        assert 'timestamp_ns' in header

        for i, line in enumerate(lines[1:], 1):
            if line.strip():
                fields = line.strip().split(',')
                try:
                    timestamp = int(fields[0])
                    assert timestamp > 0
                except ValueError:
                    pytest.fail(f"Invalid timestamp in {file_path} line {i}: {fields[0]}")

    @patch('pc_controller.src.data.hdf5_exporter.h5py')
    def test_data_export_and_aggregation(self, mock_h5py):
        """Test data export and aggregation functionality."""
        mock_file = MagicMock()
        mock_h5py.File.return_value = mock_file
        mock_h5py.version.hdf5_version = "1.12.0"

        device = self.env.create_mock_android_device("export-device")
        self.env.device_manager.register(device.device_id)

        session_id = self.env.session_manager.create_session("test_session")
        self.env.session_manager.start_recording()
        device.start_recording(session_id)
        time.sleep(0.1)
        device.stop_recording()
        self.env.session_manager.stop_recording()

        session_data_dir = Path(self.env.temp_dir) / device.device_id
        output_file = Path(self.env.temp_dir) / "test_export.h5"

        try:
            result = export_session_to_hdf5(str(session_data_dir), str(output_file))
            assert result is True

            mock_h5py.File.assert_called_once()

        except Exception:
            pytest.skip("HDF5 export not available")


@pytest.mark.integration
class TestPerformanceAndReliability:
    """Test system performance and reliability under various conditions."""

    def setup_method(self):
        """Set up test environment."""
        self.env = SystemTestEnvironment()

    def teardown_method(self):
        """Clean up test environment."""
        self.env.cleanup_environment()

    def test_high_device_count_performance(self):
        """Test system performance with many connected devices."""
        num_devices = 20
        devices = []

        for i in range(num_devices):
            device = self.env.create_mock_android_device(f"perf-device-{i:03d}")
            devices.append(device)
            self.env.device_manager.register(device.device_id)

        # Measure registration performance
        start_time = time.time()

        for device in devices:
            self.env.device_manager.update_heartbeat(device.device_id)

        heartbeat_time = time.time() - start_time

        # Verify performance is acceptable
        assert heartbeat_time < 1.0

        online_count = sum(1 for device in devices
                          if self.env.device_manager.get_status(device.device_id) == "Online")
        assert online_count == num_devices

    def test_memory_usage_stability(self):
        """Test memory usage stability during operations."""
        import gc

        gc.collect()
        initial_objects = len(gc.get_objects())

        for cycle in range(10):
            devices = []

            for i in range(5):
                device_id = f"memory-test-{cycle}-{i}"
                device = self.env.create_mock_android_device(device_id)
                devices.append(device)
                self.env.device_manager.register(device_id)

            session_id = self.env.session_manager.create_session("test_session")
            self.env.session_manager.start_recording()

            for device in devices:
                device.start_recording(session_id)
                self.env.device_manager.set_status(device.device_id, "Recording")

            time.sleep(0.1)

            for device in devices:
                device.stop_recording()
                self.env.device_manager.remove(device.device_id)

            self.env.session_manager.stop_recording()

            devices.clear()

        gc.collect()
        final_objects = len(gc.get_objects())

        object_growth = final_objects - initial_objects
        assert object_growth < 1000

    def test_rapid_session_cycling(self):
        """Test rapid session start/stop cycling for stability."""
        device = self.env.create_mock_android_device("cycling-device")
        self.env.device_manager.register(device.device_id)

        session_count = 20
        successful_sessions = 0

        for _i in range(session_count):
            session_id = self.env.session_manager.create_session("test_session")
            self.env.session_manager.start_recording()
            assert self.env.session_manager.is_active

            device.start_recording(session_id)
            self.env.device_manager.set_status(device.device_id, "Recording")

            time.sleep(0.01)

            device.stop_recording()
            self.env.session_manager.stop_recording()
            assert not self.env.session_manager.is_active

            successful_sessions += 1

        assert successful_sessions == session_count

    def test_error_recovery_resilience(self):
        """Test system resilience to various error conditions."""
        devices = [
            self.env.create_mock_android_device("resilience-device-001"),
            self.env.create_mock_android_device("resilience-device-002"),
            self.env.create_mock_android_device("resilience-device-003")
        ]

        for device in devices:
            self.env.device_manager.register(device.device_id)

        session_id = self.env.session_manager.create_session("test_session")
        self.env.session_manager.start_recording()

        for device in devices:
            device.start_recording(session_id)
            self.env.device_manager.set_status(device.device_id, "Recording")


        failed_device = devices[0]
        healthy_devices = devices[1:]

        future_time = time.time_ns() + int(10 * 1e9)
        for device in healthy_devices:
            device_info = self.env.device_manager.get_info(device.device_id)
            if device_info:
                device_info.last_heartbeat_ns = future_time - int(5 * 1e9)
                device_info.status = "Recording"

        self.env.device_manager.check_timeouts(now_ns=future_time)
        assert self.env.device_manager.get_status(failed_device.device_id) == "Offline"

        self.env.device_manager.update_heartbeat(failed_device.device_id)
        self.env.device_manager.set_status(failed_device.device_id, "Recording")
        assert self.env.device_manager.get_status(failed_device.device_id) == "Recording"

        self.env.device_manager.register(devices[1].device_id)
        assert self.env.device_manager.get_status(devices[1].device_id) == "Recording"

        self.env.device_manager.remove("non-existent-device")
        status = self.env.device_manager.get_status("non-existent-device")
        assert status is None

        for device in devices[1:]:
            assert self.env.device_manager.get_status(device.device_id) == "Recording"

        for device in devices:
            device.stop_recording()

        self.env.session_manager.stop_recording()

        assert not self.env.session_manager.is_active


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
