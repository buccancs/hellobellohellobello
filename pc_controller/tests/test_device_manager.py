from __future__ import annotations

import time
from unittest.mock import patch

from pc_controller.src.core.device_manager import DeviceInfo, DeviceManager


def test_heartbeat_updates_and_timeout() -> None:
    dm = DeviceManager(heartbeat_timeout_seconds=1)
    dev = "dev123"
    assert dm.get_status(dev) is None

    dm.update_heartbeat(dev)
    assert dm.get_status(dev) == "Online"

    dm.check_timeouts(now_ns=time.time_ns())
    assert dm.get_status(dev) == "Online"

    past = time.time_ns() + int(1.5 * 1e9)
    dm.check_timeouts(now_ns=past)
    assert dm.get_status(dev) == "Offline"

    dm.update_heartbeat(dev)
    assert dm.get_status(dev) == "Online"


def test_device_registration_and_info() -> None:
    """Test device registration and info retrieval."""
    dm = DeviceManager(heartbeat_timeout_seconds=10)
    device_id = "test-device-001"

    assert dm.get_info(device_id) is None
    assert dm.get_status(device_id) is None

    dm.register(device_id)
    info = dm.get_info(device_id)
    assert info is not None
    assert info.device_id == device_id
    assert info.status == "Online"
    assert info.first_seen_ns > 0
    assert info.last_heartbeat_ns > 0

    old_first_seen = info.first_seen_ns
    dm.register(device_id)
    info2 = dm.get_info(device_id)
    assert info2.first_seen_ns == old_first_seen


def test_device_status_management() -> None:
    """Test setting and getting device status."""
    dm = DeviceManager(heartbeat_timeout_seconds=5)
    device_id = "status-test-device"

    dm.set_status(device_id, "Recording")
    assert dm.get_status(device_id) == "Recording"
    assert dm.get_info(device_id) is not None

    dm.set_status(device_id, "Idle")
    assert dm.get_status(device_id) == "Idle"

    # Setting status updates heartbeat
    old_heartbeat = dm.get_info(device_id).last_heartbeat_ns
    time.sleep(0.001)
    dm.set_status(device_id, "Processing")
    new_heartbeat = dm.get_info(device_id).last_heartbeat_ns
    assert new_heartbeat > old_heartbeat


def test_device_removal() -> None:
    """Test device removal functionality."""
    dm = DeviceManager(heartbeat_timeout_seconds=5)
    device_id = "removable-device"

    dm.register(device_id)
    assert dm.get_status(device_id) == "Online"

    dm.remove(device_id)
    assert dm.get_status(device_id) is None
    assert dm.get_info(device_id) is None

    dm.remove("non-existent-device")


def test_list_devices() -> None:
    """Test listing all registered devices."""
    dm = DeviceManager(heartbeat_timeout_seconds=10)

    assert len(dm.list_devices()) == 0

    devices = ["device-1", "device-2", "device-3"]
    for device_id in devices:
        dm.register(device_id)

    device_list = dm.list_devices()
    assert len(device_list) == 3
    for device_id in devices:
        assert device_id in device_list
        assert isinstance(device_list[device_id], DeviceInfo)


def test_timeout_property() -> None:
    """Test timeout seconds property."""
    dm = DeviceManager(heartbeat_timeout_seconds=7)
    assert dm.timeout_seconds == 7.0

    dm2 = DeviceManager(heartbeat_timeout_seconds=15)
    assert dm2.timeout_seconds == 15.0


def test_multiple_device_timeout_scenarios() -> None:
    """Test timeout scenarios with multiple devices."""
    dm = DeviceManager(heartbeat_timeout_seconds=2)

    base_time = time.time_ns()

    dm.update_heartbeat("device-1")
    time.sleep(0.001)
    dm.update_heartbeat("device-2")
    time.sleep(0.001)
    dm.update_heartbeat("device-3")

    assert dm.get_status("device-1") == "Online"
    assert dm.get_status("device-2") == "Online"
    assert dm.get_status("device-3") == "Online"

    future_time = base_time + int(1.5 * 1e9)
    dm.check_timeouts(now_ns=future_time)

    assert dm.get_status("device-1") == "Online"
    assert dm.get_status("device-2") == "Online"
    assert dm.get_status("device-3") == "Online"

    timeout_time = base_time + int(3 * 1e9)
    dm.check_timeouts(now_ns=timeout_time)

    assert dm.get_status("device-1") == "Offline"
    assert dm.get_status("device-2") == "Offline"
    assert dm.get_status("device-3") == "Offline"


def test_device_info_to_dict() -> None:
    """Test DeviceInfo serialization."""
    device_info = DeviceInfo(
        device_id="test-device",
        first_seen_ns=1234567890000000000,
        last_heartbeat_ns=1234567891000000000,
        status="Recording"
    )

    result = device_info.to_dict()
    expected = {
        "device_id": "test-device",
        "first_seen_ns": 1234567890000000000,
        "last_heartbeat_ns": 1234567891000000000,
        "status": "Recording"
    }
    assert result == expected


@patch('pc_controller.src.core.device_manager.cfg_get')
def test_default_timeout_from_config(mock_cfg_get) -> None:
    """Test default timeout value from configuration."""
    # Test with config value
    mock_cfg_get.return_value = "30"
    dm = DeviceManager()
    assert dm.timeout_seconds == 30.0
    mock_cfg_get.assert_called_once_with("heartbeat_timeout_seconds", 10)

    mock_cfg_get.reset_mock()
    mock_cfg_get.return_value = 15
    dm2 = DeviceManager()
    assert dm2.timeout_seconds == 15.0


def test_edge_cases_and_boundary_conditions() -> None:
    """Test edge cases and boundary conditions."""
    dm = DeviceManager(heartbeat_timeout_seconds=0.001)
    device_id = "edge-case-device"

    dm.register(device_id)
    assert dm.get_status(device_id) == "Online"

    time.sleep(0.002)
    dm.check_timeouts()
    assert dm.get_status(device_id) == "Offline"

    dm_zero = DeviceManager(heartbeat_timeout_seconds=0)
    dm_zero.register("zero-timeout-device")
    dm_zero.check_timeouts()
    assert dm_zero.get_status("zero-timeout-device") == "Offline"

    dm_large = DeviceManager(heartbeat_timeout_seconds=999999)
    dm_large.register("large-timeout-device")
    future_time = time.time_ns() + int(100 * 1e9)
    dm_large.check_timeouts(now_ns=future_time)
    assert dm_large.get_status("large-timeout-device") == "Online"


def test_concurrent_device_operations() -> None:
    """Test concurrent-like operations on device manager."""
    dm = DeviceManager(heartbeat_timeout_seconds=5)
    device_id = "concurrent-test-device"

    dm.register(device_id)
    dm.update_heartbeat(device_id)
    dm.set_status(device_id, "Processing")
    dm.update_heartbeat(device_id)

    info = dm.get_info(device_id)
    assert info.status == "Processing"
    assert info.device_id == device_id


def test_empty_and_whitespace_device_ids() -> None:
    """Test handling of edge case device IDs."""
    dm = DeviceManager(heartbeat_timeout_seconds=10)

    dm.register("")
    assert dm.get_status("") == "Online"

    dm.register("   ")
    assert dm.get_status("   ") == "Online"

    assert len(dm.list_devices()) == 2
