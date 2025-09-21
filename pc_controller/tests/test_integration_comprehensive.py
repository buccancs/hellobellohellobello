"""Comprehensive integration tests for Hub-Spoke communication and multi-component interactions.

This module focuses on testing the integration between different components of the system:
- Network communication between PC Controller and Android devices
- Data flow and synchronization
- Session management across multiple devices
- Error handling and recovery scenarios
- Performance under load
"""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import pytest

from pc_controller.src.core.device_manager import DeviceManager
from pc_controller.src.core.session_manager import SessionManager
from pc_controller.src.network.heartbeat_manager import HeartbeatManager
from pc_controller.src.network.protocol import (
    COMMAND_QUERY_CAPABILITIES,
    COMMAND_START_RECORDING,
    COMMAND_STOP_RECORDING,
    build_query_capabilities,
    build_start_recording,
    build_stop_recording,
    parse_json_line,
)
from pc_controller.src.network.time_server import TimeSyncServer


class MockAndroidDevice:
    """Mock Android device for integration testing."""

    def __init__(self, device_id: str, host: str = "localhost", port: int = 0):
        self.device_id = device_id
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.recording = False
        self.capabilities = {
            "device_id": device_id,
            "cameras": ["rgb", "thermal"],
            "sensors": ["gsr"],
            "version": "1.0.0"
        }

    def start_server(self) -> int:
        """Start mock Android device server. Returns the actual port."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        actual_port = self.socket.getsockname()[1]

        def handle_client():
            while self.connected:
                try:
                    client, addr = self.socket.accept()
                    self._handle_connection(client)
                except OSError:
                    break

        self.connected = True
        self.thread = threading.Thread(target=handle_client, daemon=True)
        self.thread.start()
        return actual_port

    def _handle_connection(self, client: socket.socket) -> None:
        """Handle incoming connection from PC Controller."""
        try:
            while self.connected:
                data = client.recv(1024)
                if not data:
                    break

                message = data.decode('utf-8').strip()
                if not message:
                    continue

                try:
                    parsed = parse_json_line(message)
                    response = self._handle_command(parsed)
                    if response:
                        client.send((json.dumps(response) + '\n').encode('utf-8'))
                except Exception as e:
                    error_response = {
                        "error": str(e),
                        "id": parsed.get("id", -1) if 'parsed' in locals() else -1
                    }
                    client.send((json.dumps(error_response) + '\n').encode('utf-8'))
        finally:
            client.close()

    def _handle_command(self, command: dict) -> dict:
        """Handle specific commands from PC Controller."""
        cmd_type = command.get("command")
        cmd_id = command.get("id", -1)

        if cmd_type == COMMAND_QUERY_CAPABILITIES:
            return {
                "id": cmd_id,
                "status": "success",
                "capabilities": self.capabilities
            }
        elif cmd_type == COMMAND_START_RECORDING:
            self.recording = True
            return {
                "id": cmd_id,
                "status": "recording_started",
                "session_id": command.get("session_id")
            }
        elif cmd_type == COMMAND_STOP_RECORDING:
            self.recording = False
            return {
                "id": cmd_id,
                "status": "recording_stopped"
            }
        else:
            return {
                "id": cmd_id,
                "status": "unknown_command",
                "command": cmd_type
            }

    def stop_server(self) -> None:
        """Stop the mock server."""
        self.connected = False
        if self.socket:
            self.socket.close()


class TestHubSpokeIntegration:
    """Test Hub-Spoke communication patterns."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_devices = []
        self.device_manager = DeviceManager(heartbeat_timeout_seconds=5)
        self.session_manager = SessionManager()

    def teardown_method(self):
        """Clean up test fixtures."""
        for device in self.mock_devices:
            device.stop_server()

    def test_device_discovery_and_capability_exchange(self):
        """Test device discovery and capability exchange."""
        mock_device = MockAndroidDevice("test-device-001")
        port = mock_device.start_server()
        self.mock_devices.append(mock_device)

        time.sleep(0.1)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect(("localhost", port))

            query = build_query_capabilities(1)
            client.send(query.encode('utf-8'))

            response = client.recv(1024).decode('utf-8')
            parsed_response = json.loads(response.strip())

            assert parsed_response["id"] == 1
            assert parsed_response["status"] == "success"
            assert "capabilities" in parsed_response
            assert parsed_response["capabilities"]["device_id"] == "test-device-001"

    def test_synchronized_recording_start_stop(self):
        """Test synchronized recording across multiple devices."""
        devices = []
        for i in range(3):
            device = MockAndroidDevice(f"device-{i+1}")
            port = device.start_server()
            devices.append((device, port))
            self.mock_devices.append(device)

        time.sleep(0.1)

        clients = []
        for _, port in devices:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(("localhost", port))
            clients.append(client)

        try:
            session_id = "test-session-123"

            for i, client in enumerate(clients):
                start_cmd = build_start_recording(session_id, i + 1)
                client.send(start_cmd.encode('utf-8'))

            for i, client in enumerate(clients):
                response = client.recv(1024).decode('utf-8')
                parsed = json.loads(response.strip())
                assert parsed["status"] == "recording_started"
                assert parsed["session_id"] == session_id
                assert devices[i][0].recording is True

            for i, client in enumerate(clients):
                stop_cmd = build_stop_recording(i + 10)
                client.send(stop_cmd.encode('utf-8'))

            for i, client in enumerate(clients):
                response = client.recv(1024).decode('utf-8')
                parsed = json.loads(response.strip())
                assert parsed["status"] == "recording_stopped"
                assert devices[i][0].recording is False

        finally:
            for client in clients:
                client.close()

    def test_device_timeout_and_recovery(self):
        """Test device timeout detection and recovery."""
        device_id = "timeout-test-device"

        self.device_manager.register(device_id)
        assert self.device_manager.get_status(device_id) == "Online"

        future_time = time.time_ns() + int(10 * 1e9)
        self.device_manager.check_timeouts(now_ns=future_time)
        assert self.device_manager.get_status(device_id) == "Offline"

        self.device_manager.update_heartbeat(device_id)
        assert self.device_manager.get_status(device_id) == "Online"

    def test_error_handling_in_communication(self):
        """Test error handling during Hub-Spoke communication."""
        mock_device = MockAndroidDevice("error-test-device")
        port = mock_device.start_server()
        self.mock_devices.append(mock_device)

        time.sleep(0.1)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect(("localhost", port))

            client.send(b"invalid json data\n")

            response = client.recv(1024).decode('utf-8')
            parsed = json.loads(response.strip())
            assert "error" in parsed

    def test_concurrent_device_communication(self):
        """Test concurrent communication with multiple devices."""
        num_devices = 5
        devices_and_ports = []

        for i in range(num_devices):
            device = MockAndroidDevice(f"concurrent-device-{i+1}")
            port = device.start_server()
            devices_and_ports.append((device, port))
            self.mock_devices.append(device)

        time.sleep(0.1)

        def communicate_with_device(device_port_tuple, device_index):
            device, port = device_port_tuple
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.connect(("localhost", port))

                query = build_query_capabilities(device_index)
                client.send(query.encode('utf-8'))
                response = client.recv(1024).decode('utf-8')
                parsed = json.loads(response.strip())

                assert parsed["id"] == device_index
                assert parsed["status"] == "success"
                return parsed

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_devices) as executor:
            futures = []
            for i, device_port in enumerate(devices_and_ports):
                future = executor.submit(communicate_with_device, device_port, i + 1)
                futures.append(future)

            results = [future.result() for future in concurrent.futures.as_completed(futures)]

            assert len(results) == num_devices
            for result in results:
                assert result["status"] == "success"


class TestMultiComponentIntegration:
    """Test integration between multiple system components."""

    @pytest.mark.asyncio
    async def test_time_synchronization_integration(self):
        """Test time synchronization between Hub and Spokes."""
        time_server = TimeSyncServer(port=12345)

        await time_server.start()

        await asyncio.sleep(0.1)

        transport, protocol = await asyncio.get_event_loop().create_datagram_endpoint(
            asyncio.DatagramProtocol, remote_addr=('localhost', 12345)
        )

        try:
            transport.sendto(b'time_sync')

            await asyncio.sleep(0.1)

            assert time_server.is_running()

        finally:
            transport.close()
            await time_server.stop()

    def test_session_manager_device_coordination(self):
        """Test session management coordinating multiple devices."""
        session_manager = SessionManager()
        device_manager = DeviceManager()

        devices = ["device-1", "device-2", "device-3"]
        for device_id in devices:
            device_manager.register(device_id)
            device_manager.set_status(device_id, "Connected")

        session_manager.create_session("test_session")
        session_manager.start_recording()
        assert session_manager.is_active

        # Simulate setting recording status for devices
        for device_id in devices:
            device_manager.set_status(device_id, "Recording")

        recording_devices = [
            device_id for device_id in devices
            if device_manager.get_status(device_id) == "Recording"
        ]

        assert len(recording_devices) == len(devices)

        session_manager.stop_recording()
        assert not session_manager.is_active

    def test_heartbeat_manager_integration_with_device_manager(self):
        """Test heartbeat manager integration with device manager."""
        device_manager = DeviceManager(heartbeat_timeout_seconds=2)

        def heartbeat_callback(device_id: str):
            device_manager.update_heartbeat(device_id)

        heartbeat_manager = HeartbeatManager(callback=heartbeat_callback)

        devices = ["hb-device-1", "hb-device-2"]
        for device_id in devices:
            device_manager.register(device_id)
            heartbeat_manager.record_heartbeat(device_id)

        for device_id in devices:
            assert device_manager.get_status(device_id) == "Online"

        future_time = time.time_ns() + int(3 * 1e9)
        device_manager.check_timeouts(now_ns=future_time)

        for device_id in devices:
            assert device_manager.get_status(device_id) == "Offline"


@pytest.mark.integration
class TestSystemIntegration:
    """System-level integration tests."""

    def test_end_to_end_recording_workflow(self):
        """Test complete end-to-end recording workflow."""

        device_manager = DeviceManager()
        session_manager = SessionManager()

        test_devices = ["android-1", "android-2"]
        for device_id in test_devices:
            device_manager.register(device_id)
            assert device_manager.get_status(device_id) == "Online"


        session_manager.create_session("endtoend_test")
        session_manager.start_recording()
        assert session_manager.is_active

        for device_id in test_devices:
            device_manager.set_status(device_id, "Recording")

        recording_count = sum(1 for device_id in test_devices
                            if device_manager.get_status(device_id) == "Recording")
        assert recording_count == len(test_devices)

        session_manager.stop_recording()
        assert not session_manager.is_active

        for device_id in test_devices:
            device_manager.set_status(device_id, "Idle")

    def test_fault_tolerance_and_recovery(self):
        """Test system fault tolerance and recovery mechanisms."""
        device_manager = DeviceManager(heartbeat_timeout_seconds=1)
        session_manager = SessionManager()

        devices = ["fault-device-1", "fault-device-2", "fault-device-3"]
        for device_id in devices:
            device_manager.register(device_id)
            device_manager.set_status(device_id, "Recording")

        session_manager.create_session("fault_tolerance_test")
        session_manager.start_recording()

        failed_device = devices[1]
        future_time = time.time_ns() + int(2 * 1e9)

        for device_id in [devices[0], devices[2]]:
            device_info = device_manager.get_info(device_id)
            if device_info:
                device_info.last_heartbeat_ns = future_time - int(0.5 * 1e9)
                device_info.status = "Recording"

        device_manager.check_timeouts(now_ns=future_time)

        assert device_manager.get_status(failed_device) == "Offline"

        for device_id in [devices[0], devices[2]]:
            assert device_manager.get_status(device_id) == "Recording"

        device_manager.update_heartbeat(failed_device)
        device_manager.set_status(failed_device, "Recording")
        assert device_manager.get_status(failed_device) == "Recording"

        assert session_manager.is_active

    def test_performance_under_load(self):
        """Test system performance with many devices and operations."""
        device_manager = DeviceManager(heartbeat_timeout_seconds=10)

        num_devices = 50
        devices = [f"perf-device-{i:03d}" for i in range(num_devices)]

        start_time = time.time()

        for device_id in devices:
            device_manager.register(device_id)

        registration_time = time.time() - start_time

        start_time = time.time()
        for device_id in devices:
            device_manager.update_heartbeat(device_id)

        heartbeat_time = time.time() - start_time

        online_count = sum(1 for device_id in devices
                          if device_manager.get_status(device_id) == "Online")
        assert online_count == num_devices

        # Performance assertions (should handle 50 devices quickly)
        assert registration_time < 1.0
        assert heartbeat_time < 1.0

        # Test timeout checking performance
        start_time = time.time()
        device_manager.check_timeouts()
        timeout_check_time = time.time() - start_time

        assert timeout_check_time < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
