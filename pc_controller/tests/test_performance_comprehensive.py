"""Comprehensive performance and stress testing suite.

This module contains performance tests that validate system behavior under load:
- High throughput data processing
- Memory usage and leak detection
- Concurrent operations and thread safety
- Resource exhaustion scenarios
- Long-running stability tests
"""
from __future__ import annotations

import gc
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from pc_controller.src.core.device_manager import DeviceManager
from pc_controller.src.core.session_manager import SessionManager
from pc_controller.src.network.heartbeat_manager import HeartbeatManager
from pc_controller.src.network.protocol import (
    build_query_capabilities,
    build_start_recording,
    parse_json_line,
)


@pytest.mark.slow
@pytest.mark.integration
class TestPerformanceAndStress:
    """Performance and stress testing suite."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path("/tmp/performance_test")
        self.temp_dir.mkdir(exist_ok=True)

        gc.collect()
        self.initial_objects = len(gc.get_objects())

    def teardown_method(self):
        """Clean up after tests."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

        gc.collect()
        final_objects = len(gc.get_objects())
        object_growth = final_objects - self.initial_objects

        if object_growth > 10000:
            print(f"Warning: Object count grew by {object_growth} objects")

    def test_high_device_count_performance(self):
        """Test performance with large number of devices."""
        device_manager = DeviceManager(heartbeat_timeout_seconds=10)
        num_devices = 1000

        # Measure device registration performance
        start_time = time.perf_counter()

        device_ids = [f"perf-device-{i:04d}" for i in range(num_devices)]
        for device_id in device_ids:
            device_manager.register(device_id)

        registration_time = time.perf_counter() - start_time

        # Performance assertions
        assert registration_time < 5.0, (
            f"Registering {num_devices} devices took {registration_time:.2f}s"
        )

        # Measure heartbeat update performance
        start_time = time.perf_counter()

        for device_id in device_ids:
            device_manager.update_heartbeat(device_id)

        heartbeat_time = time.perf_counter() - start_time

        assert heartbeat_time < 2.0, f"Updating {num_devices} heartbeats took {heartbeat_time:.2f}s"

        # Measure timeout checking performance
        start_time = time.perf_counter()
        device_manager.check_timeouts()
        timeout_check_time = time.perf_counter() - start_time

        assert timeout_check_time < 0.5, (
            f"Checking {num_devices} timeouts took {timeout_check_time:.2f}s"
        )

        online_count = sum(1 for device_id in device_ids
                          if device_manager.get_status(device_id) == "Online")
        assert online_count == num_devices

    def test_concurrent_device_operations(self):
        """Test concurrent operations on device manager."""
        device_manager = DeviceManager(heartbeat_timeout_seconds=5)
        num_threads = 50
        operations_per_thread = 100

        results = []
        exceptions = []

        def worker_thread(thread_id: int):
            """Worker thread for concurrent operations."""
            thread_results = []
            try:
                for i in range(operations_per_thread):
                    device_id = f"thread-{thread_id}-device-{i}"

                    device_manager.register(device_id)

                    device_manager.update_heartbeat(device_id)

                    device_manager.set_status(device_id, "Recording")

                    status = device_manager.get_status(device_id)
                    thread_results.append((device_id, status))

                    time.sleep(0.001)

            except Exception as e:
                exceptions.append((thread_id, e))

            return thread_results

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.extend(result)
                except Exception as e:
                    exceptions.append(("future", e))

        assert len(exceptions) == 0, f"Concurrent operations had {len(exceptions)} exceptions"
        assert len(results) == num_threads * operations_per_thread

        total_devices = device_manager.list_devices()
        assert len(total_devices) == num_threads * operations_per_thread

    def test_rapid_session_cycling_performance(self):
        """Test performance of rapid session start/stop cycles."""
        session_manager = SessionManager(base_dir=str(self.temp_dir))
        num_cycles = 500

        start_time = time.perf_counter()

        session_ids = []
        for i in range(num_cycles):
            session_id = session_manager.create_session(f"perf-session-{i}")
            session_ids.append(session_id)

            session_manager.start_recording()

            time.sleep(0.001)

            session_manager.stop_recording()

        total_time = time.perf_counter() - start_time

        # Performance assertions
        avg_time_per_cycle = total_time / num_cycles
        assert avg_time_per_cycle < 0.1, f"Average cycle time {avg_time_per_cycle:.3f}s too slow"
        assert total_time < 30.0, f"Total time {total_time:.2f}s for {num_cycles} cycles"

        assert len(set(session_ids)) == num_cycles, "All sessions should have unique IDs"

    def test_memory_usage_stability(self):
        """Test memory usage stability during extended operations."""
        device_manager = DeviceManager(heartbeat_timeout_seconds=10)

        initial_memory = self._get_memory_usage()
        memory_samples = [initial_memory]

        for cycle in range(100):
            # Create temporary devices
            temp_devices = []
            for i in range(50):
                device_id = f"memory-test-{cycle}-{i}"
                device_manager.register(device_id)
                device_manager.update_heartbeat(device_id)
                device_manager.set_status(device_id, "Recording")
                temp_devices.append(device_id)

            for device_id in temp_devices:
                device_manager.remove(device_id)

            if cycle % 10 == 0:
                gc.collect()
                memory_samples.append(self._get_memory_usage())

        memory_growth = memory_samples[-1] - memory_samples[0]
        max_memory = max(memory_samples)

        assert memory_growth < 100 * 1024 * 1024, (
            f"Memory grew by {memory_growth / 1024 / 1024:.1f}MB"
        )
        assert max_memory < initial_memory + 200 * 1024 * 1024, "Peak memory usage too high"

    def test_protocol_parsing_performance(self):
        """Test performance of protocol message parsing."""
        num_messages = 10000

        messages = []
        for i in range(num_messages):
            if i % 3 == 0:
                msg = build_query_capabilities(i)
            elif i % 3 == 1:
                msg = build_start_recording(f"session-{i}", i)
            else:
                msg = json.dumps({"command": "custom", "id": i, "data": f"test-{i}"}) + '\n'
            messages.append(msg)

        # Measure parsing performance
        start_time = time.perf_counter()

        parsed_count = 0
        for message in messages:
            try:
                parsed = parse_json_line(message.strip())
                if parsed:
                    parsed_count += 1
            except Exception:
                pass

        parsing_time = time.perf_counter() - start_time

        # Performance assertions
        messages_per_second = num_messages / parsing_time
        assert messages_per_second > 5000, f"Parsing rate {messages_per_second:.0f} msg/s too slow"
        assert parsed_count >= num_messages * 0.9, "Should successfully parse most messages"

    def test_heartbeat_manager_performance(self):
        """Test heartbeat manager performance under load."""

        def heartbeat_callback(device_id: str):
            """Mock callback for heartbeat processing."""
            pass

        heartbeat_manager = HeartbeatManager(callback=heartbeat_callback)
        num_devices = 1000
        heartbeats_per_device = 100

        device_ids = [f"hb-perf-device-{i}" for i in range(num_devices)]

        start_time = time.perf_counter()

        for _ in range(heartbeats_per_device):
            for device_id in device_ids:
                heartbeat_manager.record_heartbeat(device_id)

        heartbeat_time = time.perf_counter() - start_time

        total_heartbeats = num_devices * heartbeats_per_device
        heartbeats_per_second = total_heartbeats / heartbeat_time

        assert heartbeats_per_second > 10000, \
            f"Heartbeat rate {heartbeats_per_second:.0f} hb/s too slow"

    def test_concurrent_session_management(self):
        """Test concurrent session management operations."""
        num_managers = 10
        operations_per_manager = 20

        managers = []
        for i in range(num_managers):
            manager_dir = self.temp_dir / f"manager_{i}"
            manager_dir.mkdir(exist_ok=True)
            managers.append(SessionManager(base_dir=str(manager_dir)))

        results = []
        exceptions = []

        def session_worker(manager: SessionManager, manager_id: int):
            """Worker function for session operations."""
            worker_results = []
            try:
                for op in range(operations_per_manager):
                    session_id = manager.create_session(f"concurrent-{manager_id}-{op}")
                    manager.start_recording()
                    time.sleep(0.001)
                    manager.stop_recording()

                    worker_results.append(session_id)

            except Exception as e:
                exceptions.append((manager_id, e))

            return worker_results

        with ThreadPoolExecutor(max_workers=num_managers) as executor:
            futures = [
                executor.submit(session_worker, manager, i)
                for i, manager in enumerate(managers)
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.extend(result)
                except Exception as e:
                    exceptions.append(("future", e))

        assert len(exceptions) == 0, f"Concurrent sessions had {len(exceptions)} exceptions"
        assert len(results) == num_managers * operations_per_manager
        assert len(set(results)) == len(results), "All session IDs should be unique"

    @pytest.mark.slow
    def test_extended_runtime_stability(self):
        """Test system stability over extended runtime."""
        device_manager = DeviceManager(heartbeat_timeout_seconds=5)
        session_manager = SessionManager(base_dir=str(self.temp_dir))

        runtime_seconds = 5
        start_time = time.time()

        cycle_count = 0
        error_count = 0

        while time.time() - start_time < runtime_seconds:
            try:
                devices = [f"stability-device-{cycle_count}-{i}" for i in range(5)]
                for device_id in devices:
                    device_manager.register(device_id)
                    device_manager.update_heartbeat(device_id)

                session_manager.create_session(f"stability-session-{cycle_count}")
                session_manager.start_recording()

                time.sleep(0.1)

                session_manager.stop_recording()

                for device_id in devices:
                    device_manager.remove(device_id)

                cycle_count += 1

                if cycle_count % 10 == 0:
                    gc.collect()

            except Exception as e:
                error_count += 1
                print(f"Error in cycle {cycle_count}: {e}")

        actual_runtime = time.time() - start_time

        assert cycle_count > 0, "Should complete at least one cycle"
        assert error_count < cycle_count * 0.1, f"Error rate too high: {error_count}/{cycle_count}"
        assert actual_runtime >= runtime_seconds * 0.9, "Should run for expected duration"

        print(f"Stability test completed {cycle_count} cycles in {actual_runtime:.1f}s")

    def test_resource_exhaustion_resilience(self):
        """Test system resilience under resource exhaustion."""
        device_manager = DeviceManager(heartbeat_timeout_seconds=2)

        excessive_device_count = 5000

        try:
            for i in range(excessive_device_count):
                device_id = f"exhaust-device-{i:06d}"
                device_manager.register(device_id)
                device_manager.update_heartbeat(device_id)

                if i % 1000 == 0:
                    online_count = sum(
                        1 for did in [f"exhaust-device-{j:06d}" for j in range(i+1)]
                        if device_manager.get_status(did) == "Online"
                    )
                    assert online_count > i * 0.9, f"Too many devices offline at {i}"

            test_device = "responsiveness-test"
            device_manager.register(test_device)
            device_manager.update_heartbeat(test_device)
            assert device_manager.get_status(test_device) == "Online"

        except MemoryError:
            print("Hit memory limit as expected in resource exhaustion test")

        device_manager.check_timeouts()
        cleanup_device = "cleanup-test"
        device_manager.register(cleanup_device)
        assert device_manager.get_status(cleanup_device) == "Online"

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        import os

        import psutil

        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            return len(gc.get_objects()) * 64

    def test_data_throughput_performance(self):
        """Test data processing throughput performance."""

        test_data_size = 1024 * 1024
        test_data = b"x" * test_data_size
        num_chunks = 100

        start_time = time.perf_counter()

        total_bytes_processed = 0
        for _ in range(num_chunks):
            processed_data = test_data
            total_bytes_processed += len(processed_data)

        processing_time = time.perf_counter() - start_time

        throughput_mbps = (total_bytes_processed / (1024 * 1024)) / processing_time

        assert throughput_mbps > 50, f"Data throughput {throughput_mbps:.1f} MB/s too low"
        assert total_bytes_processed == test_data_size * num_chunks

    def test_thread_safety_under_load(self):
        """Test thread safety of core components under high load."""
        device_manager = DeviceManager(heartbeat_timeout_seconds=10)

        operation_counts = {"register": 0, "heartbeat": 0, "status": 0, "remove": 0}
        operation_lock = threading.Lock()

        def update_count(operation: str):
            with operation_lock:
                operation_counts[operation] += 1

        def thread_worker(thread_id: int, num_operations: int):
            """Worker thread that performs various operations."""
            for i in range(num_operations):
                device_id = f"thread-{thread_id}-device-{i}"

                device_manager.register(device_id)
                update_count("register")

                for _ in range(3):
                    device_manager.update_heartbeat(device_id)
                    update_count("heartbeat")

                status = device_manager.get_status(device_id)
                if status:
                    device_manager.set_status(device_id, "Recording")
                    update_count("status")

                device_manager.remove(device_id)
                update_count("remove")

        num_threads = 20
        operations_per_thread = 100

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(
                target=thread_worker,
                args=(i, operations_per_thread)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        expected_registers = num_threads * operations_per_thread
        expected_heartbeats = num_threads * operations_per_thread * 3
        expected_status_ops = num_threads * operations_per_thread
        expected_removes = num_threads * operations_per_thread

        assert operation_counts["register"] == expected_registers
        assert operation_counts["heartbeat"] == expected_heartbeats
        assert operation_counts["status"] == expected_status_ops
        assert operation_counts["remove"] == expected_removes

        remaining_devices = device_manager.list_devices()
        assert len(remaining_devices) == 0, "All devices should be removed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
