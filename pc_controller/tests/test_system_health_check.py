"""
Tests for System Health Check Utility
"""

import json
import tempfile
from pathlib import Path

import pytest

from pc_controller.src.utils.system_health_check import (
    HealthCheckResult,
    SystemHealthChecker,
    SystemHealthReport,
)


class TestSystemHealthChecker:
    """Test suite for SystemHealthChecker."""

    @pytest.fixture
    def health_checker(self):
        """Create a health checker instance."""
        return SystemHealthChecker()

    def test_health_check_result_creation(self):
        """Test HealthCheckResult dataclass creation."""
        result = HealthCheckResult(
            component="Test Component",
            status="pass",
            message="Test passed successfully"
        )

        assert result.component == "Test Component"
        assert result.status == "pass"
        assert result.message == "Test passed successfully"
        assert result.timestamp

    def test_system_health_report_creation(self):
        """Test SystemHealthReport dataclass creation."""
        results = [
            HealthCheckResult("Component1", "pass", "OK"),
            HealthCheckResult("Component2", "warning", "Minor issue")
        ]

        report = SystemHealthReport(
            overall_status="warning",
            check_timestamp="",
            system_info={"platform": "test"},
            check_results=results,
            recommendations=["Fix minor issue"]
        )

        assert report.overall_status == "warning"
        assert report.check_timestamp
        assert len(report.check_results) == 2

    @pytest.mark.asyncio
    async def test_collect_system_info(self, health_checker):
        """Test system information collection."""
        system_info = health_checker._collect_system_info()

        required_keys = [
            "platform", "system", "python_version",
            "hostname", "cpu_count", "memory_total_gb"
        ]

        for key in required_keys:
            assert key in system_info
            assert system_info[key] is not None

    @pytest.mark.asyncio
    async def test_hardware_connectivity_check(self, health_checker):
        """Test hardware connectivity check."""
        result = await health_checker._check_hardware_connectivity()

        assert isinstance(result, HealthCheckResult)
        assert result.component == "Hardware Connectivity"
        assert result.status in ["pass", "warning", "fail"]
        assert result.message
        assert isinstance(result.details, dict)

    @pytest.mark.asyncio
    async def test_network_connectivity_check(self, health_checker):
        """Test network connectivity check."""
        result = await health_checker._check_network_connectivity()

        assert isinstance(result, HealthCheckResult)
        assert result.component == "Network Connectivity"
        assert result.status in ["pass", "warning", "fail"]
        assert "active_interfaces" in result.details

    @pytest.mark.asyncio
    async def test_storage_health_check(self, health_checker):
        """Test storage health check."""
        result = await health_checker._check_storage_health()

        assert isinstance(result, HealthCheckResult)
        assert result.component == "Storage Health"
        assert result.status in ["pass", "warning", "fail"]
        assert "disk_free_gb" in result.details
        assert "disk_used_percent" in result.details

    @pytest.mark.asyncio
    async def test_memory_status_check(self, health_checker):
        """Test memory status check."""
        result = await health_checker._check_memory_status()

        assert isinstance(result, HealthCheckResult)
        assert result.component == "Memory Status"
        assert result.status in ["pass", "warning", "fail"]
        assert "memory_total_gb" in result.details
        assert "memory_used_percent" in result.details

    @pytest.mark.asyncio
    async def test_process_health_check(self, health_checker):
        """Test process health check."""
        result = await health_checker._check_process_health()

        assert isinstance(result, HealthCheckResult)
        assert result.component == "Process Health"
        assert result.status in ["pass", "warning", "fail"]
        assert "total_processes" in result.details

    @pytest.mark.asyncio
    async def test_performance_check(self, health_checker):
        """Test performance check."""
        result = await health_checker._check_performance()

        assert isinstance(result, HealthCheckResult)
        assert result.component == "Performance"
        assert result.status in ["pass", "warning", "fail"]
        assert "cpu_benchmark_ms" in result.details

    @pytest.mark.asyncio
    async def test_dependencies_check(self, health_checker):
        """Test dependencies check."""
        result = await health_checker._check_dependencies()

        assert isinstance(result, HealthCheckResult)
        assert result.component == "Dependencies"
        assert result.status in ["pass", "warning", "fail"]
        assert "missing_modules" in result.details

    @pytest.mark.asyncio
    async def test_temporal_accuracy_check(self, health_checker):
        """Test temporal accuracy check."""
        result = await health_checker._check_temporal_accuracy()

        assert isinstance(result, HealthCheckResult)
        assert result.component == "Temporal Synchronization"
        assert result.status in ["pass", "warning", "fail"]
        assert "timing_accuracy_ms" in result.details

    def test_determine_overall_status(self, health_checker):
        """Test overall status determination."""
        results_pass = [
            HealthCheckResult("Test1", "pass", "OK"),
            HealthCheckResult("Test2", "pass", "OK")
        ]
        assert health_checker._determine_overall_status(results_pass) == "healthy"

        # Some warnings
        results_warning = [
            HealthCheckResult("Test1", "pass", "OK"),
            HealthCheckResult("Test2", "warning", "Minor issue")
        ]
        assert health_checker._determine_overall_status(results_warning) == "healthy"

        # Many warnings
        results_many_warnings = [
            HealthCheckResult("Test1", "warning", "Issue 1"),
            HealthCheckResult("Test2", "warning", "Issue 2"),
            HealthCheckResult("Test3", "warning", "Issue 3")
        ]
        assert health_checker._determine_overall_status(results_many_warnings) == "warning"

        results_fail = [
            HealthCheckResult("Test1", "pass", "OK"),
            HealthCheckResult("Test2", "fail", "Critical issue")
        ]
        assert health_checker._determine_overall_status(results_fail) == "critical"

    def test_generate_recommendations(self, health_checker):
        """Test recommendation generation."""
        results = [
            HealthCheckResult("Test1", "fail", "Failed", {
                "issues": ["High memory usage", "Missing dependencies"]
            }),
            HealthCheckResult("Test2", "warning", "Warning", {
                "issues": ["Network connectivity issue"]
            })
        ]

        recommendations = health_checker._generate_recommendations(results)

        assert len(recommendations) > 0
        memory_rec = any("memory" in rec.lower() for rec in recommendations)
        network_rec = any("network" in rec.lower() for rec in recommendations)
        deps_rec = any("dependencies" in rec.lower() for rec in recommendations)

        assert memory_rec or network_rec or deps_rec

    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self, health_checker):
        """Test complete health check workflow."""
        report = await health_checker.run_comprehensive_health_check()

        assert isinstance(report, SystemHealthReport)
        assert report.overall_status in ["healthy", "warning", "critical"]
        assert len(report.check_results) >= 6
        assert isinstance(report.system_info, dict)
        assert isinstance(report.recommendations, list)

    def test_save_report(self, health_checker):
        """Test report saving functionality."""
        results = [HealthCheckResult("Test", "pass", "OK")]
        report = SystemHealthReport(
            overall_status="healthy",
            check_timestamp="2024-01-01T00:00:00",
            system_info={"test": True},
            check_results=results,
            recommendations=[]
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            health_checker.save_report(report, tmp_path)

            assert tmp_path.exists()

            with open(tmp_path) as f:
                saved_data = json.load(f)

            assert saved_data["overall_status"] == "healthy"
            assert len(saved_data["check_results"]) == 1
            assert saved_data["system_info"]["test"] is True

        finally:
            tmp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_emit_progress(self, health_checker):
        """Test progress emission (basic functionality)."""
        health_checker.emit_progress(50, "Test progress")


class TestIntegration:
    """Integration tests for the health check system."""

    @pytest.mark.asyncio
    async def test_health_check_with_mocked_failures(self):
        """Test health check with simulated system issues."""
        checker = SystemHealthChecker()

        async def mock_failing_check():
            return HealthCheckResult(
                component="Mock Component",
                status="fail",
                message="Simulated failure",
                details={"issues": ["Mock issue"]}
            )

        original_check = checker._check_hardware_connectivity
        checker._check_hardware_connectivity = mock_failing_check

        try:
            report = await checker.run_comprehensive_health_check()

            assert report.overall_status == "critical"
            assert any(r.status == "fail" for r in report.check_results)
            assert len(report.recommendations) > 0

        finally:
            checker._check_hardware_connectivity = original_check


if __name__ == "__main__":
    pytest.main([__file__])
