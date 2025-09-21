"""
System Health Check Utility for Multi-Modal Physiological Sensing Platform

Production-ready health monitoring and diagnostic tool for comprehensive
system validation before and during data collection sessions.

Features:
- Hardware connectivity verification
- Network interface validation
- Storage and memory health checks
- Temporal synchronization accuracy monitoring
- Sensor data quality validation
- Performance benchmarking
"""

import asyncio
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

try:
    from PyQt6.QtCore import QObject, pyqtSignal

    HAS_QT = True
except ImportError:
    HAS_QT = False

    class _DummyQObject:
        def __init__(self):
            pass

    QObject = _DummyQObject  # type: ignore

    def pyqtSignal(*args):
        return lambda: None


@dataclass
class HealthCheckResult:
    """Container for system health check results."""

    component: str
    status: str  # "pass", "warning", "fail"
    message: str
    details: dict[str, Any] | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


@dataclass
class SystemHealthReport:
    """Complete system health assessment report."""

    overall_status: str  # "healthy", "warning", "critical"
    check_timestamp: str
    system_info: dict[str, Any]
    check_results: list[HealthCheckResult]
    recommendations: list[str]

    def __post_init__(self):
        if not self.check_timestamp:
            self.check_timestamp = datetime.now(UTC).isoformat()


class SystemHealthChecker(QObject if HAS_QT else object):
    """Comprehensive system health monitoring for production deployment."""

    if HAS_QT:
        health_check_progress = pyqtSignal(int, str)
        health_check_completed = pyqtSignal(dict)
        health_check_failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.checks = []
        self.recommendations = []

    def emit_progress(self, percent: int, message: str):
        """Emit progress or print to console."""
        if HAS_QT and hasattr(self, "health_check_progress"):
            self.health_check_progress.emit(percent, message)
        else:
            print(f"[{percent}%] {message}")

    async def run_comprehensive_health_check(self) -> SystemHealthReport:
        """Execute complete system health assessment."""
        self.emit_progress(0, "Starting comprehensive health check...")

        system_info = self._collect_system_info()
        self.emit_progress(10, "System information collected")

        checks = [
            ("Hardware Detection", self._check_hardware_connectivity),
            ("Network Interfaces", self._check_network_connectivity),
            ("Storage Health", self._check_storage_health),
            ("Memory Status", self._check_memory_status),
            ("Process Health", self._check_process_health),
            ("Performance", self._check_performance),
            ("Dependencies", self._check_dependencies),
            ("Temporal Sync", self._check_temporal_accuracy),
        ]

        results = []
        for i, (name, check_func) in enumerate(checks):
            self.emit_progress(10 + int(80 * i / len(checks)), f"Checking {name}...")
            try:
                result = await check_func()
                results.append(result)
            except Exception as e:
                results.append(
                    HealthCheckResult(
                        component=name, status="fail", message=f"Check failed: {e!s}"
                    )
                )

        self.emit_progress(90, "Analyzing results...")

        overall_status = self._determine_overall_status(results)

        recommendations = self._generate_recommendations(results)

        report = SystemHealthReport(
            overall_status=overall_status,
            check_timestamp=datetime.now(UTC).isoformat(),
            system_info=system_info,
            check_results=results,
            recommendations=recommendations,
        )

        self.emit_progress(100, "Health check completed")

        if HAS_QT and hasattr(self, "health_check_completed"):
            self.health_check_completed.emit(asdict(report))

        return report

    def _collect_system_info(self) -> dict[str, Any]:
        """Collect basic system information."""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_total_gb": round(psutil.disk_usage("/").total / (1024**3), 2),
        }

    async def _check_hardware_connectivity(self) -> HealthCheckResult:
        """Check hardware device connectivity."""
        details = {}
        issues = []

        try:
            import serial.tools.list_ports

            ports = list(serial.tools.list_ports.comports())
            port_info = [
                {"port": p.device, "description": p.description} for p in ports
            ]
            details["serial_ports"] = port_info
            if len(ports) == 0:
                issues.append(
                    "No serial ports detected - Shimmer devices may not be connectable"
                )
        except ImportError:
            issues.append("pyserial not available for hardware detection")

        try:
            result = subprocess.run(
                ["lsusb"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                usb_count = len(result.stdout.strip().split("\n"))
                details["usb_devices_count"] = usb_count
                if usb_count < 3:
                    issues.append("Very few USB devices detected")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            details["usb_check"] = "lsusb not available (Windows/macOS)"

        try:
            import cv2

            camera_count = 0
            for i in range(5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_count += 1
                    cap.release()
            details["camera_count"] = camera_count
            if camera_count == 0:
                issues.append("No camera devices detected")
        except ImportError:
            issues.append("OpenCV not available for camera detection")

        status = "fail" if len(issues) > 2 else ("warning" if issues else "pass")
        if issues:
            message = f"Hardware check: {len(issues)} issues found"
        else:
            message = "All hardware checks passed"

        return HealthCheckResult(
            component="Hardware Connectivity",
            status=status,
            message=message,
            details={"issues": issues, **details},
        )

    async def _check_network_connectivity(self) -> HealthCheckResult:
        """Check network interface status."""
        details = {}
        issues = []

        interfaces = psutil.net_if_addrs()
        active_interfaces = [
            {"name": interface_name, "ip": addr.address, "netmask": addr.netmask}
            for interface_name, addresses in interfaces.items()
            if not interface_name.startswith("lo")
            for addr in addresses
            if addr.family == socket.AF_INET
        ]

        details["active_interfaces"] = active_interfaces

        if len(active_interfaces) == 0:
            issues.append("No active network interfaces detected")
            status = "fail"
        elif len(active_interfaces) == 1:
            issues.append("Only one network interface - consider redundancy")
            status = "warning"
        else:
            status = "pass"

        try:
            start_time = time.time()
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            latency = (time.time() - start_time) * 1000
            details["internet_latency_ms"] = round(latency, 2)
            if latency > 100:
                issues.append(f"High internet latency: {latency:.1f}ms")
        except OSError:
            issues.append("No internet connectivity detected")
            if status == "pass":
                status = "warning"

        message = f"Network: {len(active_interfaces)} interfaces, {len(issues)} issues"

        return HealthCheckResult(
            component="Network Connectivity",
            status=status,
            message=message,
            details={"issues": issues, **details},
        )

    async def _check_storage_health(self) -> HealthCheckResult:
        """Check storage space and performance."""
        details = {}
        issues = []

        disk_usage = psutil.disk_usage("/")
        free_gb = disk_usage.free / (1024**3)
        used_percent = (disk_usage.used / disk_usage.total) * 100

        details["disk_free_gb"] = round(free_gb, 2)
        details["disk_used_percent"] = round(used_percent, 1)

        if free_gb < 1:
            issues.append(f"Very low disk space: {free_gb:.1f}GB free")
        elif free_gb < 10:
            issues.append(f"Low disk space: {free_gb:.1f}GB free")

        if used_percent > 95:
            issues.append(f"Disk usage critical: {used_percent:.1f}% used")
        elif used_percent > 85:
            issues.append(f"Disk usage high: {used_percent:.1f}% used")

        # Test write performance
        test_file = Path("/tmp/health_check_write_test.dat")
        try:
            test_data = b"0" * (1024 * 1024)
            start_time = time.time()
            with open(test_file, "wb") as f:
                f.write(test_data)
                f.flush()
                os.fsync(f.fileno())
            write_time = time.time() - start_time
            write_speed_mbps = 1 / write_time

            details["write_speed_mbps"] = round(write_speed_mbps, 1)

            if write_speed_mbps < 10:
                issues.append(f"Slow storage write speed: {write_speed_mbps:.1f} MB/s")

            test_file.unlink(missing_ok=True)
        except Exception as e:
            issues.append(f"Storage write test failed: {e!s}")

        status = (
            "fail"
            if any("critical" in issue.lower() for issue in issues)
            else ("warning" if issues else "pass")
        )
        message = f"Storage: {free_gb:.1f}GB free, {len(issues)} issues"

        return HealthCheckResult(
            component="Storage Health",
            status=status,
            message=message,
            details={"issues": issues, **details},
        )

    async def _check_memory_status(self) -> HealthCheckResult:
        """Check system memory status."""
        details = {}
        issues = []

        memory = psutil.virtual_memory()
        details["memory_total_gb"] = round(memory.total / (1024**3), 2)
        details["memory_available_gb"] = round(memory.available / (1024**3), 2)
        details["memory_used_percent"] = round(memory.percent, 1)

        if memory.percent > 90:
            issues.append(f"Critical memory usage: {memory.percent:.1f}%")
        elif memory.percent > 75:
            issues.append(f"High memory usage: {memory.percent:.1f}%")

        if memory.available < 1024**3:
            issues.append(f"Low available memory: {memory.available / (1024**3):.1f}GB")

        status = (
            "fail"
            if any("critical" in issue.lower() for issue in issues)
            else ("warning" if issues else "pass")
        )
        message = f"Memory: {memory.percent:.1f}% used, {len(issues)} issues"

        return HealthCheckResult(
            component="Memory Status",
            status=status,
            message=message,
            details={"issues": issues, **details},
        )

    async def _check_process_health(self) -> HealthCheckResult:
        """Check for any critical process issues."""
        details = {}
        issues = []

        high_cpu_processes = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
            try:
                if proc.info["cpu_percent"] > 50:
                    high_cpu_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if high_cpu_processes:
            details["high_cpu_processes"] = high_cpu_processes
            issues.append(f"{len(high_cpu_processes)} processes using >50% CPU")

        process_count = len(psutil.pids())
        details["total_processes"] = process_count

        if process_count > 500:
            issues.append(f"High process count: {process_count}")

        status = "warning" if issues else "pass"
        message = f"Processes: {process_count} total, {len(issues)} issues"

        return HealthCheckResult(
            component="Process Health",
            status=status,
            message=message,
            details={"issues": issues, **details},
        )

    async def _check_performance(self) -> HealthCheckResult:
        """Check system performance benchmarks."""
        details = {}
        issues = []

        start_time = time.time()
        _ = sum(i * i for i in range(100000))
        cpu_time = time.time() - start_time
        details["cpu_benchmark_ms"] = round(cpu_time * 1000, 2)

        if cpu_time > 0.1:
            issues.append(f"Slow CPU performance: {cpu_time*1000:.1f}ms")

        try:
            load_avg = psutil.getloadavg()
            details["load_average"] = [round(x, 2) for x in load_avg]
            if load_avg[0] > psutil.cpu_count() * 2:
                issues.append(f"High system load: {load_avg[0]:.1f}")
        except AttributeError:
            details["load_average"] = "Not available on this platform"

        status = "warning" if issues else "pass"
        message = f"Performance: {len(issues)} issues detected"

        return HealthCheckResult(
            component="Performance",
            status=status,
            message=message,
            details={"issues": issues, **details},
        )

    async def _check_dependencies(self) -> HealthCheckResult:
        """Check critical software dependencies."""
        details = {}
        issues = []

        required_modules = [
            "PyQt6",
            "numpy",
            "opencv-python",
            "pandas",
            "h5py",
            "pyqtgraph",
            "zeroconf",
            "psutil",
        ]

        missing_modules = []
        for module in required_modules:
            try:
                __import__(module.lower().replace("-", "_"))
            except ImportError:
                missing_modules.append(module)

        if missing_modules:
            issues.append(f"Missing Python modules: {', '.join(missing_modules)}")

        details["missing_modules"] = missing_modules
        details["python_path"] = sys.executable if "sys" in globals() else "Unknown"

        try:
            from pc_controller.native_backend import __version__, shimmer_capi_enabled

            details["native_backend_version"] = __version__
            details["shimmer_capi_enabled"] = shimmer_capi_enabled
            if not shimmer_capi_enabled:
                issues.append("Shimmer C-API not enabled - only simulation available")
        except ImportError:
            issues.append("Native backend not available")

        status = "fail" if missing_modules else ("warning" if issues else "pass")
        message = (
            f"Dependencies: {len(missing_modules)} missing, {len(issues)} total issues"
        )

        return HealthCheckResult(
            component="Dependencies",
            status=status,
            message=message,
            details={"issues": issues, **details},
        )

    async def _check_temporal_accuracy(self) -> HealthCheckResult:
        """Check temporal synchronization accuracy."""
        details = {}
        issues = []

        timestamps = []
        for _ in range(100):
            timestamps.append(time.time_ns())
            await asyncio.sleep(0.001)

        intervals = [
            timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
        ]
        mean_interval = sum(intervals) / len(intervals) / 1_000_000

        details["mean_interval_ms"] = round(mean_interval, 3)
        details["target_interval_ms"] = 1.0

        accuracy = abs(mean_interval - 1.0)
        details["timing_accuracy_ms"] = round(accuracy, 3)

        if accuracy > 0.5:
            issues.append(f"Poor timing accuracy: {accuracy:.3f}ms deviation")
        elif accuracy > 0.1:
            issues.append(f"Timing accuracy warning: {accuracy:.3f}ms deviation")

        system_times = []
        monotonic_times = []
        for _ in range(10):
            system_times.append(time.time())
            monotonic_times.append(time.monotonic())
            await asyncio.sleep(0.01)

        status = (
            "fail"
            if any("Poor" in issue for issue in issues)
            else ("warning" if issues else "pass")
        )
        message = f"Temporal accuracy: {accuracy:.3f}ms deviation, {len(issues)} issues"

        return HealthCheckResult(
            component="Temporal Synchronization",
            status=status,
            message=message,
            details={"issues": issues, **details},
        )

    def _determine_overall_status(self, results: list[HealthCheckResult]) -> str:
        """Determine overall system health status."""
        fail_count = sum(1 for r in results if r.status == "fail")
        warning_count = sum(1 for r in results if r.status == "warning")

        if fail_count > 0:
            return "critical"
        elif warning_count > 2:
            return "warning"
        else:
            return "healthy"

    def _generate_recommendations(self, results: list[HealthCheckResult]) -> list[str]:
        """Generate actionable recommendations based on health check results."""
        recommendations = []

        for result in results:
            has_issues = (
                result.status in ["fail", "warning"]
                and result.details
                and "issues" in result.details
            )
            if has_issues:
                for issue in result.details["issues"]:
                    if "memory" in issue.lower():
                        msg = "Consider closing unnecessary applications to free memory"
                        recommendations.append(msg)
                    elif "disk" in issue.lower():
                        recommendations.append(
                            "Free up disk space or add additional storage"
                        )
                    elif "network" in issue.lower():
                        recommendations.append(
                            "Check network configuration and connectivity"
                        )
                    elif "missing" in issue.lower():
                        msg = "Install missing dependencies with: pip install -r requirements.txt"
                        recommendations.append(msg)
                    elif "hardware" in issue.lower():
                        recommendations.append(
                            "Check hardware connections and device drivers"
                        )
                    elif "timing" in issue.lower():
                        msg = "Consider system optimization for real-time performance"
                        recommendations.append(msg)

        return list(set(recommendations))

    def save_report(self, report: SystemHealthReport, output_path: Path) -> None:
        """Save health check report to file."""
        report_data = asdict(report)

        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif hasattr(obj, "tolist"):
                return obj.tolist()
            else:
                return obj

        report_data = convert_numpy(report_data)

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)


def main():
    """Command-line interface for system health check."""
    import argparse
    import sys

    description = "System Health Check for Multi-Modal Physiological Sensing Platform"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", "-o", help="Output file for health report (JSON)")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode - minimal output"
    )

    args = parser.parse_args()

    async def run_check():
        checker = SystemHealthChecker()
        report = await checker.run_comprehensive_health_check()

        if args.output:
            checker.save_report(report, Path(args.output))
            if not args.quiet:
                print(f"Health check report saved to: {args.output}")

        if not args.quiet:
            print(f"\nOverall System Status: {report.overall_status.upper()}")
            print(f"Checks completed: {len(report.check_results)}")
            print(
                f"Issues found: {sum(1 for r in report.check_results if r.status != 'pass')}"
            )

            if report.recommendations:
                print("\nRecommendations:")
                for i, rec in enumerate(report.recommendations, 1):
                    print(f"  {i}. {rec}")

        if report.overall_status == "critical":
            sys.exit(2)
        elif report.overall_status == "warning":
            sys.exit(1)
        else:
            sys.exit(0)

    asyncio.run(run_check())


if __name__ == "__main__":
    main()
