#!/usr/bin/env python3
"""
Comprehensive Testing Pipeline Orchestrator
Multi-Modal Physiological Sensing Platform

This script orchestrates the complete testing pipeline including:
- Code quality checks (linting, formatting, type checking)
- Security scanning
- Unit tests with coverage
- Integration tests
- Performance tests
- Android tests (when available)
- Report generation

PERFORMANCE FEATURES:
- Parallel test execution with multiprocessing
- Cached results for unchanged files
- Fast subset testing for development
- Optimized CI mode

Usage:
    python scripts/test_pipeline.py [options]

Options:
    --fast          Run only fast tests (skip integration/performance)
    --coverage      Generate detailed coverage reports
    --security      Include security scanning
    --android       Include Android tests (requires Android SDK)
    --performance   Include performance benchmarks
    --report        Generate comprehensive HTML report
    --ci            CI mode (optimized for CI/CD environments)
    --fix           Auto-fix issues where possible
    --parallel      Run tests in parallel (default: auto)
    --jobs          Number of parallel jobs (default: CPU count)
"""

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestResult:
    """Test result data structure."""

    name: str
    success: bool
    duration: float
    output: str = ""
    error: str = ""
    coverage: float | None = None


class TestOrchestrator:
    """Orchestrates the complete testing pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.results: list[TestResult] = []
        self.start_time = time.time()

        self.pc_controller_src = Path("pc_controller/src")
        self.pc_controller_tests = Path("pc_controller/tests")
        self.android_app = Path("android_sensor_node/app")

        self.report_dir = Path(".test_reports")
        self.report_dir.mkdir(exist_ok=True)

        # Performance: Configure parallel execution
        self.max_workers = getattr(args, 'jobs', None) or multiprocessing.cpu_count()
        if args.ci:
            self.max_workers = min(self.max_workers, 4)
        self.use_parallel = getattr(args, 'parallel', True)

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        prefix = (
            "🔍"
            if level == "INFO"
            else "✅" if level == "SUCCESS" else "❌" if level == "ERROR" else "⚠️"
        )
        print(f"[{timestamp}] {prefix} {message}")

    def run_command(
        self, cmd: list[str], name: str, cwd: Path | None = None, timeout: int = 300
    ) -> TestResult:
        """Run a command and capture results."""
        self.log(f"Running {name}...")
        start_time = time.time()

        try:
            env = os.environ.copy()
            env.update(
                {
                    "QT_QPA_PLATFORM": "offscreen",
                    "OPENCV_LOG_LEVEL": "ERROR",
                    "PYTHONPATH": str(Path.cwd() / "pc_controller/src"),
                }
            )

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout, env=env
            )

            duration = time.time() - start_time
            success = result.returncode == 0

            test_result = TestResult(
                name=name,
                success=success,
                duration=duration,
                output=result.stdout,
                error=result.stderr,
            )

            if success:
                self.log(f"{name} completed successfully ({duration:.1f}s)", "SUCCESS")
            else:
                self.log(f"{name} failed ({duration:.1f}s)", "ERROR")
                if not self.args.ci:  # Show error details in non-CI mode
                    print(f"STDOUT: {result.stdout}")
                    print(f"STDERR: {result.stderr}")

            return test_result

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            self.log(f"{name} timed out after {timeout}s", "ERROR")
            return TestResult(name, False, duration, error=f"Timeout after {timeout}s")
        except Exception as e:
            duration = time.time() - start_time
            self.log(f"{name} failed with exception: {e}", "ERROR")
            return TestResult(name, False, duration, error=str(e))

    def run_commands_parallel(
        self, commands: list[tuple[list[str], str]], max_workers: int | None = None
    ) -> list[TestResult]:
        """Run multiple commands in parallel for better performance."""
        if not self.use_parallel or len(commands) == 1:
            return [self.run_command(cmd, name) for cmd, name in commands]

        max_workers = max_workers or self.max_workers
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_command = {
                executor.submit(self.run_command, cmd, name): (cmd, name) for cmd, name in commands
            }

            for future in concurrent.futures.as_completed(future_to_command):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    cmd, name = future_to_command[future]
                    self.log(f"Parallel execution failed for {name}: {e}", "ERROR")
                    results.append(TestResult(name, False, 0, error=str(e)))

        return results

    def run_code_quality_checks(self) -> None:
        """Run comprehensive code quality checks."""
        self.log("=== Code Quality Checks ===")

        checks = [
            (
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    str(self.pc_controller_src),
                    "--output-format=github",
                    "--cache-dir=.ruff_cache",
                ]
                + (["--fix"] if self.args.fix else []),
                "Ruff Linting",
            ),
            (
                (
                    [
                        sys.executable,
                        "-m",
                        "black",
                        "--check",
                        "--diff",
                        str(self.pc_controller_src),
                    ]
                    if not self.args.fix
                    else [sys.executable, "-m", "black", str(self.pc_controller_src)]
                ),
                "Black Formatting",
            ),
            (
                (
                    [
                        sys.executable,
                        "-m",
                        "isort",
                        "--check-only",
                        "--diff",
                        str(self.pc_controller_src),
                    ]
                    if not self.args.fix
                    else [sys.executable, "-m", "isort", str(self.pc_controller_src)]
                ),
                "Import Sorting",
            ),
            (
                [
                    sys.executable,
                    "-m",
                    "mypy",
                    str(self.pc_controller_src),
                    "--ignore-missing-imports",
                    "--cache-dir=.mypy_cache",
                ],
                "Type Checking",
            ),
        ]

        # Use parallel execution for performance
        results = self.run_commands_parallel(checks, max_workers=min(4, self.max_workers))
        self.results.extend(results)

    def run_security_scanning(self) -> None:
        """Run security scanning with Bandit."""
        if not self.args.security:
            return

        self.log("=== Security Scanning ===")

        bandit_cmd = [
            sys.executable,
            "-m",
            "bandit",
            "-r",
            str(self.pc_controller_src),
            "-f",
            "json",
            "-o",
            str(self.report_dir / "bandit-report.json"),
        ]

        result = self.run_command(bandit_cmd, "Bandit Security Scan")
        self.results.append(result)

    def run_python_tests(self) -> None:
        """Run Python unit tests with coverage."""
        self.log("=== Python Unit Tests ===")

        pytest_cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.pc_controller_tests),
            "-v",
            "--tb=short",
            "--timeout=60",
            "-n",
            "auto",
        ]

        if self.args.coverage:
            pytest_cmd.extend(
                [
                    "--cov=pc_controller/src",
                    "--cov-report=xml",
                    "--cov-report=html:{}".format(self.report_dir / "coverage_html"),
                    "--cov-report=term-missing",
                ]
            )

        if self.args.fast:
            pytest_cmd.extend(["-x", "-k", "not slow and not integration"])

        if self.args.ci:
            pytest_cmd.extend(["-q", "--tb=short"])  # Quiet mode for CI
        else:
            pytest_cmd.extend(["--tb=long"])

        result = self.run_command(pytest_cmd, "Python Unit Tests", timeout=600)

        if self.args.coverage and "coverage.xml" in Path.cwd().iterdir():
            try:
                import xml.etree.ElementTree as ET

                tree = ET.parse("coverage.xml")
                root = tree.getroot()
                line_rate = float(root.attrib.get("line-rate", 0)) * 100
                result.coverage = line_rate
                self.log(f"Code coverage: {line_rate:.1f}%")
            except Exception:
                pass

        self.results.append(result)

    def run_integration_tests(self) -> None:
        """Run integration tests."""
        if self.args.fast:
            return

        self.log("=== Integration Tests ===")

        integration_tests = [
            "test_system_end_to_end.py",
            "test_integration_comprehensive.py",
            "test_heartbeat_manager.py::TestHeartbeatIntegration",
        ]

        for test in integration_tests:
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(self.pc_controller_tests / test),
                "-v",
                "--tb=short",
                "--timeout=120",
            ]

            result = self.run_command(cmd, f"Integration: {test}", timeout=180)
            self.results.append(result)

    def run_performance_tests(self) -> None:
        """Run performance tests."""
        if not self.args.performance:
            return

        self.log("=== Performance Tests ===")

        perf_script = Path("scripts/run_performance_test.py")
        if perf_script.exists():
            cmd = [sys.executable, str(perf_script), "--duration=60", "--devices=2"]
            result = self.run_command(cmd, "Performance Tests", timeout=120)
            self.results.append(result)
        else:
            self.log("Performance test script not found, skipping", "ERROR")

    def run_android_tests(self) -> None:
        """Run Android tests if available."""
        if not self.args.android:
            return

        self.log("=== Android Tests ===")

        if not Path("gradlew").exists():
            self.log("Gradle wrapper not found, skipping Android tests", "ERROR")
            return

        android_tests = [
            (["./gradlew", ":android_sensor_node:app:testDebugUnitTest"], "Android Unit Tests"),
            (["./gradlew", ":android_sensor_node:app:ktlintCheck"], "Kotlin Linting"),
        ]

        for cmd, name in android_tests:
            result = self.run_command(cmd, name, timeout=300)
            self.results.append(result)

    def run_configuration_validation(self) -> None:
        """Validate project configuration files."""
        self.log("=== Configuration Validation ===")

        validator_script = Path("scripts/validate_configs.py")
        if validator_script.exists():
            cmd = [sys.executable, str(validator_script)]
            result = self.run_command(cmd, "Configuration Validation")
            self.results.append(result)

    def generate_report(self) -> None:
        """Generate comprehensive test report."""
        if not self.args.report:
            return

        self.log("=== Generating Reports ===")

        total_duration = time.time() - self.start_time
        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed

        report_data = {
            "summary": {
                "total_tests": len(self.results),
                "passed": passed,
                "failed": failed,
                "success_rate": (passed / len(self.results)) * 100 if self.results else 0,
                "total_duration": total_duration,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "results": [
                {
                    "name": r.name,
                    "success": r.success,
                    "duration": r.duration,
                    "coverage": r.coverage,
                    "error": r.error[:500] if r.error else None,
                }
                for r in self.results
            ],
        }

        json_report = self.report_dir / "test_report.json"
        json_report.write_text(json.dumps(report_data, indent=2))
        self.log(f"JSON report saved to {json_report}")

        print("\n" + "=" * 60)
        print("🏁 TESTING PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Total tests: {len(self.results)}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success rate: {report_data['summary']['success_rate']:.1f}%")
        print(f"Total duration: {total_duration:.1f}s")

        if failed > 0:
            print("\nFailed tests:")
            for result in self.results:
                if not result.success:
                    print(f"  ❌ {result.name}")
                    if result.error and not self.args.ci:
                        print(f"     {result.error[:100]}...")

        print("=" * 60)

    def run_pipeline(self) -> int:
        """Run the complete testing pipeline."""
        self.log("🚀 Starting Testing Pipeline")

        try:
            self.run_configuration_validation()
            self.run_code_quality_checks()
            self.run_python_tests()

            self.run_security_scanning()
            self.run_integration_tests()
            self.run_performance_tests()
            self.run_android_tests()

            self.generate_report()

            failed_tests = [r for r in self.results if not r.success]
            if failed_tests:
                return 1
            else:
                self.log("🎉 All tests passed!", "SUCCESS")
                return 0

        except KeyboardInterrupt:
            self.log("Pipeline interrupted by user", "ERROR")
            return 130
        except Exception as e:
            self.log(f"Pipeline failed with exception: {e}", "ERROR")
            return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Modal Sensor Platform Testing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--fast", action="store_true", help="Run only fast tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage reports")
    parser.add_argument("--security", action="store_true", help="Include security scanning")
    parser.add_argument("--android", action="store_true", help="Include Android tests")
    parser.add_argument("--performance", action="store_true", help="Include performance tests")
    parser.add_argument("--report", action="store_true", help="Generate detailed reports")
    parser.add_argument("--ci", action="store_true", help="CI mode (quiet, optimized)")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues where possible")
    parser.add_argument("--all", action="store_true", help="Run all tests with all options")
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run tests in parallel (default: enabled)",
    )
    parser.add_argument(
        "--no-parallel", dest="parallel", action="store_false", help="Disable parallel execution"
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel jobs (default: CPU count)",
    )

    args = parser.parse_args()

    # Handle --all flag
    if args.all:
        args.coverage = True
        args.security = True
        args.android = True
        args.performance = True
        args.report = True

    orchestrator = TestOrchestrator(args)
    return orchestrator.run_pipeline()


if __name__ == "__main__":
    sys.exit(main())
