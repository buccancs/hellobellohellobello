"""Test suite organization and execution framework.

This module provides utilities for organizing and executing the comprehensive test suite:
- Test categorization and filtering
- Test execution reporting
- Coverage analysis and validation
- Test suite orchestration
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class ExecutionResult:
    """Represents the result of a test execution."""
    test_name: str
    category: str
    status: str
    duration_seconds: float
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuiteExecutionReport:
    """Comprehensive test suite execution report."""
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_seconds: float
    categories: dict[str, dict[str, int]]
    test_results: list[ExecutionResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "test_results": [result.to_dict() for result in self.test_results]
        }


class SuiteOrganizer:
    """Organizes and categorizes tests for execution."""

    TEST_CATEGORIES: ClassVar[dict[str, dict[str, Any]]] = {
        "unit": {
            "description": "Unit tests for individual components",
            "markers": [],
            "patterns": ["test_*.py", "*_test.py"],
            "exclude_patterns": ["integration", "system", "performance"]
        },
        "integration": {
            "description": "Integration tests for component interactions",
            "markers": ["integration"],
            "patterns": ["*integration*.py"],
            "exclude_patterns": ["system", "performance"]
        },
        "system": {
            "description": "End-to-end system tests",
            "markers": ["integration"],
            "patterns": ["*system*.py"],
            "exclude_patterns": ["performance"]
        },
        "performance": {
            "description": "Performance and stress tests",
            "markers": ["slow", "integration"],
            "patterns": ["*performance*.py"],
            "exclude_patterns": []
        },
        "android": {
            "description": "Android-specific tests",
            "markers": [],
            "patterns": ["*.kt", "*.java"],
            "exclude_patterns": []
        }
    }

    def __init__(self, test_root: Path):
        self.test_root = Path(test_root)
        self.pc_tests_dir = self.test_root / "pc_controller" / "tests"
        self.android_tests_dir = self.test_root / "android_sensor_node" / "app" / "src" / "test"

    def categorize_tests(self) -> dict[str, list[Path]]:
        """Categorize all tests by type."""
        categorized = {}

        for category, config in self.TEST_CATEGORIES.items():
            categorized[category] = []

            if category == "android":
                if self.android_tests_dir.exists():
                    for pattern in config["patterns"]:
                        categorized[category].extend(
                            self.android_tests_dir.rglob(pattern)
                        )
            else:
                if self.pc_tests_dir.exists():
                    for test_file in self.pc_tests_dir.glob("*.py"):
                        if self._matches_category(test_file, config):
                            categorized[category].append(test_file)

        return categorized

    def _matches_category(self, test_file: Path, config: dict[str, Any]) -> bool:
        """Check if a test file matches a category configuration."""
        file_name = test_file.name.lower()

        include_match = any(
            any(pattern_part in file_name for pattern_part in pattern.replace("*", "").split("."))
            for pattern in config["patterns"]
        )

        exclude_match = any(
            exclude_pattern.lower() in file_name
            for exclude_pattern in config["exclude_patterns"]
        )

        return include_match and not exclude_match

    def get_test_execution_plan(self, categories: list[str] | None = None) -> dict[str, list[Path]]:
        """Get execution plan for specified categories."""
        all_categorized = self.categorize_tests()

        if categories is None:
            return all_categorized

        return {
            category: tests
            for category, tests in all_categorized.items()
            if category in categories
        }


class SuiteExecutor:
    """Executes tests and generates reports."""

    def __init__(self, test_root: Path):
        self.test_root = Path(test_root)
        self.organizer = SuiteOrganizer(test_root)

    def execute_python_tests(
        self,
        test_files: list[Path],
        category: str,
        markers: list[str] | None = None
    ) -> list[ExecutionResult]:
        """Execute Python tests using pytest."""
        if not test_files:
            return []

        cmd = ["python", "-m", "pytest", "-v", "--tb=short"]

        if markers:
            marker_expr = " and ".join(markers)
            cmd.extend(["-m", marker_expr])

        cmd.extend([str(f) for f in test_files])

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=self.test_root,
                capture_output=True,
                text=True,
                timeout=300
            )

            duration = time.time() - start_time

            test_results = self._parse_pytest_output(result.stdout, category, duration)

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            test_results = [
                ExecutionResult(
                    test_name=f"{category}_tests",
                    category=category,
                    status="ERROR",
                    duration_seconds=duration,
                    error_message="Test execution timed out"
                )
            ]
        except Exception as e:
            duration = time.time() - start_time
            test_results = [
                ExecutionResult(
                    test_name=f"{category}_tests",
                    category=category,
                    status="ERROR",
                    duration_seconds=duration,
                    error_message=str(e)
                )
            ]

        return test_results

    def execute_android_tests(self, test_files: list[Path]) -> list[ExecutionResult]:
        """Execute Android tests using Gradle."""
        if not test_files:
            return []

        android_root = self.test_root / "android_sensor_node"
        if not android_root.exists():
            return [
                ExecutionResult(
                    test_name="android_tests",
                    category="android",
                    status="SKIPPED",
                    duration_seconds=0.0,
                    error_message="Android project not found"
                )
            ]

        start_time = time.time()

        try:
            result = subprocess.run(
                ["./gradlew", "testDebugUnitTest"],
                cwd=android_root,
                capture_output=True,
                text=True,
                timeout=600
            )

            duration = time.time() - start_time

            test_results = self._parse_android_output(result.stdout, duration)

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            test_results = [
                ExecutionResult(
                    test_name="android_tests",
                    category="android",
                    status="ERROR",
                    duration_seconds=duration,
                    error_message="Android test execution timed out"
                )
            ]
        except Exception as e:
            duration = time.time() - start_time
            test_results = [
                ExecutionResult(
                    test_name="android_tests",
                    category="android",
                    status="ERROR",
                    duration_seconds=duration,
                    error_message=str(e)
                )
            ]

        return test_results

    def _parse_pytest_output(
        self, output: str, category: str, total_duration: float
    ) -> list[ExecutionResult]:
        """Parse pytest output to extract test results."""
        results = []
        lines = output.split('\n')

        for line in lines:
            line = line.strip()

            if '::test_' in line and ('PASSED' in line or 'FAILED' in line or 'SKIPPED' in line):
                parts = line.split(' ')
                test_name = parts[0] if parts else f"unknown_test_{len(results)}"

                if 'PASSED' in line:
                    status = 'PASSED'
                elif 'FAILED' in line:
                    status = 'FAILED'
                elif 'SKIPPED' in line:
                    status = 'SKIPPED'
                else:
                    status = 'ERROR'

                duration = 0.0
                for part in parts:
                    if part.endswith('s') and part[:-1].replace('.', '').isdigit():
                        duration = float(part[:-1])
                        break

                results.append(ExecutionResult(
                    test_name=test_name,
                    category=category,
                    status=status,
                    duration_seconds=duration
                ))

        if not results:
            if 'failed' in output.lower():
                status = 'FAILED'
            elif 'error' in output.lower():
                status = 'ERROR'
            else:
                status = 'PASSED'

            results.append(ExecutionResult(
                test_name=f"{category}_suite",
                category=category,
                status=status,
                duration_seconds=total_duration
            ))

        return results

    def _parse_android_output(self, output: str, total_duration: float) -> list[ExecutionResult]:
        """Parse Android test output to extract results."""
        if "BUILD SUCCESSFUL" in output:
            status = "PASSED"
            error_msg = None
        elif "BUILD FAILED" in output:
            status = "FAILED"
            error_msg = "Android build failed"
        else:
            status = "ERROR"
            error_msg = "Unknown Android test result"

        return [ExecutionResult(
            test_name="android_unit_tests",
            category="android",
            status=status,
            duration_seconds=total_duration,
            error_message=error_msg
        )]

    def execute_comprehensive_suite(
        self,
        categories: list[str] | None = None,
        output_file: Path | None = None
    ) -> SuiteExecutionReport:
        """Execute comprehensive test suite and generate report."""
        start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        execution_plan = self.organizer.get_test_execution_plan(categories)

        all_results = []
        category_stats = {}

        for category, test_files in execution_plan.items():
            print(f"Executing {category} tests...")

            if category == "android":
                results = self.execute_android_tests(test_files)
            else:
                config = self.organizer.TEST_CATEGORIES[category]
                results = self.execute_python_tests(test_files, category, config.get("markers"))

            all_results.extend(results)

            category_stats[category] = {
                "total": len(results),
                "passed": len([r for r in results if r.status == "PASSED"]),
                "failed": len([r for r in results if r.status == "FAILED"]),
                "skipped": len([r for r in results if r.status == "SKIPPED"]),
                "errors": len([r for r in results if r.status == "ERROR"]),
            }

        total_duration = time.time() - start_time

        report = SuiteExecutionReport(
            timestamp=timestamp,
            total_tests=len(all_results),
            passed=len([r for r in all_results if r.status == "PASSED"]),
            failed=len([r for r in all_results if r.status == "FAILED"]),
            skipped=len([r for r in all_results if r.status == "SKIPPED"]),
            errors=len([r for r in all_results if r.status == "ERROR"]),
            duration_seconds=total_duration,
            categories=category_stats,
            test_results=all_results
        )

        if output_file:
            self._save_report(report, output_file)

        return report

    def _save_report(self, report: SuiteExecutionReport, output_file: Path) -> None:
        """Save test report to file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)


class CoverageAnalyzer:
    """Analyzes test coverage and generates recommendations."""

    def __init__(self, test_root: Path):
        self.test_root = Path(test_root)
        self.pc_src_dir = self.test_root / "pc_controller" / "src"
        self.android_src_dir = self.test_root / "android_sensor_node" / "app" / "src" / "main"

    def analyze_python_coverage(self) -> dict[str, Any]:
        """Analyze Python test coverage."""
        try:
            subprocess.run(
                ["python", "-m", "pytest", "--cov=pc_controller/src", "--cov-report=json"],
                cwd=self.test_root,
                capture_output=True,
                text=True,
                timeout=300
            )

            coverage_file = self.test_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)

                return {
                    "total_coverage": coverage_data.get("totals", {}).get("percent_covered", 0),
                    "files": coverage_data.get("files", {}),
                    "missing_lines": self._extract_missing_lines(coverage_data)
                }

        except Exception as e:
            print(f"Coverage analysis failed: {e}")

        return {"error": "Coverage analysis not available"}

    def _extract_missing_lines(self, coverage_data: dict[str, Any]) -> dict[str, list[int]]:
        """Extract missing lines from coverage data."""
        missing = {}

        for filepath, file_data in coverage_data.get("files", {}).items():
            missing_lines = file_data.get("missing_lines", [])
            if missing_lines:
                missing[filepath] = missing_lines

        return missing

    def generate_coverage_report(self) -> dict[str, Any]:
        """Generate comprehensive coverage report."""
        python_coverage = self.analyze_python_coverage()

        return {
            "python": python_coverage,
            "recommendations": self._generate_recommendations(python_coverage)
        }

    def _generate_recommendations(self, python_coverage: dict[str, Any]) -> list[str]:
        """Generate testing recommendations based on coverage analysis."""
        recommendations = []

        total_coverage = python_coverage.get("total_coverage", 0)

        if total_coverage < 80:
            recommendations.append(
                f"Overall coverage ({total_coverage:.1f}%) is below recommended 80%. "
                "Focus on adding unit tests for uncovered code paths."
            )

        missing_lines = python_coverage.get("missing_lines", {})
        if missing_lines:
            recommendations.append(
                f"Found {len(missing_lines)} files with missing coverage. "
                "Review these files and add targeted unit tests."
            )

        if not recommendations:
            recommendations.append("Test coverage is good! Consider adding more integration tests.")

        return recommendations


def main():
    """Main entry point for comprehensive test execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Test Suite Executor")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["unit", "integration", "system", "performance", "android"],
        help="Test categories to execute"
    )
    parser.add_argument("--output", type=Path, help="Output file for test report")
    parser.add_argument("--coverage", action="store_true", help="Include coverage analysis")
    parser.add_argument(
        "--test-root", type=Path, default=Path.cwd(), help="Root directory of tests"
    )

    args = parser.parse_args()

    executor = SuiteExecutor(args.test_root)

    print("Starting comprehensive test suite execution...")
    report = executor.execute_comprehensive_suite(
        categories=args.categories,
        output_file=args.output
    )

    print("\n" + "="*50)
    print("TEST EXECUTION SUMMARY")
    print("="*50)
    print(f"Total Tests: {report.total_tests}")
    print(f"Passed: {report.passed}")
    print(f"Failed: {report.failed}")
    print(f"Skipped: {report.skipped}")
    print(f"Errors: {report.errors}")
    print(f"Duration: {report.duration_seconds:.1f}s")
    print(f"Success Rate: {(report.passed / report.total_tests * 100):.1f}%")

    print("\nCATEGORY BREAKDOWN:")
    for category, stats in report.categories.items():
        print(f"  {category.capitalize()}: {stats['passed']}/{stats['total']} passed")

    if args.coverage:
        print("\nCOVERAGE ANALYSIS:")
        analyzer = CoverageAnalyzer(args.test_root)
        coverage_report = analyzer.generate_coverage_report()

        python_coverage = coverage_report["python"].get("total_coverage", 0)
        print(f"Python Coverage: {python_coverage:.1f}%")

        print("\nRECOMMENDAT&IONS:")
        for rec in coverage_report["recommendations"]:
            print(f"  - {rec}")

    exit_code = 0 if report.failed == 0 and report.errors == 0 else 1
    print(f"\nTest execution completed with exit code {exit_code}")
    exit(exit_code)


if __name__ == "__main__":
    main()
