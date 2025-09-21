"""Pytest configuration for Python tests.

Ensures the repository root is present on sys.path so that imports like
`pc_controller.src...` resolve regardless of invocation location.

Also emits explicit per-test start and result markers to improve CI/console
visibility of which test is running and its outcome.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
# Disable OpenGL to prevent EGL issues in CI environments
os.environ.setdefault("QT_QUICK_BACKEND", "software")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_LOGGING_RULES", "*=false")

try:
    import cv2  # type: ignore
    try:
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    try:
        if hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
            cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
        if hasattr(cv2, "LOG_LEVEL_SILENT"):
            cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
    except Exception:
        pass
except Exception:
    pass


def _ts() -> str:
    """Return a human-readable local timestamp for log lines."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int, str] | None) -> None:
    """Log the start of each test node.

    Parameters
    ----------
    nodeid: str
        The pytest node id of the test (e.g., path::TestClass::test_name)
    location: Optional[Tuple[str, int, str]]
        File path, line, and test name as provided by pytest.
    """
    print(f"[TEST_START] {_ts()} {nodeid}")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Log the final result of each test's call phase.

    Parameters
    ----------
    report: pytest.TestReport
        The test report emitted by pytest for each phase (setup/call/teardown).
    """
    if report.when == "call":
        outcome = report.outcome.upper()
        duration = getattr(report, "duration", 0.0)
        print(f"[TEST_RESULT] {_ts()} {report.nodeid} -> {outcome} ({duration:.3f}s)")


def pytest_collection_modifyitems(config, items):
    """Add timeout markers and handle GUI tests appropriately."""
    for item in items:
        if not item.get_closest_marker("timeout"):
            item.add_marker(pytest.mark.timeout(30))

        if any(keyword in item.name.lower() for keyword in ["gui", "preview", "ui_smoke"]):
            is_ci = os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")

            try:
                import PyQt6.QtCore
                import PyQt6.QtWidgets

                app = PyQt6.QtWidgets.QApplication.instance()
                if app is None:
                    test_app = PyQt6.QtWidgets.QApplication([])
                    test_app.setAttribute(
                        test_app.ApplicationAttribute.AA_DontShowIconsInMenus, True
                    )

            except Exception as e:
                if is_ci:
                    item.add_marker(pytest.mark.skip(reason=f"GUI testing skipped in CI: {e}"))
                else:
                    item.add_marker(pytest.mark.skip(reason=f"GUI testing unavailable: {e}"))
