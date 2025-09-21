import base64
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import contextlib

import pytest

try:
    PyQt6 = pytest.importorskip("PyQt6")
    from PyQt6.QtCore import QObject, pyqtSignal
    from PyQt6.QtWidgets import QApplication

    test_app = QApplication.instance()
    if test_app is None:
        test_app = QApplication([])

except ImportError as e:
    if (
        "libEGL" in str(e) or
        "cannot open shared object" in str(e) or
        "qt.qpa.plugin" in str(e).lower()
    ):
        pytest.skip(f"GUI libraries not available: {e}", allow_module_level=True)
    else:
        raise

sys.path.append(os.path.join(os.getcwd(), "pc_controller", "src"))

_PNG_1x1_B64 = (
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO9s6d8AAAAASUVORK5CYII="
)
_PNG_1x1 = base64.b64decode(_PNG_1x1_B64)


class _StubNetwork(QObject):
    device_discovered = pyqtSignal(object)
    device_removed = pyqtSignal(str)
    log = pyqtSignal(str)
    preview_frame = pyqtSignal(str, bytes, int)

    def __init__(self) -> None:
        super().__init__()
        self.started = False

    def start(self) -> None:
        self.started = True
        with contextlib.suppress(Exception):
            self.log.emit("Stub network started")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_remote_preview_throttling(qapp, monkeypatch):
    from gui import gui_manager as gm  # type: ignore

    # Disable local devices; we only test remote throttling here
    monkeypatch.setattr(gm, "WebcamInterface", None, raising=False)
    monkeypatch.setattr(gm, "ShimmerInterface", None, raising=False)

    net = _StubNetwork()
    ui = gm.GUIManager(network=net)
    try:
        device = "dev1"
        start = time.monotonic()
        for _ in range(200):
            net.preview_frame.emit(device, _PNG_1x1, 0)
            qapp.processEvents()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0
        drops = ui._remote_drop_counts.get(device, 0)  # type: ignore[attr-defined]
        assert drops >= 1
    finally:
        ui.close()


def test_local_preview_throttling_logic(qapp, monkeypatch):
    import numpy as np
    from gui import gui_manager as gm  # type: ignore

    net = _StubNetwork()
    ui = gm.GUIManager(network=net)
    try:
        class _FakeCam:
            def get_latest_frame(self):
                return np.zeros((480, 640, 3), dtype=np.uint8)
        ui.webcam = _FakeCam()  # type: ignore

        for _ in range(50):
            ui._on_video_timer()  # type: ignore
        drops = ui._video_drop_count  # type: ignore[attr-defined]
        assert drops >= 1
    finally:
        ui.close()
