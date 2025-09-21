"""Local device interfaces for Phase 3 (PC Controller).

This module provides thin Python wrappers around the native C++ backends
if available, and otherwise provides robust Python fallbacks to allow the
GUI to function without compiled extensions or hardware present.

Interfaces:
- ShimmerInterface: exposes start(), stop(), and get_latest_samples() -> (ts, vals)
  where ts is a numpy array of monotonic timestamps (seconds) and vals are
  GSR values in microsiemens. Target sample rate: 128 Hz.
- WebcamInterface: exposes start(), stop(), and get_latest_frame() -> numpy ndarray
  in BGR uint8 shape (H, W, 3).

Both implementations run background threads and are safe to call from the GUI
thread. No blocking operations are performed on the GUI thread.
"""

from __future__ import annotations

import contextlib

import importlib
import threading
import time
from collections import deque

import numpy as np

_ns_cls = None
_nw_cls = None
try:
    nb_mod = importlib.import_module("pc_controller.native_backend.native_backend")
    _ns_cls = getattr(nb_mod, "NativeShimmer", None)
    _nw_cls = getattr(nb_mod, "NativeWebcam", None)
except Exception:
    try:
        nb_pkg = importlib.import_module("pc_controller.native_backend")
        _ns_cls = getattr(nb_pkg, "NativeShimmer", None)
        _nw_cls = getattr(nb_pkg, "NativeWebcam", None)
    except Exception:
        _ns_cls = None
        _nw_cls = None


class ShimmerInterface:
    """Local Shimmer GSR access via native backend or simulated fallback.

    If the native backend is present, it will be used to connect to the wired
    Shimmer dock. Otherwise, a simulated 128 Hz signal is generated.
    """

    def __init__(self, port: str | None = None) -> None:
        self._port = port or "COM3"
        self._use_native = _ns_cls is not None
        self._lock = threading.Lock()
        self._running = False
        self._buf_ts: deque[float] = deque(maxlen=4096)
        self._buf_vals: deque[float] = deque(maxlen=4096)
        self._thread: threading.Thread | None = None
        self._native: object | None = None
        
        # Performance tracking for native backend demonstration
        self._samples_processed = 0
        self._native_backend_active = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self._use_native:
            try:
                self._native = _ns_cls()  # type: ignore[operator]
                self._native.connect(self._port)
                self._native.start_streaming()
                self._native_backend_active = True
                self._thread = threading.Thread(target=self._native_loop, daemon=True)
                self._thread.start()
                print(f"ShimmerInterface: Started with native C++ backend for high-performance GSR capture")
                return
            except Exception as e:
                self._use_native = False
                print(f"ShimmerInterface: Native backend failed ({e}), falling back to simulation")
        self._thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._thread.start()
        print("ShimmerInterface: Started with Python simulation backend")
        with self._lock:
            now = time.monotonic()
            self._buf_ts.append(now)
            self._buf_vals.append(10.0)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None
        if self._native is not None:
            with contextlib.suppress(Exception):
                self._native.stop_streaming()
            self._native = None

    def get_latest_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """Return all currently buffered samples and clear internal buffers.

        The returned arrays are sorted by timestamp to guarantee monotonic
        order for plotting and tests.
        """
        with self._lock:
            if not self._buf_ts:
                return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
            ts = np.fromiter(self._buf_ts, dtype=np.float64, count=len(self._buf_ts))
            vals = np.fromiter(
                self._buf_vals, dtype=np.float64, count=len(self._buf_vals)
            )
            self._buf_ts.clear()
            self._buf_vals.clear()
        if ts.size > 1:
            order = np.argsort(ts)
            ts = ts[order]
            vals = vals[order]
        return ts, vals

    def get_performance_stats(self) -> dict[str, any]:
        """Get performance statistics for native backend demonstration."""
        return {
            "native_backend_active": self._native_backend_active,
            "samples_processed": self._samples_processed,
            "buffer_size": len(self._buf_ts),
            "backend_type": "C++ Native" if self._native_backend_active else "Python Simulation"
        }

    def _native_loop(self) -> None:
        poll_dt = 0.005
        while self._running:
            try:
                samples = self._native.get_latest_samples()  # type: ignore[attr-defined]
                if samples:
                    with self._lock:
                        for t, v in samples:
                            self._buf_ts.append(float(t))
                            self._buf_vals.append(float(v))
                            self._samples_processed += 1
            except Exception as e:
                print(f"ShimmerInterface: Native loop failed ({e}), falling back to simulation")
                self._use_native = False
                self._native_backend_active = False
                self._native = None
                self._sim_loop()
                return
            time.sleep(poll_dt)

    def _sim_loop(self) -> None:
        rate = 128.0
        dt = 1.0 / rate
        phase = 0.0
        two_pi = 2.0 * np.pi
        t_next = time.monotonic()
        while self._running:
            now = time.monotonic()
            if now < t_next:
                time.sleep(max(0.0, t_next - now))
                continue
            val = 10.0 + 2.0 * np.sin(phase) + np.random.normal(0.0, 0.05)
            with self._lock:
                self._buf_ts.append(now)
                self._buf_vals.append(val)
                self._samples_processed += 1
            phase = (phase + two_pi * dt * 1.2) % two_pi
            t_next += dt


class WebcamInterface:
    """Local webcam access via native backend, OpenCV, or synthetic frames."""

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480) -> None:
        self._device_id = device_id
        self._width = width
        self._height = height
        self._use_native = _nw_cls is not None
        self._lock = threading.Lock()
        self._running = False
        self._frame: np.ndarray | None = None
        self._thread: threading.Thread | None = None
        self._native: object | None = None
        self._cap = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self._use_native:
            try:
                self._native = _nw_cls(self._device_id)  # type: ignore[operator]
                self._native.start_capture()
                self._thread = threading.Thread(target=self._native_loop, daemon=True)
                self._thread.start()
                with self._lock:
                    if self._frame is None:
                        self._frame = np.zeros(
                            (self._height, self._width, 3), dtype=np.uint8
                        )
                return
            except Exception:
                self._use_native = False
        try:
            import cv2  # type: ignore

            self._cap = cv2.VideoCapture(self._device_id)
            if self._cap.isOpened():
                try:
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
                except Exception:
                    pass
                self._thread = threading.Thread(target=self._cv_loop, daemon=True)
                self._thread.start()
                with self._lock:
                    if self._frame is None:
                        self._frame = np.zeros(
                            (self._height, self._width, 3), dtype=np.uint8
                        )
                return
        except Exception:
            self._cap = None
        self._thread = threading.Thread(target=self._synthetic_loop, daemon=True)
        self._thread.start()
        with self._lock:
            if self._frame is None:
                self._frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None
        if self._native is not None:
            with contextlib.suppress(Exception):
                self._native.stop_capture()
            self._native = None
        if self._cap is not None:
            with contextlib.suppress(Exception):
                self._cap.release()
            self._cap = None

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def _native_loop(self) -> None:
        poll_dt = 0.01
        while self._running:
            try:
                frame = self._native.get_latest_frame()  # type: ignore[attr-defined]
                if frame is not None:
                    with self._lock:
                        self._frame = frame
            except Exception:
                self._use_native = False
                self._synthetic_loop()
                return
            time.sleep(poll_dt)

    def _cv_loop(self) -> None:

        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def _synthetic_loop(self) -> None:
        try:
            import cv2  # type: ignore
        except Exception:
            cv2 = None
        t0 = time.monotonic()
        x = np.linspace(0, 255, self._width, dtype=np.uint8)
        base = np.tile(x, (self._height, 1))
        while self._running:
            dt = time.monotonic() - t0
            shift = int((dt * 60) % self._width)
            img = np.roll(base, shift=shift, axis=1)
            frame = np.stack([img, np.flipud(img), img], axis=2)
            if cv2 is not None:
                try:
                    ts = time.strftime("%H:%M:%S") + f".{int((dt % 1) * 1000):03d}"
                    cv2.putText(
                        frame,
                        ts,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                except Exception:
                    pass
            with self._lock:
                self._frame = frame.astype(np.uint8)
            time.sleep(1.0 / 30.0)
