"""Unit tests for local interfaces (Phase 3).

These tests validate the Python fallback behavior so they do not require
compiled native extensions or actual hardware.
"""
from __future__ import annotations

import time

import pytest

np = pytest.importorskip("numpy")

from pc_controller.src.core.local_interfaces import ShimmerInterface, WebcamInterface


def test_shimmer_interface_produces_samples() -> None:
    shimmer = ShimmerInterface()
    shimmer.start()
    try:
        time.sleep(0.15)
        ts, vals = shimmer.get_latest_samples()
        # [DEBUG_LOG] sizes
        print(f"[DEBUG_LOG] shimmer samples: {ts.size}")
        assert isinstance(ts, np.ndarray) and isinstance(vals, np.ndarray)
        assert ts.dtype == np.float64
        assert vals.dtype == np.float64
        assert ts.size > 0
        assert np.all(np.diff(ts) >= 0)
        assert np.all(np.isfinite(vals))
    finally:
        shimmer.stop()


def test_webcam_interface_returns_frame() -> None:
    cam = WebcamInterface()
    cam.start()
    try:
        for _ in range(20):
            frame = cam.get_latest_frame()
            if frame is not None:
                break
            time.sleep(0.05)
        # [DEBUG_LOG] frame shape
        if frame is not None:
            print(f"[DEBUG_LOG] webcam frame: {frame.shape}")
        assert frame is not None, "Expected a frame from synthetic or OpenCV fallback"
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype == np.uint8
    finally:
        cam.stop()
