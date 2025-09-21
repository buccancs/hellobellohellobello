from __future__ import annotations

from tools.validate_sync_core import (
    StreamDetection,
    compute_validation_report,
    detect_flash_indices_from_brightness,
    estimate_T0_ns,
)


def test_detect_flash_indices_from_brightness_simple():
    x = [10.0] * 200
    x[30] = 200.0
    x[100] = 220.0
    x[170] = 210.0
    peaks = detect_flash_indices_from_brightness(x, n_events=3, min_separation=5)
    assert len(peaks) == 3
    assert peaks[0] == 30
    assert peaks[1] == 100
    assert peaks[2] == 170


def test_estimate_T0_and_compute_validation_report_pass():
    # Reference aligned events on master clock (ns)
    ref_events = [1_000_000_000, 2_000_000_000, 3_000_000_000]
    det_pc = StreamDetection(name="PC", frame_indices=[100, 200, 300], fps=100.0)
    det_dev = StreamDetection(name="devA", frame_indices=[100, 200, 300], fps=100.0)
    detections = {"PC": det_pc, "devA": det_dev}

    # Only one device with events needed as reference for compute_validation_report
    aligned_by_device = {"devA": list(ref_events)}

    T0_pc = estimate_T0_ns(ref_events, det_pc.rel_times_ns)
    T0_dev = estimate_T0_ns(ref_events, det_dev.rel_times_ns)
    assert T0_pc == 0
    assert T0_dev == 0

    result = compute_validation_report(aligned_by_device, detections, tolerance_ms=5.0)
    assert result.passed is True
    assert result.overall_max_ms <= 0.001
