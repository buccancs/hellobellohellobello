"""Validation core utilities for Flash Sync alignment.

This module provides helper functions used by scripts/validate_sync.py to
- detect flash frames (luminance spikes) in videos
- align per-device flash timestamps to the PC master clock
- estimate absolute video time origins (T0) on the master clock
- compute per-event timing differences and PASS/FAIL verdicts

All functions are pure and unit-testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class StreamDetection:
    name: str
    frame_indices: list[int]
    fps: float

    @property
    def rel_times_ns(self) -> np.ndarray:
        if self.fps <= 0:
            return np.array([], dtype=np.int64)
        return (
            np.array(self.frame_indices, dtype=np.float64) / float(self.fps) * 1e9
        ).astype(np.int64)


def detect_flash_indices_from_brightness(
    brightness: Sequence[float], n_events: int, min_separation: int = 3
) -> list[int]:
    """Detect indices of flash events from a per-frame brightness sequence.

    Parameters
    ----------
    brightness: Sequence[float]
        Mean luminance per frame (e.g., grayscale average).
    n_events: int
        Expected number of flash events (from logs). If 0, returns empty list.
    min_separation: int
        Minimum frame separation between peaks to avoid double-counting.

    Returns
    -------
    List[int]
        Sorted list of peak indices (length <= n_events). If fewer peaks are found
        robustly, returns what is available.
    """
    if n_events <= 0 or not brightness:
        return []
    x = np.asarray(brightness, dtype=np.float64)
    if x.size == 0:
        return []
    mu = float(np.mean(x))
    sigma = float(np.std(x)) or 1.0
    z = (x - mu) / sigma
    dz = np.diff(x, prepend=x[0])
    dz_sigma = float(np.std(dz)) or 1.0
    dz_z = dz / dz_sigma
    score = 0.6 * z + 0.4 * dz_z
    idxs = list(np.argsort(-score))
    selected: list[int] = []
    used = np.zeros_like(x, dtype=bool)
    for idx in idxs:
        if len(selected) >= n_events:
            break
        if used[idx]:
            continue
        lo = max(0, idx - min_separation)
        hi = min(len(x), idx + min_separation + 1)
        selected.append(int(idx))
        used[lo:hi] = True
    selected.sort()
    if selected:
        thresh = max(2.5, float(np.percentile(score, 90)))
        selected = [i for i in selected if score[i] >= thresh]
        selected = selected[:n_events]
    return selected


def estimate_T0_ns(
    aligned_event_ts_ns: Sequence[int], rel_times_ns: Sequence[int]
) -> int:
    """Estimate absolute video start time T0 on the master clock.

    Uses the relation aligned_ts[k] ~= T0 + rel_times_ns[k]. If multiple events
    are available, computes the least squares estimate.
    """
    a = np.asarray(aligned_event_ts_ns, dtype=np.int64)
    r = np.asarray(rel_times_ns, dtype=np.int64)
    n = min(a.size, r.size)
    if n == 0:
        return 0
    a = a[:n]
    r = r[:n]
    return int(np.round(float(np.mean(a.astype(np.float64) - r.astype(np.float64)))))


def choose_offset_direction(
    ref_ts_ns: Sequence[int], device_ts_ns: Sequence[int], offset_ns: int
) -> int:
    """Choose +offset or -offset for alignment by minimizing median absolute diff.

    Returns +1 if device_ts + offset matches ref better; -1 if device_ts - offset matches better.
    """
    ref = np.asarray(ref_ts_ns, dtype=np.int64)
    dev = np.asarray(device_ts_ns, dtype=np.int64)
    m = min(ref.size, dev.size)
    if m == 0:
        return 1
    ref = ref[:m]
    dev = dev[:m]
    plus = np.median(np.abs((dev + offset_ns) - ref))
    minus = np.median(np.abs((dev - offset_ns) - ref))
    return 1 if plus <= minus else -1


@dataclass
class ValidationResult:
    per_event_ranges_ms: list[float]
    overall_max_ms: float
    passed: bool
    details: dict[str, dict[str, int]]


def compute_validation_report(
    aligned_events_by_device: dict[str, list[int]],
    detections_by_stream: dict[str, StreamDetection],
    tolerance_ms: float = 5.0,
) -> ValidationResult:
    """Compute per-event time spreads across all provided streams.

    Parameters
    ----------
    aligned_events_by_device: Dict[str, List[int]]
        Absolute aligned timestamps (ns) per Android device (after choosing offset sign).
    detections_by_stream: Dict[str, StreamDetection]
        Detected flash frames and fps for each video stream (e.g., PC webcam, Android RGB videos).
        Stream names should match keys like "PC" and device ids for clarity.
    tolerance_ms: float
        PASS threshold.

    Returns
    -------
    ValidationResult
    """
    # Reference event schedule: pick first device
    if not aligned_events_by_device:
        return ValidationResult([], 0.0, False, details={})
    ref_name = next(iter(aligned_events_by_device.keys()))
    ref_events = np.asarray(aligned_events_by_device[ref_name], dtype=np.int64)
    n_events = ref_events.size

    # Estimate T0 for each stream vs reference timeline
    T0_by_stream: dict[str, int] = {}
    for sname, det in detections_by_stream.items():
        T0_by_stream[sname] = estimate_T0_ns(ref_events, det.rel_times_ns)

    per_event_ranges_ms: list[float] = []
    for k in range(n_events):
        times_ns: list[int] = []
        times_ns.extend(
            int(arr[k]) for arr in aligned_events_by_device.values() if k < len(arr)
        )
        times_ns.extend(
            int(T0_by_stream[sname] + det.rel_times_ns[k])
            for sname, det in detections_by_stream.items()
            if k < len(det.rel_times_ns)
        )
        if times_ns:
            tmin = min(times_ns)
            tmax = max(times_ns)
            per_event_ranges_ms.append((tmax - tmin) / 1e6)
        else:
            per_event_ranges_ms.append(float("inf"))

    overall = max(per_event_ranges_ms) if per_event_ranges_ms else float("inf")
    return ValidationResult(
        per_event_ranges_ms=per_event_ranges_ms,
        overall_max_ms=overall,
        passed=bool(per_event_ranges_ms and overall <= tolerance_ms),
        details={
            "T0_by_stream": {k: int(v) for k, v in T0_by_stream.items()},
        },
    )
