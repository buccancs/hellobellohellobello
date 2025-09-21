#!/usr/bin/env python3
"""Flash Sync Validation CLI

Command-line tool to validate end-to-end temporal synchronization for a
recorded session using the Hardware Validation Protocol.

Usage:
  python scripts\\validate_sync.py --session-id 20250101_120000 \
    --base-dir ./pc_controller_data --tolerance-ms 5.0

This script will:
- Locate the session directory under base-dir by session_id
- Load session_metadata.json to obtain per-device clock offsets
- For each Android device subdirectory, load flash_sync_events.csv and the
  RGB video (rgb/video.*)
- Detect flash frames in videos using luminance spikes
- Apply clock offsets to align device flash timestamps onto the PC master
  clock, choosing offset sign that best matches the reference device
- Estimate each video stream's absolute start time and compute the time
  spread across all streams for each flash event
- Print a Validation Report and exit with code 0 (PASS) or 1 (FAIL)

Requirements: OpenCV (opencv-python), numpy, pandas (indirect), available via
pc_controller/requirements.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.append(os.path.join(os.getcwd(), "pc_controller", "src"))
from tools.validate_sync_core import (  # type: ignore
    StreamDetection,
    choose_offset_direction,
    compute_validation_report,
    detect_flash_indices_from_brightness,
)



@dataclass
class DeviceSession:
    name: str
    path: str
    flash_csv: str | None
    video_path: str | None
    raw_events_ns: list[int]
    aligned_events_ns: list[int]
    offset_ns: int
    offset_sign: int


SUPPORTED_VID_EXT = (".mp4", ".avi", ".mkv")


def _normalize_name(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _find_file_case_insensitive(root: str, rel: str) -> str | None:
    full = os.path.join(root, rel)
    if os.path.exists(full):
        return full
    base = os.path.dirname(full)
    name = os.path.basename(full).lower()
    if os.path.isdir(base):
        for fn in os.listdir(base):
            if fn.lower() == name:
                return os.path.join(base, fn)
    return None


def _find_flash_csv(dev_dir: str) -> str | None:
    cand = _find_file_case_insensitive(dev_dir, "flash_sync_events.csv")
    if cand:
        return cand
    for root, _dirs, files in os.walk(dev_dir):
        for f in files:
            if f.lower() == "flash_sync_events.csv":
                return os.path.join(root, f)
    return None


def _find_android_video(dev_dir: str) -> str | None:
    rgb_dir = os.path.join(dev_dir, "rgb")
    if os.path.isdir(rgb_dir):
        for ext in SUPPORTED_VID_EXT:
            p = os.path.join(rgb_dir, "video" + ext)
            if os.path.exists(p):
                return p
    for root, _dirs, files in os.walk(dev_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_VID_EXT:
                return os.path.join(root, f)
    return None


def _load_flash_csv(path: str) -> list[int]:
    arr: list[int] = []
    try:
        with open(path, encoding="utf-8") as f:
            f.readline()
            for line in f:
                s = line.strip().split(",")
                if not s:
                    continue
                val = s[0]
                try:
                    arr.append(int(val))
                except Exception:
                    continue
    except Exception:
        return []
    return arr


def _map_offsets_to_devices(session_dir: str, offsets: dict[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    norm_offset: dict[str, tuple[str, int]] = {
        _normalize_name(k): (k, v) for k, v in offsets.items()
    }
    for name in sorted(os.listdir(session_dir)):
        dev_dir = os.path.join(session_dir, name)
        if not os.path.isdir(dev_dir):
            continue
        if name.lower().startswith("rgb") or name.lower().startswith("gsr"):
            continue
        norm = _normalize_name(name)
        match_key = None
        if norm in norm_offset:
            match_key = norm
        else:
            for k in norm_offset:
                if norm in k or k in norm:
                    match_key = k
                    break
        if match_key is not None:
            result[name] = norm_offset[match_key][1]
        else:
            result[name] = 0
    return result


def _read_video_brightness(path: str) -> tuple[list[float], float]:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "OpenCV (cv2) is required to analyze videos. "
            "Install opencv-python as per pc_controller/requirements.txt."
        ) from exc
    cap = cv2.VideoCapture(path)
    if cap is None or not cap.isOpened():
        return [], 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    means: list[float] = []
    ok, frame = cap.read()
    while ok:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            means.append(float(np.mean(gray)))
        except Exception:
            means.append(float(np.mean(frame)))
        ok, frame = cap.read()
    cap.release()
    return means, fps


def build_device_sessions(
    session_dir: str, offsets: dict[str, int]
) -> tuple[list[DeviceSession], dict[str, int]]:
    mapped_offsets = _map_offsets_to_devices(session_dir, offsets)
    sessions: list[DeviceSession] = []
    for name, offset in mapped_offsets.items():
        dev_dir = os.path.join(session_dir, name)
        if not os.path.isdir(dev_dir):
            continue
        flash_csv = _find_flash_csv(dev_dir)
        video_path = _find_android_video(dev_dir)
        raw = _load_flash_csv(flash_csv) if flash_csv else []
        sessions.append(
            DeviceSession(
                name=name,
                path=dev_dir,
                flash_csv=flash_csv,
                video_path=video_path,
                raw_events_ns=raw,
                aligned_events_ns=[],
                offset_ns=offset,
                offset_sign=1,
            )
        )
    return sessions, mapped_offsets


def main() -> int:
    parser = argparse.ArgumentParser(description="Flash Sync Validation")
    parser.add_argument(
        "--session-id", required=True, help="Session ID (folder name under base dir)"
    )
    default_base_dir = os.path.join(os.getcwd(), "pc_controller_data")
    parser.add_argument(
        "--base-dir", default=default_base_dir, help="Base directory where sessions are stored"
    )
    parser.add_argument(
        "--tolerance-ms", type=float, default=5.0, help="PASS/FAIL tolerance in milliseconds"
    )
    args = parser.parse_args()

    session_dir = os.path.join(args.base_dir, args.session_id)
    if not os.path.isdir(session_dir):
        print(f"ERROR: Session directory not found: {session_dir}", file=sys.stderr)
        return 2

    meta_path = os.path.join(session_dir, "session_metadata.json")
    offsets: dict[str, int] = {}
    clock_sync: dict[str, dict] = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            offsets = {str(k): int(v) for k, v in (meta.get("clock_offsets_ns") or {}).items()}
            try:
                clock_sync = {str(k): dict(v) for k, v in (meta.get("clock_sync") or {}).items()}
            except Exception:
                clock_sync = {}
        except Exception as exc:
            print(f"WARNING: Failed to load session metadata offsets: {exc}")
    else:
        print("WARNING: session_metadata.json not found; proceeding with zero offsets.")

    devices, mapped = build_device_sessions(session_dir, offsets)
    n_events = 0
    for d in devices:
        if d.raw_events_ns:
            n_events = max(n_events, len(d.raw_events_ns))
    if n_events == 0:
        msg = "ERROR: No flash_sync_events.csv timestamps found in device folders."
        print(msg, file=sys.stderr)
        return 2

    # Choose a reference device with events
    ref_idx = next((i for i, d in enumerate(devices) if d.raw_events_ns), None)
    if ref_idx is None:
        print("ERROR: No device has flash events.", file=sys.stderr)
        return 2
    ref_dev = devices[ref_idx]
    # Reference aligned events assume +sign for reference (offset applied +)
    ref_dev.offset_sign = 1
    ref_dev.aligned_events_ns = [int(v + ref_dev.offset_ns) for v in ref_dev.raw_events_ns]

    for i, dev in enumerate(devices):
        if i == ref_idx:
            continue
        if not dev.raw_events_ns:
            continue
        sign = choose_offset_direction(ref_dev.raw_events_ns, dev.raw_events_ns, dev.offset_ns)
        dev.offset_sign = sign
        dev.aligned_events_ns = [int(v + sign * dev.offset_ns) for v in dev.raw_events_ns]

    aligned_by_device: dict[str, list[int]] = {
        d.name: d.aligned_events_ns for d in devices if d.aligned_events_ns
    }

    detections: dict[str, StreamDetection] = {}

    # PC webcam video (optional)
    pc_webcam_path = None
    for ext in SUPPORTED_VID_EXT:
        p = os.path.join(session_dir, "webcam" + ext if ext != ".avi" else "webcam.avi")
        if os.path.exists(p):
            pc_webcam_path = p
            break
    if pc_webcam_path is None:
        p = os.path.join(session_dir, "webcam.avi")
        if os.path.exists(p):
            pc_webcam_path = p
    if pc_webcam_path and os.path.exists(pc_webcam_path):
        means, fps = _read_video_brightness(pc_webcam_path)
        idxs = detect_flash_indices_from_brightness(means, n_events)
        detections["PC"] = StreamDetection(name="PC", frame_indices=idxs, fps=fps)

    for dev in devices:
        if dev.video_path and os.path.exists(dev.video_path):
            means, fps = _read_video_brightness(dev.video_path)
            idxs = detect_flash_indices_from_brightness(means, n_events)
            detections[dev.name] = StreamDetection(name=dev.name, frame_indices=idxs, fps=fps)

    if not aligned_by_device:
        print("ERROR: No aligned device events; cannot compute validation.", file=sys.stderr)
        return 2

    tolerance = args.tolerance_ms
    result = compute_validation_report(aligned_by_device, detections, tolerance_ms=tolerance)

    print("=== Flash Sync Validation Report ===")
    print(f"Session: {args.session_id}")
    print(f"Base Dir: {args.base_dir}")
    print("\nDevices and Offsets (ns):")
    for d in devices:
        video_status = 'yes' if d.video_path else 'no'
        print(
            f"- {d.name}: offset={d.offset_ns} sign={d.offset_sign} "
            f"events={len(d.aligned_events_ns)} video={video_status}"
        )
    if 'clock_sync' in locals() and clock_sync:
        print("\nClock Sync Stats (from session_metadata.json):")
        for dev_name, st in clock_sync.items():
            try:
                off = int(st.get('offset_ns', 0))
                dly = int(st.get('delay_ns', 0))
                sd = int(st.get('std_dev', st.get('std_dev_ns', 0)))
                tri = int(st.get('trials', 0))
                ts = int(st.get('timestamp_ns', 0))
                delay_ms = dly / 1e6
                print(
                    f"- {dev_name}: offset={off} ns, min_delay={delay_ms:.3f} ms, "
                    f"std_dev={sd} ns, trials={tri}, ts={ts}"
                )
            except Exception:
                print(f"- {dev_name}: {st}")
    if detections:
        print("\nDetected Streams:")
        for s, det in detections.items():
            print(f"- {s}: fps={det.fps:.2f} peaks={len(det.frame_indices)}")
    print("\nPer-Event Spread (ms):")
    for i, v in enumerate(result.per_event_ranges_ms, 1):
        print(f"  Event {i}: {v:.3f} ms")
    print(f"\nOverall Max Spread: {result.overall_max_ms:.3f} ms")
    verdict = "PASS" if result.passed else "FAIL"
    print(f"Verdict: {verdict} (tolerance {args.tolerance_ms:.3f} ms)")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
