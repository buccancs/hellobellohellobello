from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

from core.session_manager import SessionManager
from core.shimmer_manager import SimulatedShimmer


def test_fr4_session_management_end_to_end(tmp_path: Path) -> None:
    """FR4: System-wide verification of session lifecycle and metadata finalization.

    - Create a session and verify directory + metadata.json are created.
    - Start and stop recording; verify metadata updated with timestamps and final state.
    """
    base_dir = tmp_path / "pc_data"
    sm = SessionManager(base_dir=str(base_dir))

    sid = sm.create_session("e2e_fr4")
    assert sm.session_id == sid

    sdir = sm.session_dir
    assert sdir is not None and sdir.exists()

    meta_path = sdir / "metadata.json"
    assert meta_path.exists(), "metadata.json should be created immediately after session creation"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["session_id"] == sid
    assert meta["state"] == "Created"
    assert isinstance(meta["created_at_ns"], int)

    sm.start_recording()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["state"] == "Recording"
    assert isinstance(meta.get("start_time_ns"), int)

    sm.stop_recording()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["state"] == "Stopped"
    assert isinstance(meta.get("end_time_ns"), int)
    assert meta.get("end_time_ns") >= meta.get("start_time_ns", 0)


def test_fr1_simulation_mode_generates_gsr_csv(tmp_path: Path) -> None:
    """FR1: Run a short simulated session using SimulatedShimmer and confirm CSV with dummy GSR.

    This test uses SimulatedShimmer to generate callback samples quickly and writes them to CSV file
    within the active session directory. It does not require any hardware or Android device.
    """
    base_dir = tmp_path / "pc_data"
    sm = SessionManager(base_dir=str(base_dir))

    sm.create_session("e2e_fr1_simulated_gsr")
    sm.start_recording()

    sdir = sm.session_dir
    assert sdir is not None

    csv_path = sdir / "gsr.csv"

    lock = threading.Lock()
    count = {"n": 0}

    f = csv_path.open("w", encoding="utf-8", newline="")
    writer = csv.writer(f)
    writer.writerow(["timestamp_ns", "gsr_dummy"])

    def cb(ts_ns: int, value: float) -> None:
        with lock:
            writer.writerow([ts_ns, float(value)])
            count["n"] += 1

    shimmer = SimulatedShimmer(sample_rate_hz=256)
    assert shimmer.connect() is True
    shimmer.start_streaming(cb)

    t0 = time.time()
    while True:
        with lock:
            n = count["n"]
        if n >= 25:
            break
        if time.time() - t0 > 3.0:
            break
        time.sleep(0.01)

    shimmer.stop_streaming()

    f.flush()
    f.close()

    sm.stop_recording()

    assert csv_path.exists(), "CSV should be created in the session directory"
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2, f"Expected at least 2 lines (header + data), got {len(lines)}"
