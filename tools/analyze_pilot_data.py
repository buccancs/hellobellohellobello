#!/usr/bin/env python3
"""
Pilot Study Data Analysis Script (Objective 3)

Generates a combined plot of GSR and thermal-derived feature on a common time axis.
- Loads synchronized data streams from a session directory using DataLoader.
- Thermal feature: mean pixel value per frame from thermal.csv (default), or ROI
  mean if --roi x y w h is provided.
- GSR processing: low-pass filter via SciPy if available; otherwise moving-average
  smoothing based on cutoff or window seconds.
- Outputs a single PNG plotting both signals with twin y-axes.

Usage:
  python3 scripts/analyze_pilot_data.py \
    --session /path/to/session_dir \
    --out /path/to/output.png \
    [--roi 120 80 30 30] \
    [--gsr-cutoff-hz 0.5] [--gsr-window-sec 2.0] [--show]

Notes:
- The script does not modify session files.
- Exits non-zero on missing required inputs.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PC_SRC = REPO_ROOT / "pc_controller" / "src"
if str(PC_SRC) not in sys.path:
    sys.path.insert(0, str(PC_SRC))

from data.data_loader import DataLoader  # type: ignore  # noqa: E402

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


@dataclass
class Series:
    t_s: list[float]
    y: list[float]


def _find_first_matching(d: dict, name: str) -> str | None:
    if name in d:
        return d[name]
    lower = name.lower()
    for rel, p in d.items():
        if rel.lower().endswith("/" + lower) or rel.lower() == lower:
            return p
    return None


def load_gsr(session_dir: str, column: str = "gsr") -> Series:
    dl = DataLoader(session_dir)
    idx = dl.index_files()
    gsr_path = _find_first_matching(idx.csv_files, "gsr.csv")
    if not gsr_path:
        raise FileNotFoundError("gsr.csv not found in session")
    df = dl.load_csv(os.path.relpath(gsr_path, session_dir))
    if df.index.name is None:
        for cand in ("timestamp_ns", "ts_ns", "timestamp", "time_ns"):
            if cand in df.columns:
                df = df.set_index(cand)
                break
    if column not in df.columns:
        cols = [c for c in df.columns if c not in ("timestamp_ns", "ts_ns", "timestamp", "time_ns")]
        if not cols:
            raise ValueError("GSR CSV has no value columns")
        column = cols[0]
    ts_ns = df.index.astype("int64").to_numpy()
    y = df[column].astype("float64").to_numpy()
    t0 = int(ts_ns[0])
    t_s = (ts_ns - t0) / 1e9
    return Series(t_s=list(map(float, t_s)), y=list(map(float, y)))


def _roi_mean(
    flat_vals: Sequence[float], w: int, h: int, roi: tuple[int, int, int, int] | None
) -> float:
    if roi is None:
        return float(sum(flat_vals) / max(1, len(flat_vals)))
    x, y, rw, rh = roi
    if x < 0 or y < 0 or rw <= 0 or rh <= 0 or x + rw > w or y + rh > h:
        return float(sum(flat_vals) / max(1, len(flat_vals)))
    s = 0.0
    count = 0
    for yy in range(y, y + rh):
        base = yy * w
        start = base + x
        end = start + rw
        row_slice = flat_vals[start:end]
        s += sum(row_slice)
        count += rw
    return s / max(1, count)


def load_thermal_means(session_dir: str, roi: tuple[int, int, int, int] | None) -> Series:
    dl = DataLoader(session_dir)
    idx = dl.index_files()
    tpath = _find_first_matching(idx.csv_files, "thermal.csv")
    if not tpath:
        raise FileNotFoundError("thermal.csv not found in session")

    t_s: list[float] = []
    y: list[float] = []
    with open(tpath, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("thermal.csv is empty")
        try:
            ts_idx = header.index("timestamp_ns")
        except ValueError:
            ts_idx = 0
        try:
            w_idx = header.index("w")
            h_idx = header.index("h")
        except ValueError:
            w_idx, h_idx = 1, 2

        for row in reader:
            if not row:
                continue
            try:
                ts_ns = int(row[ts_idx])
                w = int(row[w_idx])
                h = int(row[h_idx])
                start_vals = max(ts_idx, w_idx, h_idx) + 1
                vals = [float(v) for v in row[start_vals:]]
                if len(vals) < w * h:
                    continue
                mean_val = _roi_mean(vals, w, h, roi)
            except Exception:
                continue
            t_s.append(float(ts_ns))
            y.append(float(mean_val))

    if not t_s:
        raise ValueError("No thermal rows parsed from thermal.csv")

    t0 = t_s[0]
    t_s = [(t - t0) / 1e9 for t in t_s]
    return Series(t_s=t_s, y=y)


def _estimate_sample_rate(t_s: list[float]) -> float:
    if len(t_s) < 2:
        return 1.0
    ds = sorted([t_s[i + 1] - t_s[i] for i in range(len(t_s) - 1) if t_s[i + 1] > t_s[i] > 0])
    if not ds:
        return 1.0
    med = ds[len(ds) // 2]
    if med <= 0:
        return 1.0
    return 1.0 / med


def smooth_gsr(series: Series, cutoff_hz: float | None, window_sec: float | None) -> Series:
    if cutoff_hz is None and window_sec is None:
        return series
    try:
        import scipy.signal as sig  # type: ignore

        use_scipy = True
    except Exception:
        use_scipy = False
    t_s = series.t_s
    y = series.y
    if use_scipy and cutoff_hz and len(y) > 4:
        fs = _estimate_sample_rate(t_s)
        nyq = 0.5 * fs
        norm = max(1e-6, min(0.99, cutoff_hz / nyq))
        b, a = sig.butter(2, norm, btype="low")
        y_f = sig.filtfilt(b, a, y)
        return Series(t_s=t_s, y=list(map(float, y_f)))
    if window_sec is None and cutoff_hz:
        window_sec = max(0.5, 1.0 / max(1e-3, cutoff_hz))
    if window_sec is None:
        return series
    fs = _estimate_sample_rate(t_s)
    win = max(1, round(fs * float(window_sec)))
    if win <= 1 or win >= len(y):
        return series
    acc: list[float] = []
    s = 0.0
    for i, v in enumerate(y):
        s += v
        if i >= win:
            s -= y[i - win]
        acc.append(s / min(win, i + 1))
    return Series(t_s=t_s, y=acc)


def plot_combined(out_path: str, gsr: Series, thermal: Series, show: bool) -> None:
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_title("Pilot Study: GSR vs Thermal Mean")
    ax1.set_xlabel("Time (s)")

    ax1.plot(gsr.t_s, gsr.y, color="tab:blue", label="GSR (a.u.)", linewidth=1.0)
    ax1.set_ylabel("GSR (a.u.)", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(
        thermal.t_s, thermal.y, color="tab:red", label="Thermal mean", linewidth=1.0, alpha=0.8
    )
    ax2.set_ylabel("Thermal mean (raw units)", color="tab:red")
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _synthesize_series(n: int = 300, fs: float = 10.0) -> tuple[Series, Series]:
    dt = 1.0 / fs
    t = [i * dt for i in range(n)]
    gsr = [0.5 * math.sin(2 * math.pi * 0.05 * ti) + 0.01 * ti for ti in t]
    thermal = [30.0 - 0.3 * math.sin(2 * math.pi * 0.05 * ti + 0.5) for ti in t]
    return Series(t_s=t, y=gsr), Series(t_s=t, y=thermal)


def _dry_run(out_path: str | None) -> int:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        can_plot = True
    except Exception:
        can_plot = False

    gsr, thermal = _synthesize_series(n=200, fs=10.0)

    if can_plot:
        if not out_path:
            out_path = os.path.join(tempfile.gettempdir(), "pilot_dry_run.png")
        try:
            plot_combined(out_path, gsr, thermal, show=False)
            print(f"[DRY-RUN] analyze_pilot_data: generated {out_path}")
            return 0
        except Exception as e:
            print(f"[DRY-RUN] analyze_pilot_data: plot failed: {e}")
            return 1
    else:
        print("[DRY-RUN] analyze_pilot_data: matplotlib not available; synthesis OK")
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Analyze pilot session data and produce a combined GSR vs Thermal plot."
    )
    p.add_argument("--session", help="Path to session directory")
    p.add_argument("--out", help="Path to output PNG file")
    p.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Thermal ROI as x y w h",
        default=None,
    )
    p.add_argument("--gsr-column", default="gsr", help="Column in gsr.csv to plot/filter")
    p.add_argument(
        "--gsr-cutoff-hz",
        type=float,
        default=None,
        help="Low-pass cutoff in Hz for GSR; if SciPy unavailable, used to derive moving-average window",
    )
    p.add_argument(
        "--gsr-window-sec",
        type=float,
        default=None,
        help="Window seconds for moving-average fallback if SciPy unavailable",
    )
    p.add_argument("--show", action="store_true", help="Display the plot window after saving")
    p.add_argument(
        "--dry-run", action="store_true", help="Run a self-check without requiring session files"
    )
    args = p.parse_args(argv)

    if args.dry_run:
        return _dry_run(args.out)

    if not args.session or not args.out:
        print("Error: --session and --out are required unless --dry-run is used.")
        return 2

    session = os.path.abspath(args.session)
    if not os.path.isdir(session):
        print(f"Error: session not found: {session}")
        return 2

    try:
        gsr = load_gsr(session, column=args.gsr_column)
    except Exception as e:
        print(f"Error loading GSR: {e}")
        return 1
    try:
        thermal = load_thermal_means(session, roi=tuple(args.roi) if args.roi else None)
    except Exception as e:
        print(f"Error loading Thermal: {e}")
        return 1

    gsr_sm = smooth_gsr(gsr, cutoff_hz=args.gsr_cutoff_hz, window_sec=args.gsr_window_sec)

    try:
        plot_combined(args.out, gsr_sm, thermal, show=args.show)
    except Exception as e:
        print(f"Error generating plot: {e}")
        return 1
    print(f"Saved plot to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
