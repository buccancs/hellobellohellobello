"""HDF5 Exporter (Phase 5).

Compiles all CSVs in a session directory into a single HDF5 file
organized by device and modality.
"""

from __future__ import annotations

import contextlib
import glob
import json
import os

import h5py
import numpy as np
import pandas as pd


def export_session_to_hdf5(
    session_dir: str,
    output_path: str,
    metadata: dict | None = None,
    annotations: dict | None = None,
) -> str:
    """Export a session directory to an HDF5 file.

    The HDF5 layout is /<device>/<modality>/ with datasets derived from CSV columns.
    If a timestamp column is present (timestamp_ns, ts_ns, timestamp, time_ns),
    it will be used as the index and stored as a separate dataset named 'timestamp_ns'.

    Returns the output_path written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with h5py.File(output_path, "w") as hf:
        if metadata is not None:
            hf.attrs["session_metadata_json"] = json.dumps(metadata)
        if annotations is not None:
            hf.attrs["annotations_json"] = json.dumps(annotations)
        for csv_path in glob.glob(
            os.path.join(session_dir, "**", "*.csv"), recursive=True
        ):
            rel = os.path.relpath(csv_path, session_dir)
            parts = rel.replace("\\", "/").split("/")
            device = parts[0] if len(parts) >= 2 else "PC"
            modality_name = os.path.splitext(parts[-1])[0]
            group = hf.require_group(f"/{device}/{modality_name}")
            df = pd.read_csv(csv_path)
            ts_col = None
            for cand in ("timestamp_ns", "ts_ns", "timestamp", "time_ns"):
                if cand in df.columns:
                    ts_col = cand
                    break
            if ts_col is not None:
                ts_ser = pd.to_numeric(df[ts_col], errors="coerce")
                mask = ts_ser.notna()
                df = df.loc[mask].copy()
                ts_np = ts_ser.loc[mask].astype("int64").to_numpy()
                ts_ds = group.create_dataset(
                    "timestamp_ns", data=ts_np, compression="gzip", compression_opts=4
                )
                with contextlib.suppress(Exception):
                    ts_ds.attrs["units"] = "ns"
                sample_rate_hz = None
                try:
                    if ts_np.size >= 2:
                        ts_sorted = np.sort(ts_np.astype(np.int64), axis=0)
                        diffs = np.diff(ts_sorted)
                        diffs = diffs[diffs > 0]
                        if diffs.size:
                            lo, hi = np.percentile(diffs, [10, 90])
                            core = diffs[(diffs >= lo) & (diffs <= hi)]
                            md = float(np.median(core if core.size else diffs))
                        else:
                            md = 0.0
                        if md > 0:
                            sample_rate_hz = float(1e9 / md)
                            group.attrs["sample_rate_hz"] = sample_rate_hz
                except Exception:
                    sample_rate_hz = None
                data_cols = [c for c in df.columns if c != ts_col]
            else:
                sample_rate_hz = None
                data_cols = list(df.columns)
            for col in data_cols:
                ds = None
                data = None
                try:
                    num = pd.to_numeric(df[col], errors="coerce")
                    if num.notna().any():
                        data = num.astype("float64").to_numpy()
                        ds = group.create_dataset(
                            col, data=data, compression="gzip", compression_opts=4
                        )
                    else:
                        data = df[col].astype(str).to_numpy(dtype="S")
                        ds = group.create_dataset(
                            col, data=data, compression="gzip", compression_opts=4
                        )
                except Exception:
                    data = df[col].astype(str).to_numpy(dtype="S")
                    ds = group.create_dataset(
                        col, data=data, compression="gzip", compression_opts=4
                    )
                try:
                    lname = col.lower()
                    if lname == "gsr_microsiemens":
                        ds.attrs["units"] = "microsiemens"
                    elif lname == "ppg_raw":
                        ds.attrs["units"] = "raw_counts"
                    elif lname in ("w", "width") or lname in ("h", "height"):
                        ds.attrs["units"] = "pixels"
                    if (
                        sample_rate_hz is not None
                        and data is not None
                        and hasattr(data, "dtype")
                        and np.issubdtype(data.dtype, np.number)
                    ):
                        ds.attrs["sample_rate_hz"] = float(sample_rate_hz)
                except Exception:
                    pass
        try:
            meta_src = metadata if isinstance(metadata, dict) else None
            if meta_src is None:
                meta_path = os.path.join(session_dir, "session_metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, encoding="utf-8") as mf:
                        meta_src = json.load(mf)
            if isinstance(meta_src, dict):
                offsets = meta_src.get("clock_offsets_ns", {})
                if isinstance(offsets, dict) and len(offsets):
                    sync_grp = hf.require_group("/sync")
                    try:
                        import numpy as _np

                        str_dtype = h5py.string_dtype(encoding="utf-8")
                        dev_ids = list(offsets.keys())
                        vals = [_np.int64(int(offsets[k])) for k in dev_ids]
                        sync_grp.create_dataset(
                            "device_ids",
                            data=_np.array(dev_ids, dtype=str_dtype),
                            compression="gzip",
                            compression_opts=4,
                        )
                        sync_grp.create_dataset(
                            "clock_offsets_ns",
                            data=_np.array(vals, dtype=_np.int64),
                            compression="gzip",
                            compression_opts=4,
                        )
                    except Exception:
                        pass
                # Optional sync stats: store as JSON dataset if available
                stats_obj = meta_src.get("sync_stats", None)
                if stats_obj is None:
                    stats_obj = meta_src.get("clock_sync", None)
                if stats_obj is not None:
                    sync_grp = hf.require_group("/sync")
                    try:
                        s = json.dumps(stats_obj)
                        str_dtype = h5py.string_dtype(encoding="utf-8")
                        sync_grp.create_dataset("stats_json", data=s, dtype=str_dtype)
                    except Exception:
                        pass
        except Exception:
            pass
    return output_path
