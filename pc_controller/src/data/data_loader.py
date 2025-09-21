"""DataLoader for Playback & Annotation (Phase 5).

Loads sensor CSVs via pandas and locates videos for OpenCV playback.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class SessionData:
    session_dir: str
    csv_files: dict[str, str]
    video_files: dict[str, str]


class DataLoader:
    """Utility to index session directory and load data files."""

    SUPPORTED_VIDEO_EXT = (".avi", ".mp4", ".mkv")

    def __init__(self, session_dir: str) -> None:
        self.session_dir = session_dir

    def index_files(self) -> SessionData:
        csv_files: dict[str, str] = {}
        video_files: dict[str, str] = {}
        for path in glob.glob(
            os.path.join(self.session_dir, "**", "*.csv"), recursive=True
        ):
            name = os.path.relpath(path, self.session_dir)
            csv_files[name.replace("\\", "/")] = path
        for ext in self.SUPPORTED_VIDEO_EXT:
            for path in glob.glob(
                os.path.join(self.session_dir, "**", f"*{ext}"), recursive=True
            ):
                name = os.path.relpath(path, self.session_dir)
                video_files[name.replace("\\", "/")] = path
        return SessionData(self.session_dir, csv_files, video_files)

    def load_csv(self, rel_name: str) -> pd.DataFrame:
        """Load a CSV into a DataFrame with timestamp_ns as int index if present."""
        full = os.path.join(self.session_dir, rel_name)
        if not os.path.exists(full):
            full = rel_name
        df = pd.read_csv(full)
        ts_col = None
        for cand in ("timestamp_ns", "ts_ns", "timestamp", "time_ns"):
            if cand in df.columns:
                ts_col = cand
                break
        if ts_col is not None:
            df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce").astype("Int64")
            df = df.dropna(subset=[ts_col])
            df = df.set_index(ts_col)
        return df
