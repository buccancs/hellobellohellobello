"""Thread-safe CSV writer helper for simulated or real GSR streams.

Usage:
    from core.gsr_csv import GsrCsvWriter

    with GsrCsvWriter(path) as w:
        w.write(ts_ns, value)

This module keeps dependencies minimal and is suitable for tests and simple
recording pipelines (e.g., SimulatedShimmer callbacks).
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any, TextIO


class GsrCsvWriter:
    def __init__(self, file_path: str | Path, newline: str = "") -> None:
        self._path = Path(file_path)
        self._lock = threading.Lock()
        self._fh: TextIO | None = None
        self._writer: Any | None = None
        self._newline = newline

    def open(self) -> None:
        if self._fh is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", encoding="utf-8", newline=self._newline)
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["timestamp_ns", "gsr"])
        self._fh.flush()

    def write(self, ts_ns: int, value: float) -> None:
        if self._fh is None or self._writer is None:
            self.open()
        assert self._writer is not None
        with self._lock:
            self._writer.writerow([int(ts_ns), float(value)])
            if self._fh:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                if self._fh:
                    self._fh.flush()
            finally:
                if self._fh:
                    self._fh.close()
                self._fh = None
                self._writer = None

    def __enter__(self) -> GsrCsvWriter:
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
