"""Unit tests for DataLoader (Phase 5).

These tests validate indexing and CSV loading behavior while being
robust to environments lacking heavy dependencies by skipping if
pandas is unavailable.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from pc_controller.src.data.data_loader import DataLoader


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(r + "\n")


def test_index_and_load_csv_with_timestamp_index() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sess = root / "20250101_010101"
        csv_path = sess / "Pixel_7" / "gsr.csv"
        _write_csv(csv_path, "timestamp_ns,gsr_microsiemens", [
            "1000000000,1.23",
            "2000000000,1.25",
            "3000000000,1.28",
        ])

        loader = DataLoader(str(sess))
        sd = loader.index_files()
        assert any(k.endswith("Pixel_7/gsr.csv") for k in sd.csv_files.keys())

        df = loader.load_csv(next(iter(sd.csv_files.keys())))
        assert not df.empty
        assert df.index.name in ("timestamp_ns", "ts_ns", "timestamp", "time_ns")
        assert int(df.index[0]) == 1000000000
        assert "gsr_microsiemens" in df.columns
        assert pytest.approx(float(df.iloc[0]["gsr_microsiemens"])) == 1.23
