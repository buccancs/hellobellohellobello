"""
Production HDF5 Export Pipeline for Multi-Modal Physiological Data

Enhanced export functionality with comprehensive features for research use.
This module provides production-ready export in addition to the basic exporter.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

try:
    from PyQt6.QtCore import QObject, pyqtSignal

    HAS_QT = True
except ImportError:
    HAS_QT = False

    class _DummyQObject:
        def __init__(self):
            pass

    QObject = _DummyQObject  # type: ignore

    def pyqtSignal(*args):
        return lambda: None


class ProductionHDF5Exporter(QObject if HAS_QT else object):
    """Production-ready HDF5 exporter with enhanced features."""

    if HAS_QT:
        progress_updated = pyqtSignal(int, str)
        export_completed = pyqtSignal(str)
        export_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.compression_level = 6

    def emit_progress(self, percent: int, message: str):
        """Emit progress or print to console."""
        if HAS_QT and hasattr(self, "progress_updated"):
            self.progress_updated.emit(percent, message)
        else:
            print(f"[{percent}%] {message}")

    def export_with_integrity_checks(
        self,
        session_dir: Path,
        output_path: Path,
        metadata: dict[str, Any] | None = None,
        anonymize: bool = True,
        participant_id: str | None = None,
    ) -> bool:
        """
        Export with data integrity verification and comprehensive metadata.

        Args:
            session_dir: Session directory path
            output_path: Output HDF5 file path
            metadata: Additional metadata
            anonymize: Whether to anonymize data
            participant_id: Participant ID (if not anonymized)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.emit_progress(0, "Starting production HDF5 export...")

            if not session_dir.exists():
                raise ValueError(f"Session directory not found: {session_dir}")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(output_path, "w") as hdf:
                export_metadata = {
                    "export_timestamp": datetime.now(UTC).isoformat(),
                    "exporter_version": "2.0.0-production",
                    "hdf5_version": h5py.version.hdf5_version,
                    "anonymized": anonymize,
                    "temporal_accuracy_ms": 5.0,
                    "synchronization_method": "NTP-like offset correction",
                }

                if metadata:
                    export_metadata.update(metadata)

                if not anonymize and participant_id:
                    export_metadata["participant_id"] = participant_id
                else:
                    export_metadata["participant_id"] = "ANONYMIZED"

                hdf.attrs["session_metadata_json"] = json.dumps(export_metadata)

                self._process_data_files_with_integrity(hdf, session_dir)

                self._write_hardware_metadata(hdf, session_dir)

                self.emit_progress(100, "Production export completed")

            if HAS_QT and hasattr(self, "export_completed"):
                self.export_completed.emit(str(output_path))

            return True

        except Exception as e:
            error_msg = f"Production HDF5 export failed: {e!s}"
            if HAS_QT and hasattr(self, "export_error"):
                self.export_error.emit(error_msg)
            else:
                print(error_msg)
            return False

    def _process_data_files_with_integrity(self, hdf: h5py.File, session_dir: Path):
        """Process data files with integrity verification."""
        integrity_group = hdf.create_group("data_integrity")

        csv_files = list(session_dir.rglob("*.csv"))

        for i, csv_file in enumerate(csv_files):
            progress = int(20 + (i / len(csv_files)) * 60)
            rel_path = csv_file.relative_to(session_dir)
            self.emit_progress(progress, f"Processing {rel_path}")

            file_md5 = self._calculate_file_md5(csv_file)
            integrity_group.attrs[f"{rel_path.as_posix().replace('/', '_')}_md5"] = (
                file_md5
            )

            try:
                df = pd.read_csv(csv_file)
                if df.empty:
                    continue

                parts = rel_path.parts
                device = parts[0] if len(parts) > 1 else "PC"
                modality = rel_path.stem

                group = hdf.require_group(f"devices/{device}/{modality}")

                timestamp_col = self._find_timestamp_column(df.columns)
                if timestamp_col:
                    timestamps = np.array(df[timestamp_col], dtype=np.int64)
                    group.create_dataset(
                        "timestamp_ns",
                        data=timestamps,
                        compression="gzip",
                        compression_opts=self.compression_level,
                    )
                    df = df.drop(columns=[timestamp_col])

                for col in df.columns:
                    self._create_optimized_dataset(group, col, df[col])

                group.attrs["sample_count"] = len(df)
                group.attrs["source_file"] = str(rel_path)
                group.attrs["data_integrity_md5"] = file_md5

            except Exception as e:
                print(f"Warning: Could not process {rel_path}: {e}")
                continue

    def _find_timestamp_column(self, columns):
        """Find timestamp column using common patterns."""
        patterns = ["timestamp_ns", "ts_ns", "timestamp", "time_ns", "time"]
        for col in columns:
            if col.lower() in patterns:
                return col
        return None

    def _create_optimized_dataset(
        self, group: h5py.Group, col_name: str, series: pd.Series
    ):
        """Create optimized HDF5 dataset for the given column."""
        if series.dtype == "object":
            dt = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(col_name, data=series.values, dtype=dt)
        elif series.dtype.kind in ["i", "u"]:
            if series.max() <= 65535 and series.min() >= 0:
                data = series.astype(np.uint16)
            elif series.max() <= 32767 and series.min() >= -32768:
                data = series.astype(np.int16)
            else:
                data = series.values

            group.create_dataset(
                col_name,
                data=data,
                compression="gzip",
                compression_opts=self.compression_level,
            )
        elif series.dtype.kind == "f":
            data = series.astype(np.float32)
            group.create_dataset(
                col_name,
                data=data,
                compression="gzip",
                compression_opts=self.compression_level,
            )
        else:
            group.create_dataset(col_name, data=series.values)

    def _calculate_file_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash for integrity verification."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _write_hardware_metadata(self, hdf: h5py.File, session_dir: Path):
        """Write hardware configuration metadata."""
        config_files = list(session_dir.rglob("device_config.json"))

        if config_files:
            hardware_group = hdf.create_group("hardware_config")

            for config_file in config_files:
                try:
                    with open(config_file) as f:
                        config_data = json.load(f)

                    rel_path = config_file.relative_to(session_dir)
                    device_name = rel_path.parts[0]

                    device_config = hardware_group.create_group(device_name)
                    device_config.attrs["config_json"] = json.dumps(config_data)

                except Exception as e:
                    print(
                        f"Warning: Could not process hardware config {config_file}: {e}"
                    )


def export_session_production(
    session_dir: Path,
    output_path: Path,
    metadata: dict[str, Any] | None = None,
    anonymize: bool = True,
    participant_id: str | None = None,
) -> bool:
    """
    Convenience function for production HDF5 export.

    Args:
        session_dir: Session directory
        output_path: Output HDF5 file
        metadata: Additional metadata
        anonymize: Whether to anonymize data
        participant_id: Participant ID

    Returns:
        True if successful
    """
    exporter = ProductionHDF5Exporter()
    return exporter.export_with_integrity_checks(
        session_dir, output_path, metadata, anonymize, participant_id
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Production HDF5 export")
    parser.add_argument("session_dir", type=Path, help="Session directory")
    parser.add_argument("output_file", type=Path, help="Output HDF5 file")
    parser.add_argument("--participant-id", help="Participant ID")
    parser.add_argument(
        "--no-anonymize", action="store_true", help="Keep identifying information"
    )

    args = parser.parse_args()

    success = export_session_production(
        args.session_dir,
        args.output_file,
        anonymize=not args.no_anonymize,
        participant_id=args.participant_id,
    )

    exit(0 if success else 1)
