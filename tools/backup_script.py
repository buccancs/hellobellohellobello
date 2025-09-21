"""Backup Script for Local Data (3-2-1 Strategy Helper).

Usage (Windows PowerShell):
    python scripts\backup_script.py --source "C:\\Data\\PhysioSessions" --dest "E:\\ResearchBackups\\PhysioPlatform" --log "C:\\Data\\backup_logs\\backup.log"

This script copies files from the source directory to a destination
folder, preserving timestamps (copy2). It logs progress and errors.

Note: This is a local-copy helper. Ensure you also implement off-site
backups per BACKUP_STRATEGY.md.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path


def setup_logger(log_path: Path | None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path) if log_path else logging.StreamHandler()],
    )


def copy_tree(src: Path, dst: Path) -> None:
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            s = Path(root) / name
            d = target_dir / name
            try:
                shutil.copy2(s, d)
                logging.info("Copied: %s -> %s", s, d)
            except Exception as exc:
                logging.error("Failed to copy %s: %s", s, exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local backup copy script")
    parser.add_argument("--source", required=True, help="Source directory path")
    parser.add_argument("--dest", required=True, help="Destination directory path")
    parser.add_argument("--log", help="Optional log file path")
    args = parser.parse_args()

    src = Path(args.source)
    dst = Path(args.dest)
    log_path = Path(args.log) if args.log else None

    if not src.exists() or not src.is_dir():
        print(f"Source does not exist or is not a directory: {src}")
        return 1

    dst.mkdir(parents=True, exist_ok=True)
    setup_logger(log_path)

    logging.info("Starting backup: %s -> %s", src, dst)
    copy_tree(src, dst)
    logging.info("Backup completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
