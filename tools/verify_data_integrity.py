#!/usr/bin/env python3
"""
Post-hoc Data Integrity Verifier (NFR4)

Verifies file checksums recorded in session metadata. This script is designed to
work even if your current metadata does not include checksums: in that case it
prints an informative message and can optionally generate checksums to stdout so
you can capture them for your thesis appendix.

What it does:
- Loads <session_dir>/metadata.json (root session metadata).
- Recursively scans for other metadata.json files within the session (device subfolders).
- Looks for checksum entries in common schemas:
  - file_manifest: [{ filename|path, size?, checksum|sha256|md5, alg|algorithm? }]
  - received_files: [{ filename, size?, checksum?, alg? }]
- Recomputes checksums of the referenced files on disk and compares.
- Prints a per-file report with OK/FAILED/MISSING.
- Exits with code 0 if all verified entries are OK (or if there are no checksum entries),
  1 if any verification failed, or 2 on invalid input.

Usage:
  python3 scripts/verify_data_integrity.py --session /path/to/session \
      [--alg sha256] [--generate]

Notes:
- Default algorithm for recomputation is sha256; when an entry contains its own
  algorithm, that takes precedence.
- "--generate" computes and prints checksums for common session files even if no
  checksums are present in metadata, without modifying files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass

SUPPORTED_ALGS = {"sha256", "md5"}


@dataclass
class Entry:
    meta_path: str
    session_dir: str
    filename: str
    abs_path: str
    size_meta: int | None
    alg: str
    expected: str | None


def _iter_metadata_files(session_dir: str) -> Iterable[str]:
    root = os.path.join(session_dir, "metadata.json")
    if os.path.exists(root):
        yield root
    for r, _dirs, files in os.walk(session_dir):
        for f in files:
            if f == "metadata.json":
                p = os.path.join(r, f)
                if os.path.abspath(p) != os.path.abspath(root):
                    yield p


def _normalize_alg(s: str | None, default_alg: str) -> str:
    if not s:
        return default_alg
    s = s.lower().strip()
    if s in SUPPORTED_ALGS:
        return s
    if s in {"sha-256", "sha_256"}:
        return "sha256"
    return default_alg


def _digest_file(path: str, alg: str) -> str:
    h = hashlib.new(alg)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_entries(meta_path: str, session_dir: str, default_alg: str) -> list[Entry]:
    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    entries: list[Entry] = []

    def add(filename: str, expected: str | None, alg: str | None, size: int | None) -> None:
        if not filename:
            return
        abs_path = os.path.join(session_dir, filename)
        entries.append(
            Entry(
                meta_path=meta_path,
                session_dir=session_dir,
                filename=filename,
                abs_path=abs_path,
                size_meta=int(size) if size is not None else None,
                alg=_normalize_alg(alg, default_alg),
                expected=(expected.lower() if isinstance(expected, str) else None),
            )
        )

    fm = data.get("file_manifest")
    if isinstance(fm, list):
        for it in fm:
            if not isinstance(it, dict):
                continue
            filename = it.get("path") or it.get("filename") or ""
            expected = it.get("checksum") or it.get("sha256") or it.get("md5")
            alg = it.get("alg") or it.get("algorithm")
            size = it.get("size")
            add(filename, expected, alg, size)

    rf = data.get("received_files")
    if isinstance(rf, list):
        for it in rf:
            if not isinstance(it, dict):
                continue
            filename = it.get("path") or it.get("filename") or ""
            expected = it.get("checksum") or it.get("sha256") or it.get("md5")
            alg = it.get("alg") or it.get("algorithm")
            size = it.get("size")
            add(filename, expected, alg, size)

    return entries


def _format_status(ok: bool | None) -> str:
    if ok is True:
        return "OK"
    if ok is False:
        return "FAILED"
    return "MISSING"


def _find_common_files(session_dir: str) -> list[str]:
    cands = [
        "gsr.csv",
        "thermal.csv",
        os.path.join("rgb", "video.mp4"),
        os.path.join("rgb", "video.avi"),
        os.path.join("rgb", "video.mkv"),
    ]
    found: list[str] = []
    for root, _dirs, files in os.walk(session_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), session_dir)
            if rel in cands or os.path.basename(rel) in {"gsr.csv", "thermal.csv"}:
                found.append(rel)
    return sorted(set(found))


def verify(session_dir: str, default_alg: str, generate: bool) -> int:
    if not os.path.isdir(session_dir):
        print(f"Error: session directory not found: {session_dir}")
        return 2

    all_entries: list[Entry] = []
    for meta_path in _iter_metadata_files(session_dir):
        all_entries.extend(_collect_entries(meta_path, session_dir, default_alg))

    if not all_entries:
        print("[INFO] No checksum entries found in metadata.")
        if generate:
            print("[INFO] Generating checksums (no metadata modification):")
            for rel in _find_common_files(session_dir):
                abs_path = os.path.join(session_dir, rel)
                if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
                    continue
                try:
                    digest = _digest_file(abs_path, default_alg)
                    size = os.path.getsize(abs_path)
                    print(f"GEN {default_alg} {digest}  {size:10d}  {rel}")
                except Exception as e:
                    print(f"GEN ERROR {rel}: {e}")
        return 0

    print(f"[INFO] Verifying {len(all_entries)} file entr{'y' if len(all_entries)==1 else 'ies'}:")
    failures = 0
    for ent in all_entries:
        exists = os.path.exists(ent.abs_path) and os.path.isfile(ent.abs_path)
        ok: bool | None = None
        reason = ""
        if not exists:
            ok = None
            reason = "file missing"
        else:
            try:
                actual = _digest_file(ent.abs_path, ent.alg)
                if ent.expected is None:
                    ok = None
                    reason = "no expected checksum in metadata"
                else:
                    ok = actual.lower() == ent.expected.lower()
                    if not ok:
                        reason = f"mismatch (expected {ent.expected}, got {actual})"
            except ValueError:
                ok = None
                reason = f"unsupported algorithm: {ent.alg}"
            except Exception as e:
                ok = False
                reason = f"error: {e}"

        status = _format_status(ok)
        size_str = f"{ent.size_meta}" if ent.size_meta is not None else "-"
        print(f"[{status}] {ent.alg:<6} size={size_str:>10}  {ent.filename}  ({reason})")
        if ok is False:
            failures += 1

    if failures:
        print(f"[SUMMARY] FAILED: {failures} entr{'y' if failures==1 else 'ies'} mismatched.")
        return 1
    print("[SUMMARY] All checksum verifications passed or were missing.")
    return 0


def _dry_run(default_alg: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        fpath = os.path.join(tmp, "test.bin")
        content = b"hello world\n"
        with open(fpath, "wb") as f:
            f.write(content)
        digest = _digest_file(fpath, default_alg)
        meta = {
            "file_manifest": [
                {
                    "filename": "test.bin",
                    "size": len(content),
                    "checksum": digest,
                    "alg": default_alg,
                }
            ]
        }
        with open(os.path.join(tmp, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        rc = verify(tmp, default_alg, generate=False)
        if rc == 0:
            print("[DRY-RUN] verify_data_integrity: OK")
        else:
            print(f"[DRY-RUN] verify_data_integrity: FAILED (rc={rc})")
        return rc


def main(argv: list[str] | None = None) -> int:
    description = "Verify session data integrity against metadata checksums."
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--session", help="Path to the session directory")
    default_help = "Default checksum algorithm to use when metadata does not specify one"
    p.add_argument("--alg", default="sha256", choices=sorted(SUPPORTED_ALGS), help=default_help)
    generate_help = (
        "If no checksums found in metadata, compute and print checksums "
        "for common files (no file modifications)"
    )
    p.add_argument("--generate", action="store_true", help=generate_help)
    dry_run_help = "Run a self-check without requiring any real data"
    p.add_argument("--dry-run", action="store_true", help=dry_run_help)
    args = p.parse_args(argv)

    if args.dry_run:
        return _dry_run(args.alg)

    if not args.session:
        print("Error: --session is required unless --dry-run is used.")
        return 2

    return verify(args.session, args.alg, args.generate)


if __name__ == "__main__":
    raise SystemExit(main())
