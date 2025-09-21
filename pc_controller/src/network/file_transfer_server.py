"""FileTransferServer for automatic data transfer (FR10).

Listens for TCP connections on a configured port and receives files from
Android devices. Each connection sends a one-line JSON header followed by
raw file bytes. The server writes the file into the appropriate session
folder and updates session metadata to record received files (NFR4).

Header JSON fields:
- session_id: str (required)
- device_id: str (optional, used for logs)
- filename: str (required)
- size: int (optional, if provided server reads exactly this many bytes)

Protocol per connection:
1) Client sends header as a single UTF-8 line terminated by \n.
2) Client streams file content. If size is present, the server reads exactly
   that many bytes; otherwise, reads until connection close.

The server creates pc_controller_data/<session_id>/ if missing and writes
file there. After saving, it updates metadata.json in that session folder
appending the received file details.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import threading
import time
from dataclasses import dataclass


@dataclass
class _Header:
    session_id: str
    filename: str
    size: int | None
    device_id: str

    @staticmethod
    def parse(line: str) -> _Header:
        obj = json.loads(line)
        session_id = str(obj.get("session_id") or "unknown_session")
        filename = str(obj.get("filename") or "data.bin")
        size = obj.get("size")

        size_i: int | None = None
        if size is not None:
            try:
                size_i = int(size)
            except (ValueError, TypeError):
                # Log warning but continue with None size
                import logging

                logging.warning(f"Invalid size value in file transfer header: {size}")

        device_id = str(obj.get("device_id") or "unknown_device")
        return _Header(
            session_id=session_id, filename=filename, size=size_i, device_id=device_id
        )


class FileTransferServer:
    def __init__(self, base_dir: str) -> None:
        self._base_dir = os.path.abspath(base_dir)
        os.makedirs(self._base_dir, exist_ok=True)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_ev = threading.Event()
        self._port: int | None = None

    def start(self, port: int) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._port = int(port)
        self._stop_ev.clear()
        self._thread = threading.Thread(
            target=self._serve, name="FileTransferServer", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_ev.set()
        try:
            if self._sock:
                with contextlib.suppress(Exception):
                    self._sock.close()
        finally:
            if self._thread:
                self._thread.join(timeout=1.0)
                self._thread = None

    def _ensure_session_dir(self, session_id: str) -> str:
        d = os.path.join(self._base_dir, session_id)
        os.makedirs(d, exist_ok=True)
        return d

    def _update_metadata(
        self, session_dir: str, filename: str, size: int, device_id: str
    ) -> None:
        meta_path = os.path.join(session_dir, "metadata.json")
        data = {}
        try:
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    data = json.load(f)
        except Exception:
            data = {}
        files = data.get("received_files")
        if not isinstance(files, list):
            files = []
        files.append(
            {
                "filename": filename,
                "size": int(size),
                "device_id": device_id,
                "received_at_ns": int(time.time_ns()),
            }
        )
        data["received_files"] = files
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _serve(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", int(self._port or 0)))
                s.listen(5)
                self._sock = s
                s.settimeout(1.0)
                while not self._stop_ev.is_set():
                    try:
                        conn, _ = s.accept()
                    except TimeoutError:
                        continue
                    except Exception:
                        if self._stop_ev.is_set():
                            break
                        continue
                    try:
                        with conn:
                            conn.settimeout(10.0)
                            header_line = self._read_line(conn)
                            if not header_line:
                                continue
                            try:
                                header = _Header.parse(header_line)
                            except Exception:
                                continue
                            session_dir = self._ensure_session_dir(header.session_id)
                            target_path = os.path.join(session_dir, header.filename)
                            bytes_written = 0
                            with open(target_path, "wb") as f:
                                if header.size is not None:
                                    remaining = header.size
                                    while remaining > 0:
                                        chunk = conn.recv(min(65536, remaining))
                                        if not chunk:
                                            break
                                        f.write(chunk)
                                        bytes_written += len(chunk)
                                        remaining -= len(chunk)
                                else:
                                    while True:
                                        chunk = conn.recv(65536)
                                        if not chunk:
                                            break
                                        f.write(chunk)
                                        bytes_written += len(chunk)
                            with contextlib.suppress(Exception):
                                self._update_metadata(
                                    session_dir,
                                    header.filename,
                                    bytes_written,
                                    header.device_id,
                                )
                    except Exception:
                        continue
        except Exception:
            return

    def _read_line(self, conn: socket.socket) -> str | None:
        buf = bytearray()
        while True:
            try:
                b = conn.recv(1)
            except Exception:
                return None
            if not b:
                break
            if b == b"\n":
                break
            if b != b"\r":
                buf += b
            if len(buf) > 1024 * 1024:
                return None
        if not buf:
            return None
        try:
            return buf.decode("utf-8", errors="replace")
        except Exception:
            return None
