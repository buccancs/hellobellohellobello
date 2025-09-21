"""Data Aggregator and File Receiver (Phase 5).

This module implements a dedicated file-receiving server to collect
zipped session archives from Android Spokes and unpack them into the
proper session directory on the PC Hub.

It follows the design in docs/4_5_phase.md Task 5.1.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import zipfile
from dataclasses import dataclass

from PyQt6.QtCore import QObject, QThread, pyqtSignal

# TLS optional server context
try:
    from network.tls_utils import create_server_ssl_context
except Exception:  # pragma: no cover - optional import guard

    def create_server_ssl_context():
        return None


def get_local_ip() -> str:
    """Best-effort local IPv4 address for outward connections.

    Uses a UDP socket trick to determine the preferred outbound IP.
    Falls back to 127.0.0.1 if it fails.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        return ip
    except Exception:  # pragma: no cover - environment-specific
        return "127.0.0.1"


@dataclass
class _ClientHeader:
    session_id: str
    device_id: str
    filename: str
    size: int | None

    @staticmethod
    def parse(text: str) -> _ClientHeader:
        obj = json.loads(text)
        return _ClientHeader(
            session_id=str(obj.get("session_id", "unknown_session")),
            device_id=str(obj.get("device_id", "unknown_device")),
            filename=str(obj.get("filename", "data.zip")),
            size=(int(obj["size"]) if obj.get("size") is not None else None),
        )


class FileReceiverServer(QThread):
    """A simple line-prefixed JSON header + raw-bytes receiver.

    Protocol per connection:
      1) Client sends a single JSON line with keys: session_id, device_id,
         filename, size (optional).
      2) Then client streams raw bytes (ZIP archive). If `size` is provided,
         we read exactly `size` bytes; otherwise we read until the socket closes.
    After receiving, we unpack into base_dir/<session_id>/<device_id>/ and delete
    the temporary archive.
    """

    log = pyqtSignal(str)
    progress = pyqtSignal(
        str, int, int
    )
    file_received = pyqtSignal(str, str)

    def __init__(
        self, base_dir: str, port: int = 9001, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._base_dir = base_dir
        self._port = port
        self._sock: socket.socket | None = None
        self._stopped = threading.Event()

    @property
    def port(self) -> int:
        return self._port

    def stop(self) -> None:
        self._stopped.set()
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass

    def _ensure_dirs(self, session_id: str, device_id: str) -> str:
        d = os.path.join(self._base_dir, session_id, device_id)
        os.makedirs(d, exist_ok=True)
        return d

    def _recv_exact(self, conn: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(min(65536, n - len(buf)))
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf)

    def run(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", self._port))
                s.listen(5)
                self._sock = s
                self.log.emit(f"FileReceiver listening on port {self._port}")
                # Prepare optional TLS server context
                ssl_ctx = None
                try:
                    ssl_ctx = create_server_ssl_context()
                except Exception:
                    ssl_ctx = None
                while not self._stopped.is_set():
                    try:
                        s.settimeout(1.0)
                        conn, addr = s.accept()
                    except TimeoutError:
                        continue
                    except Exception as exc:
                        self.log.emit(f"Accept error: {exc}")
                        continue
                    with conn:
                        # Wrap with TLS if configured
                        try:
                            if ssl_ctx is not None:
                                try:
                                    conn = ssl_ctx.wrap_socket(conn, server_side=True)
                                except Exception as exc:
                                    self.log.emit(f"TLS wrap failed from {addr}: {exc}")
                                    continue
                            conn.settimeout(10.0)
                            header_line = b""
                            while b"\n" not in header_line:
                                chunk = conn.recv(4096)
                                if not chunk:
                                    break
                                header_line += chunk
                                if len(header_line) > 1024 * 1024:
                                    raise RuntimeError("Header too large")
                            header_text, _, remainder = header_line.partition(b"\n")
                            header = _ClientHeader.parse(
                                header_text.decode("utf-8", errors="replace")
                            )
                            target_dir = self._ensure_dirs(
                                header.session_id, header.device_id
                            )
                            tmp_zip_path = os.path.join(
                                target_dir, header.filename or "data.zip"
                            )
                            bytes_written = 0
                            total = header.size if header.size is not None else -1
                            with open(tmp_zip_path, "wb") as f:
                                if remainder:
                                    f.write(remainder)
                                    bytes_written += len(remainder)
                                    self.progress.emit(
                                        header.device_id, bytes_written, total
                                    )
                                if header.size is not None:
                                    to_read = header.size - len(remainder)
                                    while to_read > 0:
                                        chunk = conn.recv(min(65536, to_read))
                                        if not chunk:
                                            break
                                        f.write(chunk)
                                        bytes_written += len(chunk)
                                        to_read -= len(chunk)
                                        self.progress.emit(
                                            header.device_id, bytes_written, total
                                        )
                                else:
                                    while True:
                                        chunk = conn.recv(65536)
                                        if not chunk:
                                            break
                                        f.write(chunk)
                                        bytes_written += len(chunk)
                                        self.progress.emit(
                                            header.device_id, bytes_written, total
                                        )
                            try:
                                with zipfile.ZipFile(tmp_zip_path, "r") as zf:
                                    zf.extractall(target_dir)
                                os.remove(tmp_zip_path)
                                self.file_received.emit(
                                    header.session_id, header.device_id
                                )
                                self.log.emit(
                                    f"Received and unpacked {bytes_written} bytes "
                                    f"from {header.device_id}"
                                )
                            except Exception as exc:
                                self.log.emit(
                                    f"Failed to unpack zip from {header.device_id}: {exc}"
                                )
                        except Exception as exc:
                            self.log.emit(f"Receiver error from {addr}: {exc}")
        except Exception as exc:
            self.log.emit(f"FileReceiver fatal error: {exc}")


class DataAggregator(QObject):
    """High-level manager for the FileReceiver server lifecycle."""

    log = pyqtSignal(str)
    progress = pyqtSignal(str, int, int)
    file_received = pyqtSignal(str, str)

    def __init__(self, base_dir: str) -> None:
        super().__init__()
        self._base_dir = base_dir
        self._server: FileReceiverServer | None = None

    def start_server(self, port: int = 9001) -> int:
        if self._server and self._server.isRunning():
            return self._server.port
        srv = FileReceiverServer(self._base_dir, port)
        srv.log.connect(self.log.emit)
        srv.progress.connect(self.progress.emit)
        srv.file_received.connect(self.file_received.emit)
        srv.start()
        self._server = srv
        self.log.emit(f"DataAggregator server started on port {port}")
        return port

    def stop_server(self) -> None:
        if self._server:
            try:
                self._server.stop()
                self._server.wait(1000)
            except Exception:
                pass
            self._server = None
            self.log.emit("DataAggregator server stopped")
