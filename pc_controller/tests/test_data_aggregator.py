import io
import json
import socket
import time
import zipfile
from pathlib import Path

import pytest

PyQt6 = pytest.importorskip("PyQt6")

from data.data_aggregator import FileReceiverServer, _ClientHeader  # type: ignore  # noqa: E402


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


def test_client_header_parse_roundtrip():
    header = {
        "session_id": "sess_123",
        "device_id": "Pixel_7",
        "filename": "Pixel_7_data.zip",
        "size": 42,
    }
    parsed = _ClientHeader.parse(json.dumps(header))
    assert parsed.session_id == header["session_id"]
    assert parsed.device_id == header["device_id"]
    assert parsed.filename == header["filename"]
    assert parsed.size == header["size"]


def test_file_receiver_server_receives_and_unpacks(tmp_path: Path):
    port = _find_free_port()
    server = FileReceiverServer(base_dir=str(tmp_path), port=port)
    server.start()

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("rgb/file1.txt", "hello")
    zip_bytes = mem.getvalue()

    session_id = "sessA"
    device_id = "DeviceA"
    filename = f"{device_id}_data.zip"

    time.sleep(0.2)

    with socket.create_connection(("127.0.0.1", port), timeout=2.0) as sock:
        header = {
            "session_id": session_id,
            "device_id": device_id,
            "filename": filename,
            "size": len(zip_bytes),
        }
        sock.sendall(json.dumps(header).encode("utf-8") + b"\n" + zip_bytes)

    target = tmp_path / session_id / device_id / "rgb" / "file1.txt"
    deadline = time.time() + 3.0
    while time.time() < deadline and not target.exists():
        time.sleep(0.05)
    assert target.exists(), f"Expected file not found: {target}"
    assert target.read_text() == "hello"

    server.stop()
    time.sleep(0.1)
