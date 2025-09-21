"""Protocol definitions and helpers for PC<->Android communication (Phase 1-5).

This module defines message formats and utilities for:
- Legacy line-delimited JSON (Phase 1-4)
- Preferred length-prefixed framing: "${length}\n{json}" (Phase 5)

See also docs/markdown for protocol phases.

Service Type (mDNS/Zeroconf):
- _gsr-controller._tcp.local.
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass
from typing import Any

QUERY_CMD_ID = 1
COMMAND_QUERY_CAPABILITIES = "query_capabilities"

COMMAND_START_RECORDING = "start_recording"
COMMAND_STOP_RECORDING = "stop_recording"
COMMAND_FLASH_SYNC = "flash_sync"
COMMAND_TIME_SYNC = "time_sync"
COMMAND_PREVIEW_REQUEST = "preview_request"

COMMAND_TRANSFER_FILES = "transfer_files"

COMMAND_REJOIN_SESSION = "rejoin_session"

V1 = 1


def build_query_capabilities(cmd_id: int | None = None) -> str:
    """Return a JSON line string for the capabilities query (legacy).

    The payload matches the early spec and ends with a newline suitable
    for line-oriented TCP protocols.

    Args:
        cmd_id: Optional command ID. If not provided, uses QUERY_CMD_ID.
    """
    payload = {"id": cmd_id or QUERY_CMD_ID, "command": COMMAND_QUERY_CAPABILITIES}
    return json.dumps(payload) + "\n"




def encode_frame(obj: dict[str, Any]) -> bytes:
    """Encode a single JSON message using length-prefix framing.

    Format: b"{length}\n{json_bytes}"
    where length is the number of bytes in json_bytes (ASCII digits).
    """
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    prefix = f"{len(data)}\n".encode("ascii")
    return prefix + data


@dataclass
class DecodeResult:
    messages: list[dict[str, Any]]
    remainder: bytes


def decode_frames(buffer: bytes) -> DecodeResult:
    """Decode as many length-prefixed JSON frames from buffer as possible.

    Returns a DecodeResult with parsed messages and the remaining unconsumed
    bytes. If the buffer does not start with digits+"\n", no frames are
    decoded (messages=[]), and the remainder is the original buffer.
    """
    msgs: list[dict[str, Any]] = []
    i = 0
    n = len(buffer)
    while True:
        j = buffer.find(b"\n", i)
        if j == -1:
            break
        length_field = buffer[i:j]
        if not length_field.isdigit():
            break
        length = int(length_field)
        start = j + 1
        end = start + length
        if end > n:
            break
        payload_bytes = buffer[start:end]
        try:
            msg = json.loads(payload_bytes.decode("utf-8"))
            if isinstance(msg, dict):
                msgs.append(msg)
        except Exception:
            pass
        i = end
        if i >= n:
            break
    remainder = buffer[i:]
    return DecodeResult(messages=msgs, remainder=remainder)




def build_v1_cmd(command: str, msg_id: int, **kwargs: Any) -> dict[str, Any]:
    return {"v": V1, "id": int(msg_id), "type": "cmd", "command": command, **kwargs}


def build_v1_query_capabilities(msg_id: int) -> dict[str, Any]:
    return build_v1_cmd(COMMAND_QUERY_CAPABILITIES, msg_id)


def build_v1_time_sync_req(
    msg_id: int, seq: int, t0_ns: int | None = None
) -> dict[str, Any]:
    t0 = int(t0_ns if t0_ns is not None else time.time_ns())
    return build_v1_cmd(COMMAND_TIME_SYNC, msg_id, seq=int(seq), t0=t0)


def build_v1_start_recording(msg_id: int, session_id: str) -> dict[str, Any]:
    return build_v1_cmd(COMMAND_START_RECORDING, msg_id, session_id=session_id)


def build_v1_preview_frame(
    device_id: str, jpeg_base64: str, ts_ns: int
) -> dict[str, Any]:
    return {
        "v": V1,
        "type": "event",
        "name": "preview_frame",
        "device_id": device_id,
        "jpeg_base64": jpeg_base64,
        "ts": int(ts_ns),
    }


def build_v1_ack(ack_id: int, status: str = "ok", **kwargs: Any) -> dict[str, Any]:
    return {"v": V1, "ack_id": int(ack_id), "type": "ack", "status": status, **kwargs}


def build_v1_error(ack_id: int, code: str, message: str) -> dict[str, Any]:
    """Standardized error envelope.

    Example: {"v":1, "ack_id":..., "type":"error", "code":"E_BAD_PARAM", "message":"..."}
    """
    return {
        "v": V1,
        "ack_id": int(ack_id),
        "type": "error",
        "code": str(code),
        "message": str(message),
    }




def build_start_recording(session_id: str, msg_id: int) -> str:
    """Build a start_recording command with a session_id (legacy)."""
    payload = {
        "id": msg_id,
        "command": COMMAND_START_RECORDING,
        "session_id": session_id,
    }
    return json.dumps(payload) + "\n"


def build_stop_recording(msg_id: int) -> str:
    """Build a stop_recording command (legacy)."""
    payload = {"id": msg_id, "command": COMMAND_STOP_RECORDING}
    return json.dumps(payload) + "\n"


def build_flash_sync(msg_id: int) -> str:
    """Build a flash_sync command (legacy)."""
    payload = {"id": msg_id, "command": COMMAND_FLASH_SYNC}
    return json.dumps(payload) + "\n"


def build_time_sync_request(msg_id: int, t0_ns: int | None = None) -> str:
    """Build a time_sync request with PC timestamp t0 in nanoseconds (legacy)."""
    t0 = int(t0_ns if t0_ns is not None else time.time_ns())
    payload = {"id": msg_id, "command": COMMAND_TIME_SYNC, "t0": t0}
    return json.dumps(payload) + "\n"


def build_transfer_files(host: str, port: int, session_id: str, msg_id: int) -> str:
    """Build a transfer_files command with receiver host/port and session id (legacy)."""
    payload = {
        "id": msg_id,
        "command": COMMAND_TRANSFER_FILES,
        "host": host,
        "port": int(port),
        "session_id": session_id,
    }
    return json.dumps(payload) + "\n"


def parse_json_line(line: str) -> dict[str, Any]:
    """Parse a single JSON line into a dictionary.

    Primary parser is strict JSON. If that fails (e.g., tests provide
    Python-literal dicts with True/False), fall back to ast.literal_eval.
    """
    try:
        return json.loads(line)
    except Exception:
        obj = ast.literal_eval(line)
        if not isinstance(obj, dict):
            raise ValueError("Parsed object is not a dict") from None
        return obj


def compute_time_sync(t0: int, t1: int, t2: int, t3: int) -> tuple[int, int]:
    """Compute (offset_ns, delay_ns) using NTP-like algorithm.

    offset = ((t1 - t0) + (t2 - t3)) / 2
    delay = (t3 - t0) - (t2 - t1)

    Returns a tuple of integers (offset_ns, delay_ns).
    """
    offset = ((t1 - t0) + (t2 - t3)) // 2
    delay = (t3 - t0) - (t2 - t1)
    return int(offset), int(delay)


def compute_time_sync_stats(
    offsets: list[int], delays: list[int], trim_ratio: float = 0.1
) -> tuple[int, int, int, int]:
    """Aggregate multi-trial time sync results robustly.

    Parameters
    ----------
    offsets: list[int]
        Per-trial offset estimates in nanoseconds.
    delays: list[int]
        Per-trial RTT estimates in nanoseconds.
    trim_ratio: float
        Fraction to trim from each tail when computing robust statistics (0-0.45).

    Returns
    -------
    Tuple[int, int, int, int]
        (median_offset_ns, min_delay_ns, std_dev_ns, trials_used)

    Notes
    -----
    - Trims the same number of elements from both tails of the sorted offsets.
    - Uses population standard deviation on the trimmed offsets (ns).
    - If inputs are empty, returns (0, 0, 0, 0).
    """
    n = min(len(offsets), len(delays))
    if n == 0:
        return 0, 0, 0, 0
    so = sorted(offsets[:n])
    trim_ratio = max(0.0, min(0.45, float(trim_ratio)))
    k = round(n * trim_ratio)
    if k * 2 >= n:
        k = max(0, (n - 1) // 2)
    trimmed = so[k : n - k] if k > 0 else so
    m_idx = len(trimmed) // 2
    if len(trimmed) % 2 == 1:
        median_offset = int(trimmed[m_idx])
    else:
        median_offset = int((trimmed[m_idx - 1] + trimmed[m_idx]) // 2)
    if len(trimmed) <= 1:
        std_dev = 0
    else:
        mu = sum(trimmed) / float(len(trimmed))
        var = sum((x - mu) ** 2 for x in trimmed) / float(len(trimmed))
        std_dev = round(var**0.5)
    min_delay = int(min(delays[:n]))
    return int(median_offset), int(min_delay), int(std_dev), len(trimmed)


def compute_backoff_schedule(
    base_ms: int, attempts: int, factor: float = 2.0
) -> list[int]:
    """Compute an exponential backoff schedule in milliseconds without jitter.

    Example: base_ms=100, attempts=3 -> [100, 200, 400]
    """
    if attempts <= 0:
        return []
    base = max(1, int(base_ms))
    sched: list[int] = []
    delay = base
    for _ in range(attempts):
        sched.append(int(delay))
        delay = max(1, int(delay * factor))
    return sched
