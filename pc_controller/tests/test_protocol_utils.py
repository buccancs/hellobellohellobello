from __future__ import annotations

import base64
import itertools

from network.protocol import (
    build_v1_preview_frame,
    compute_backoff_schedule,
    decode_frames,
    encode_frame,
)


def test_encode_decode_roundtrip_single() -> None:
    msg: dict[str, object] = {"v": 1, "type": "cmd", "id": 42, "command": "ping", "x": 123}
    data = encode_frame(msg)
    res = decode_frames(data)
    assert res.messages == [msg]
    assert res.remainder == b""


def test_encode_decode_roundtrip_concatenated() -> None:
    m1 = {"v": 1, "type": "cmd", "id": 1, "command": "a"}
    m2 = {"v": 1, "type": "cmd", "id": 2, "command": "b", "payload": {"k": "v"}}
    buf = encode_frame(m1) + encode_frame(m2)
    res = decode_frames(buf)
    assert res.messages == [m1, m2]
    assert res.remainder == b""


def test_backoff_schedule_properties() -> None:
    sched = compute_backoff_schedule(100, 4)
    assert sched == [100, 200, 400, 800]
    assert all(isinstance(x, int) and x > 0 for x in sched)
    assert all(b <= a * 2 for a, b in itertools.pairwise(sched))
    assert compute_backoff_schedule(100, 0) == []
    assert compute_backoff_schedule(100, -3) == []


def test_preview_frame_builder_fields() -> None:
    # Sanity check builder for preview frames used by performance scripts
    jpeg_b64 = base64.b64encode(b"fake").decode("ascii")
    msg = build_v1_preview_frame("dev1", jpeg_b64, 123456789)
    assert msg.get("v") == 1
    assert msg.get("type") == "event"
    assert msg.get("name") == "preview_frame"
    assert msg.get("device_id") == "dev1"
    assert msg.get("jpeg_base64") == jpeg_b64
    assert msg.get("ts") == 123456789
