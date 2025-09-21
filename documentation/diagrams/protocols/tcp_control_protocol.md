# TCP Control Protocol Specification

**Purpose**: Definitive reference for implementers and testers of the length-prefixed JSON control protocol.

**Placement**: Protocols appendix or Architecture chapter.

**Content**: Complete message format specification with v=1 framing and legacy support.

## Message Framing

### Length-Prefixed JSON (v=1, Preferred)
```
${length}\n{json_payload}
```
- First line: ASCII decimal length of JSON payload in bytes + newline
- Next `${length}` bytes: UTF-8 JSON document
- Example: `43\n{"v":1,"type":"cmd","id":1,"command":"ping"}`

### Legacy Newline-Delimited (Fallback)
```
{json_payload}\n
```
- Single JSON object terminated by newline
- Example: `{"id":1,"command":"ping"}\n`

## Message Schema (v=1)

### Common Fields
| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `v` | integer | Protocol version (1) | Yes |
| `type` | string | Message type: "cmd", "ack", "error", "event" | Yes |
| `id` | integer | Request ID (PC-generated, for commands only) | Commands only |
| `ack_id` | integer | Echoed request ID (for responses) | Responses only |

## Command Reference

### query_capabilities

**Purpose**: Discover Android device hardware capabilities and configuration.

**Request** (PC → Android):
```json
{
  "v": 1,
  "type": "cmd",
  "id": 1,
  "command": "query_capabilities"
}
```

**Response** (Android → PC):
```json
{
  "v": 1,
  "type": "ack",
  "ack_id": 1,
  "status": "ok",
  "capabilities": {
    "device_id": "Pixel_7_ab12cd34",
    "device_model": "Pixel 7",
    "android_sdk": 34,
    "android_release": "14",
    "service_port": 8080,
    "has_rgb": true,
    "has_thermal": false,
    "has_gsr": true,
    "cameras": [
      {
        "id": "0",
        "facing": "BACK",
        "resolutions": ["1920x1080", "1280x720"]
      },
      {
        "id": "1",
        "facing": "FRONT",
        "resolutions": ["1920x1080"]
      }
    ]
  }
}
```

### time_sync

**Purpose**: TCP-based time synchronization for offset calculation.

**Request** (PC → Android):
```json
{
  "v": 1,
  "type": "cmd",
  "id": 2,
  "command": "time_sync"
}
```

**Response** (Android → PC):
```json
{
  "v": 1,
  "type": "ack",
  "ack_id": 2,
  "status": "ok",
  "t1": 1692374212450000000,
  "t2": 1692374212451234567
}
```

**Fields**:
- `t1`: Android timestamp when request received (nanoseconds)
- `t2`: Android timestamp when response sent (nanoseconds)

### start_recording

**Purpose**: Initiate recording session on Android device.

**Request** (PC → Android):
```json
{
  "v": 1,
  "type": "cmd",
  "id": 3,
  "command": "start_recording",
  "session_id": "20250818_173012_123_DeviceX_ab12cd34"
}
```

**Response** (Android → PC):
```json
{
  "v": 1,
  "type": "ack",
  "ack_id": 3,
  "status": "ok"
}
```

**Error Response**:
```json
{
  "v": 1,
  "type": "error",
  "ack_id": 3,
  "code": "E_RECORDING_ACTIVE",
  "message": "Recording already in progress"
}
```

### stop_recording

**Purpose**: Stop current recording session.

**Request** (PC → Android):
```json
{
  "v": 1,
  "type": "cmd",
  "id": 4,
  "command": "stop_recording"
}
```

**Response** (Android → PC):
```json
{
  "v": 1,
  "type": "ack",
  "ack_id": 4,
  "status": "ok"
}
```

### flash_sync

**Purpose**: Trigger flash synchronization event for temporal alignment.

**Request** (PC → Android):
```json
{
  "v": 1,
  "type": "cmd",
  "id": 5,
  "command": "flash_sync"
}
```

**Response** (Android → PC):
```json
{
  "v": 1,
  "type": "ack",
  "ack_id": 5,
  "status": "ok",
  "ts": 1692374212450000000
}
```

**Fields**:
- `ts`: Precise nanosecond timestamp when flash event occurred

### transfer_files

**Purpose**: Initiate file transfer from Android to PC.

**Request** (PC → Android):
```json
{
  "v": 1,
  "type": "cmd",
  "id": 6,
  "command": "transfer_files",
  "host": "192.168.1.100",
  "port": 8090,
  "session_id": "20250818_173012_123_DeviceX_ab12cd34"
}
```

**Response** (Android → PC):
```json
{
  "v": 1,
  "type": "ack",
  "ack_id": 6,
  "status": "ok"
}
```

**Error Response**:
```json
{
  "v": 1,
  "type": "error",
  "ack_id": 6,
  "code": "E_BAD_PARAM",
  "message": "Session directory not found"
}
```

### rejoin_session (Future Feature)

**Purpose**: Reconnect to existing recording session after network interruption.

**Request** (Android → PC):
```json
{
  "v": 1,
  "type": "cmd",
  "command": "rejoin_session",
  "session_id": "20250818_173012_123_DeviceX_ab12cd34",
  "device_id": "Pixel_7_ab12cd34",
  "recording": true
}
```

## Event Messages (Android → PC)

### preview_frame

**Purpose**: Stream downsampled camera preview frames to PC dashboard.

```json
{
  "v": 1,
  "type": "event",
  "name": "preview_frame",
  "device_id": "Pixel_7_ab12cd34",
  "jpeg_base64": "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBD...",
  "ts": 1692374212450000000
}
```

**Rate Limiting**: ~6-8 FPS via PreviewBus throttling (150ms gate)

## Error Codes

| Code | Description |
|------|-------------|
| `E_BAD_PARAM` | Invalid or missing command parameters |
| `E_RECORDING_ACTIVE` | Cannot start recording when already active |
| `E_NOT_RECORDING` | Cannot stop recording when not active |
| `E_SENSOR_ERROR` | Hardware sensor initialization failed |
| `E_STORAGE_ERROR` | File system or storage access error |
| `E_NETWORK_ERROR` | Network connection or transfer error |

## Implementation Notes

- **Backward Compatibility**: Receivers must accept both framed and newline-delimited formats
- **Forward Compatibility**: Unknown fields in JSON messages must be ignored
- **Timeout Handling**: Commands should timeout after 10 seconds
- **Connection Management**: TCP connections are persistent during session
- **Encoding**: All text data uses UTF-8 encoding
