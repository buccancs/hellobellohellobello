# Timing and Synchronization Diagrams

**Purpose**: Show derivation of offset and round-trip handling for both UDP echo and TCP time_sync protocols.

**Placement**: Chapter 3: NFRs timing subsection.

## UDP Time Synchronization Timeline

**Purpose**: Demonstrate NTP-like protocol for cross-device clock alignment.

### Sequence Diagram

```mermaid
sequenceDiagram
    participant A as Android TimeManager
    participant P as PC TimeServer

    Note over A,P: UDP Time Synchronization Protocol

    A->>A: t1 = System.nanoTime()
    A->>P: UDP packet [1 byte probe]
    Note over P: Receive at T_server
    P->>P: T_server = time.monotonic_ns()
    P->>A: UDP response [T_server as ASCII bytes]
    A->>A: t2 = System.nanoTime()

    Note over A: Calculate offset
    Note over A: rtt = t2 - t1
    Note over A: network_delay = rtt / 2
    Note over A: offset = (T_server + network_delay) - t2
    Note over A: offset = T_server + (t2 - t1) / 2 - t2
    Note over A: offsetNs = offset (store for future use)

    A->>A: Synchronized timestamp = System.nanoTime() + offsetNs
```

### Mathematical Formula

**Offset Calculation**:
```
Given:
  t1 = Android timestamp when packet sent
  t2 = Android timestamp when response received
  T_server = PC server timestamp when packet processed

Round-trip time:
  rtt = t2 - t1

Network delay (assuming symmetric):
  network_delay = rtt / 2

Server time when response received:
  T_server_adjusted = T_server + network_delay
                    = T_server + (t2 - t1) / 2

Clock offset (PC time - Android time):
  offset = T_server_adjusted - t2
         = T_server + (t2 - t1) / 2 - t2
         = T_server - (t2 + t1) / 2
```

**Synchronized Timestamp**:
```kotlin
fun getSyncedTimestamp(): Long {
    val now = System.nanoTime()
    return now + offsetNs
}
```

## TCP Time Sync Command

**Purpose**: Alternative time synchronization via control channel.

### Sequence Diagram

```mermaid
sequenceDiagram
    participant PC as PC Controller
    participant RS as Android RecordingService
    participant TM as TimeManager

    Note over PC,TM: TCP Time Sync Command

    PC->>RS: {"v":1,"type":"cmd","id":2,"command":"time_sync"}
    RS->>TM: t1 = TimeManager.nowNanos()
    Note over RS: Process command
    RS->>TM: t2 = TimeManager.nowNanos()
    RS->>PC: {"v":1,"type":"ack","ack_id":2,"status":"ok","t1":t1,"t2":t2}

    Note over PC: Calculate round-trip time
    Note over PC: Can estimate network delay from t2 - t1
    Note over PC: Store timing data for validation
```

### Usage Comparison

| Protocol | Purpose | Accuracy | Overhead | Use Case |
|----------|---------|----------|----------|----------|
| **UDP Echo** | Primary sync | High | Low | Real-time timestamp alignment |
| **TCP Command** | Validation | Medium | Higher | Sync quality assessment |

## Preview Frame Throttling

**Purpose**: Show PreviewBus throttling curve to explain bandwidth management.

### Throttling Timeline

```mermaid
gantt
    title Preview Frame Throttling (6-8 FPS Target)
    dateFormat X
    axisFormat %L ms

    section Camera Capture (30 FPS)
    Frame 1: 0, 33
    Frame 2: 33, 67
    Frame 3: 67, 100
    Frame 4: 100, 133
    Frame 5: 133, 167
    Frame 6: 167, 200

    section PreviewBus Emission (6-8 FPS)
    Emit Frame 1: 0, 10
    Throttled: 33, 150
    Emit Frame 4: 150, 160
    Throttled: 167, 300
    Emit Frame 6: 300, 310
```

### Throttling Algorithm

```kotlin
// In RgbCameraRecorder
private var lastPreviewNs: Long = 0L
private val PREVIEW_THROTTLE_NS = 150_000_000L // 150ms = 6.67 FPS

private fun emitPreviewIfThrottled(jpegBytes: ByteArray) {
    val now = TimeManager.nowNanos()
    if (now - lastPreviewNs >= PREVIEW_THROTTLE_NS) {
        PreviewBus.emit(jpegBytes, now)
        lastPreviewNs = now
    }
    // Otherwise frame is dropped (network bandwidth conservation)
}
```

### Performance Characteristics

```mermaid
xychart-beta
    title "Preview FPS vs Network Bandwidth"
    x-axis "Preview FPS" [1, 2, 4, 6, 8, 10, 15, 20, 30]
    y-axis "Bandwidth (Mbps)" 0 --> 60
    line [2, 4, 8, 12, 16, 20, 30, 40, 60]
```

**Assumptions**:
- JPEG frame size: ~50KB (640x480, quality=70)
- Base64 encoding overhead: ~33%
- Network frame size: ~67KB per preview
- Total bandwidth = FPS × 67KB × 8 bits/byte

**Throttling Benefits**:
- Prevents network congestion with multiple devices
- Reduces CPU usage for base64 encoding
- Maintains responsive control channel communication
- Preserves bandwidth for high-resolution recording data

## Time Sync Accuracy Analysis

### Error Sources

| Source | Typical Error | Mitigation |
|--------|---------------|------------|
| **Network Jitter** | 1-5ms | Multiple samples, outlier filtering |
| **Clock Drift** | 10-50ppm | Periodic re-synchronization |
| **Processing Delay** | 0.1-1ms | High-priority threads |
| **Asymmetric Network** | 0.5-2ms | Statistical analysis |

### Synchronization Quality Metrics

```mermaid
xychart-beta
    title "Time Sync Accuracy Over Session Duration"
    x-axis "Time (minutes)" [0, 5, 10, 15, 20, 25, 30]
    y-axis "Sync Error (ms)" 0 --> 10
    line [1, 1.5, 2.1, 2.8, 3.2, 3.8, 4.1]
```

**Acceptable Thresholds**:
- **Excellent**: <2ms error
- **Good**: <5ms error (meets NFR2 requirement)
- **Warning**: 5-10ms error
- **Poor**: >10ms error (re-sync recommended)

### Multi-Device Synchronization

```mermaid
sequenceDiagram
    participant PC as PC TimeServer
    participant A1 as Android Device 1
    participant A2 as Android Device 2
    participant A3 as Android Device 3

    Note over PC,A3: Simultaneous Flash Sync Event

    PC->>A1: flash_sync command
    PC->>A2: flash_sync command
    PC->>A3: flash_sync command

    par Flash Event Processing
        A1->>A1: Record flash timestamp
        A1->>PC: ACK with ts=1692374212450000000
    and
        A2->>A2: Record flash timestamp
        A2->>PC: ACK with ts=1692374212450001200
    and
        A3->>A3: Record flash timestamp
        A3->>PC: ACK with ts=1692374212449999800
    end

    Note over PC: Analyze timestamp spread
    Note over PC: Max deviation: 1.4ms (acceptable)
    Note over PC: All devices synchronized within 5ms target
```

**Cross-Device Validation**:
- Flash sync events used as ground truth
- Expected timestamp spread: <5ms for synchronized devices
- Statistical analysis of drift patterns
- Automatic re-sync if deviation exceeds thresholds

## Implementation Notes

### Android TimeManager.kt
- Uses `System.nanoTime()` for monotonic timestamps
- Stores offset from UDP sync in volatile variable
- Thread-safe access to synchronized timestamps
- Background sync with configurable intervals

### PC TimeServer Implementation
- Uses `time.monotonic_ns()` for high precision
- UDP socket with 1-second timeout
- Handles concurrent sync requests from multiple devices
- Logs sync statistics for quality monitoring

### Validation Tools
- `validate_sync_core.py` analyzes flash sync timestamp distributions
- Cross-device synchronization quality assessment
- Drift detection and correction recommendations
- Integration with automated testing pipeline
