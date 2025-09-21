# Sequence Diagrams for Key Workflows

**Purpose**: Show command exchange and internal orchestration for critical system operations.

**Placement**: Chapter 4: Protocols or Use Cases section.

## Start/Stop Recording Sequence

**Purpose**: Demonstrate PC-initiated recording control and Android internal coordination.

### PlantUML Diagram

```plantuml
@startuml
actor Operator
participant PC as "PC Controller"
participant RS as "RecordingService"
participant UI as "MainActivity"
participant RC as "RecordingController"
participant RGB as "RgbCameraRecorder"
participant TH as "ThermalCameraRecorder"
participant GSR as "ShimmerRecorder"

== Start Recording ==
Operator -> PC: Click Start Session
activate PC

PC -> RS: {"v":1,"type":"cmd","command":"start_recording","id":1,"session_id":"20250818_173012_123_DeviceX_ab12cd34"}
activate RS

RS -> UI: Broadcast ACTION_START_RECORDING
activate UI

UI -> RC: startSession("20250818_173012_123_DeviceX_ab12cd34")
activate RC

note over RC: State: IDLE → PREPARING
RC -> RC: Create session directory
RC -> RGB: start(sessionDir/rgb)
activate RGB
RGB -> RGB: Initialize CameraX\n(VideoCapture + ImageCapture)
return

RC -> TH: start(sessionDir/thermal)
activate TH
TH -> TH: Create thermal.csv\nand metadata.json
return

RC -> GSR: start(sessionDir/gsr)
activate GSR
GSR -> GSR: Connect Shimmer BLE\nStart GSR streaming
return

note over RC: State: PREPARING → RECORDING
return

return

RS -> PC: {"v":1,"type":"ack","ack_id":1,"status":"ok"}
return

Operator <-- PC: Recording Started

== Recording Active ==
RGB -> RGB: Capture MP4 video\n+ JPEG frames + CSV index
TH -> TH: Generate thermal CSV data
GSR -> GSR: Stream GSR/PPG data to CSV

== Stop Recording ==
Operator -> PC: Click Stop Session
activate PC

PC -> RS: {"v":1,"type":"cmd","command":"stop_recording","id":2}
activate RS

RS -> UI: Broadcast ACTION_STOP_RECORDING
activate UI

UI -> RC: stopSession()
activate RC

note over RC: State: RECORDING → STOPPING
RC -> RGB: stop()
activate RGB
RGB -> RGB: Finalize MP4\nFlush CSV buffer
return

RC -> TH: stop()
activate TH
TH -> TH: Close CSV files
return

RC -> GSR: stop()
activate GSR
GSR -> GSR: Disconnect BLE\nClose CSV files
return

note over RC: State: STOPPING → IDLE
return

return

RS -> PC: {"v":1,"type":"ack","ack_id":2,"status":"ok"}
return

Operator <-- PC: Recording Stopped
@enduml
```

### Mermaid Alternative

```mermaid
sequenceDiagram
    participant O as Operator
    participant PC as PC Controller
    participant RS as RecordingService
    participant UI as MainActivity
    participant RC as RecordingController
    participant RGB as RgbCameraRecorder
    participant TH as ThermalRecorder
    participant GSR as ShimmerRecorder

    Note over O,GSR: Start Recording Flow
    O->>PC: Click Start Session
    PC->>RS: start_recording command (v=1, JSON)
    RS->>UI: ACTION_START_RECORDING broadcast
    UI->>RC: startSession(sessionId)

    Note over RC: State: IDLE → PREPARING
    RC->>RC: Create session directory

    par Parallel Sensor Initialization
        RC->>RGB: start(sessionDir/rgb)
        RGB->>RGB: Initialize CameraX
    and
        RC->>TH: start(sessionDir/thermal)
        TH->>TH: Create CSV + metadata
    and
        RC->>GSR: start(sessionDir/gsr)
        GSR->>GSR: Connect Shimmer BLE
    end

    Note over RC: State: PREPARING → RECORDING
    RS-->>PC: ACK (status: ok)
    PC-->>O: Recording Started

    Note over O,GSR: Recording Active
    RGB->>RGB: Capture MP4 + JPEGs + CSV
    TH->>TH: Generate thermal CSV
    GSR->>GSR: Stream GSR/PPG data

    Note over O,GSR: Stop Recording Flow
    O->>PC: Click Stop Session
    PC->>RS: stop_recording command
    RS->>UI: ACTION_STOP_RECORDING broadcast
    UI->>RC: stopSession()

    Note over RC: State: RECORDING → STOPPING
    par Parallel Sensor Shutdown
        RC->>RGB: stop()
        RGB->>RGB: Finalize MP4, flush CSV
    and
        RC->>TH: stop()
        TH->>TH: Close CSV files
    and
        RC->>GSR: stop()
        GSR->>GSR: Disconnect BLE, close CSV
    end

    Note over RC: State: STOPPING → IDLE
    RS-->>PC: ACK (status: ok)
    PC-->>O: Recording Stopped
```

## File Transfer Sequence

**Purpose**: Show push model from Android to PC receiver including ZIP streaming.

### PlantUML Diagram

```plantuml
@startuml
participant PC as "PC Controller"
participant RS as "RecordingService"
participant FTM as "FileTransferManager"
participant FTS as "FileTransferServer"
participant Storage as "PC Storage"

== Transfer Initiation ==
PC -> RS: {"v":1,"type":"cmd","command":"transfer_files","id":3,"host":"192.168.1.100","port":8090,"session_id":"20250818_173012_123"}
activate RS

RS -> FTM: transferSession("20250818_173012_123", "192.168.1.100", 8090)
activate FTM

RS -> PC: {"v":1,"type":"ack","ack_id":3,"status":"ok"}

== File Transfer Process ==
FTM -> FTS: TCP Connect (host:8090)
note over FTM: 2s connect timeout

FTM -> FTS: JSON header line\n{"session_id":"20250818_173012_123","filename":"session_data.zip","device_id":"Pixel_7_ab12cd34"}

FTM -> FTM: zipDirectoryContents()\nRecursive directory scan

loop For each file in session directory
    note over FTM: Including:\n• rgb/ (MP4 + JPEGs + CSV)\n• thermal/ (CSV + metadata.json)\n• gsr/ (CSV)\n• flash_sync_events.csv (if exists)
    FTM -> FTS: ZIP entry header
    FTM -> FTS: File content (streamed)
end

FTM -> FTS: ZIP end marker
FTM -> FTS: Close connection

note over FTS: 5s read timeout per chunk
FTS -> Storage: Write session files\n./sessions/20250818_173012_123/

FTS -> FTS: Update metadata\nsession_transfers.json

deactivate FTM

note over RS: Transfer complete
deactivate RS

PC <-- FTS: Transfer completion\n(via file system monitoring)
@enduml
```

### Key Transfer Details

**Header Format**:
```json
{"session_id": "20250818_173012_123", "filename": "session_data.zip", "size": 45678901, "device_id": "Pixel_7_ab12cd34"}
```

**ZIP Stream Contents**:
- `rgb/video_1692374212345678901.mp4` - H.264 video recording
- `rgb/frames/frame_*.jpg` - High-resolution JPEG stills
- `rgb/rgb.csv` - Frame timestamp index
- `thermal/thermal.csv` - Thermal sensor data (CSV)
- `thermal/metadata.json` - Sensor configuration
- `gsr/gsr.csv` - GSR and PPG measurements
- `flash_sync_events.csv` - Flash synchronization timestamps

**Error Handling**:
- Connection timeout: 2 seconds
- Read timeout: 5 seconds per chunk
- Automatic retry with exponential backoff
- Partial transfer detection and cleanup

**Performance Considerations**:
- No temporary ZIP file creation (streaming)
- ~10-50 MB/s transfer rate depending on network
- Memory usage ~1-2 MB (streaming buffers)

## Preview Frame Streaming

**Purpose**: Show continuous preview frame broadcast for dashboard display.

### Mermaid Diagram

```mermaid
sequenceDiagram
    participant RGB as RgbCameraRecorder
    participant PB as PreviewBus
    participant RS as RecordingService
    participant PC as PC Controller
    participant GUI as PyQt6 GUI

    Note over RGB,GUI: Preview Streaming (During Recording)

    loop Every ~150ms (6-8 FPS)
        RGB->>RGB: Capture preview frame
        RGB->>RGB: Downsample to 640x480
        RGB->>RGB: Encode as JPEG (quality=70)
        RGB->>PB: emit(jpegBytes, timestampNs)

        PB->>RS: Preview frame callback
        RS->>RS: Base64 encode JPEG
        RS->>PC: {"v":1,"type":"event","name":"preview_frame","device_id":"Pixel_7","jpeg_base64":"...","ts":1692374212450}

        PC->>PC: Decode base64 JPEG
        PC->>PC: Convert to QImage
        PC->>GUI: Update device preview (QLabel pixmap)
        GUI->>GUI: Display in dashboard grid
    end

    Note over RGB,GUI: Throttling prevents network overload
```

**Throttling Logic**:
- PreviewBus implements 150ms minimum interval
- Last preview timestamp stored per recorder
- Frames dropped if interval < 150ms
- Target rate: 6-8 FPS for bandwidth efficiency

**Performance Impact**:
- Preview generation: ~5-10ms per frame
- Base64 encoding: ~2-5ms per frame
- Network transmission: ~10-20ms per frame
- Total latency: ~50-100ms end-to-end
