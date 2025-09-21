# System Flows and State Management

This document details the runtime sequences, state machines, and interaction flows within the multi-modal physiological sensing platform.

## Table of Contents

1. [Core Sequence Diagrams](#core-sequence-diagrams)
2. [State Machine Definitions](#state-machine-definitions)
3. [Error and Recovery Flows](#error-and-recovery-flows)
4. [Network Protocol Flows](#network-protocol-flows)

---

## Core Sequence Diagrams

### Start Recording End-to-End Flow

```mermaid
sequenceDiagram
    participant R as Researcher
    participant PC as PC Controller
    participant RS as RecordingService
    participant MA as MainActivity
    participant RC as RecordingController
    participant RGB as RgbCameraRecorder
    participant TH as ThermalRecorder
    participant GSR as ShimmerRecorder
    participant PB as PreviewBus
    participant FS as File System

    R->>PC: Click "Start Recording"
    PC->>PC: Generate session_id
    PC->>RS: JSON cmd {v:1, type:"cmd", command:"start_recording", session_id:"20241218_143052_001"}

    RS->>RS: Parse & validate command
    RS->>MA: Broadcast ACTION_START_RECORDING

    MA->>RC: controlReceiver.onReceive()
    MA->>RC: ensureController()
    MA->>RC: startSession(session_id)

    Note over RC: State: IDLE → PREPARING

    RC->>FS: Create sessions/20241218_143052_001/
    RC->>FS: Create rgb/, thermal/, gsr/ subdirs

    par Start All Recorders
        RC->>RGB: start(sessions/20241218_143052_001/rgb)
        RC->>TH: start(sessions/20241218_143052_001/thermal)
        RC->>GSR: start(sessions/20241218_143052_001/gsr)
    end

    Note over RC: State: PREPARING → RECORDING

    RGB->>RGB: CameraX bind + VideoCapture start
    RGB->>RGB: Begin still capture loop (150ms interval)
    RGB->>FS: Create video_20241218143052123456789.mp4
    RGB->>FS: Open rgb.csv writer

    TH->>TH: Initialize Topdon SDK
    TH->>TH: Start thermal stream
    TH->>FS: Open thermal.csv writer
    TH->>FS: Create metadata.json

    GSR->>GSR: Connect BLE Shimmer device
    GSR->>GSR: Configure sampling parameters
    GSR->>FS: Open gsr.csv writer

    loop Every 150ms (while recording)
        RGB->>RGB: Capture still frame
        RGB->>RGB: Compress to JPEG
        RGB->>PB: Emit preview frame
        RGB->>FS: Save frame_<timestamp_ns>.jpg
        RGB->>FS: Write CSV row (timestamp_ns, filename)

        PB->>RS: Forward preview frame
        RS->>PC: TCP frame {type:"preview_frame", jpeg_base64:"...", ts:timestamp_ns}
    end

    loop Continuous (while recording)
        TH->>TH: Read thermal frame (256x192)
        TH->>FS: Write CSV row (timestamp_ns, w, h, v0...v49151)

        GSR->>GSR: Read GSR/PPG samples
        GSR->>FS: Write CSV row (timestamp_ns, gsr_microsiemens, ppg_raw)
    end

    RS->>PC: Ack {ack_id:1, status:"ok", message:"Recording started"}
    PC->>R: Update UI - "Recording Active"
```

### Stop Recording Flow

```mermaid
sequenceDiagram
    participant R as Researcher
    participant PC as PC Controller
    participant RS as RecordingService
    participant MA as MainActivity
    participant RC as RecordingController
    participant RGB as RgbCameraRecorder
    participant TH as ThermalRecorder
    participant GSR as ShimmerRecorder
    participant FS as File System

    R->>PC: Click "Stop Recording"
    PC->>RS: JSON cmd {v:1, type:"cmd", command:"stop_recording"}

    RS->>MA: Broadcast ACTION_STOP_RECORDING
    MA->>RC: stopSession()

    Note over RC: State: RECORDING → STOPPING

    par Stop All Recorders Gracefully
        RC->>RGB: runCatching { recorder.stop() }
        RC->>TH: runCatching { recorder.stop() }
        RC->>GSR: runCatching { recorder.stop() }
    end

    RGB->>RGB: Stop still capture loop
    RGB->>RGB: Release CameraX resources
    RGB->>FS: Close CSV writer & flush
    RGB->>FS: Stop MP4 recording

    TH->>TH: Stop thermal stream
    TH->>FS: Close CSV writer & flush
    TH->>FS: Finalize metadata.json

    GSR->>GSR: Disconnect BLE device
    GSR->>FS: Close CSV writer & flush

    Note over RC: State: STOPPING → IDLE
    RC->>RC: Reset session state
    RC->>RC: Clear currentSessionId

    RS->>PC: Ack {ack_id:2, status:"ok", message:"Recording stopped", session_id:"20241218_143052_001"}
    PC->>R: Update UI - "Recording Stopped"
```

### File Transfer Flow (Device → PC)

```mermaid
sequenceDiagram
    participant PC as PC Controller
    participant RS as RecordingService
    participant FT as FileTransferManager
    participant FS as Android FileSystem
    participant FR as PC FileReceiver
    participant PFS as PC FileSystem

    PC->>RS: JSON cmd {v:1, type:"cmd", command:"transfer_files", session_id:"20241218_143052_001"}

    RS->>FT: transferSessionToPC(session_id, pc_ip, pc_port)

    FT->>FS: Validate session directory exists
    FS-->>FT: Directory contents confirmed

    FT->>FT: Connect to PC FileReceiver socket

    Note over FT,FR: TCP Connection Established

    FT->>FR: Header JSON line: {"session_id":"20241218_143052_001","device_id":"pixel7_abc123","filename":"pixel7_abc123_data.zip"}

    FT->>FT: Initialize ZipOutputStream

    loop For each file in session directory
        FT->>FS: Read file content
        FS-->>FT: File bytes
        FT->>FT: Add to ZIP stream
        FT->>FR: Stream ZIP bytes
    end

    Note over FT: Include flash_sync_events.csv if exists
    FT->>FS: Check for flash_sync_events.csv
    alt Flash sync file exists
        FS-->>FT: Flash sync data
        FT->>FT: Add flash_sync_events.csv to ZIP
        FT->>FR: Stream additional bytes
    end

    FT->>FR: Close ZIP stream
    FT->>FT: Close TCP connection

    FR->>PFS: Write complete ZIP file
    FR->>PC: Notify transfer complete

    FT->>RS: Transfer completion status
    RS->>PC: Ack {ack_id:3, status:"ok", message:"Transfer complete", bytes_sent:1048576}
```

### Time Synchronization Flow

```mermaid
sequenceDiagram
    participant PC as PC TimeSync Service
    participant TM as Android TimeManager

    Note over PC: Start UDP time server on port 3333
    PC->>PC: Listen for sync requests

    TM->>TM: sync_with_server(server_ip, 3333)

    Note over TM: Record T1 = System.nanoTime()
    TM->>PC: UDP packet [0x01] (minimal probe)

    PC->>PC: Receive request
    PC->>PC: server_time = time.monotonic_ns()
    PC->>TM: UDP response: ASCII string "1703856123456789012"

    Note over TM: Record T2 = System.nanoTime()
    TM->>TM: Parse server_time from response
    TM->>TM: Calculate offset = (server_time + (T2-T1)/2) - T2
    TM->>TM: Store offsetNs for future timestamps

    Note over TM: getSyncedTimestamp() = System.nanoTime() + offsetNs
```

### Client Rejoin Flow

```mermaid
sequenceDiagram
    participant A as Android Device
    participant RS as RecordingService
    participant PC as PC Controller
    participant RC as RecordingController

    Note over A: Device reconnects after network interruption

    A->>PC: TCP connection established
    PC->>RS: Connection accepted

    RS->>RS: Check if recording in progress
    alt Currently recording
        RS->>RS: Get current session state
        RS->>PC: {v:1, type:"event", event:"rejoin_session", session_id:"20241218_143052_001", device_id:"pixel7_abc123", recording:true}
        PC->>PC: Update device status: "Reconnected - Recording"
        PC->>RS: Continue with existing session
    else Not recording
        RS->>PC: {v:1, type:"event", event:"rejoin_session", device_id:"pixel7_abc123", recording:false}
        PC->>PC: Update device status: "Reconnected - Idle"
    end
```

---

## State Machine Definitions

### RecordingController State Machine

```mermaid
stateDiagram-v2
    [*] --> IDLE

    IDLE --> PREPARING : startSession() / sessionId valid
    IDLE --> IDLE : startSession() / already recording (ignore)
    IDLE --> IDLE : stopSession() / not recording (ignore)

    PREPARING --> RECORDING : All recorders started successfully
    PREPARING --> IDLE : Error during startup / safeStopAll()
    PREPARING --> PREPARING : stopSession() / ignore during preparation

    RECORDING --> STOPPING : stopSession()
    RECORDING --> STOPPING : Error in any recorder / safeStopAll()
    RECORDING --> RECORDING : startSession() / ignore (already recording)

    STOPPING --> IDLE : All recorders stopped & cleanup complete
    STOPPING --> IDLE : Timeout or error / force cleanup
    STOPPING --> STOPPING : startSession() / ignore during shutdown

    state IDLE {
        [*] --> Ready
        Ready : currentSessionId = null
        Ready : sessionRootDir = null
        Ready : All recorders inactive
    }

    state PREPARING {
        [*] --> CreatingDirectories
        CreatingDirectories --> StartingRecorders : Directories created
        StartingRecorders --> ValidatingRecorders : All start() calls complete
        ValidatingRecorders --> [*] : All recorders report active

        CreatingDirectories --> [*] : Directory creation fails
        StartingRecorders --> [*] : Any recorder.start() throws
        ValidatingRecorders --> [*] : Any recorder not active
    }

    state RECORDING {
        [*] --> ActiveRecording
        ActiveRecording : currentSessionId = valid
        ActiveRecording : sessionRootDir = valid
        ActiveRecording : All recorders active
        ActiveRecording : Preview bus emitting frames
    }

    state STOPPING {
        [*] --> StoppingRecorders
        StoppingRecorders --> CleaningUp : All stop() calls complete
        CleaningUp --> [*] : State reset complete

        StoppingRecorders --> CleaningUp : Timeout (force cleanup)
    }
```

**State Descriptions:**

- **IDLE**: No active session, ready to start recording
- **PREPARING**: Session directory created, starting all sensor recorders
- **RECORDING**: All recorders active, data collection in progress
- **STOPPING**: Gracefully shutting down recorders and finalizing files

**Concurrency Rules:**
- Only one session active at a time per device
- State transitions are atomic and thread-safe via StateFlow
- Error during PREPARING or RECORDING triggers automatic cleanup via `safeStopAll()`

### PC Session Manager State Machine

```mermaid
stateDiagram-v2
    [*] --> NO_SESSION

    NO_SESSION --> CREATING : createSession()
    NO_SESSION --> NO_SESSION : All other operations (ignore)

    CREATING --> SESSION_READY : Session directory & metadata created
    CREATING --> NO_SESSION : Creation failed / cleanup

    SESSION_READY --> STARTING : startRecording()
    SESSION_READY --> NO_SESSION : cancelSession()
    SESSION_READY --> SESSION_READY : Other operations (queue or ignore)

    STARTING --> RECORDING : All devices acknowledge start
    STARTING --> SESSION_READY : Start failed / rollback
    STARTING --> STARTING : Retry on partial failure

    RECORDING --> STOPPING : stopRecording()
    RECORDING --> STOPPING : Error condition / emergency stop
    RECORDING --> RECORDING : Other operations (ignore)

    STOPPING --> TRANSFERRING : All devices stopped successfully
    STOPPING --> SESSION_READY : Stop failed / retry

    TRANSFERRING --> PROCESSING : All files received
    TRANSFERRING --> TRANSFERRING : Retry failed transfers
    TRANSFERRING --> STOPPING : Transfer timeout / manual intervention

    PROCESSING --> COMPLETE : Data validation & export finished
    PROCESSING --> TRANSFERRING : Processing failed / re-transfer

    COMPLETE --> NO_SESSION : Session archived
    COMPLETE --> COMPLETE : Export operations
```

---

## Error and Recovery Flows

### Network Disconnection Recovery

```mermaid
sequenceDiagram
    participant A as Android Device
    participant RS as RecordingService
    participant NC as NetworkClient
    participant PC as PC Controller
    participant RC as RecordingController

    Note over A,PC: Normal operation - recording in progress

    A->>X: Network interruption
    NC->>NC: Detect connection loss
    NC->>RS: Connection lost event

    RS->>RC: Continue local recording
    RC->>RC: Maintain RECORDING state

    Note over A: Device continues recording locally

    loop Reconnection attempts
        NC->>NC: Attempt reconnection
        NC->>PC: Try to establish connection
        alt Connection successful
            PC->>RS: Connection accepted
            RS->>PC: Rejoin message with current state
            PC->>PC: Update device status
        else Connection failed
            NC->>NC: Wait exponential backoff
        end
    end

    Note over A,PC: Connection restored - sync status
```

### Recorder Failure Recovery

```mermaid
stateDiagram-v2
    [*] --> NormalRecording

    NormalRecording --> RecorderFailed : Exception in any SensorRecorder

    RecorderFailed --> StoppingAll : safeStopAll() triggered
    StoppingAll --> PartialCleanup : Some recorders stopped successfully
    StoppingAll --> ForceCleanup : Timeout or multiple failures

    PartialCleanup --> SessionPreserved : Essential data saved
    ForceCleanup --> SessionCorrupted : Data integrity compromised

    SessionPreserved --> [*] : Log error, notify PC
    SessionCorrupted --> [*] : Log critical error, notify PC
```

### Storage Space Recovery

```mermaid
flowchart TD
    A[Monitor Available Storage] --> B{Space < 1GB?}
    B -->|No| A
    B -->|Yes| C[Warn User - Low Storage]
    C --> D{Space < 500MB?}
    D -->|No| A
    D -->|Yes| E[Block New Sessions]
    E --> F{Space < 100MB?}
    F -->|No| G[Allow Emergency Stop Only]
    F -->|Yes| H[Force Stop Current Recording]
    H --> I[Emergency Session Cleanup]
    I --> J[Critical Storage Alert]
    G --> A
    J --> A
```

---

## Network Protocol Flows

### mDNS Service Discovery Flow

```mermaid
sequenceDiagram
    participant A as Android Device
    participant mDNS as mDNS Service
    participant PC as PC Controller
    participant ZC as Zeroconf Browser

    A->>A: RecordingService starts
    A->>mDNS: Register service "_gsr-controller._tcp.local."
    A->>mDNS: Service name: "SensorSpoke - Pixel_7"
    A->>mDNS: Port: 8080, IP: 192.168.1.100

    PC->>PC: Start device discovery
    PC->>ZC: Browse for "_gsr-controller._tcp.local."
    ZC->>mDNS: Query for matching services

    mDNS-->>ZC: Service discovered: "SensorSpoke - Pixel_7"
    ZC-->>PC: Device found - IP: 192.168.1.100, Port: 8080

    PC->>A: TCP connection attempt to 192.168.1.100:8080
    A-->>PC: Connection accepted
    PC->>A: Query capabilities command
    A-->>PC: Device capabilities response
    PC->>PC: Add device to active list
```

### Command Processing Flow

```mermaid
flowchart TD
    A[Receive TCP Message] --> B{Valid Framing?}
    B -->|No| C[Log Error & Ignore]
    B -->|Yes| D[Parse JSON]
    D --> E{Valid JSON?}
    E -->|No| F[Send Error Ack]
    E -->|Yes| G{Has Required Fields?}
    G -->|No| F
    G -->|Yes| H[Route by Command Type]

    H --> I[query_capabilities]
    H --> J[time_sync]
    H --> K[start_recording]
    H --> L[stop_recording]
    H --> M[flash_sync]
    H --> N[transfer_files]

    I --> O[Generate Capabilities Response]
    J --> P[Process Time Sync]
    K --> Q[Execute Start Recording]
    L --> R[Execute Stop Recording]
    M --> S[Trigger Flash Sync]
    N --> T[Initiate File Transfer]

    O --> U[Send Success Ack]
    P --> U
    Q --> U
    R --> U
    S --> U
    T --> U

    F --> V[Connection Maintained]
    U --> V
    C --> V
```

This comprehensive flow documentation provides clear understanding of system interactions, state management, and error handling across all major operational scenarios.
