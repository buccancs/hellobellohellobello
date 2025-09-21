# Ready-to-Adapt Diagram Examples

**Purpose**: Compact Mermaid/PlantUML code snippets for immediate use in reports and presentations.

**Usage**: Copy and paste into Markdown documents, adjust labels and styling as needed.

## System Architecture Examples

### High-Level System (Mermaid)
```mermaid
flowchart LR
  subgraph Android["Android Sensor Node"]
    UI[MainActivity] <--> Svc[RecordingService]
    Svc --> RC[RecordingController]
    RC --> RGB[RgbCameraRecorder]
    RC --> TH[ThermalRecorder]
    RC --> GSR[ShimmerRecorder]
  end

  subgraph PC["PC Controller"]
    NC[NetworkController] --> GUI[PyQt6 GUI]
    FTS[FileTransferServer] --> Storage[(Sessions)]
  end

  PC <-->|TCP Control| Android
  Android -->|File Transfer| PC
```

### Deployment View (Mermaid)
```mermaid
graph TB
  subgraph WiFi["Local Network"]
    A[Android Device<br/>192.168.1.100]
    P[PC Controller<br/>192.168.1.10]
  end
  A -.->|mDNS Discovery| P
  P -->|TCP 8080+| A
  A -->|TCP 8090| P
  P -->|UDP 9999| A
```

## Protocol Examples

### Command Sequence (PlantUML)
```plantuml
@startuml
PC -> Android: start_recording
Android -> Android: Initialize sensors
Android -> PC: ACK (success)
note right: Recording active
PC -> Android: stop_recording
Android -> Android: Finalize files
Android -> PC: ACK (complete)
@enduml
```

### State Machine (Mermaid)
```mermaid
stateDiagram-v2
  [*] --> IDLE
  IDLE --> PREPARING: startSession()
  PREPARING --> RECORDING: sensors ready
  RECORDING --> STOPPING: stopSession()
  STOPPING --> IDLE: cleanup complete
  PREPARING --> IDLE: error
```

## Data Flow Examples

### Preview Pipeline (Mermaid)
```mermaid
sequenceDiagram
  Camera->>Recorder: Capture frame
  Recorder->>Recorder: Downsample + encode
  Recorder->>PreviewBus: emit(jpeg, timestamp)
  PreviewBus->>Service: Frame event
  Service->>PC: preview_frame message
  PC->>GUI: Update display
```

### File Structure (Tree View)
```
sessions/
└── 20250818_173012_123_Device_ab12cd34/
    ├── rgb/
    │   ├── video_1692374212345678901.mp4
    │   ├── frames/frame_*.jpg
    │   └── rgb.csv
    ├── thermal/
    │   ├── thermal.csv
    │   └── metadata.json
    ├── gsr/
    │   └── gsr.csv
    └── flash_sync_events.csv
```

## Performance Examples

### Timeline Chart (Mermaid)
```mermaid
gantt
  title Recording Session Timeline
  dateFormat X
  axisFormat %M:%S

  section RGB Camera
  Video Recording: 0, 600000
  JPEG Capture: 0, 600000

  section GSR Sensor
  Data Streaming: 0, 600000

  section File Transfer
  ZIP Creation: 600000, 630000
  Network Transfer: 630000, 660000
```

### Performance Chart (Mermaid)
```mermaid
xychart-beta
  title "Transfer Rate vs File Size"
  x-axis [100, 250, 500, 1000, 2000]
  y-axis "MB/s" 0 --> 60
  line [45, 52, 58, 60, 55]
```

## Class Diagram (Mermaid)
```mermaid
classDiagram
  class RecordingController {
    +State state
    +startSession(id)
    +stopSession()
  }

  class SensorRecorder {
    <<interface>>
    +start(dir)
    +stop()
  }

  RecordingController o-- SensorRecorder
  SensorRecorder <|.. RgbCameraRecorder
  SensorRecorder <|.. ThermalRecorder
```

## Error Handling (Mermaid)
```mermaid
flowchart TD
  A[Error Detected] --> B{Critical?}
  B -->|Yes| C[Stop Recording]
  B -->|No| D[Continue with Warning]
  C --> E[Preserve Data]
  E --> F[Notify User]
  D --> G[Log Error]
  F --> G
  G --> H[Continue Operation]
```

## Network Protocol (Table Format)

| Command | Request | Response |
|---------|---------|----------|
| `query_capabilities` | `{"v":1,"type":"cmd","id":1}` | `{"v":1,"type":"ack","ack_id":1,"capabilities":{...}}` |
| `start_recording` | `{"v":1,"type":"cmd","id":2,"session_id":"..."}` | `{"v":1,"type":"ack","ack_id":2,"status":"ok"}` |
| `time_sync` | `{"v":1,"type":"cmd","id":3}` | `{"v":1,"type":"ack","ack_id":3,"t1":...,"t2":...}` |

## Quick Reference

### Styling Options (Mermaid)
```mermaid
graph LR
  A[Normal] --> B[Process]
  C[(Database)] --> D{{Decision}}
  E([Start/End]) --> F[[Subroutine]]

  classDef errorClass fill:#ff6b6b
  classDef successClass fill:#51cf66
  class A errorClass
  class B successClass
```

### PlantUML Skinparams
```plantuml
@startuml
!theme plain
skinparam backgroundColor white
skinparam handwritten false
skinparam monochrome false

participant "PC Controller" as PC
participant "Android Device" as A
PC -> A: Command
A --> PC: Response
@enduml
```

### Mermaid Configuration
```yaml
# In frontmatter or config
mermaid:
  theme: default
  themeVariables:
    primaryColor: "#00b4d8"
    primaryTextColor: "#023e8a"
    primaryBorderColor: "#0077b6"
```

## Usage Tips

1. **Copy-Paste Ready**: All examples use standard syntax
2. **Customizable**: Change labels, colors, and layout as needed
3. **Scalable**: Add or remove components easily
4. **Multiple Formats**: Both Mermaid and PlantUML versions provided
5. **Documented**: Each example includes purpose and context

## Rendering Tools

- **VS Code**: Mermaid Preview, PlantUML extensions
- **GitHub**: Native Mermaid support in markdown
- **Online**: mermaid.live, plantuml.com
- **Static Sites**: Hugo, Jekyll, Docusaurus support
- **Export**: SVG, PNG, PDF formats available
