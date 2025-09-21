# Chapter 4: Design and Implementation Visualizations

## Figure 4.1: Detailed System Architecture

```mermaid
flowchart TD
    %% PC Controller Detail
    subgraph PC["[PC] PC Controller (Python 3.8+)"]
        %% GUI Layer
        subgraph GUI_LAYER["[DESKTOP] Presentation Layer"]
            MAIN_WIN["MainWindow (PyQt6)<br/>[UNICODE] Session controls<br/>[UNICODE] Device status display<br/>[UNICODE] Real-time monitoring<br/>[UNICODE] Settings management"]

            DEVICE_PANEL["DevicePanel<br/>[UNICODE] Connection status<br/>[UNICODE] Signal preview<br/>[UNICODE] Configuration UI<br/>[UNICODE] Error indicators"]

            DATA_VIZ["DataVisualization<br/>[UNICODE] Live sensor plots<br/>[UNICODE] Timeline display<br/>[UNICODE] Quality metrics<br/>[UNICODE] Export controls"]
        end

        %% Business Logic Layer
        subgraph BUSINESS["[UNIT] Business Logic Layer"]
            DEVICE_MGR["DeviceManager<br/>[UNICODE] Discovery coordination<br/>[UNICODE] Connection pooling<br/>[UNICODE] Health monitoring<br/>[UNICODE] Command routing"]

            SESSION_MGR["SessionManager<br/>[UNICODE] Lifecycle control<br/>[UNICODE] Data aggregation<br/>[UNICODE] Metadata management<br/>[UNICODE] Export coordination"]

            SYNC_CTRL["TimeSync Controller<br/>[UNICODE] UDP echo protocol<br/>[UNICODE] Clock offset calculation<br/>[UNICODE] Drift compensation<br/>[UNICODE] Accuracy validation"]
        end

        %% Network Layer
        subgraph NETWORK["[PROTOCOL] Network Layer"]
            NSD_CLIENT["NSDClient<br/>[UNICODE] Service discovery<br/>[UNICODE] Device enumeration<br/>[UNICODE] Address resolution<br/>[UNICODE] Service monitoring"]

            TCP_SERVER["TCPServer<br/>[UNICODE] Command/response handling<br/>[UNICODE] JSON message processing<br/>[UNICODE] Optional TLS encryption<br/>[UNICODE] Connection management"]

            FILE_SERVER["FileTransferServer<br/>[UNICODE] ZIP stream handling<br/>[UNICODE] Progress tracking<br/>[UNICODE] Integrity validation<br/>[UNICODE] Storage coordination"]
        end

        %% Data Layer
        subgraph DATA_LAYER["[DATA] Data Layer"]
            CSV_EXPORTER["CSVExporter<br/>[UNICODE] Multi-format export<br/>[UNICODE] Schema validation<br/>[UNICODE] Timestamp alignment<br/>[UNICODE] Quality metrics"]

            FILE_MGR["FileManager<br/>[UNICODE] Directory organization<br/>[UNICODE] Session archival<br/>[UNICODE] Backup coordination<br/>[UNICODE] Metadata storage"]

            CONFIG_MGR["ConfigManager<br/>[UNICODE] Settings persistence<br/>[UNICODE] Device profiles<br/>[UNICODE] Security configuration<br/>[UNICODE] Performance tuning"]
        end
    end

    %% Android Application Detail
    subgraph ANDROID["[ANDROID] Android Application (Kotlin, API 26+)"]
        %% UI Layer
        subgraph UI_LAYER["[ANDROID] UI Layer"]
            MAIN_ACTIVITY["MainActivity<br/>[UNICODE] Connection interface<br/>[UNICODE] Manual IP entry<br/>[UNICODE] Status indicators<br/>[UNICODE] Settings access"]

            RECORDING_FRAG["RecordingFragment<br/>[UNICODE] Session controls<br/>[UNICODE] Sensor status<br/>[UNICODE] Preview display<br/>[UNICODE] Error handling"]
        end

        %% Service Layer
        subgraph SERVICE_LAYER["[INTEGRATION] Service Layer"]
            RECORDING_SERVICE["RecordingService<br/>[UNICODE] Foreground operation<br/>[UNICODE] Lifecycle management<br/>[UNICODE] Notification handling<br/>[UNICODE] Resource coordination"]

            NETWORK_CLIENT["NetworkClient<br/>[UNICODE] TCP connection<br/>[UNICODE] JSON messaging<br/>[UNICODE] Auto-reconnection<br/>[UNICODE] Command processing"]
        end

        %% Recording Layer
        subgraph REC_LAYER["[UNICODE] Recording Layer"]
            REC_CONTROLLER["RecordingController<br/>[UNICODE] State management<br/>[UNICODE] Sensor coordination<br/>[UNICODE] Timeline synchronization<br/>[UNICODE] Error recovery"]

            RGB_RECORDER["RgbCameraRecorder<br/>[UNICODE] Camera2 API<br/>[UNICODE] Dual pipeline (MP4+JPEG)<br/>[UNICODE] Preview generation<br/>[UNICODE] Quality control"]

            THERMAL_RECORDER["ThermalCameraRecorder<br/>[UNICODE] Topdon SDK integration<br/>[UNICODE] Radiometric data<br/>[UNICODE] ROI processing<br/>[UNICODE] Temperature calibration"]

            GSR_RECORDER["ShimmerRecorder<br/>[UNICODE] Bluetooth integration<br/>[UNICODE] 128Hz sampling<br/>[UNICODE] Real-time streaming<br/>[UNICODE] Battery monitoring"]
        end

        %% Storage Layer
        subgraph STORAGE_LAYER["[DATA] Storage Layer"]
            SESSION_STORAGE["SessionStorage<br/>[UNICODE] File organization<br/>[UNICODE] Metadata tracking<br/>[UNICODE] Compression<br/>[UNICODE] Transfer preparation"]

            PREVIEW_BUS["PreviewBus<br/>[UNICODE] Frame throttling<br/>[UNICODE] Network streaming<br/>[UNICODE] Quality adaptation<br/>[UNICODE] Buffer management"]
        end
    end

    %% External Systems
    subgraph EXTERNAL["[SENSOR] External Systems"]
        SHIMMER["Shimmer3 GSR+<br/>[UNICODE] Bluetooth LE<br/>[UNICODE] Real-time streaming<br/>[UNICODE] Battery status<br/>[UNICODE] Configuration API"]

        TOPDON["Topdon TC001<br/>[UNICODE] USB-C interface<br/>[UNICODE] SDK integration<br/>[UNICODE] Radiometric output<br/>[UNICODE] Calibration data"]

        NETWORK_INFRA["Network Infrastructure<br/>[UNICODE] WiFi 802.11n+<br/>[UNICODE] NSD/mDNS support<br/>[UNICODE] TCP/UDP protocols<br/>[UNICODE] Optional internet"]
    end

    %% Communication Flows
    MAIN_WIN --> DEVICE_MGR
    DEVICE_MGR --> SESSION_MGR
    SESSION_MGR --> SYNC_CTRL

    NSD_CLIENT --> TCP_SERVER
    TCP_SERVER --> FILE_SERVER

    CSV_EXPORTER --> FILE_MGR
    FILE_MGR --> CONFIG_MGR

    %% Android internal flows
    MAIN_ACTIVITY --> RECORDING_SERVICE
    RECORDING_SERVICE --> REC_CONTROLLER
    REC_CONTROLLER --> RGB_RECORDER
    REC_CONTROLLER --> THERMAL_RECORDER
    REC_CONTROLLER --> GSR_RECORDER

    %% Cross-system communication
    TCP_SERVER <--> NETWORK_CLIENT
    FILE_SERVER <--> SESSION_STORAGE
    SYNC_CTRL <--> REC_CONTROLLER

    %% External connections
    GSR_RECORDER <--> SHIMMER
    THERMAL_RECORDER <--> TOPDON
    NSD_CLIENT <--> NETWORK_INFRA
    NETWORK_CLIENT <--> NETWORK_INFRA

    %% Data flow
    RGB_RECORDER --> SESSION_STORAGE
    THERMAL_RECORDER --> SESSION_STORAGE
    GSR_RECORDER --> SESSION_STORAGE
    SESSION_STORAGE --> FILE_SERVER
    PREVIEW_BUS --> TCP_SERVER

    %% Component styling
    classDef guiStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef businessStyle fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef networkStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef dataStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef androidUIStyle fill:#e0f2f1,stroke:#00796b,stroke-width:2px
    classDef androidServiceStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef androidRecStyle fill:#f1f8e9,stroke:#689f38,stroke-width:2px
    classDef androidStorageStyle fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    classDef externalStyle fill:#ffebee,stroke:#d32f2f,stroke-width:2px

    class GUI_LAYER,MAIN_WIN,DEVICE_PANEL,DATA_VIZ guiStyle
    class BUSINESS,DEVICE_MGR,SESSION_MGR,SYNC_CTRL businessStyle
    class NETWORK,NSD_CLIENT,TCP_SERVER,FILE_SERVER networkStyle
    class DATA_LAYER,CSV_EXPORTER,FILE_MGR,CONFIG_MGR dataStyle
    class UI_LAYER,MAIN_ACTIVITY,RECORDING_FRAG androidUIStyle
    class SERVICE_LAYER,RECORDING_SERVICE,NETWORK_CLIENT androidServiceStyle
    class REC_LAYER,REC_CONTROLLER,RGB_RECORDER,THERMAL_RECORDER,GSR_RECORDER androidRecStyle
    class STORAGE_LAYER,SESSION_STORAGE,PREVIEW_BUS androidStorageStyle
    class EXTERNAL,SHIMMER,TOPDON,NETWORK_INFRA externalStyle
```

## Figure 4.2: Android Application Architecture

```mermaid
flowchart TD
    %% Activity/Fragment Layer
    subgraph PRESENTATION["[ANDROID] Presentation Layer"]
        MAIN_ACT["MainActivity<br/>[UNICODE] Entry point & navigation<br/>[UNICODE] Connection setup UI<br/>[UNICODE] Manual IP configuration<br/>[UNICODE] Settings access<br/>[UNICODE] Status monitoring"]

        REC_FRAG["RecordingFragment<br/>[UNICODE] Session control interface<br/>[UNICODE] Start/stop recording<br/>[UNICODE] Sensor status display<br/>[UNICODE] Error notifications<br/>[UNICODE] Preview window"]

        SETTINGS_ACT["SettingsActivity<br/>[UNIT] Configuration UI<br/>[UNICODE] Network parameters<br/>[UNICODE] Sensor calibration<br/>[UNICODE] Storage preferences<br/>[UNICODE] Debug options"]
    end

    %% Service Layer
    subgraph SERVICE["[INTEGRATION] Service Layer (Background Operations)"]
        REC_SERVICE["RecordingService<br/>[TARGET] Foreground service<br/>[UNICODE] Lifecycle management<br/>[UNICODE] System notifications<br/>[UNICODE] Resource coordination<br/>[UNICODE] Process isolation"]

        NET_CLIENT["NetworkClient<br/>[PROTOCOL] Communication manager<br/>[UNICODE] TCP connection handling<br/>[UNICODE] JSON message protocol<br/>[UNICODE] Auto-reconnection logic<br/>[UNICODE] Command queue management"]

        DEVICE_MONITOR["DeviceMonitor<br/>[DATA] Health supervisor<br/>[UNICODE] Connection monitoring<br/>[UNICODE] Performance tracking<br/>[UNICODE] Error detection<br/>[UNICODE] Recovery coordination"]
    end

    %% Controller Layer
    subgraph CONTROLLER["[UNIT] Controller Layer (Business Logic)"]
        REC_CTRL["RecordingController<br/>[UNICODE] Central orchestrator<br/>[UNICODE] State machine management<br/>[UNICODE] Sensor coordination<br/>[UNICODE] Timeline synchronization<br/>[UNICODE] Error recovery"]

        subgraph STATE_MACHINE["[LIST] State Management"]
            IDLE["IDLE<br/>Waiting for commands"]
            PREPARING["PREPARING<br/>Initializing sensors"]
            RECORDING["RECORDING<br/>Active data capture"]
            STOPPING["STOPPING<br/>Cleanup & finalization"]
            ERROR["ERROR<br/>Fault recovery"]

            IDLE --> PREPARING
            PREPARING --> RECORDING
            RECORDING --> STOPPING
            STOPPING --> IDLE
            PREPARING --> ERROR
            RECORDING --> ERROR
            ERROR --> IDLE
        end
    end

    %% Sensor Recorder Layer
    subgraph RECORDERS["[UNICODE] Sensor Recorder Layer"]
        RGB_REC["RgbCameraRecorder<br/>[UNICODE] Camera management<br/>[UNICODE] Camera2 API integration<br/>[UNICODE] Dual pipeline (MP4 + JPEG)<br/>[UNICODE] Preview frame generation<br/>[UNICODE] Quality control & settings"]

        THERMAL_REC["ThermalCameraRecorder<br/>[THERMAL] Thermal imaging<br/>[UNICODE] Topdon SDK integration<br/>[UNICODE] Radiometric data capture<br/>[UNICODE] ROI temperature tracking<br/>[UNICODE] Calibration management"]

        SHIMMER_REC["ShimmerRecorder<br/>[SIGNAL] GSR sensor interface<br/>[UNICODE] Bluetooth LE communication<br/>[UNICODE] 128Hz data sampling<br/>[UNICODE] Real-time streaming<br/>[UNICODE] Battery monitoring"]

        subgraph RECORDER_INTERFACE["[LIST] Common Interface"]
            INIT["initialize(): Boolean<br/>Setup sensor connection"]
            START["startRecording(sessionDir): Boolean<br/>Begin data capture"]
            STOP["stopRecording(): Boolean<br/>End capture & cleanup"]
            STATUS["getStatus(): RecorderStatus<br/>Current state information"]
        end

        RGB_REC -.-> RECORDER_INTERFACE
        THERMAL_REC -.-> RECORDER_INTERFACE
        SHIMMER_REC -.-> RECORDER_INTERFACE
    end

    %% Data Management Layer
    subgraph DATA_MGMT["[DATA] Data Management Layer"]
        SESSION_STORAGE["SessionStorage<br/>[UNICODE] File organization<br/>[UNICODE] Directory structure creation<br/>[UNICODE] Metadata file generation<br/>[UNICODE] Data compression<br/>[UNICODE] Transfer preparation"]

        PREVIEW_BUS["PreviewBus<br/>[UNICODE] Real-time streaming<br/>[UNICODE] Frame throttling (6-8 FPS)<br/>[UNICODE] Network transmission<br/>[UNICODE] Quality adaptation<br/>[UNICODE] Buffer management"]

        METADATA_MGR["MetadataManager<br/>[DATA] Session information<br/>[UNICODE] Timestamp coordination<br/>[UNICODE] Device configuration<br/>[UNICODE] Quality metrics<br/>[UNICODE] Export preparation"]
    end

    %% External Interface Layer
    subgraph EXTERNAL_IF["[SENSOR] External Interface Layer"]
        BT_MANAGER["BluetoothManager<br/>[NETWORK] BLE communication<br/>[UNICODE] Device discovery<br/>[UNICODE] Pairing management<br/>[UNICODE] Connection stability<br/>[UNICODE] Data streaming"]

        USB_CONTROLLER["USBController<br/>[SENSOR] USB device handling<br/>[UNICODE] Permission management<br/>[UNICODE] Device enumeration<br/>[UNICODE] Data transfer<br/>[UNICODE] Hotplug detection"]

        NETWORK_IF["NetworkInterface<br/>[PROTOCOL] Network abstraction<br/>[UNICODE] WiFi management<br/>[UNICODE] Connection monitoring<br/>[UNICODE] Protocol handling<br/>[UNICODE] Error recovery"]
    end

    %% Connections and Data Flow
    MAIN_ACT --> REC_FRAG
    MAIN_ACT --> SETTINGS_ACT
    REC_FRAG --> REC_SERVICE

    REC_SERVICE --> REC_CTRL
    REC_SERVICE --> NET_CLIENT
    REC_SERVICE --> DEVICE_MONITOR

    REC_CTRL --> STATE_MACHINE
    REC_CTRL --> RGB_REC
    REC_CTRL --> THERMAL_REC
    REC_CTRL --> SHIMMER_REC

    RGB_REC --> SESSION_STORAGE
    THERMAL_REC --> SESSION_STORAGE
    SHIMMER_REC --> SESSION_STORAGE

    RGB_REC --> PREVIEW_BUS
    THERMAL_REC --> PREVIEW_BUS

    SESSION_STORAGE --> METADATA_MGR
    PREVIEW_BUS --> NET_CLIENT

    SHIMMER_REC --> BT_MANAGER
    THERMAL_REC --> USB_CONTROLLER
    NET_CLIENT --> NETWORK_IF

    %% Architecture Principles
    PRINCIPLES["[ARCH] Architecture Principles<br/>[UNICODE] Layered separation of concerns<br/>[UNICODE] Dependency injection for testing<br/>[UNICODE] Observer pattern for state updates<br/>[UNICODE] Command pattern for operations<br/>[UNICODE] Strategy pattern for sensor types"]

    %% Threading Model
    THREADING["[UNICODE] Threading Model<br/>[UNICODE] Main UI thread (presentation)<br/>[UNICODE] Background service thread<br/>[UNICODE] Network I/O thread pool<br/>[UNICODE] Sensor data collection threads<br/>[UNICODE] File I/O worker thread"]

    CONTROLLER --> PRINCIPLES
    DATA_MGMT --> THREADING

    %% Component Styling
    classDef presentationStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef serviceStyle fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef controllerStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef recorderStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef dataStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef externalStyle fill:#e0f2f1,stroke:#00796b,stroke-width:2px
    classDef stateStyle fill:#f1f8e9,stroke:#689f38,stroke-width:1px
    classDef interfaceStyle fill:#e8eaf6,stroke:#3f51b5,stroke-width:1px,stroke-dasharray: 5 5
    classDef principleStyle fill:#ffebee,stroke:#d32f2f,stroke-width:2px,stroke-dasharray: 10 5

    class PRESENTATION,MAIN_ACT,REC_FRAG,SETTINGS_ACT presentationStyle
    class SERVICE,REC_SERVICE,NET_CLIENT,DEVICE_MONITOR serviceStyle
    class CONTROLLER,REC_CTRL controllerStyle
    class RECORDERS,RGB_REC,THERMAL_REC,SHIMMER_REC recorderStyle
    class DATA_MGMT,SESSION_STORAGE,PREVIEW_BUS,METADATA_MGR dataStyle
    class EXTERNAL_IF,BT_MANAGER,USB_CONTROLLER,NETWORK_IF externalStyle
    class STATE_MACHINE,IDLE,PREPARING,RECORDING,STOPPING,ERROR stateStyle
    class RECORDER_INTERFACE,INIT,START,STOP,STATUS interfaceStyle
    class PRINCIPLES,THREADING principleStyle
```

## Figure 4.3: PC Controller Threading Model (Intended Design)

```mermaid
flowchart TD
    %% Main Thread
    subgraph MAIN["[DESKTOP] Main UI Thread (PyQt6)"]
        EVENT_LOOP["Qt Event Loop<br/>[UNICODE] GUI event processing<br/>[UNICODE] User interactions<br/>[UNICODE] Timer events<br/>[UNICODE] Signal/slot connections"]

        UI_COMPONENTS["UI Components<br/>[UNICODE] MainWindow updates<br/>[UNICODE] Device status display<br/>[UNICODE] Real-time charts<br/>[UNICODE] User input handling"]

        UI_CONTROLLERS["UI Controllers<br/>[UNICODE] Session control logic<br/>[UNICODE] Settings management<br/>[UNICODE] Error dialog display<br/>[UNICODE] Progress indicators"]
    end

    %% Worker Threads
    subgraph WORKERS["[UNICODE] Worker Threads (QThread)"]

        subgraph NETWORK_WORKER["[PROTOCOL] Network Worker"]
            TCP_THREAD["TCP Server Thread<br/>[UNICODE] Command handling<br/>[UNICODE] Connection management<br/>[UNICODE] JSON processing<br/>[UNICODE] Response generation"]

            UDP_THREAD["UDP Sync Thread<br/>[UNICODE] Time sync protocol<br/>[UNICODE] Clock offset calculation<br/>[UNICODE] Drift monitoring<br/>[UNICODE] Accuracy validation"]

            FILE_THREAD["File Transfer Thread<br/>[UNICODE] ZIP stream processing<br/>[UNICODE] Progress tracking<br/>[UNICODE] Data validation<br/>[UNICODE] Storage coordination"]
        end

        subgraph DATA_WORKER["[DATA] Data Processing Worker"]
            STREAM_PROC["Stream Processor<br/>[UNICODE] Real-time data parsing<br/>[UNICODE] Quality validation<br/>[UNICODE] Buffer management<br/>[UNICODE] Preview generation"]

            EXPORT_PROC["Export Processor<br/>[UNICODE] CSV generation<br/>[UNICODE] Format conversion<br/>[UNICODE] Metadata aggregation<br/>[UNICODE] Archive creation"]
        end

        subgraph DEVICE_WORKER["[SENSOR] Device Management Worker"]
            DISCOVERY["Device Discovery<br/>[UNICODE] NSD scanning<br/>[UNICODE] Service enumeration<br/>[UNICODE] Address resolution<br/>[UNICODE] Availability monitoring"]

            HEALTH_MON["Health Monitor<br/>[UNICODE] Connection testing<br/>[UNICODE] Performance tracking<br/>[UNICODE] Error detection<br/>[UNICODE] Recovery coordination"]
        end
    end

    %% Thread Pool
    subgraph THREAD_POOL["[UNICODE] Thread Pool (QThreadPool)"]
        IO_TASKS["I/O Tasks<br/>[UNICODE] File operations<br/>[UNICODE] Database queries<br/>[UNICODE] Configuration loading<br/>[UNICODE] Log writing"]

        COMPUTE_TASKS["Compute Tasks<br/>[UNICODE] Data analysis<br/>[UNICODE] Statistical calculations<br/>[UNICODE] Image processing<br/>[UNICODE] Compression operations"]
    end

    %% Signal/Slot Communication
    subgraph COMMUNICATION["[NETWORK] Signal/Slot Communication"]

        subgraph SIGNALS["[UNICODE] Custom Signals"]
            DEVICE_SIGNALS["Device Signals<br/>[UNICODE] deviceConnected(info)<br/>[UNICODE] deviceDisconnected(id)<br/>[UNICODE] deviceError(error)<br/>[UNICODE] deviceStatusChanged(status)"]

            DATA_SIGNALS["Data Signals<br/>[UNICODE] dataReceived(stream)<br/>[UNICODE] sessionStarted(id)<br/>[UNICODE] sessionStopped(id)<br/>[UNICODE] exportCompleted(path)"]

            ERROR_SIGNALS["Error Signals<br/>[UNICODE] networkError(msg)<br/>[UNICODE] storageError(msg)<br/>[UNICODE] syncError(msg)<br/>[UNICODE] recoveryRequired(type)"]
        end

        subgraph SLOTS["[UNICODE] UI Slot Handlers"]
            UPDATE_SLOTS["Update Slots<br/>[UNICODE] updateDeviceStatus()<br/>[UNICODE] updateDataView()<br/>[UNICODE] showErrorMessage()<br/>[UNICODE] refreshDisplay()"]

            CONTROL_SLOTS["Control Slots<br/>[UNICODE] startSession()<br/>[UNICODE] stopSession()<br/>[UNICODE] connectDevice()<br/>[UNICODE] exportData()"]
        end
    end

    %% Thread Communication Rules
    subgraph RULES["[LIST] Threading Rules & Best Practices"]
        RULE1["[FAIL] NEVER: Direct UI updates from worker threads<br/>Use signals/slots instead"]

        RULE2["[OK] ALWAYS: Move heavy operations to workers<br/>Keep UI thread responsive"]

        RULE3["[INTEGRATION] PATTERN: Worker emits signal -> UI slot updates<br/>Thread-safe communication"]

        RULE4["[SECURITY] SAFETY: Use QMutex for shared data<br/>Protect critical sections"]

        RULE5["[SIGNAL] PERFORMANCE: Use QThreadPool for short tasks<br/>QThread for long-running operations"]
    end

    %% Problem Areas (Current Implementation Issues)
    subgraph PROBLEMS["[WARNING] Current Implementation Issues"]
        BLOCKING_UI["[UNICODE] Blocking UI Operations<br/>[UNICODE] DeviceManager.scan_network()<br/>[UNICODE] Synchronous file operations<br/>[UNICODE] Direct database queries<br/>[UNICODE] Network timeouts"]

        THREAD_MIXING["[UNICODE] Thread Safety Issues<br/>[UNICODE] GUI updates from workers<br/>[UNICODE] Shared state access<br/>[UNICODE] Race conditions<br/>[UNICODE] Deadlock potential"]

        POOR_ERROR["[UNICODE] Error Handling<br/>[UNICODE] Unhandled worker exceptions<br/>[UNICODE] UI freezing on errors<br/>[UNICODE] Resource leaks<br/>[UNICODE] Recovery failures"]
    end

    %% Connections
    EVENT_LOOP --> UI_COMPONENTS
    UI_COMPONENTS --> UI_CONTROLLERS

    %% Worker to Main communication
    TCP_THREAD --> DEVICE_SIGNALS
    UDP_THREAD --> DATA_SIGNALS
    FILE_THREAD --> DATA_SIGNALS
    STREAM_PROC --> DATA_SIGNALS
    DISCOVERY --> DEVICE_SIGNALS
    HEALTH_MON --> ERROR_SIGNALS

    %% Signals to Slots
    DEVICE_SIGNALS --> UPDATE_SLOTS
    DATA_SIGNALS --> UPDATE_SLOTS
    ERROR_SIGNALS --> UPDATE_SLOTS

    UI_CONTROLLERS --> CONTROL_SLOTS
    CONTROL_SLOTS --> TCP_THREAD
    CONTROL_SLOTS --> DISCOVERY

    %% Thread Pool usage
    EXPORT_PROC --> IO_TASKS
    STREAM_PROC --> COMPUTE_TASKS

    %% Problem indicators
    BLOCKING_UI -.->|Causes| UI_COMPONENTS
    THREAD_MIXING -.->|Affects| COMMUNICATION
    POOR_ERROR -.->|Impacts| ERROR_SIGNALS

    %% Styling
    classDef mainStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef workerStyle fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef poolStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef commStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef signalStyle fill:#e0f2f1,stroke:#00796b,stroke-width:2px
    classDef slotStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef ruleStyle fill:#f1f8e9,stroke:#689f38,stroke-width:2px
    classDef problemStyle fill:#ffebee,stroke:#d32f2f,stroke-width:2px,stroke-dasharray: 5 5

    class MAIN,EVENT_LOOP,UI_COMPONENTS,UI_CONTROLLERS mainStyle
    class WORKERS,NETWORK_WORKER,DATA_WORKER,DEVICE_WORKER,TCP_THREAD,UDP_THREAD,FILE_THREAD,STREAM_PROC,EXPORT_PROC,DISCOVERY,HEALTH_MON workerStyle
    class THREAD_POOL,IO_TASKS,COMPUTE_TASKS poolStyle
    class COMMUNICATION commStyle
    class SIGNALS,DEVICE_SIGNALS,DATA_SIGNALS,ERROR_SIGNALS signalStyle
    class SLOTS,UPDATE_SLOTS,CONTROL_SLOTS slotStyle
    class RULES,RULE1,RULE2,RULE3,RULE4,RULE5 ruleStyle
    class PROBLEMS,BLOCKING_UI,THREAD_MIXING,POOR_ERROR problemStyle
```

**Key Implementation Notes:**
- **Current Issue**: `DeviceManager.scan_network()` runs on main UI thread causing freezing
- **Solution**: Move to `DISCOVERY` worker thread, emit `deviceFound` signals
- **Pattern**: All network I/O, file operations, and computationally expensive tasks must run on worker threads
- **Communication**: Workers never directly update UI - only through Qt's signal/slot mechanism
