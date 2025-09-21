# Chapter 3: Requirements and Analysis Visualizations

## Figure 3.1: High-Level System Architecture

```mermaid
flowchart TD
    %% Research Context
    RESEARCHER["[UNICODE][TEST] Researcher<br/>Controls experiments<br/>Reviews synchronized data"]

    %% Core System Components
    subgraph SYSTEM["[TEST] Multi-Modal Recording Platform"]

        %% PC Controller (Hub)
        subgraph PC["[PC] PC Controller (Central Hub)"]
            GUI["[DESKTOP] PyQt6 Desktop GUI<br/>[UNICODE] Session management<br/>[UNICODE] Device discovery<br/>[UNICODE] Real-time monitoring<br/>[UNICODE] Data visualization"]

            CONTROL["[UNIT] Control Services<br/>[UNICODE] NetworkClient (TCP/JSON)<br/>[UNICODE] FileTransferServer<br/>[UNICODE] TimeSync (NTP/UDP)<br/>[UNICODE] DeviceManager"]

            DATA_MGR["[DATA] Data Management<br/>[UNICODE] SessionManager<br/>[UNICODE] CSV export/import<br/>[UNICODE] Metadata tracking<br/>[UNICODE] Quality validation"]

            STORAGE["[DATA] Data Storage<br/>[UNICODE] Local session folders<br/>[UNICODE] Synchronized timestamps<br/>[UNICODE] Multi-format export<br/>[UNICODE] Backup integration"]
        end

        %% Android Sensor Nodes
        subgraph ANDROID["[ANDROID] Android Sensor Nodes (Distributed)"]

            subgraph NODE1["[ANDROID] Node 1: RGB + Thermal"]
                MAIN1["MainActivity<br/>Connection + UI"]
                SERVICE1["RecordingService<br/>Background operation"]
                RGB_REC["RgbCameraRecorder<br/>30fps + preview"]
                THERMAL_REC["ThermalCameraRecorder<br/>Topdon TC001 integration"]
            end

            subgraph NODE2["[ANDROID] Node 2: Additional Angles"]
                MAIN2["MainActivity<br/>Multi-angle setup"]
                SERVICE2["RecordingService<br/>Coordinated recording"]
                RGB_REC2["RgbCameraRecorder<br/>Secondary viewpoint"]
            end
        end

        %% External Sensors
        subgraph SENSORS["[SIGNAL] External Sensors"]
            GSR["[SENSOR] Shimmer3 GSR+<br/>[UNICODE] Bluetooth connection<br/>[UNICODE] 128Hz sampling<br/>[UNICODE] Real-time streaming<br/>[UNICODE] Battery monitoring"]

            ADDITIONAL["[SENSOR] Future Sensors<br/>[UNICODE] Heart rate monitors<br/>[UNICODE] Accelerometers<br/>[UNICODE] Environmental sensors<br/>[UNICODE] Custom devices"]
        end
    end

    %% Network Architecture
    subgraph NETWORK["[PROTOCOL] Communication Architecture"]
        DISCOVERY["[NETWORK] Service Discovery<br/>[UNICODE] NSD (Network Service Discovery)<br/>[UNICODE] _gsr-controller._tcp<br/>[UNICODE] Automatic device finding<br/>[UNICODE] Dynamic IP handling"]

        CONTROL_PROTO["[INTEGRATION] Control Protocol<br/>[UNICODE] TCP JSON messages<br/>[UNICODE] Command/response pairs<br/>[UNICODE] Optional TLS encryption<br/>[UNICODE] Error handling"]

        SYNC_PROTO["[TIME] Time Synchronization<br/>[UNICODE] UDP echo protocol<br/>[UNICODE] Cross-device alignment<br/>[UNICODE] +/-3.2ms accuracy<br/>[UNICODE] Drift compensation"]

        FILE_PROTO["[UNICODE] File Transfer<br/>[UNICODE] TCP ZIP streaming<br/>[UNICODE] Session data upload<br/>[UNICODE] Progress monitoring<br/>[UNICODE] Integrity validation"]
    end

    %% Data Flow
    subgraph DATAFLOW["[PERFORMANCE] Synchronized Data Streams"]
        TIMELINE["[UNICODE] Master Timeline<br/>Hardware timestamps<br/>Cross-device synchronization"]

        RGB_DATA["[UNICODE] RGB Video<br/>[UNICODE] 30fps MP4 recording<br/>[UNICODE] JPEG preview frames<br/>[UNICODE] Facial analysis ready"]

        THERMAL_DATA["[THERMAL] Thermal Data<br/>[UNICODE] 25Hz radiometric<br/>[UNICODE] ROI temperature tracking<br/>[UNICODE] CSV + thermal video"]

        GSR_DATA["[SIGNAL] GSR Signal<br/>[UNICODE] 128Hz skin conductance<br/>[UNICODE] Phasic/tonic components<br/>[UNICODE] Event markers"]

        TIMELINE --> RGB_DATA
        TIMELINE --> THERMAL_DATA
        TIMELINE --> GSR_DATA
    end

    %% Connections
    RESEARCHER --> GUI
    GUI --> CONTROL
    CONTROL --> DATA_MGR
    DATA_MGR --> STORAGE

    %% Network connections
    PC --> DISCOVERY
    ANDROID --> DISCOVERY
    DISCOVERY --> CONTROL_PROTO
    CONTROL_PROTO --> SYNC_PROTO
    SYNC_PROTO --> FILE_PROTO

    %% Sensor connections
    GSR -.->|Bluetooth| NODE1
    RGB_REC --> RGB_DATA
    THERMAL_REC --> THERMAL_DATA
    GSR --> GSR_DATA

    %% Data to PC
    RGB_DATA --> STORAGE
    THERMAL_DATA --> STORAGE
    GSR_DATA --> STORAGE

    %% System Requirements Callouts
    RELIABILITY["[SECURITY] Reliability Requirements<br/>[UNICODE] 99% uptime during sessions<br/>[UNICODE] Automatic error recovery<br/>[UNICODE] Data integrity validation<br/>[UNICODE] Graceful degradation"]

    PERFORMANCE["[PERFORMANCE] Performance Requirements<br/>[UNICODE] <50ms command latency<br/>[UNICODE] Concurrent multi-device<br/>[UNICODE] 8+ hour operation<br/>[UNICODE] Minimal resource usage"]

    USABILITY["[UNICODE] Usability Requirements<br/>[UNICODE] One-click session start<br/>[UNICODE] Automatic device discovery<br/>[UNICODE] Real-time status feedback<br/>[UNICODE] Intuitive error messages"]

    SYSTEM --> RELIABILITY
    SYSTEM --> PERFORMANCE
    SYSTEM --> USABILITY

    %% Styling
    classDef pcStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef androidStyle fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef sensorStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef networkStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef dataStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef reqStyle fill:#f1f8e9,stroke:#689f38,stroke-width:2px,stroke-dasharray: 5 5

    class PC,GUI,CONTROL,DATA_MGR,STORAGE pcStyle
    class ANDROID,NODE1,NODE2,MAIN1,MAIN2,SERVICE1,SERVICE2,RGB_REC,RGB_REC2,THERMAL_REC androidStyle
    class SENSORS,GSR,ADDITIONAL sensorStyle
    class NETWORK,DISCOVERY,CONTROL_PROTO,SYNC_PROTO,FILE_PROTO networkStyle
    class DATAFLOW,TIMELINE,RGB_DATA,THERMAL_DATA,GSR_DATA dataStyle
    class RELIABILITY,PERFORMANCE,USABILITY reqStyle
```

## Figure 3.2: UML Use Case Diagram

```mermaid
graph TD
    %% Actors
    RESEARCHER["[UNICODE][TEST] Researcher<br/>(Primary User)"]
    PARTICIPANT["[UNICODE] Research Participant<br/>(Data Subject)"]
    ADMIN["[CONFIG] System Administrator<br/>(Technical Support)"]

    %% System Boundary
    subgraph PLATFORM["[TEST] Multi-Modal Recording Platform"]

        %% Primary Use Cases
        UC1["[LIST] Plan Recording Session<br/>[UNICODE] Define session parameters<br/>[UNICODE] Select sensor modalities<br/>[UNICODE] Configure devices<br/>[UNICODE] Set experiment protocols"]

        UC2["[ANALYSIS] Discover and Connect Devices<br/>[UNICODE] Auto-detect Android nodes<br/>[UNICODE] Pair GSR sensor via Bluetooth<br/>[UNICODE] Verify device capabilities<br/>[UNICODE] Test connections"]

        UC3["[TIME] Synchronize Device Clocks<br/>[UNICODE] Establish master timeline<br/>[UNICODE] Align timestamps<br/>[UNICODE] Compensate for drift<br/>[UNICODE] Validate synchronization"]

        UC4["[UNICODE] Conduct Recording Session<br/>[UNICODE] Start synchronized recording<br/>[UNICODE] Monitor real-time status<br/>[UNICODE] Handle interruptions<br/>[UNICODE] Stop and save data"]

        UC5["[DATA] Monitor Data Quality<br/>[UNICODE] View live sensor streams<br/>[UNICODE] Check signal quality<br/>[UNICODE] Detect anomalies<br/>[UNICODE] Adjust parameters"]

        UC6["[INTEGRATION] Transfer and Process Data<br/>[UNICODE] Download session files<br/>[UNICODE] Validate data integrity<br/>[UNICODE] Export to analysis formats<br/>[UNICODE] Generate reports"]

        UC7["[CONFIG] Calibrate Sensors<br/>[UNICODE] Thermal camera calibration<br/>[UNICODE] GSR baseline measurement<br/>[UNICODE] RGB camera settings<br/>[UNICODE] Environmental compensation"]

        UC8["[UNIT] Configure System Settings<br/>[UNICODE] Network parameters<br/>[UNICODE] Security settings<br/>[UNICODE] Data storage paths<br/>[UNICODE] Performance optimization"]

        %% Secondary Use Cases
        UC9["[PERFORMANCE] Analyze Session Data<br/>[UNICODE] Load recorded sessions<br/>[UNICODE] Visualize multi-modal data<br/>[UNICODE] Export analysis results<br/>[UNICODE] Generate research reports"]

        UC10["[SECURITY] Manage System Security<br/>[UNICODE] Configure TLS encryption<br/>[UNICODE] Manage certificates<br/>[UNICODE] Set access permissions<br/>[UNICODE] Audit security logs"]

        UC11["[ANDROID] Operate Android Interface<br/>[UNICODE] Connect to PC controller<br/>[UNICODE] Start local recording<br/>[UNICODE] Monitor sensor status<br/>[UNICODE] Handle error conditions"]
    end

    %% Actor-Use Case Relationships
    RESEARCHER --> UC1
    RESEARCHER --> UC2
    RESEARCHER --> UC3
    RESEARCHER --> UC4
    RESEARCHER --> UC5
    RESEARCHER --> UC6
    RESEARCHER --> UC7
    RESEARCHER --> UC9

    ADMIN --> UC8
    ADMIN --> UC10

    PARTICIPANT --> UC11

    %% Use Case Dependencies (includes, extends)
    UC2 -.->|<<includes>>| UC3
    UC4 -.->|<<includes>>| UC2
    UC4 -.->|<<includes>>| UC5
    UC6 -.->|<<includes>>| UC4
    UC7 -.->|<<extends>>| UC2
    UC9 -.->|<<includes>>| UC6

    %% Error Handling Extensions
    ERROR1["[WARNING] Handle Connection Failures<br/>[UNICODE] Network timeouts<br/>[UNICODE] Device disconnections<br/>[UNICODE] Recovery procedures<br/>[UNICODE] User notifications"]

    ERROR2["[UNICODE] Manage Recording Errors<br/>[UNICODE] Sensor malfunctions<br/>[UNICODE] Storage failures<br/>[UNICODE] Synchronization loss<br/>[UNICODE] Graceful degradation"]

    UC2 -.->|<<extends>>| ERROR1
    UC4 -.->|<<extends>>| ERROR2

    %% Styling
    classDef actorStyle fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef primaryUCStyle fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef secondaryUCStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef errorUCStyle fill:#ffebee,stroke:#d32f2f,stroke-width:2px

    class RESEARCHER,PARTICIPANT,ADMIN actorStyle
    class UC1,UC2,UC3,UC4,UC5,UC6,UC7 primaryUCStyle
    class UC8,UC9,UC10,UC11 secondaryUCStyle
    class ERROR1,ERROR2 errorUCStyle
```

## Table 3.1: Summary of Functional Requirements

| ID | Requirement | Priority | Complexity | Dependencies |
|----|-------------|----------|------------|--------------|
| **FR1** | **Multi-Device Recording Coordination** | Critical | High | Network connectivity |
| FR1.1 | Discover Android devices via NSD | High | Medium | WiFi network |
| FR1.2 | Synchronize recording start/stop commands | Critical | High | Time synchronization |
| FR1.3 | Handle device disconnections gracefully | High | High | Error recovery |
| **FR2** | **Time Synchronization** | Critical | High | UDP protocol support |
| FR2.1 | Align timestamps across devices (+/-5ms) | Critical | High | Network latency compensation |
| FR2.2 | Detect and compensate clock drift | Medium | Medium | Continuous monitoring |
| FR2.3 | Validate synchronization accuracy | High | Medium | Statistical analysis |
| **FR3** | **Sensor Integration** | Critical | High | Hardware compatibility |
| FR3.1 | GSR data acquisition via Bluetooth | Critical | High | Shimmer SDK |
| FR3.2 | Thermal camera integration (USB-C) | High | Medium | Topdon SDK |
| FR3.3 | RGB video recording (30fps) | High | Low | Standard Android APIs |
| **FR4** | **Data Management** | High | Medium | File system access |
| FR4.1 | Session-based data organization | High | Medium | Directory management |
| FR4.2 | Multi-format export (CSV, JSON, MP4) | Medium | Low | File format libraries |
| FR4.3 | Data integrity validation | High | Medium | Checksum algorithms |
| **FR5** | **Real-Time Monitoring** | Medium | Medium | Network streaming |
| FR5.1 | Live sensor data preview | Medium | Medium | Data compression |
| FR5.2 | Device status monitoring | High | Low | Heartbeat protocol |
| FR5.3 | Error notification system | High | Medium | Event management |
| **FR6** | **User Interface** | Medium | Medium | GUI framework |
| FR6.1 | Session control interface | High | Low | PyQt6 implementation |
| FR6.2 | Device management dashboard | Medium | Medium | Real-time updates |
| FR6.3 | Data visualization tools | Low | High | Plotting libraries |
| **FR7** | **Security** | Medium | High | Cryptographic libraries |
| FR7.1 | Optional TLS encryption | Medium | High | Certificate management |
| FR7.2 | Device authentication | Medium | Medium | Token-based auth |
| FR7.3 | Data access control | Low | Medium | User management |

## Table 3.2: Summary of Non-Functional Requirements

| ID | Requirement | Target Value | Measurement Method | Priority |
|----|-------------|--------------|-------------------|----------|
| **NFR1** | **Performance** | | | |
| NFR1.1 | Command response latency | <50ms | Network latency tests | High |
| NFR1.2 | Concurrent device support | 8+ devices | Load testing | Medium |
| NFR1.3 | Session duration capacity | 8+ hours | Endurance testing | High |
| NFR1.4 | Memory usage (PC) | <2GB during operation | Resource monitoring | Medium |
| NFR1.5 | CPU utilization (Android) | <30% average | Performance profiling | Medium |
| **NFR2** | **Reliability** | | | |
| NFR2.1 | System uptime | 99% during sessions | Failure tracking | Critical |
| NFR2.2 | Data loss prevention | 0% acceptable loss | Data validation | Critical |
| NFR2.3 | Error recovery time | <30 seconds | Recovery testing | High |
| NFR2.4 | Network fault tolerance | Auto-reconnect | Connection testing | High |
| **NFR3** | **Usability** | | | |
| NFR3.1 | Setup time (experienced user) | <5 minutes | User testing | Medium |
| NFR3.2 | Setup time (new user) | <15 minutes | User testing | Low |
| NFR3.3 | Error message clarity | 90% user comprehension | Usability testing | Medium |
| NFR3.4 | Interface responsiveness | <200ms UI updates | Performance testing | Medium |
| **NFR4** | **Accuracy** | | | |
| NFR4.1 | Time synchronization | +/-3.2ms median | Statistical analysis | Critical |
| NFR4.2 | GSR sampling accuracy | 128Hz +/-1% | Signal validation | High |
| NFR4.3 | Thermal measurement | +/-2degC or +/-2% | Calibration testing | Medium |
| NFR4.4 | Video frame timing | +/-33ms (30fps) | Frame analysis | Medium |
| **NFR5** | **Compatibility** | | | |
| NFR5.1 | Android version support | Android 8.0+ (API 26+) | Device testing | High |
| NFR5.2 | PC platform support | Windows 10+, Linux, macOS | Cross-platform testing | Medium |
| NFR5.3 | Network requirements | WiFi 802.11n minimum | Network testing | Medium |
| NFR5.4 | Hardware requirements | 4GB RAM, 100GB storage | System testing | Medium |

## Caption Information

**Figure 3.1**: High-level system architecture showing the distributed hub-and-spoke model with PC controller managing multiple Android sensor nodes and external sensors. The architecture emphasizes synchronized data collection through standardized communication protocols and centralized data management.

**Figure 3.2**: UML use case diagram illustrating researcher interactions with the multi-modal recording platform. Primary use cases cover the complete research workflow from session planning through data analysis, with error handling extensions ensuring robust operation.

**Table 3.1**: Comprehensive functional requirements organized by system component, showing critical path dependencies and implementation complexity. Requirements prioritization ensures core recording capabilities are delivered before enhanced features.

**Table 3.2**: Non-functional requirements with quantitative targets and measurement methods. Performance and reliability requirements ensure research-grade data quality, while usability requirements support efficient experimental workflows.

**Thesis Placement**:
- Chapter 3, Section 3.1 (System Architecture Overview)
- Chapter 3, Section 3.2 (Use Case Analysis)
- Chapter 3, Section 3.3 (Functional Requirements)
- Chapter 3, Section 3.4 (Non-Functional Requirements)
