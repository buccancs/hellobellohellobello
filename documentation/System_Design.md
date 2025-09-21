# System Design - Multi-Modal Physiological Sensing Platform

This document provides comprehensive system design details for the multi-modal physiological sensing platform, consolidating high-level architecture, detailed component specifications, and implementation approach.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [C4 Architectural Views](#c4-architectural-views)
3. [Detailed Component Design](#detailed-component-design)
4. [Communication Protocol](#communication-protocol)
5. [Data Flow and Management](#data-flow-and-management)
6. [Deployment Architecture](#deployment-architecture)
7. [Implementation Strategy](#implementation-strategy)

---

## High-Level Architecture

### Hub-and-Spoke Model

The system employs a **Hub-and-Spoke architecture** with a central PC acting as the "Hub" and multiple Android devices functioning as mobile sensor "Spokes."

**Architectural Benefits:**
- **Centralized Control**: Single PC Hub provides unified interface for session management
- **Scalability**: Easily accommodates multiple Android spokes without architectural changes
- **Modularity**: Hub and Spokes are distinct, self-contained applications
- **Fault Tolerance**: System continues operation if individual spokes fail

### System Components

```
┌─────────────────┐         ┌─────────────────┐
│   PC Hub        │◄────────┤ Android Spoke 1  │
│   (Controller)  │         │  (Sensor Node)  │
└─────────────────┘         └─────────────────┘
         ▲
         │                  ┌─────────────────┐
         └──────────────────┤ Android Spoke 2  │
         │                  │  (Sensor Node)  │
         │                  └─────────────────┘
         │
         │                  ┌─────────────────┐
         └──────────────────┤ Android Spoke N  │
                            │  (Sensor Node)  │
                            └─────────────────┘
```

---

## C4 Architectural Views

The platform architecture follows the C4 model for comprehensive system visualization, from context through code-level details.

### C1 - System Context

The multi-modal physiological sensing platform operates within a research environment where investigators collect synchronized physiological data using multiple sensor modalities.

```mermaid
graph TB
    subgraph "Research Environment"
        R[Researcher/Investigator]
        S[Study Participants]
    end

    subgraph "Core Platform"
        PC[PC Hub Controller<br/>Session Management & Data Aggregation]
        A1[Android Sensor Node 1<br/>RGB + Thermal + GSR]
        A2[Android Sensor Node 2<br/>RGB + Thermal + GSR]
        AN[Android Sensor Node N<br/>RGB + Thermal + GSR]
    end

    subgraph "External Devices"
        TC[Topdon TC001<br/>Thermal Camera]
        GS[Shimmer GSR+<br/>Physiological Sensor]
    end

    subgraph "Storage & Analysis"
        LS[Local File System<br/>Session Data Storage]
        AS[Analysis Software<br/>MATLAB/Python/R]
    end

    R -->|controls sessions| PC
    R -->|monitors participants| S
    S -->|worn by| GS
    S -->|observed by| A1
    S -->|observed by| A2
    S -->|observed by| AN

    PC <-->|TCP/IP commands & sync| A1
    PC <-->|TCP/IP commands & sync| A2
    PC <-->|TCP/IP commands & sync| AN

    A1 <-->|USB-OTG| TC
    A1 <-->|BLE| GS
    A2 <-->|USB-OTG| TC
    A2 <-->|BLE| GS

    PC -->|aggregated data| LS
    LS -->|export formats| AS
```

**Key External Actors:**
- **Researcher/Investigator**: Controls recording sessions, monitors system status, analyzes collected data
- **Study Participants**: Subjects being monitored, wear/interact with physiological sensors

**Key External Systems:**
- **Topdon TC001 Thermal Camera**: USB-connected thermal imaging device providing temperature matrices
- **Shimmer GSR+ Sensor**: BLE-connected galvanic skin response and photoplethysmography sensor
- **Local File System**: Persistent storage for session data, metadata, and exports
- **Analysis Software**: External tools (MATLAB, Python, R) for post-processing and research analysis

### C2 - Container Architecture

The platform consists of two primary application containers running on different deployment nodes, connected via TCP/IP networking.

---

## Detailed Component Design

### PC Controller (Hub) Architecture

The PC Controller is a Python-based application with PyQt6 GUI, responsible for session orchestration and data aggregation.

#### Core Software Modules

**GUI Manager (PyQt6)**
- **Dashboard Tab**: Dynamic grid displaying connected devices with live previews and status indicators
- **Logs Tab**: Real-time system messages, warnings, and errors for debugging
- **Playback & Annotation Tab**: Post-session analysis with synchronized timeline and annotation capabilities

**Network Controller**
- Main server thread listening for Android Spoke connections
- Dedicated `WorkerThread` (QThread) for each connected device to prevent UI blocking
- Zeroconf (mDNS) integration for automatic device discovery (`_gsr-controller._tcp`)
- JSON message serialization/deserialization

**Session Manager**
- Session lifecycle control (create, start, stop, finalize)
- Unique timestamped directory creation for each session
- Session metadata management (`session_metadata.json`)
- State tracking: `IDLE` → `CREATED` → `RECORDING` → `STOPPED` → `COMPLETE`

**Time Synchronization Service**
- NTP-like protocol implementation for cross-device clock alignment
- Clock offset calculation and storage for each connected Spoke
- High-precision timestamp generation for <5ms accuracy requirement
- Continuous sync maintenance during sessions

**Sensor Manager**
- Direct PC sensor integration (webcam, Shimmer dock)
- **NativeShimmer Module**: C++ backend for high-integrity GSR data capture
- Real-time data processing and visualization
- Hardware abstraction layer for different sensor types

**Data Aggregator**
- File reception and organization from Android Spokes
- Session directory structure management
- Data integrity validation and corruption detection
- Multi-format export capabilities (CSV, JSON, MAT)

#### Technical Implementation Details

**Threading Model**:
- Main UI thread for PyQt6 GUI responsiveness
- Dedicated network threads for each connected device
- Background threads for data processing and file I/O
- C++ threads for real-time sensor data acquisition

**Memory Management**:
- Streaming data processing to minimize memory footprint
- Circular buffers for real-time sensor data
- Automatic garbage collection for completed sessions
- Resource monitoring and leak prevention

**Error Handling**:
- Comprehensive exception handling at all API boundaries
- Graceful degradation for non-critical component failures
- Automatic recovery mechanisms for network interruptions
- Detailed logging for debugging and support

### Android Sensor Node (Spoke) Architecture

The Android application is a modular Kotlin-based system that handles hardware interfacing and data capture.

#### Core Software Modules

**Network Client**
- TCP/IP connection management with PC Hub
- Command reception and acknowledgment handling
- Status update transmission and heartbeat mechanism
- Automatic reconnection logic for network resilience

**Recording Controller**
- Central orchestrator for all sensor modules
- Session state management and coordination
- Thread-safe operations across multiple sensor streams
- Resource management and lifecycle control

**Sensor Modules**:

**RGB Camera Module (`RgbCameraRecorder`)**
- CameraX integration for video capture (≥1920x1080, 30 FPS)
- High-resolution still image capture capability
- Synchronized timestamping with other sensors
- Hardware-accelerated encoding when available

**Thermal Camera Module (`ThermalCameraRecorder`)** ✅ **PRODUCTION-READY**
- **True Topdon TC001 SDK Integration**: Real IRCMD, LibIRParse, LibIRProcess implementation
- **Hardware-Calibrated Temperature Processing**: ±2°C accuracy with emissivity correction
- **Professional Thermal Imaging**: Iron, Rainbow, Grayscale color palettes
- **TC001-Specific Device Detection**: VID/PID 0x0525/0xa4a2, 0x0525/0xa4a5
- **Graceful Hardware/Simulation Fallback**: Development-friendly operation

**GSR Sensor Module (`ShimmerRecorder`)** ✅ **PRODUCTION-READY**
- **Real Shimmer Android API Integration**: ShimmerBluetooth, ShimmerConfig classes
- **12-bit ADC Precision**: Correct 0-4095 range conversion (scientifically accurate)
- **128 Hz Sampling Rate**: Hardware-validated sampling frequency compliance
- **Dual-Sensor Recording**: Simultaneous GSR (microsiemens) + PPG (raw ADC)
- **Robust BLE Connection Management**: Device discovery, pairing, and reconnection

**Time Synchronization Client**
- UDP client for NTP-like protocol with PC Hub
- Local clock adjustment and offset management
- Continuous synchronization during recording sessions
- High-precision timestamp generation for all data streams

**Local Storage Manager**
- Session-based directory structure organization
- Atomic file operations to prevent data corruption
- Storage space monitoring and management
- Data compression and archival for transfer

#### Android-Specific Optimizations

**Battery Optimization**:
- Background service management for continuous recording
- CPU wake lock management to prevent interruptions
- Power-efficient sensor sampling strategies
- Battery level monitoring and low-power mode handling

**Memory Management**:
- Streaming file I/O to minimize memory usage
- Native memory management for sensor data buffers
- Automatic cleanup of completed session data
- Memory pressure monitoring and adaptive behavior

**Hardware Integration**: ✅ **PRODUCTION-READY**
- **CameraX API**: Professional RGB camera control with dual-pipeline recording
- **True Topdon TC001 SDK**: Real thermal camera connectivity via USB Host API
- **Real Shimmer Android API**: Production BLE communication for GSR sensors
- **Hardware abstraction**: Unified interface with graceful simulation fallback

---

## Communication Protocol

### Protocol Overview

The system uses JSON-based messages over TCP/IP for command and control, with UDP for time synchronization.

### Core Message Types

**Device Discovery and Connection**:
```json
{
  "type": "DEVICE_ANNOUNCEMENT",
  "device_id": "android_spoke_001",
  "device_name": "Samsung Galaxy S21",
  "capabilities": ["rgb_camera", "thermal_camera", "gsr_sensor"],
  "ip_address": "192.168.1.101",
  "port": 8080,
  "protocol_version": "2.0"
}

{
  "type": "CONNECTION_REQUEST",
  "hub_id": "pc_hub_001",
  "session_id": "session_20231201_143022"
}

{
  "type": "CONNECTION_RESPONSE",
  "status": "accepted",
  "device_capabilities": ["rgb_camera", "thermal_camera"],
  "sync_offset_ms": 2.3
}
```

**Session Management**:
```json
{
  "type": "SESSION_START",
  "session_id": "session_20231201_143022",
  "timestamp": "2023-12-01T14:30:22.123456Z",
  "recording_duration_s": 300
}

{
  "type": "SESSION_STOP",
  "session_id": "session_20231201_143022",
  "final_timestamp": "2023-12-01T14:35:22.123456Z"
}
```

**Synchronization Commands**:
```json
{
  "type": "SYNC_FLASH",
  "flash_id": "flash_001",
  "timestamp": "2023-12-01T14:30:25.456789Z"
}

{
  "type": "TIME_SYNC_REQUEST",
  "client_timestamp": "2023-12-01T14:30:22.123456Z"
}

{
  "type": "TIME_SYNC_RESPONSE",
  "server_timestamp": "2023-12-01T14:30:22.124789Z",
  "client_timestamp": "2023-12-01T14:30:22.123456Z"
}
```

**Status and Monitoring**:
```json
{
  "type": "HEARTBEAT",
  "device_id": "android_spoke_001",
  "timestamp": "2023-12-01T14:30:22.123456Z",
  "battery_level": 85,
  "storage_free_mb": 2048,
  "recording_status": "active"
}

{
  "type": "ERROR_REPORT",
  "error_code": "SENSOR_DISCONNECTED",
  "error_message": "Shimmer sensor lost BLE connection",
  "timestamp": "2023-12-01T14:30:22.123456Z"
}
```

### Protocol Features

**Reliability**:
- Message acknowledgment for critical commands
- Sequence numbering for message ordering
- Timeout and retry mechanisms
- Connection health monitoring

**Security**:
- TLS encryption for all TCP communications
- Device authentication using certificates
- Message integrity validation
- Protection against replay attacks

**Performance**:
- Efficient JSON serialization/deserialization
- Message batching for high-frequency data
- Compression for large data transfers
- Adaptive message prioritization

---

## Data Flow and Management

### Session Data Workflow

```
1. Session Creation
   ├── PC Hub creates session directory
   ├── Generates unique session ID
   └── Initializes metadata.json

2. Device Synchronization
   ├── Clock synchronization across all devices
   ├── Capability verification
   └── Recording parameter configuration

3. Data Recording Phase
   ├── Synchronized start across all devices
   ├── Continuous data capture to local storage
   ├── Real-time status monitoring
   └── Flash sync events for temporal validation

4. Session Termination
   ├── Synchronized stop command
   ├── Local data finalization
   └── Transfer preparation

5. Data Transfer
   ├── Android devices compress session data
   ├── Automatic transfer to PC Hub
   ├── Data integrity validation
   └── Session completion confirmation

6. Data Organization
   ├── Extraction and organization on PC
   ├── Cross-device data alignment
   ├── Metadata consolidation
   └── Analysis-ready data preparation
```

### Storage Architecture

**PC Hub Storage Structure**:
```
pc_controller_data/
├── sessions/
│   └── session_YYYYMMDD_HHMMSS/
│       ├── metadata.json
│       ├── hub_data/
│       │   ├── shimmer_gsr.csv
│       │   └── sync_events.csv
│       └── devices/
│           ├── android_spoke_001/
│           │   ├── video.mp4
│           │   ├── thermal.csv
│           │   └── device_metadata.json
│           └── android_spoke_002/
│               ├── video.mp4
│               └── gsr.csv
```

**Android Local Storage**:
```
/Android/data/com.example.sensorspoke/files/
└── sessions/
    └── session_YYYYMMDD_HHMMSS/
        ├── video.mp4
        ├── thermal.csv
        ├── gsr.csv
        ├── sync_events.csv
        └── metadata.json
```

### Data Quality and Integrity

**Validation Mechanisms**:
- Checksum validation for all transferred files
- Temporal consistency checking across devices
- Missing data detection and reporting
- Correlation validation between data streams

**Quality Metrics**:
- Synchronization accuracy measurement (<5ms requirement)
- Data completeness assessment
- Signal quality indicators for each sensor
- System performance metrics (latency, throughput)

---

## Deployment Architecture

### Physical Deployment Model

The platform deploys across multiple heterogeneous devices within a research environment:

```
Research Laboratory Environment
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌─────────────────┐    WiFi/Ethernet    ┌─────────────┐ │
│  │   PC Controller │◄─────────────────────┤   Router    │ │
│  │   (Windows)     │                      │             │ │
│  │                 │                      └─────────────┘ │
│  │ ┌─────────────┐ │                              │       │
│  │ │Session Mgmt │ │                              │       │
│  │ │Data Aggreg │ │                              │       │
│  │ │GUI Dashboard│ │                      ┌───────▼─────┐ │
│  │ └─────────────┘ │                      │Android Node1│ │
│  └─────────────────┘                      │             │ │
│           │                               │┌───────────┐│ │
│           │                               ││Sensor Rec ││ │
│    ┌──────▼──────┐                       ││Network Mgr││ │
│    │Local Storage│                       │└───────────┘│ │
│    │Sessions/    │                       └─────────────┘ │
│    │Exports      │                               │       │
│    └─────────────┘                       ┌───────▼─────┐ │
│                                          │Android NodeN│ │
│                                          │             │ │
│                                          └─────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Deployment Specifications

**PC Controller Requirements:**
- **OS**: Windows 10/11 (primary), Linux/macOS (development)
- **RAM**: 8GB minimum, 16GB recommended for multi-node sessions
- **Storage**: 500GB SSD for session data buffering
- **Network**: Gigabit Ethernet or WiFi 6 for optimal throughput
- **USB**: Multiple ports for Shimmer dock connections

**Android Sensor Node Requirements:**
- **OS**: Android 8.0+ (API 26+) for CameraX compatibility
- **RAM**: 4GB minimum for thermal processing and video encoding
- **Storage**: 64GB minimum for local session buffering
- **Camera**: Back camera with manual focus control
- **USB**: USB-C with OTG support for thermal camera
- **BLE**: Bluetooth 5.0+ for Shimmer GSR connectivity

**Network Infrastructure:**
- **Bandwidth**: 100 Mbps minimum for real-time streaming
- **Latency**: <10ms for synchronization accuracy
- **Topology**: Star configuration with dedicated research network segment
- **Security**: WPA3 encryption, network segmentation from general internet access

---

## Implementation Strategy

### Development Phases

The implementation follows a structured 6-phase approach:

1. **Foundation Phase**: Basic communication and project scaffolding
2. **Sensor Integration**: Individual sensor module development and testing
3. **Advanced Networking**: Time synchronization and robust communication
4. **GUI and Session Management**: Complete user interface and workflow
5. **Integration and Testing**: End-to-end system validation
6. **Optimization and Deployment**: Performance tuning and production readiness

### Technology Stack

**PC Hub (Python)**:
- **Framework**: PyQt6 for GUI, asyncio for concurrent networking
- **Networking**: TCP/IP sockets, Zeroconf for service discovery
- **Data Processing**: NumPy, Pandas for analysis, PyQtGraph for visualization
- **Hardware**: PySerial for Shimmer dock, OpenCV for image processing
- **Testing**: pytest for unit testing, coverage analysis

**Android Spoke (Kotlin)**:
- **Framework**: Android SDK with Architecture Components (ViewModel, LiveData)
- **Camera**: CameraX for video capture, Camera2 API for advanced features
- **Connectivity**: Android Network Service Discovery, standard TCP/IP sockets
- **Hardware**: USB Host API, Bluetooth Low Energy APIs
- **Testing**: JUnit for unit tests, Robolectric for framework testing

**Cross-Platform**:
- **Serialization**: JSON for protocol messages, Protocol Buffers for high-performance data
- **Security**: TLS 1.3 for encrypted communication, certificate-based authentication
- **Time Sync**: Custom NTP-like protocol optimized for local network
- **Data Formats**: CSV for time series, MP4 for video, custom binary for high-frequency data

### Quality Assurance

**Testing Strategy**:
- Comprehensive unit tests for all major components
- Integration tests for cross-component interactions
- System tests under realistic conditions
- Performance tests with multiple devices
- Long-duration stability tests

**Code Quality**:
- Static analysis tools (pylint, ktlint)
- Code review requirements for all changes
- Continuous integration with automated testing
- Documentation standards and API documentation

**Validation Approach**:
- Empirical validation of synchronization accuracy
- Real-world testing with target user groups
- Performance benchmarking under various conditions
- Security assessment and penetration testing

This comprehensive design provides a robust foundation for implementing a research-grade multi-modal physiological sensing platform that meets all functional and non-functional requirements while maintaining scalability, reliability, and usability.
