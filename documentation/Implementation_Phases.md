# Implementation Phases - Multi-Modal Physiological Sensing Platform

This document consolidates all implementation phases for the systematic development of the multi-modal physiological sensing platform. The project follows a structured 6-phase approach from foundational infrastructure to complete system deployment.

## Overview

The implementation is organized into phases that build upon each other, ensuring systematic development and validation at each stage:

1. **Phase 1**: Foundation and basic communication
2. **Phase 2**: Data capture and local storage
3. **Phase 3**: Advanced networking features
4. **Phase 4**: PC Hub GUI and session management
5. **Phase 5**: Integration and deployment preparation
6. **Phase 6**: Optimization and production readiness

---

## Phase 1: Foundation and Communication

**Objective**: Establish foundational project structures and implement core network communication for device discovery and connection.

### Task 1.1: Project Scaffolding and Version Control

**PC Controller (Hub)**:
- Technology: Python 3, PyQt6
- Actions:
  1. Initialize Git repository
  2. Set up Python virtual environment
  3. Create `requirements.txt` with dependencies (`PyQt6`, `zeroconf`)
  4. Create directory structure:
     ```
     /pc_controller/
     ├── src/
     │   ├── main.py           # Main application entry point
     │   ├── gui/              # GUI-related modules
     │   ├── network/          # Network communication modules
     │   └── core/             # Core logic (session management)
     └── tests/                # Unit tests
     ```

**Android Sensor Node (Spoke)**:
- Technology: Kotlin, Android Studio
- Actions:
  1. Initialize Git repository/directory
  2. Create Android Studio project (modern API level)
  3. Add permissions to `AndroidManifest.xml` (`INTERNET`, `ACCESS_WIFI_STATE`, `CHANGE_WIFI_MULTICAST_STATE`)
  4. Create package structure:
     ```
     /com.example.sensorspoke/
     ├── ui/                   # Activities and Fragments
     ├── network/              # Network client and service discovery
     └── service/              # Background services for recording
     ```

### Task 1.2: Communication Protocol Definition (Version 1.0)

**Technology**: JSON message payloads

**Protocol Messages**:

```json
// Device Discovery
{
  "type": "DEVICE_ANNOUNCEMENT",
  "device_id": "android_spoke_001",
  "device_name": "Samsung Galaxy S21",
  "capabilities": ["rgb_camera", "thermal_camera_topdon_tc001", "gsr_sensor_shimmer3"],
  "ip_address": "192.168.1.101",
  "port": 8080
}

// Connection Establishment
{
  "type": "CONNECTION_REQUEST",
  "hub_id": "pc_hub_001"
}

{
  "type": "CONNECTION_RESPONSE",
  "status": "accepted",
  "session_id": "session_20231201_143022"
}

// Basic Commands
{
  "type": "PING",
  "timestamp": "2023-12-01T14:30:22.123Z"
}

{
  "type": "PONG",
  "timestamp": "2023-12-01T14:30:22.125Z"
}
```

### Task 1.3: Service Discovery Implementation

**PC Hub**:
- Implement mDNS/Zeroconf browser using `zeroconf` library
- Listen for Android device announcements
- Maintain discovered device registry

**Android Spoke**:
- Implement mDNS/Zeroconf service advertisement
- Broadcast device capabilities and connection info
- Handle discovery responses from PC Hub

### Task 1.4: Basic TCP Communication

**Implementation**:
- Establish reliable TCP connections between Hub and Spokes
- Implement JSON message serialization/deserialization
- Create basic message routing and handling framework
- Add connection lifecycle management (connect, disconnect, error handling)

---

## Phase 2: Data Capture and Local Storage

**Objective**: Implement data capture and local storage for each sensor modality within the Android application.

### Task 2.1: Central RecordingController

**Technology**: Kotlin, Android Architecture Components (ViewModel, LiveData)

**Implementation**:
1. Create `RecordingController` class managed by ViewModel
2. Define recording states: `IDLE`, `PREPARING`, `RECORDING`, `STOPPING`
3. Implement core methods:
   - `startSession(sessionId: String)`: Creates session directory, starts all sensor modules
   - `stopSession()`: Stops all modules, finalizes session
4. Create simple UI with "Start Recording"/"Stop Recording" buttons for testing

### Task 2.2: RGB Camera Module (RgbCameraRecorder)

**Technology**: Android Camera2 API

**Implementation**:
- High-resolution video recording (≥1920x1080, 30 FPS)
- Still image capture capability
- Thread-safe recording operations
- Proper resource management (camera, file handles)

**Output**:
- `video.mp4`: Continuous video recording
- `images/`: Directory for timestamped still images

### Task 2.3: Thermal Camera Module (ThermalCameraRecorder) ✅ **COMPLETED**

**Technology**: ✅ **TRUE Topdon TC001 SDK Integration**

**✅ IMPLEMENTED Features**:
- ✅ **Real SDK Integration**: IRCMD, LibIRParse, LibIRProcess classes
- ✅ **Hardware-Specific Detection**: TC001 VID/PID 0x0525/0xa4a2, 0x0525/0xa4a5
- ✅ **Calibrated Temperature Processing**: ±2°C accuracy with emissivity correction
- ✅ **Professional Thermal Imaging**: Iron, Rainbow, Grayscale color palettes
- ✅ **Graceful Hardware/Simulation Fallback**: Development-friendly operation

**✅ PRODUCTION OUTPUTS**:
- `thermal.csv`: Hardware-calibrated timestamped thermal readings with ±2°C accuracy
- `thermal_images/`: Professional thermal image snapshots with real color palettes

### Task 2.4: GSR Sensor Module (ShimmerRecorder) ✅ **COMPLETED**

**Technology**: ✅ **REAL Shimmer3 GSR+ Android SDK Integration**

**✅ IMPLEMENTED Features**:
- ✅ **Production SDK Integration**: ShimmerBluetooth, ShimmerConfig classes
- ✅ **12-bit ADC Precision**: Scientifically accurate 0-4095 range conversion (not 16-bit)
- ✅ **128 Hz Sampling Rate**: Hardware-validated sampling frequency compliance
- ✅ **Robust BLE Connection**: Device discovery, pairing, reconnection management
- ✅ **Dual-Sensor Recording**: Simultaneous GSR (microsiemens) + PPG (raw ADC)

**✅ PRODUCTION OUTPUTS**:
- `gsr.csv`: Production-quality timestamped GSR measurements with 12-bit precision
- Header includes real sensor configuration and hardware metadata

### Task 2.5: Local Data Management

**Implementation**:
- Session-based directory structure: `/Android/data/app/sessions/{session_id}/`
- Atomic file operations to prevent corruption
- Session metadata tracking (`metadata.json`)
- Storage space monitoring and management

---

## Phase 3: Advanced Networking Features

**Objective**: Implement time synchronization, advanced protocol features, and robust network communication.

### Task 3.1: Time Synchronization Service

**Technology**: NTP-like protocol over UDP

**PC Hub Implementation**:
- Dedicated UDP time server on port 8081
- High-precision timestamp generation
- Multiple timestamp samples for accuracy

**Android Spoke Implementation**:
- UDP client for time synchronization requests
- Clock offset calculation and application
- Continuous sync maintenance during sessions

**Protocol**:
```json
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

### Task 3.2: Enhanced Protocol (Version 2.0)

**Additional Commands**:

```json
// Session Management
{
  "type": "SESSION_START",
  "session_id": "session_20231201_143022",
  "timestamp": "2023-12-01T14:30:22.123Z"
}

{
  "type": "SESSION_STOP",
  "session_id": "session_20231201_143022"
}

// Flash Synchronization
{
  "type": "SYNC_FLASH",
  "flash_id": "flash_001",
  "timestamp": "2023-12-01T14:30:25.456Z"
}

// Status Updates
{
  "type": "STATUS_UPDATE",
  "device_id": "android_spoke_001",
  "battery_level": 85,
  "storage_free_mb": 2048,
  "recording_status": "active"
}
```

### Task 3.3: Heartbeat and Connection Monitoring

**Implementation**:
- Regular heartbeat messages (every 3 seconds)
- Connection timeout detection and handling
- Automatic reconnection logic
- Device status tracking (online/offline/reconnecting)

### Task 3.4: Error Handling and Recovery

**Features**:
- Graceful handling of network interruptions
- Session state persistence across disconnections
- Automatic session rejoin capability
- Comprehensive error logging

---

## Phase 4: PC Hub GUI and Session Management

**Objective**: Develop comprehensive PC Hub GUI and complete session management functionality.

### Task 4.1: Main Dashboard GUI

**Technology**: PyQt6, Model-View architecture

**Components**:
- Device list with real-time status updates
- Session control buttons (Start, Stop, Flash Sync)
- Session progress indicators
- System status display

**Features**:
- Real-time device discovery and connection status
- Visual indicators for recording state
- Error message display and user notifications
- Session metadata display

### Task 4.2: Device Management Interface

**Implementation**:
- Detailed device information panels
- Individual device control capabilities
- Battery and storage monitoring
- Connection quality indicators

**Visual Elements**:
- Color-coded status indicators (green=online, red=offline, yellow=reconnecting)
- Device capability badges
- Real-time sensor data previews

### Task 4.3: Session Management System

**Core Features**:
- Session creation and configuration
- Directory structure management
- Session metadata tracking
- Multi-session history and navigation

**Session States**:
1. `IDLE`: No active session
2. `CREATED`: Session configured but not started
3. `RECORDING`: Active data collection
4. `STOPPED`: Recording completed, pending data transfer
5. `TRANSFERRING`: Data transfer in progress
6. `COMPLETE`: Session fully processed and archived

### Task 4.4: Data Visualization and Playback

**Implementation**:
- Real-time data stream visualization
- Post-session data playback capabilities
- Synchronized multi-stream playback
- Basic data analysis tools

### Task 4.5: Configuration Management

**Features**:
- System configuration interface
- Sensor parameter configuration
- Network settings management
- User preferences and profiles

---

## Phase 5: Integration and Deployment Preparation

**Objective**: Integrate all components, implement data transfer, and prepare for deployment.

### Task 5.1: Automatic Data Transfer System

**Technology**: Dedicated TCP file transfer protocol

**PC Hub - FileTransferServer**:
- Dedicated server on port 8082
- Multi-threaded file reception
- Progress tracking and validation
- Automatic archive extraction

**Android Spoke - FileTransferManager**:
- Session data compression (ZIP format)
- Chunked file transmission
- Transfer progress reporting
- Cleanup after successful transfer

**Transfer Protocol**:
```json
{
  "type": "TRANSFER_START",
  "session_id": "session_20231201_143022",
  "file_count": 4,
  "total_size_bytes": 1048576
}

{
  "type": "FILE_CHUNK",
  "file_name": "session_data.zip",
  "chunk_index": 1,
  "chunk_data": "<base64_encoded_data>",
  "checksum": "sha256_hash"
}
```

### Task 5.2: Calibration Tools

**Android Calibration App**: ✅ **PRODUCTION-READY**
- ✅ **Real RGB/Thermal Pairing**: True thermal camera integration enables accurate RGB/thermal image calibration
- Structured calibration pattern support
- Export calibration image sets with hardware-validated thermal data

**PC Calibration Tool**:
- OpenCV-based calibration computation
- Camera parameter estimation
- ✅ **Enhanced with Real Thermal Data**: True temperature calibration quality assessment
- Configuration file generation for production thermal accuracy

### Task 5.3: Data Export and Analysis Tools

**Implementation**:
- Multi-format data export (CSV, JSON, MAT files)
- Synchronized data alignment tools
- Basic statistical analysis functions
- Integration with common analysis platforms

### Task 5.4: Comprehensive Testing Suite

**Test Categories**:
- Unit tests for all major components
- Integration tests for end-to-end workflows
- Performance tests for multi-device scenarios
- Stress tests for long-duration sessions

### Task 5.5: Documentation and User Guides

**Documentation**:
- Complete user manual with screenshots
- Technical API documentation
- Installation and setup guides
- Troubleshooting and FAQ sections

---

## Phase 6: Optimization and Production Readiness

**Objective**: Optimize performance, enhance security, and prepare for production deployment.

### Task 6.1: Performance Optimization

**PC Hub Optimizations**:
- Multi-threading for concurrent device handling
- Memory management and leak prevention
- Network I/O optimization
- GUI responsiveness improvements

**Android Optimizations**:
- Background processing optimization
- Battery usage minimization
- Memory footprint reduction
- Storage I/O efficiency

### Task 6.2: Security Implementation

**Features**:
- TLS encryption for all network communication
- Device authentication and authorization
- Secure session token management
- Data integrity validation

**Security Protocol**:
- Certificate-based device authentication
- End-to-end encryption of sensor data
- Secure key exchange mechanisms
- Protection against common network attacks

### Task 6.3: Advanced Features

**Implemented Capabilities**:
- Multi-session concurrent recording
- Remote device monitoring and control
- Advanced synchronization algorithms
- Adaptive quality control based on network conditions

### Task 6.4: Deployment Packaging

**PC Hub Packaging**:
- Windows executable with installer
- Dependency bundling and validation
- Configuration wizard for initial setup
- Automatic update mechanism

**Android App Packaging**:
- Release-signed APK generation
- Play Store metadata and assets
- Beta testing distribution
- Production deployment pipeline

### Task 6.5: Validation and Certification

**Final Validation**:
- Complete system testing under production conditions
- Performance benchmarking and validation
- Security audit and penetration testing
- User acceptance testing with target researchers

**Quality Assurance**:
- Code review and quality metrics
- Documentation completeness validation
- Compliance checking for research standards
- Long-term reliability testing

---

## Phase Dependencies and Timeline

### Critical Path Dependencies

1. **Phase 1 → Phase 2**: Basic communication required before sensor integration
2. **Phase 2 → Phase 3**: Local recording must work before advanced networking
3. **Phase 3 → Phase 4**: Synchronization needed for GUI session management
4. **Phase 4 → Phase 5**: Complete GUI required for integration testing
5. **Phase 5 → Phase 6**: All features must be integrated before optimization

### Parallel Development Opportunities

- **Phase 2 Tasks**: Different sensor modules can be developed concurrently
- **Phase 4 Tasks**: GUI components can be developed in parallel with backend
- **Phase 5 Tasks**: Testing and documentation can overlap with feature development

### Quality Gates

Each phase includes validation criteria that must be met before proceeding:
- **Unit tests** passing for all new components
- **Integration tests** validating phase objectives
- **Code review** and documentation completion
- **Performance benchmarks** meeting requirements

---

This phased approach ensures systematic development with clear milestones, proper validation at each stage, and minimal risk of integration issues. Each phase builds incrementally toward a complete, production-ready research platform.
