# AI Agent Guidelines - Multi-Modal Physiological Sensing Platform

This document provides comprehensive guidelines for AI agents (Gemini, JetBrains Junie, GitHub Copilot) working on the multi-modal physiological sensing platform. It consolidates maintenance practices, strategic development plans, and implementation prompts.

## Table of Contents

1. [Project Maintenance and DevOps](#project-maintenance-and-devops)
2. [Strategic Development Plan](#strategic-development-plan)
3. [Implementation Prompts](#implementation-prompts)
4. [Code Quality Standards](#code-quality-standards)

---

## Project Maintenance and DevOps

AI agents are responsible for maintaining project health, documentation, and data integrity beyond initial code generation.

### Changelog Maintenance

**Requirements**:
- Maintain `CHANGELOG.md` in repository root using "Keep a Changelog" format
- Include "Unreleased" section at top for ongoing changes
- Add entries for every `feat`, `fix`, or `BREAKING CHANGE` commit
- Entries must be user-focused, describing impact rather than technical details

**Categories**: `Added`, `Changed`, `Fixed`, `Removed`, `Security`

### Data Backup Strategy

**Documentation Requirements**:
- Create `BACKUP_STRATEGY.md` outlining 3-2-1 Backup Rule:
  - 3 copies of data
  - 2 different media types
  - 1 copy off-site
- Provide clear template for researchers' specific backup implementations

**Automation**:
- Create `backup_script.py` in `/scripts` directory
- Configurable copying from main session data to local backup destination
- Well-documented with clear usage instructions

---

## Strategic Development Plan

This plan provides a systematic approach to complete the multi-modal data collection platform, focusing on implementing missing features and ensuring all MEng-level requirements are met.

### Overall Assessment

The repository has foundational structure but requires systematic implementation of missing features, core component refactoring, and rigorous testing. The plan is organized into phases from architectural validation to feature implementation and final verification.

### Requirements Implementation Status & Action Plan

| Requirement ID | Description | Target Module(s) | Status | Action Plan |
|:---|:---|:---|:---|:---|
| **FR1** | Multi-device sensor integration | `PC:ShimmerManager`, `Android:ShimmerRecorder` | Partially Implemented | 1. Finalize Bluetooth connection logic for Shimmer<br>2. Implement `SimulatedShimmer` class on PC<br>3. Ensure `DeviceManager` handles real and simulated sensors |
| **FR2** | Synchronized start/stop | `PC:SessionManager`, `Android:RecordingController` | Partially Implemented | 1. Refine `SESSION_START`/`STOP` commands<br>2. Implement robust state management in `RecordingController` |
| **FR3** | Time Synchronization Service | `PC:NetworkServer`, `Android:NetworkClient` | Not Implemented | 1. Implement NTP-like service on PC with UDP socket<br>2. Implement client-side clock offset calculation on Android |
| **FR4** | Session Management | `PC:SessionManager` | Partially Implemented | 1. Implement full lifecycle: `IDLE`→`CREATED`→`RECORDING`→`STOPPED`→`COMPLETE`<br>2. Ensure atomic session directory creation |
| **FR5** | Data Recording & Storage | All Recorder Modules | Partially Implemented | 1. Implement real-time CSV writing for GSR data<br>2. Ensure Android saves to private app directory<br>3. Verify video ≥1920×1080, 30 FPS |
| **FR6** | PC GUI Monitoring & Control | `PC:GUI` | Partially Implemented | 1. Bind device list UI to `DeviceManager`<br>2. Implement visual alerts for disconnections<br>3. Handle UI interactions on separate threads |
| **FR7** | Device Sync Signals (Flash) | `PC:SessionManager`, `Android:RecordingController` | Not Implemented | 1. Add `SYNC_FLASH` command to protocol<br>2. Implement full-screen white overlay on Android |
| **FR8** | Fault Tolerance & Recovery | `PC:DeviceManager`, `Android:NetworkClient` | Not Implemented | 1. Implement heartbeat mechanism (3s intervals)<br>2. Implement timeout detection and "Offline" status<br>3. Implement automatic reconnection logic |
| **FR9** | Calibration Utilities | `Android:CalibrationActivity`, `PC:CalibrationTool` | Not Implemented | 1. Create Android calibration Activity<br>2. Develop PC calibration tool using OpenCV |
| **FR10** | Automatic Data Transfer | `Android:FileTransferManager`, `PC:FileTransferServer` | Not Implemented | 1. Implement dedicated `FileTransferServer` on PC<br>2. Create `FileTransferManager` on Android<br>3. Update metadata after successful transfers |

### Development Phases

**Phase 1: Core Infrastructure**
- Project scaffolding and version control
- Basic TCP/IP communication
- Service discovery implementation
- Protocol definition and message handling

**Phase 2: Sensor Integration**
- Android sensor module development
- Local data recording and storage
- Session management implementation
- Hardware abstraction layers

**Phase 3: Advanced Features**
- Time synchronization service
- Fault tolerance and recovery
- Enhanced protocol features
- Error handling and logging

**Phase 4: Integration and Testing**
- End-to-end system integration
- Comprehensive testing suite
- Performance optimization
- Security implementation

**Phase 5: Deployment and Documentation**
- Production packaging
- User documentation
- Installation guides
- Maintenance procedures

---

## Implementation Prompts

These prompts are designed for AI code generation tools like JetBrains Junie, structured to build the application systematically from the ground up.

### Phase 1: Foundational Infrastructure

#### Prompt 1.1 (PC): Project Scaffolding & Configuration

Create the project structure for a Python application that will act as the central controller for a data acquisition system.

**Directory Structure**:
```
pc_controller/
├── core/
├── network/
├── sensors/
├── gui/
└── tests/
```

**Configuration File** (`config.json`):
```json
{
  "server_ip": "0.0.0.0",
  "command_port": 8080,
  "timesync_port": 8081,
  "file_transfer_port": 8082,
  "shimmer_sampling_rate": 128,
  "video_resolution": [1920, 1080],
  "video_fps": 30
}
```

#### Prompt 1.2 (Android): Project Structure & Permissions

Create Android application structure for sensor data collection.

**Package Structure**:
```
com.example.sensorspoke/
├── ui/                 # Activities and Fragments
├── network/            # Network client and discovery
├── sensors/            # Sensor recording modules
└── services/           # Background services
```

**Required Permissions**:
- `INTERNET`
- `ACCESS_WIFI_STATE`
- `CHANGE_WIFI_MULTICAST_STATE`
- `CAMERA`
- `RECORD_AUDIO`
- `WRITE_EXTERNAL_STORAGE`

#### Prompt 1.3: Communication Protocol Implementation

Implement JSON-based communication protocol for Hub-Spoke architecture.

**Core Message Types**:
- Device discovery and capability exchange
- Session control (start, stop, status)
- Time synchronization requests/responses
- Heartbeat and health monitoring
- Error reporting and recovery

### Phase 2: Sensor Integration

#### Prompt 2.1 (Android): Recording Controller

Implement central `RecordingController` class managing all sensor modules:
- Session state management (`IDLE`, `RECORDING`, `STOPPED`)
- Coordinated sensor start/stop operations
- Thread-safe operations across multiple streams
- Resource lifecycle management

#### Prompt 2.2 (Android): Camera Module

Implement RGB camera recording using CameraX:
- High-resolution video recording (≥1920x1080, 30 FPS)
- Still image capture capability
- Synchronized timestamping
- Hardware-accelerated encoding

#### Prompt 2.3 (Android): Sensor Data Management

Implement local storage system:
- Session-based directory structure
- Atomic file operations
- Storage space monitoring
- Data compression for transfer

### Phase 3: Advanced Features

#### Prompt 3.1: Time Synchronization

Implement NTP-like time synchronization:
- UDP-based protocol for low latency
- Clock offset calculation and application
- Continuous synchronization during sessions
- <5ms accuracy requirement validation

#### Prompt 3.2: Fault Tolerance

Implement robust network handling:
- Heartbeat mechanism (3-second intervals)
- Automatic reconnection logic
- Graceful degradation for device failures
- Session state persistence across disconnections

### Phase 4: PC Hub GUI

#### Prompt 4.1: Main Dashboard

Create PyQt6-based main interface:
- Real-time device status grid
- Session control buttons
- Live data visualization
- Status indicators and notifications

#### Prompt 4.2: Data Transfer System

Implement automatic file transfer:
- Dedicated TCP server for file reception
- Multi-threaded file handling
- Progress tracking and validation
- Automatic session data organization

---

## Code Quality Standards

### Python Development Standards

**Code Style**:
- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Maximum line length: 88 characters (Black formatter)
- Use descriptive variable and function names

**Error Handling**:
- Use specific exception types rather than generic `Exception`
- Implement proper logging at appropriate levels
- Graceful degradation for non-critical failures
- Comprehensive error documentation

**Testing Requirements**:
- Minimum 80% code coverage for core modules
- Unit tests for all major components
- Integration tests for cross-component interactions
- Mock external dependencies in tests

**Documentation**:
- Docstrings for all public functions and classes
- Type annotations for improved IDE support
- README files for each major module
- API documentation using Sphinx

### Android/Kotlin Development Standards

**Code Style**:
- Follow Android coding conventions
- Use ktlint for consistent formatting
- Implement proper lifecycle management
- Follow MVVM architecture pattern

**Resource Management**:
- Proper camera and sensor resource cleanup
- Background thread management for long operations
- Memory leak prevention with lifecycle awareness
- Battery optimization best practices

**Testing Strategy**:
- Unit tests using JUnit and Mockito
- UI tests using Espresso where appropriate
- Robolectric for Android framework testing
- Integration tests for sensor modules

### Security Considerations

**Network Security**:
- TLS encryption for all TCP communications
- Certificate-based device authentication
- Input validation for all network messages
- Protection against common network attacks

**Data Protection**:
- Secure local storage for sensitive data
- Data integrity validation using checksums
- Secure deletion of temporary files
- Privacy protection for research data

### Performance Requirements

**Real-time Performance**:
- <5ms synchronization accuracy across devices
- No data loss during extended recording sessions
- Responsive UI with background processing
- Efficient memory usage patterns

**Scalability**:
- Support for multiple concurrent Android devices
- Graceful handling of varying network conditions
- Adaptive quality based on system resources
- Efficient data compression and transfer

### Development Workflow

**Version Control**:
- Use conventional commit messages
- Feature branches for all major changes
- Code review requirements for all commits
- Automated testing in CI/CD pipeline

**Quality Gates**:
- All tests must pass before merge
- Static analysis tools must pass
- Code coverage thresholds must be met
- Documentation must be updated for public APIs

This comprehensive guideline ensures consistent, high-quality development across all AI agents working on the platform, maintaining professional standards suitable for academic and research environments.
    * The script's documentation must include clear instructions on how to configure the source and destination paths
      and how to schedule the script to run automatically (e.g., using `cron` on Linux/macOS or Task Scheduler on
      Windows).

##### **9.3. Other Maintenance Tasks**

* **README Maintenance:** You must keep the root `README.md` file up-to-date. If you add a new feature, dependency, or
  change the setup process, you must update the README to reflect these changes. The README should always contain clear,
  accurate instructions for setting up and running the project.
* **Dependency Management:** When you add a new library or SDK, you must add it to the appropriate dependency file (
  `requirements.txt` for Python, `app/build.gradle.kts` for Android). Keep these files clean and free of unused
  dependencies.
* **`.gitignore` Maintenance:** You must ensure that temporary files, build artifacts, IDE-specific configuration
  files (e.g., `.idea/`), and sensitive files are included in the `.gitignore` file to keep the repository clean.

-----

### Why These Additions Are Useful

* **Changelog:** This forces the AI to document its own work in a human-readable format, making it much easier for you
  and your team to track progress and prepare for new releases.
* **Backup Strategy:** While the AI can't perform the physical backups, instructing it to create the documentation and
  automation scripts is a huge value-add. It builds best practices directly into the project, protecting the invaluable
  data that the platform is designed to collect.
* **Other Tasks:** These instructions cover common developer chores that are easy to forget. By delegating them to the
  AI, you ensure the project remains well-documented, clean, and easy for new developers (or even yourself, months
  later) to set up and use.
