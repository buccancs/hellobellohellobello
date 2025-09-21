[![Copilot](https://github.com/uclgsr/hellobellohellobello/actions/workflows/copilot-swe-agent/copilot/badge.svg)](https://github.com/uclgsr/hellobellohellobello/actions/workflows/copilot-swe-agent/copilot)

# Multi-Modal Physiological Sensing Platform

This repository contains a hub-and-spoke research platform for synchronous, multi-modal data collection with **production-ready hardware integration**.

- Hub (PC Controller, Python + PyQt6): Central controller for session management, device control, data aggregation,
  playback/annotation, and export to HDF5. Performance-critical components are bridged via a C++ native backend (
  PyBind11).
- Spoke (Android Sensor Node, Kotlin/Android): Mobile client with **true SDK integration** for hardware (RGB cameras, Topdon TC001 thermal camera, Shimmer GSR sensors), featuring calibrated thermal sensing and production-ready data recording, communicating with the Hub over a secured TCP/IP socket.

See documentation/System_Design.md for architecture overview and documentation/Implementation_Phases.md for implementation details.

## Project Structure

- **pc_controller/** — Python sources, native backend (C++), tests, and Gradle tasks for Python workflows
- **android_sensor_node/** — Android application (MVVM) with **production-ready SDK integration**: CameraX RGB, **true Topdon TC001 SDK integration** (±2°C thermal accuracy), and **real Shimmer GSR SDK** with 12-bit ADC precision
- **documentation/** — All project documentation, protocol guides, validation evidence, and LaTeX thesis sources
  - documentation/diagrams/ — Comprehensive system documentation diagrams and artifacts
  - documentation/phases/ — Development phase documentation and implementation summaries
  - documentation/architecture/ — Technical architecture, protocols, and integration guides
  - documentation/development/ — Development processes, testing pipelines, and quality assurance
- **demos/** — Feature demonstrations, GUI visualizations, and system integration examples
- **tools/** — Development utilities, analysis scripts, and build automation

## Prerequisites

- Windows 10/11 for the packaged PC Controller EXE (PyInstaller). Python 3.11+ for development.
- Android Studio (Giraffe/Koala or newer), Android SDK/NDK as required by your environment.
- Git and Java 17+ for Gradle builds.

## Building and Running

### PC Controller (Hub)

- Create a virtual environment and install requirements:
    - From repository root: `gradlew.bat :pc_controller:installRequirements` (creates pc_controller/.venv and installs
      deps)
- Run tests: `gradlew.bat :pc_controller:pyTest` or simply `pytest` from the repo root (pytest.ini is configured)
- Package EXE: `gradlew.bat :pc_controller:assemblePcController`
    - Output is placed under pc_controller/build/dist/pc_controller.exe

To run from sources (development):

- Activate the venv: `pc_controller\.venv\Scripts\activate`
- Launch: `python pc_controller\src\main.py`

### Android Sensor Node (Spoke)

- Open the project in Android Studio at the repository root (multi-project Gradle). The Android module is at
  android_sensor_node/app.
- Build debug: `gradlew.bat :android_sensor_node:app:assembleDebug`
- Unit tests: `gradlew.bat :android_sensor_node:app:testDebugUnitTest`
- Release APK: `gradlew.bat :android_sensor_node:app:assembleRelease`

> Note: Building the Android module requires a properly configured Android SDK/NDK in your environment.

### Aggregate Tasks

- Run both Python and Android unit tests: `gradlew.bat checkAll`
- Package PC EXE and Android APK: `gradlew.bat packageAll`

## Communication and Time Sync

- **Enterprise TLS Security**: TLS 1.2+ TCP/IP socket with mutual authentication and secure message framing. See docs/API_Documentation.md for comprehensive security configuration.
- **Fault Tolerance System**: Automatic heartbeat monitoring with device health tracking and exponential backoff reconnection. See docs/API_Documentation.md for implementation details.
- **NTP-like Time Sync**: Hardened handshake with 10-20 trials per device to compute robust per-device clock offsets with median statistics and outlier filtering. All data is timestamped using device-local monotonic clocks; alignment is performed in post-processing.
- **Flash Sync**: The Hub can broadcast a `flash_sync` command to trigger simultaneous on-screen flashes across devices for temporal validation (docs/markdown/Flash_Sync_Validation.md).

## Data Handling

- Android records locally per session (RGB MP4 + FR5 JPEGs, thermal CSV, GSR CSV). On stop, the device can transfer a
  ZIP of the session to the Hub over TCP.
- On PC, DataLoader and HDF5 exporter utilities aggregate and export into a research-ready HDF5 file.
- Backup strategy and automation: see documentation/BACKUP_STRATEGY.md and tools/backup_script.py.

## Native Backend

- pc_controller/native_backend contains the C++ code intended for PyBind11 integration for high-performance capture (
  Shimmer via Shimmer C-API and local webcam via OpenCV). Consult pc_controller/native_backend/README.md for build
  guidance.

## Documentation

📋 **[Complete Documentation Index](docs/Documentation_Index.md)** - Comprehensive guide to all documentation in this repository

### Quick Links

**Essential Documentation:**
- **[User Manual](docs/markdown/User_Manual.md)** - End-user guide for operating the platform
- **[Developer Guide](docs/markdown/Developer_Guide.md)** - Technical implementation and development procedures
- **[Production Deployment Guide](docs/Production_Deployment_Guide.md)** - Enterprise deployment and security hardening
- **[TRUE_TOPDON_SDK_INTEGRATION.md](docs/TRUE_TOPDON_SDK_INTEGRATION.md)** - **NEW**: Complete true SDK integration documentation with ±2°C thermal accuracy

**API References:**
- **[API Documentation](docs/API_Documentation.md)** - TLS Security, Heartbeat Monitoring, and protocol APIs

**Technical Specifications:**
- **[PROTOCOL.md](PROTOCOL.md)** - Complete communication protocol specification
- **[TEST_PLAN.md](docs/TEST_PLAN.md)** - Comprehensive testing strategy and procedures
- **[BACKUP_STRATEGY.md](docs/BACKUP_STRATEGY.md)** - Data backup and recovery procedures
- **[TOPDON_SDK_INTEGRATION_ANALYSIS.md](docs/TOPDON_SDK_INTEGRATION_ANALYSIS.md)** - SDK integration analysis and comparison

### System Documentation Diagrams
- **[docs/diagrams/](docs/diagrams/)**: **NEW** - Comprehensive visual documentation including:
  - **System Architecture**: High-level component diagrams, deployment views, PC/Android module maps
  - **Protocols**: TCP control protocol tables, message formats, command reference
  - **Sequences**: Recording workflows, file transfer, preview streaming
  - **Data Formats**: Session directory structure, CSV schemas, file organization
  - **State Machines**: RecordingController lifecycle, error handling flows
  - **Performance**: Timing diagrams, synchronization analysis, test coverage maps
  - **Examples**: Ready-to-adapt Mermaid/PlantUML snippets for reports and presentations
  - **[PLACEMENT_GUIDE.md](docs/diagrams/PLACEMENT_GUIDE.md)**: Integration guide for academic reports/thesis chapters

## Security & Ethics

- **Production Security**: Enterprise-grade TLS encryption with certificate-based authentication, comprehensive input validation, and security hardening. See docs/Production_Deployment_Guide.md for security configuration.
- **Data Protection**: Local recording with optional encrypted transfer and secure data handling throughout the pipeline.
- **Privacy Compliance**: Designed for research environments with appropriate consent and data protection procedures.

- Android local storage is encrypted using Android Keystore (AES256-GCM) for sensitive artifacts. The PC Controller
  requires authentication (see GUI docs) and supports anonymized participant IDs.
- Face blurring is available in video streams to protect privacy (see GUI playback tools in docs).

## Testing

- Python unit tests use pytest (see pytest.ini; tests located under pc_controller/tests). Run via `pytest` or Gradle
  task `:pc_controller:pyTest`.
- Android unit tests use JUnit; invoke via `:android_sensor_node:app:testDebugUnitTest`.
- System-level validation for temporal accuracy is described in docs/Flash_Sync_Validation.md.

## Contributing

- Follow Conventional Commits. Update CHANGELOG.md (Keep a Changelog) under the Unreleased section for every change.
- Branching model: main (stable), develop (integration), feat/... for features.

## License

This repository is intended for research use. See headers in source files and institutional policies for licensing
details.

## Running Tests with Hardware Connected

See docs/markdown/guide_running_tests_with_hardware.md for a step-by-step two-terminal workflow on Windows, covering
prerequisites, hardware setup, ports, and safe co-execution of tests and integration sessions.

Quick commands:

- .\scripts\run_all_tests.ps1
- pytest -q
- .\gradlew.bat --no-daemon :android_sensor_node:app:testDebugUnitTest
- python pc_controller\src\main.py
- python scripts\validate_sync.py --session-id <SESSION_ID> --base-dir .\pc_controller_data


---

# Getting Started (Cross-Platform)

This section provides a simple, step-by-step path for new users and examiners to install dependencies, run a basic session, and execute validation scripts.

## PC Controller Environment Setup
- Install Python 3.11+
- (Recommended) Create and activate a virtual environment
  - Windows (PowerShell):
    - python -m venv .venv
    - .\.venv\Scripts\Activate.ps1
  - macOS/Linux (bash):
    - python3 -m venv .venv
    - source .venv/bin/activate
- Install Python dependencies:
  - pip install -r pc_controller/requirements.txt

## Android Sensor Node (APK Build in Android Studio)
1. Open Android Studio and choose "Open" then select this repository root.
2. Ensure the Android module is at android_sensor_node/app.
3. Connect an Android device with USB debugging enabled.
4. Build and run:
   - Debug build: Gradle task :android_sensor_node:app:assembleDebug, or press Run in Android Studio.
   - Release APK: Gradle task :android_sensor_node:app:assembleRelease (signing config required).

## Quick Start (Run a Basic Recording Session)
1. Launch the PC Controller from sources:
   - Windows: python pc_controller\src\main.py
   - macOS/Linux: python3 pc_controller/src/main.py
2. Ensure your Android device(s) are on the same network as the PC.
3. Start the Android app; wait until it appears in the PC Controller device list.
4. Create a new session in the PC Controller (e.g., TEST_<date>).
5. Click Start Recording to begin a synchronized session.
6. Record for ~30 seconds, then click Stop Recording.
7. After stop, confirm the session folder under pc_controller_data/<SESSION_ID> contains metadata.json and incoming files from devices.

## Run System Verification Tests
- Automated (pytest):
  - pytest -q pc_controller/tests/test_system_end_to_end.py
  - These tests validate FR4 (Session Management) and FR1 (Simulation Mode) without hardware.
- Manual Checklist:
  - Follow docs/markdown/System_Test_Checklist.md to validate FR2, FR3/NFR2, FR5, FR8, FR10, and Endurance.

## Performance & Endurance Testing
- Simulated load with multiple clients:
  - python tools/run_performance_test.py --clients 8 --rate 30 --duration 900
  - Monitors CPU/Memory; verifies NetworkController stability under load.

## Post-Session Sync Validation
- Validate cross-device timing (FR3/NFR2):
  - python tools/validate_sync.py --session-id <SESSION_ID> --base-dir ./pc_controller_data
  - PASS: |Δt| < 5 ms across compared streams.



## Repository Hygiene
- Do not commit machine-specific files like local.properties (already covered by .gitignore). If one appears in VCS, remove it from version control and keep it local-only.
- Keep build artifacts out of Git: Gradle .gradle/, **/build/, Android .cxx/, and Python caches/venvs are ignored via the root .gitignore.
- Prefer adding new ignore patterns to the root .gitignore rather than project-specific files to maintain a single source of truth.
- Before committing, run tests (pytest for PC, Gradle unit tests for Android) to avoid breaking main.
