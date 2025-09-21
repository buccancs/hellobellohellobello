# AI Agent Guidelines for the Multi-Modal Physiological Sensing Platform

## 1. Project Context and Documentation

This document contains the primary set of instructions for you, the AI coding agent. All guidelines herein are derived
from and must be consistent with the official project documentation, which serves as the ultimate source of truth:

* `docs/latex/1.tex` (Introduction and Objectives)
* `docs/latex/2.tex` (Background and Literature Review)
* `docs/latex/3.tex` (Requirements and Analysis)

[//]: # (* `docs/latex/4.tex` &#40;Design and Implementation Details&#41;)

You must adhere to the specifications, architectures, and requirements detailed in these source documents by following
the specific instructions laid out in the subsequent sections of this guideline file.

## **2. Project-Level Instructions for the AI Agent**

You are an expert software architect and developer. Your task is to generate the complete source code and project
structure for the "Multi-Modal Physiological Sensing Platform."

### **2.1. Project Overview and Goal**

The goal is to create a scientific data acquisition tool consisting of two main applications: a **PC Controller (Hub)**
and an **Android Sensor Node (Spoke)**. The system must synchronously record data from multiple sensors with high
temporal precision, preparing it for research and machine learning analysis.

### **2.2. Core Architecture: Hub-and-Spoke Model**

The system must be built on a **Hub-and-Spoke** client-server architecture.

* **Hub (PC Controller):** The central master controller. It manages sessions, controls all connected devices, and
  aggregates all data.
* **Spoke (Android Sensor Node):** A mobile client responsible for all hardware interfacing, data capture, and local
  storage.

-----

## **3. Coding Guidelines and Standards**

You must adhere to the following coding standards and best practices for all generated code.

### **3.1. General Guidelines**

* **Language:** All code, comments, and documentation must be in English.
* **Readability:** Prioritize clean, readable, and self-documenting code. Use clear and descriptive names for variables,
  functions, and classes. Keep functions short and focused on a single responsibility.
* **Documentation:**
    * Generate Python docstrings for all public modules, classes, and functions.
    * Generate KDoc comments for all public classes and methods in the Kotlin codebase.
* **Error Handling:** Implement robust error handling using `try-except` (Python) and `try-catch` (Kotlin) blocks.
  Provide meaningful error messages that can be logged and displayed to the user.
* **Logging:** Implement structured logging throughout both applications to facilitate debugging. Log key events,
  errors, and state changes.

### **3.2. Python (PC Controller) Guidelines**

* **Style Guide:** Adhere strictly to the **PEP 8** style guide for all Python code.
* **Type Hinting:** All function and method signatures **must** include type hints.
* **Threading:** All network operations and other long-running tasks must be executed in background `QThread` workers to
  ensure the PyQt6 GUI remains responsive and never blocks.
* **Performance:** For performance-critical operations (e.g., real-time sensor data processing, video frame handling),
  you must implement the logic in the C++ `native_backend` and expose it to Python via PyBind11.

### **3.3. Kotlin (Android Spoke) Guidelines**

* **Style Guide:** Adhere strictly to the official **Kotlin Style Guide** recommended by Google for Android development.
* **Architecture:** The application **must** follow the **MVVM (Model-View-ViewModel)** architecture. Separate UI
  logic (Activities/Fragments) from business logic (ViewModels) and data sources (Repositories).
* **Asynchronicity:** **All** asynchronous operations (network requests, file I/O, database access) must be handled
  using **Kotlin Coroutines**.
* **Lifecycle Awareness:** All components that interact with the Android framework must be lifecycle-aware. Use Android
  Architecture Components like `ViewModel` and `LiveData` to manage UI state and data across configuration changes.
* **Resource Management:** Use the `use` block for all `Closeable` resources (e.g., file streams) to ensure they are
  properly closed, even in the event of an exception.
* **Modularity:** Any new sensor integration must implement the common `SensorRecorder` interface to ensure a consistent
  API for the `RecordingController`.

### **3.4. Version Control (Git) Guidelines**

* **Commit Messages:** All commit messages must follow the **Conventional Commits** specification. Each message must
  have a type (e.g., `feat`, `fix`, `docs`, `refactor`) and a concise description.
    * *Example:* `feat: Implement NativeShimmer C++ backend for low-latency GSR`
* **Branching:** Use a simple GitFlow-like branching model:
    * `main`: Contains stable, production-ready code.
    * `develop`: Integration branch for new features.
    * `feat/...`: Feature branches for new development.

-----

## **4. PC Controller (Hub) Specifications**

### **4.1. Technology Stack**

* **Language:** Python 3.11+
* **GUI Framework:** PyQt6
* **Performance-Critical Backend:** C++ integrated via PyBind11
* **Dependencies:** `zeroconf`, `PyQtGraph`, `pandas`, `h5py`.

### **4.2. Project Structure**

Generate the following directory structure for the PC Controller:

```plaintext
/pc_controller/
├── src/
│   ├── main.py           # Main application entry point
│   ├── gui/              # GUI modules (GUIManager, DeviceWidget)
│   ├── network/          # Network modules (NetworkController, WorkerThread)
│   ├── core/             # Core logic (SessionManager, TimeSyncService)
│   └── data/             # Data handling (DataAggregator, HDF5Exporter)
├── native_backend/       # C++ source for PyBind11 extension
└── tests/                # Pytest unit tests
```

### 4.3. Key Module Implementation

* **`GUIManager`:** Implement a tabbed interface (Dashboard, Logs, Playback). The Dashboard must use a dynamic grid to
  display live video previews (`QLabel`) and real-time GSR plots (`PyQtGraph`).
* **`NetworkController`:** Must use `zeroconf` for device discovery. It must run a TCP server in a background `QThread`
  and spawn a new `WorkerThread` for each connected Android device to handle non-blocking communication.
* **`native_backend` (C++):**
    * Create a `NativeShimmer` class to connect to a docked Shimmer sensor via a serial port (e.g., "COM3"). This must
      run in a dedicated C++ thread and use a thread-safe, lock-free queue to pass data to Python.
    * Create a `NativeWebcam` class using OpenCV to capture frames from a local PC webcam, also in a C++ thread. Use
      zero-copy memory sharing to expose frames to Python as NumPy arrays.

-----

## **5. Android Sensor Node (Spoke) Specifications**

### **5.1. Technology Stack**

* **Language:** Kotlin
* **Architecture:** Follow the **MVVM (Model-View-ViewModel)** pattern.
* **Asynchronicity:** Use Kotlin Coroutines for all background tasks.
* **Key Libraries:** CameraX, Nordic BLE Library, `UVCCamera` (for thermal).

### **5.2. Project Structure**

Generate the following package structure for the Android application:

```plaintext
/com/yourcompany/sensorspoke/
├── ui/                 # UI Layer (MainActivity, MainViewModel)
├── service/            # Background Service Layer (RecordingService)
├── controller/         # Core Logic (RecordingController)
├── sensors/            # Sensor Integration Layer (SensorRecorder interface and implementations)
│   ├── rgb/
│   ├── thermal/
│   └── gsr/
├── network/            # Network Communication Layer (NetworkClient, FileTransferManager)
├── data/               # Data Models and Storage (model/, storage/)
└── utils/              # Utility and Helper Classes (TimeManager)
```

### **5.3. Sensor Integration Logic**

* **`RgbCameraRecorder`:** Use CameraX to configure two simultaneous streams: a 1080p MP4 video file and a continuous
  stream of high-resolution, timestamped JPEG images (FR5). [1, 1]
* **`ThermalCameraRecorder`:** Use the `UVCCamera` library to interface with the Topdon TC001. Implement the frame
  callback to parse the raw thermal data and write each frame as a new row in a CSV file, prefixed with a nanosecond
  timestamp.
* **`ShimmerRecorder`:**
    * Use the Nordic BLE library for robust communication.
    * Implement logic to send start (`0x07`) and stop (`0x20`) commands.
    * In the notification callback, parse the incoming data packets.
    * **Critical Technical Detail:** The GSR value must be calculated from the raw sensor value using the correct *
      *12-bit ADC resolution (0-4095 range)**, not 16-bit. This is a mandatory requirement for data accuracy.
    * Log the converted GSR (in microsiemens) and raw PPG values to a timestamped CSV file.

-----

## **6. Communication and Synchronization**

* **Protocol:** All communication between the Hub and Spokes must be over a **TLS 1.2+ secured TCP/IP socket**. All
  control messages must be **JSON-formatted** (FR7, NFR5). [1, 1]
* **Time Synchronization:** Implement an NTP-like handshake upon connection to calculate the clock offset between the PC
  and each Android device. All data must be timestamped using a local high-precision monotonic clock. The final
  alignment will be done in post-processing using the calculated offset, with a target accuracy of \<5 ms (FR3,
  NFR2). [1, 1]

-----

## **7. Testing and Verification**

* **Mandatory Unit Tests:** You must generate unit tests for all new code.
* **Frameworks:** Use `pytest` for the Python PC Controller and `JUnit`/`Robolectric` for the Android application.
* **Verification:** The system's end-to-end temporal synchronization must be verifiable. Implement a "Flash Sync"
  command that causes all Android screens to flash simultaneously. This will be used to confirm that timestamps across
  all video and data streams align to within the required 5ms tolerance (FR7).

-----

## **8. Security and Data Handling**

* **Data Protection:** The Android app must use AES256-GCM encryption via the Android Keystore for local storage. The PC
  controller must require authentication.
* **Anonymization:** The system must use participant ID codes and must not store personal identifiers with sensor data.
  Implement a feature for face blurring in the video streams.
* **Device Safety:** Ensure all hardware configurations adhere to safety specifications, including current limits for
  the GSR sensor and passive sensing for the thermal camera.

-----

## **9. Mandatory Libraries and SDKs**

You **must** use the following official and community-provided libraries for all hardware and data streaming
integration. Do not use generic or alternative libraries for these specific tasks unless explicitly instructed.

### **9.1. PC Controller (Hub) Libraries**

* **Shimmer Sensor Communication (Python Layer):**
    * **Repository:** `https://github.com/seemoo-lab/pyshimmer`
    * **Purpose:** This library must be used for all Python-level interactions with the Shimmer sensor, including device
      discovery, connection, and data streaming when the sensor is connected via the PC's Bluetooth or serial dock.

* **Shimmer Sensor Communication (C++ Backend):**
    * **Repository:** `https://github.com/ShimmerEngineering/Shimmer-C-API`
    * **Purpose:** The high-performance `native_backend` C++ module must be built using the official Shimmer C-API. This
      ensures the lowest-level, most reliable communication with a docked Shimmer sensor for the "High-Integrity Mode."

### **9.2. Android Sensor Node (Spoke) Libraries**

* **Shimmer Sensor Communication (Kotlin/Android):**
    * **Repository:** `https://github.com/ShimmerEngineering/ShimmerAndroidAPI`
    * **Purpose:** This is the **official and mandatory** API for all Shimmer3 GSR+ sensor interactions on Android. Use
      it for BLE connection, sending start/stop commands, and parsing incoming data packets for the "High-Mobility
      Mode."

* **Thermal Camera Integration (Kotlin/Android):**
    * **Primary Repository:** `https://github.com/buccancs/topdon-sdk`
    * **Alternative/Reference:** `https://github.com/CoderCaiSL/IRCamera`
    * **Purpose:** You must use a dedicated SDK for the Topdon TC001 thermal camera instead of a generic UVC library.
      The `topdon-sdk` is preferred as it provides a direct, stable interface to the hardware. This will ensure access
      to all device-specific features and improve reliability.

### **9.3. Data Streaming and Synchronization**

* **Lab Streaming Layer (LSL):**
    * **Repository:** `https://github.com/sccn/labstreaminglayer`
    * **Purpose:** LSL is to be used for **complementary, real-time data monitoring only**. The Android app should
      create an LSL outlet to stream live GSR data for external visualization or debugging.
    * **Constraint:** LSL **must not** be used for the primary, persistent data recording or for the core time
      synchronization between the Hub and Spokes. The primary synchronization must be handled by the custom NTP-like
      protocol.

-----

## **10. Project Maintenance, DevOps, and Documentation**

Beyond the initial code generation, you are responsible for maintaining the project's health, documentation, and data
integrity. You must adhere to the following maintenance tasks as part of your ongoing work.

### **10.1. Changelog Maintenance**

Maintaining a clear and accessible changelog is a mandatory requirement for this project. It is the primary way we
communicate changes to other developers and end-users.

* **File and Format:** You must create and maintain a file named `CHANGELOG.md` in the root of the repository. This file
  must adhere to the **"Keep a Changelog"** format.
* **Content:**
    * The changelog must have an "Unreleased" section at the top for ongoing changes.
    * For every new feature, bug fix, or change you implement, you must add a corresponding entry to the "Unreleased"
      section under the appropriate category (`Added`, `Changed`, `Fixed`, `Removed`, `Security`, etc.).
    * Your entries should be user-focused and describe the impact of the change, not just the technical details. Do not
      simply dump git logs.
* **Link to Commits:** This process is directly linked to the **Conventional Commits** standard. Every commit of type
  `feat`, `fix`, or any commit with a `BREAKING CHANGE` footer must have a corresponding entry in the changelog.

### **10.2. Data Backup Strategy**

The data collected by this platform is invaluable scientific research data. Therefore, a robust backup strategy is
critical to prevent data loss.

* **Documentation:** You must create a document named `BACKUP_STRATEGY.md` in the root of the repository.
    * This document must outline the **3-2-1 Backup Rule** (3 copies of the data, on 2 different types of media, with 1
      copy off-site) as the recommended strategy for researchers using this platform.
    * The document should provide a clear, easy-to-follow template that helps researchers plan and document their
      specific backup implementation (e.g., which cloud provider to use, where to store the off-site physical drive).
* **Automation Script:**
    * You must create a well-documented Python script named `backup_script.py` in a `/scripts` directory.
    * This script should automate the local part of the backup process. It should be configurable to copy all data from
      the main session data folder to a specified local backup destination (e.g., an external hard drive).
    * The script's documentation must include clear instructions on how to configure the source and destination paths
      and how to schedule the script to run automatically (e.g., using `cron` on Linux/macOS or Task Scheduler on
      Windows).

### **10.3. Other Maintenance Tasks**

* **README Maintenance:** You must keep the root `README.md` file up-to-date. If you add a new feature, dependency, or
  change the setup process, you must update the README to reflect these changes. The README should always contain clear,
  accurate instructions for setting up and running the project.
* **Dependency Management:** When you add a new library or SDK, you must add it to the appropriate dependency file (
  `requirements.txt` for Python, `app/build.gradle.kts` for Android). Keep these files clean and free of unused
  dependencies.
* **`.gitignore` Maintenance:** You must ensure that temporary files, build artifacts, IDE-specific configuration
  files (e.g., `.idea/`), and sensitive files are included in the `.gitignore` file to keep the repository clean.

-----
