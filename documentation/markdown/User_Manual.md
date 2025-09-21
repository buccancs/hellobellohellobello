# Multi-Modal Physiological Sensing Platform — User Manual

This manual explains how to install, run, and validate the platform for research use.

## 1. Components

- PC Controller (Hub): PyQt6 application controlling sessions, aggregating data, exporting HDF5.
- Android Sensor Node (Spoke): Records RGB/Thermal/GSR and responds to commands from the Hub.

## 2. Installation

### 2.1. Windows PC (Hub)

1. Install Python 3.11+ and Git.
2. Clone the repository.
3. Build prerequisites (first time only):
    - From repo root: `./gradlew.bat :pc_controller:installRequirements`
4. Launch (dev mode): `python pc_controller\src\main.py`

### 2.2. Android (Spoke)

1. Open the project in Android Studio (Hedgehog or newer).
2. Connect a device with USB debugging enabled.
3. Build and run the `app` module.

## 3. Running a Pilot Session

1. Start the PC Controller.
2. Ensure Android device(s) are on the same secure network.
3. Use Dashboard to discover devices; press Start Recording.
4. Use "Flash Sync" to trigger synchronized white-screen flash across nodes for temporal check.
5. Press Stop Recording; the Spoke will transfer session files automatically to the Hub.
6. Use Playback & Annotation tab to inspect streams and Export to HDF5 if needed.

## 4. Camera Calibration (FR9)

Use a checkerboard target and capture several images at different orientations.

- Use the calibration utility (Tools) to compute intrinsics and save parameters as JSON.
- Parameters are used for undistortion and improved pose accuracy.

## 5. Verifying System Accuracy (Flash Sync Validation)

Follow the Hardware Validation Protocol (docs/Hardware_Validation_Protocol.md) to run a 5-minute session with at least
five Flash Sync events.

After the Android devices finish transferring files to the PC (the PC Controller logs will indicate receipt and
unpacking):

1. Locate the session directory under: pc_controller_data/<session_id>/
2. From a PowerShell terminal at the repository root, run:

   python scripts\validate_sync.py --session-id <YOUR_SESSION_ID>

   Optional:
    - --base-dir <path> to override the default ./pc_controller_data
    - --tolerance-ms <float> to adjust the PASS/FAIL threshold (default 5.0)
3. The tool will detect flash frames in each video, align timestamps using the recorded NTP-like offsets, and print
   per-event spreads and an overall verdict.

PASS criteria: All events must be within <5 ms across streams (NFR2). If the verdict is FAIL, review camera
positioning/lighting and repeat the protocol.

## 6. Packaging

- PC Controller (Windows): `./gradlew.bat :pc_controller:assemblePcController` produces an EXE under
  `pc_controller/build/dist/`.
- Android APK: `./gradlew.bat :android_sensor_node:app:assembleRelease`.

## 7. Safety and Data Protection

- Participant IDs only; no PII stored with sensor data.
- Face blurring can be enabled in video post-processing pipeline.
- All communication is TLS 1.2+; data at rest should be encrypted on Android using Keystore-backed AES256-GCM.

## 8. Troubleshooting

- If no devices appear: check firewall, TLS certificates, and that both PC and phone are on the same network.
- If HDF5 export fails: validate CSV paths and ensure `h5py` is installed.
- If calibration cannot find corners: verify checkerboard size matches settings and lighting is adequate.



## 9. PC GUI Overview

This section provides a quick tour of the main PC Controller GUI and how to operate it during data collections.

### 9.1 Device List
- Columns typically include: Device Name, IP/Port, Status, Sync Offset, Preview.
- Status indicators:
  - Discovered: Device has been seen via Zeroconf.
  - Online: PC successfully connected and exchanged capabilities.
  - Recording: Device is actively recording (after Start Recording).
  - Offline/Disconnected: Device connection lost (see FR8 fault tolerance in the Checklist).
- Sync Offset:
  - Shows last measured clock offset (ns or ms). Values near 0 indicate good alignment.
  - Auto re-sync may trigger if delay/variance exceeds thresholds.
- Preview:
  - Thumbnails or a count of incoming preview frames for quick health checks.

### 9.2 Session Control Panel
- Create Session: Creates a new session folder and metadata.json on the PC. The new session becomes active.
- Start Recording: Broadcasts a start_recording command to all connected devices. The Shimmer (if managed on PC) also starts streaming/recording.
- Stop Recording: Broadcasts stop to all devices; finalizes metadata.json with end_time_ns and triggers data transfer (FR10).
- Transfer Files: If supported, requests devices to upload their session artifacts to the PC FileTransferServer.
- Time Sync All: Runs NTP-like handshake to estimate per-device clock offsets and update the Sync Offset column.
- Flash Sync: Triggers bright flash on devices to create a visual timing marker across streams.

### 9.3 Using Flash Sync
- Click Flash Sync while devices are online (recording optional but recommended during validation runs).
- For validation sessions, trigger several flashes spaced ~30–60 s apart.
- After the session, run scripts/validate_sync.py to quantify inter-stream timing differences.

### 9.4 Camera Calibration Workflow
1. Prepare a checkerboard target (size per your lab’s standard, e.g., 9×6 inner corners).
2. In the PC Controller Tools/Calibration view:
   - Capture several images at varying angles and distances.
   - Verify the checkerboard corners are detected in previews.
3. Run calibration to compute intrinsics. Save parameters as JSON in your project (e.g., docs/calibration/<device>.json).
4. Use saved intrinsics for undistortion and improved pose estimation in downstream analysis.

### 9.5 Locating and Interpreting Session Data
- Session Root: pc_controller_data/<SESSION_ID>/
- Contents:
  - metadata.json: Includes session_id, created_at_ns, start_time_ns, end_time_ns, and state.
  - gsr.csv: (If PC-side GSR capture is enabled/simulated) timestamp_ns, gsr_dummy (or calibrated units if real sensor).
  - device_<NAME>/ subfolders: Per-Android-device artifacts such as:
    - rgb.mp4 (or similar): RGB video.
    - thermal.csv/.mp4 or frames: Thermal data per device capability.
    - logs/*.txt: Optional logs.
- Time Semantics:
  - Timestamps are typically monotonic clock-based. Use validate_sync.py and stored offsets to align streams.
  - PASS criteria for NFR2: |Δt| < 5 ms across compared streams.
