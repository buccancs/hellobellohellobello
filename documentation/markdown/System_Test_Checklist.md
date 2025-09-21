# System Test Checklist

This checklist guides a human tester through the end-to-end verification of key Functional and Non-Functional Requirements (FR/NFR) that require physical hardware and real-world interaction. It complements automated tests and follows the scope outlined in TEST_PLAN.md and Chapter5_Evaluation_and_Testing.md.

Pre-requisites
- PC Controller application built and runnable on the test machine.
- 2 or more Android devices with the Android Sensor Node app installed.
- One Shimmer3 GSR+ sensor and its required accessories.
- Local network with stable Wi‑Fi for all devices and the PC.
- Accurate PC clock; ensure NTP is enabled.
- Sufficient storage on all devices.

Artifacts/Paths
- PC session root: as configured in the PC app; by default pc_controller_data/<SESSION_ID>.
- Android device session files: automatically transferred into the PC session folder upon stop (see FR10).
- Sync validation tool: scripts/validate_sync.py on the PC.

Section A — FR2 & FR5: Synchronized Recording with Real Hardware
Goal: Verify synchronized start/stop across multiple Android devices and the Shimmer sensor, and that all expected files are created.

Steps
1. Setup
   - Power on the Shimmer GSR+ and ensure electrodes are properly connected to the participant or a known-good dummy load.
   - Connect at least two Android devices to the same Wi‑Fi network as the PC.
   - Launch the Android Sensor Node app on each device; ensure they show Connected to the PC controller once discovered.
   - Launch the PC Controller and open the dashboard that lists connected devices.
2. Create Session
   - In the PC Controller, create a new session named SYNC_FR2_FR5_<date>.
   - Confirm a session folder appears on the PC (pc_controller_data/<SESSION_ID>) with a metadata.json file.
3. Start Recording (Synchronized)
   - On the PC, click Start Recording to broadcast to all devices and the Shimmer manager.
   - Observe:
     - PC GUI indicates Recording for each Android device and the GSR stream.
     - Live previews appear for RGB and thermal (if available), and a GSR plot starts updating.
4. Record for 5 Minutes
   - Maintain recording for at least 5 minutes. Monitor that all devices remain online and streaming.
5. Stop Recording (Synchronized)
   - On the PC, click Stop Recording. Ensure all Android devices and the GSR stream stop within a few seconds.
6. Verify Files and Alignment
   - On the PC session folder, verify presence of:
     - For each Android device: video files (RGB MP4), thermal data files (e.g., CSV/frames per design), and any device-specific logs.
     - Shimmer GSR CSV (if recorded by PC or transferred from Android, depending on your deployment).
   - Open metadata.json and confirm state is Stopped and end_time_ns is set.
   - Check that file timestamps (header times or per-row timestamps) show aligned start and stop within expected tolerances (exact per-module formats described in TEST_PLAN.md). This will be further validated in Section B.

Expected Result
- All devices recorded for approximately 5 minutes with synchronized start/stop; all relevant files exist in the PC session folder structure and show reasonable sizes and timestamps.

Section B — FR3 & NFR2: Time Sync Accuracy
Goal: Validate cross-device time synchronization accuracy better than ±5 ms after the synchronized session.

Steps
1. Ensure the session from Section A has completed and all files are present in the PC session folder.
2. On the PC, open a terminal and run the sync validation script:
   - python3 scripts/validate_sync.py --session <PATH_TO_SESSION_FOLDER>
3. Observe the output. The tool computes inter-stream offsets (e.g., PC vs. Android RGB/thermal, GSR vs. video) using event markers or timestamp streams as implemented in validate_sync.py and validate_sync_core.

Acceptance Criteria
- The reported time difference for each pairwise comparison must be less than 5 ms (|Δt| < 0.005 s). Record the values in your test report.

Troubleshooting
- If drift exceeds 5 ms, ensure:
  - PC and Android devices had NTP enabled and a stable network during recording.
  - No device experienced heavy CPU throttling.
  - Repeat the test after restarting all devices.

Section C — FR8: Fault Tolerance During Recording
Goal: Confirm the system handles transient device disconnects gracefully and auto-rejoins the session.

Steps
1. Create a new session named FAULT_FR8_<date> and start recording as in Section A.
2. During recording, disconnect one Android device from Wi‑Fi (toggle Airplane Mode or move out of range).
3. Observe the PC GUI:
   - The affected device should transition to Offline or Disconnected within a reasonable timeout.
   - Recording continues on remaining devices.
4. Reconnect the Android device to Wi‑Fi and reopen the app if required.
5. Observe the PC GUI:
   - The device should automatically reconnect and rejoin the session (status Online/Recording) without needing to stop the whole session.
6. Continue recording for an additional 1–2 minutes, then stop the session.

Expected Result
- The PC shows timely Offline status, then the device rejoins automatically and is included in the final session outputs without crashing or blocking other devices.

Section D — FR10: Post-Session Data Transfer
Goal: Confirm all device-side media and thermal data are uploaded to the PC after stopping the session.

Steps
1. After stopping the session in Section A or C, wait for the transfer process to complete. The PC may run a file receiver (see Data Aggregator/FileReceiverServer).
2. In the PC session folder, verify for each Android device a device-specific subfolder (naming per TEST_PLAN.md), containing:
   - RGB video files (e.g., .mp4) covering the entire recording duration.
   - Thermal data (CSV, images, or video as per device capability).
   - Any GSR-related files if the device logs them.
3. Check file sizes are non-zero and timestamps fall within the session start/end window as recorded in metadata.json.
4. If transfers are incomplete after a reasonable time (e.g., 2–5 minutes depending on sizes), consult the PC logs and Android app logs; re-try the stop procedure if safe.

Acceptance Criteria
- All expected files from all Android devices appear in the session folder on the PC with correct sizes and timestamps and can be opened without corruption.

Section E — Endurance Testing (NFR1, NFR7)
Goal: Validate stability and scalability by running a continuous multi-device recording session for at least 2 hours with real Android devices and monitoring for crashes or memory leaks.

Steps
1. Setup
   - Prepare at least two real Android devices with the Android Sensor Node app installed and connected to the same Wi‑Fi as the PC.
   - Ensure all devices and the PC have sufficient battery power and are connected to chargers to prevent sleep or throttling.
   - Disable battery optimizations for the Android app and keep the screen awake (developer options) as required by your deployment.
   - Confirm PC storage has enough free space for large recordings (e.g., tens of GB depending on cameras/data rates).
2. Create Session
   - In the PC Controller, create a new session named ENDURANCE_<date>.
   - Verify the session folder appears under pc_controller_data/<SESSION_ID>.
3. Start Recording
   - Start recording from the PC Controller to broadcast to all connected Android devices.
   - Confirm all devices show status Recording in the PC UI.
4. Continuous Run (>= 2 hours)
   - Maintain the session for at least 2 hours. Keep devices stationary and Wi‑Fi stable.
   - Monitoring during the run:
     - PC (choose appropriate tools):
       - Windows: Task Manager (Processes/Performance) for CPU, Memory (Working Set/RSS), Disk, and Network.
       - macOS: Activity Monitor (CPU/Memory) and/or `top`/`vm_stat` in Terminal.
       - Linux: `top`/`htop`, `free -m`, `vmstat 5`, `iostat -dx 5`, optional `pidstat -r -u -p <PID> 5`.
       - Observe the PC Controller process CPU and memory trends; note any steady unbounded growth.
     - Android devices:
       - With USB debugging enabled, periodically run `adb shell dumpsys meminfo com.yourcompany.sensorspoke` and `adb shell top -n 1 | grep sensorspoke` to spot abnormal memory growth or CPU spikes.
       - Keep `adb logcat` running to catch crashes, ANRs, or fatal exceptions.
     - General health:
       - Watch the PC UI for disconnect/reconnect events or error logs.
       - Monitor network stability (optional: continuous `ping` to devices/AP).
5. Stop Recording
   - After 2+ hours, stop the recording from the PC Controller.
   - Wait for file transfers to complete automatically (per FR10) and verify completion in the PC session folder.

Acceptance Criteria
- The recording session runs continuously for ≥ 2 hours without crashes on the PC or any Android device.
- All resulting large data files are successfully transferred to the PC session folder and can be opened.

Documentation
- Record the session ID, device IDs, pass/fail status per requirement (FR2, FR5, FR3/NFR2, FR8, FR10, NFR1, NFR7), any anomalies, logs, and screenshots.

References
- TEST_PLAN.md — System/Integration Tests definitions and acceptance criteria.
- Chapter5_Evaluation_and_Testing.md — Methodology and metrics, including sync validation approach.
- scripts/validate_sync.py — Tool to compute stream alignment and timing differences.
