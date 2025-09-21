# Hardware Validation Protocol (SOP)

This Standard Operating Procedure (SOP) defines the steps to perform a real, hardware-in-the-loop validation of the
system’s end-to-end temporal synchronization using Flash Sync events. It adheres to the project requirements and design
in docs/markdown/ProjectGuidelines_2024-25.md and docs/latex/3.tex (FR3, FR7; NFR2).

Audience: Human researchers and QA engineers validating a deployed setup.

Duration: ~20–30 minutes (including a 5-minute recording).

---

## 1. Objective

Validate that all participating streams (PC webcam and at least two Android Spokes’ RGB videos) exhibit temporal
alignment within <5 ms for visible Flash Sync events when clocks are aligned using the built-in NTP-like handshake.

## 2. Equipment and Software

- 1x PC (Windows) running the PC Controller (Hub)
    - Python environment per pc_controller/requirements.txt installed (OpenCV, pandas, numpy)
    - USB webcam connected and functional (used by the Hub for the reference stream)
- 2x Android devices (Spokes)
    - With the Sensor Spoke app installed and running
    - Back cameras unobstructed and pointed towards the same scene (so they can see each other’s flashes or a common
      reflective surface)
- 1x Shimmer sensor connected via wired dock to the PC (optional for this test; can remain connected)
- 1x Shimmer sensor connected via BLE to one Android device (optional for this test)
- Wi-Fi network allowing the Hub and Spokes to communicate over TCP/IP

## 3. Physical Setup

1. Place the two Android devices on stable mounts facing a common field of view (e.g., a white wall or a reflective
   panel) so that their screens are visible to the camera(s) when they flash or the flash is indirectly visible as scene
   illumination.
2. Position the PC’s webcam to see the same scene where flash illumination is visible (e.g., both phone screens, or
   light reflecting from a white surface).
3. Ensure sufficient ambient lighting to avoid sensor auto-exposure extremes, but not so bright that flashes are
   indistinguishable.

## 4. Software Preparation

1. Start the PC Controller application (pc_controller/src/main.py or packaged app). Ensure the “Dashboard” shows the
   local webcam preview and discovered Android devices.
2. On each Android device, launch the Sensor Spoke app (MainActivity). Keep the app in the foreground. Ensure required
   camera permissions are granted.
3. Verify network discovery: In the PC Controller Logs tab, confirm the devices are discovered. If not, ensure all
   devices share the same network and that the Android service is running (RecordingService foreground notification
   should be present).

## 5. Session Execution (5 minutes)

1. In the PC Controller toolbar, click “Start Session.” This will:
    - Perform a time sync handshake to measure per-device clock offsets.
    - Begin recording on the Android devices (RGB camera video.mp4 and periodic high-res JPEGs) and start the PC webcam
      recording.
2. Allow the session to run uninterrupted for 5 minutes.
3. During the recording, trigger Flash Sync at least five times at random intervals:
    - In the PC Controller toolbar, click “Flash Sync.”
    - Each Android device will briefly display a full-screen white overlay.
    - The Android app logs each event timestamp (flash_sync_events.csv) and records video where the flash will be
      visible.
    - The PC webcam will also capture the scene flash illumination.
4. After ~5 minutes, click “Stop Session.” The PC Controller will:
    - Stop local recording.
    - Start a file receiver and broadcast a file transfer request to all Android devices.
    - Store all received files into the session directory.

## 6. Data Locations

- PC Hub session data root: pc_controller_data/<session_id>/
    - Local PC data:
        - webcam.avi — PC webcam video
        - gsr.csv — local Shimmer samples (if active)
        - session_metadata.json — includes NTP-like clock offsets (per device)
    - Each Android device (identified by its model-based device_id):
        - rgb/video.mp4 — CameraX video
        - rgb/frames/*.jpg — periodic high-res frames
        - flash_sync_events.csv — log of flash timestamps (automatically included in transfer)

Note: If any device’s flash_sync_events.csv is missing after transfer, re-run the session with the current app version.
The Android app includes this file in the transferred archive.

## 7. Automated Analysis

Run the validation script to compute inter-stream timing differences and PASS/FAIL.

- Activate your Python environment with the dependencies in pc_controller/requirements.txt installed.
- From the repository root, run:

  python scripts\validate_sync.py --session-id <YOUR_SESSION_ID>

Optional arguments:

- --base-dir: Override the base directory for sessions (default: ./pc_controller_data)
- --tolerance-ms: Override pass/fail tolerance in milliseconds (default: 5.0)

The script will:

1. Locate the session directory using the given session_id.
2. Load session_metadata.json to obtain per-device clock offsets.
3. For each Android device, load flash_sync_events.csv and its video.
4. For the PC webcam and each Android video, detect the precise frame for every flash event (luminance spike).
5. Estimate each video’s start time on the PC master clock using the detected frames and the aligned (offset-corrected)
   flash timestamps.
6. Compute the cross-stream time differences for each event and print a Validation Report. PASS if all events are
   within <5 ms tolerance.

## 8. Acceptance Criteria

- All Flash Sync events across the PC webcam and all Android devices exhibit <5 ms absolute difference in the final
  report.
- Any missing or ambiguous detections must be flagged; a failing result requires re-collection.

## 9. Safety and Data Integrity Notes

- Ensure no personal identifiers are visible in the recordings; use anonymized IDs.
- Follow safe device handling and any Shimmer current limit recommendations.
- Back up session data per BACKUP_STRATEGY.md.

## 10. Troubleshooting

- Devices not discovered: Verify Wi-Fi network and that the Android RecordingService is running.
- No flash visible in videos: Reposition cameras to ensure flashes illuminate the scene.
- Validation script errors: Confirm OpenCV (opencv-python) is installed, and that session_metadata.json and
  flash_sync_events.csv files exist as described.
