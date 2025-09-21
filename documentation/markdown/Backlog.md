# Backlog — Multi-Modal Physiological Sensing Platform

This document tracks near-term tasks and priorities. It complements CHANGELOG.md and TEST_PLAN.md.

## Done in this session (2025-08-17)
- Added centralized pc_controller/config.json and config loader (NFR8).
- Implemented asyncio-based UDP TimeSyncServer replying with monotonic ns (FR3).
- PC main.py now starts the TimeSyncServer in a background thread and shuts it down on exit (FR3/NFR8).
- Android TimeManager: added sync_with_server() and getSyncedTimestamp() (FR3/NFR2).

## Done in this session (2025-08-16)
- Verified all Python unit tests (pc_controller) passed: 31/31 green.
- Verified all Android JVM unit tests (android_sensor_node) passed; CameraX/UI smoke tests are intentionally skipped.
- Android Spoke improvements recently landed and are green:
  - RgbCameraRecorder: timestamped MP4 filename; ~6–8 FPS downsampled preview emission.
  - ThermalCameraRecorder: thermal.csv header + metadata.json scaffold.
  - RecordingService: richer query_capabilities response (device_id, Android version, service_port, sensors, cameras).
- CHANGELOG.md updated under Unreleased with a maintenance note for this verification.

## Priority Backlog

### P3 — Android Spoke Completeness (MVVM, Sensors)
- ThermalCameraRecorder (Topdon TC001) SDK integration
  - Stream radiometric frames (256x192@25 Hz) and append CSV rows: `timestamp_ns,w,h,v0..v49151`.
  - Persist accurate metadata (emissivity, calibration, temperature units) and device settings per session.
- ShimmerRecorder (ShimmerAndroidAPI)
  - Implement BLE connect, send start (0x07) and stop (0x20) commands.
  - Parse notifications; compute GSR (μS) with correct 12-bit ADC pipeline and range handling; log `timestamp_ns,gsr_microsiemens,ppg_raw`.
  - Provide an LSL outlet for live GSR visualization (monitoring only).
- RgbCameraRecorder
  - Add instrumentation tests (Espresso/CameraX testing) for video + burst still capture on devices/emulators that support it.

### P2 — Network/Sync
- Expand robust time sync trials with per-device stats already persisted; expose re-sync during long recordings (policy exists on PC; consider Android hooks as needed).
- NSD reconnection and server auto-restart robustness in RecordingService.

### P5 — Data Management & Export
- HDF5 exporter: ensure end-to-end metadata (sync stats, units, sample_rate_hz) match latest spec; extend tests where feasible.
- Post-record automated ZIP transfer: maintain progress reporting and error handling; add retries.

### Testing & Coverage
- Maintain 100% practical coverage where feasible for pure logic; continue to isolate Android-framework code to enable JVM tests.
- Add additional unit tests for new logic introduced by Topdon/Shimmer integrations (where mockable).
- Introduce instrumentation tests for CameraX functionality on CI device farm (future).

### Documentation
- Update Developer_Guide.md with integration steps for Topdon and Shimmer SDKs.
- Update User_Manual.md with any new permissions/workflows stemming from integrations.

## Notes
- Follow MVVM structure; keep I/O and threading with Kotlin coroutines.
- Adhere to standardized CSV schemas and timestamp monotonicity (ns) across modalities.
