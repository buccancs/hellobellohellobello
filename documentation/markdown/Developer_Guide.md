# Developer Guide — Multi-Modal Physiological Sensing Platform

This guide describes the repository layout, local development workflow, testing, and packaging for release.

## 1. Repository Overview

The project uses a hub-and-spoke architecture with a multi-project Gradle build at the root.

```
/ (root)
├── android_sensor_node/              # Android app (Spoke)
│   └── app/                          # Main Android application module
├── pc_controller/                    # Python PC application (Hub)
│   ├── src/                          # PyQt6 app, core/network/data/tools
│   ├── tests/                        # pytest tests
│   └── native_backend/               # C++ (PyBind11) scaffolding
├── docs/                             # Documentation (User Manual, Flash Sync, this guide)
├── scripts/                          # Utilities (e.g., backup_script.py)
├── build.gradle.kts                  # Root orchestrator (aggregates test & packaging tasks)
├── settings.gradle.kts               # Multi-project settings
├── pytest.ini                        # Standardized pytest discovery
└── CHANGELOG.md                      # Keep a Changelog
```

## 2. Toolchains and Requirements

- Android: Android Studio Hedgehog+ (AGP 8.5+), JDK 17, Android SDK/NDK as needed.
- Python (PC Controller): Python 3.11+, pip. Optional: MSVC Build Tools for native C++ backend.
- OS: Windows 10+ recommended for packaging the PC app (EXE).

## 3. Gradle Multi-Project

The root Gradle build orchestrates tasks for both subprojects:

- `:pc_controller` – Python-specific tasks (virtualenv, install, pytest, PyInstaller)
- `:android_sensor_node:app` – Android application module

Aggregate tasks at root:

- `checkAll` – Runs Python pytest and Android unit tests
- `packageAll` – Builds the PC EXE and assembles Android release APK

## 4. Python (PC Controller) Development

### 4.1. Virtual Environment and Dependencies

From the repo root, initialize the Python environment once:

- Windows PowerShell:
    - `./gradlew.bat :pc_controller:installRequirements`

This will:

- Create `pc_controller/.venv`
- Upgrade pip
- Install `pc_controller/requirements.txt`
- Ensure `pytest` and `pyinstaller` are available

### 4.2. Running the App (Dev Mode)

- `python pc_controller\src\main.py`

### 4.3. Tests

- Use Gradle: `./gradlew.bat :pc_controller:pyTest`
- Or directly: `pytest -q`

### 4.4. Packaging (PyInstaller)

- Build EXE: `./gradlew.bat :pc_controller:assemblePcController`
- Output: `pc_controller/build/dist/pc_controller.exe`

Notes:

- The Gradle task wraps `PyInstaller -F` (single-file) build.
- For advanced control, create a `.spec` file and adapt the `pyInstaller` task.

## 5. Android (Sensor Spoke) Development

### 5.1. Build & Run

- Open the project in Android Studio.
- Select the `app` run configuration and deploy to a connected device.

### 5.2. Unit Tests

- CLI: `./gradlew.bat :android_sensor_node:app:testDebugUnitTest`

### 5.3. Release APK Assembly

- CLI: `./gradlew.bat :android_sensor_node:app:assembleRelease`
- Output under: `android_sensor_node/app/build/outputs/apk/release/`

### 5.4. Signing the APK (Release)

Option A: Android Studio (recommended)

- Build > Generate Signed Bundle / APK…
- Create/select a Keystore, set passwords and key alias
- Build the `release` variant

Option B: CLI (custom signing)

1. Create a keystore (once):
   ```bash
   keytool -genkeypair -v -keystore my-release-key.jks -keyalg RSA -keysize 2048 -validity 10000 -alias my-key
   ```
2. Configure signing in `android_sensor_node/app/build.gradle.kts` (do not commit secrets):
   ```kotlin
   android {
       signingConfigs {
           create("release") {
               storeFile = file(System.getenv("KEYSTORE_FILE") ?: "my-release-key.jks")
               storePassword = System.getenv("KEYSTORE_PASSWORD")
               keyAlias = System.getenv("KEY_ALIAS")
               keyPassword = System.getenv("KEY_PASSWORD")
           }
       }
       buildTypes {
           release {
               signingConfig = signingConfigs.getByName("release")
           }
       }
   }
   ```
3. Build signed APK:
    - `SET KEYSTORE_FILE=C:\\path\\to\\my-release-key.jks`
    - `SET KEYSTORE_PASSWORD=********`
    - `SET KEY_ALIAS=my-key`
    - `SET KEY_PASSWORD=********`
    - `./gradlew.bat :android_sensor_node:app:assembleRelease`

4. Verify signature:
   ```bash
   apksigner verify --print-certs android_sensor_node/app/build/outputs/apk/release/app-release.apk
   ```

Security: Never commit keystores or passwords. Use environment variables or local `gradle.properties` (excluded from
VCS).

## 6. Continuous Verification

- Root: `./gradlew.bat checkAll` runs Python pytest and Android unit tests.
- For CI, cache Gradle and Python pip downloads to speed up builds.

## 7. Coding Standards & Docs

- Python: PEP 8, type hints, PyQt6 GUI on main thread, background work in QThreads.
- Kotlin: MVVM, Coroutines, Lifecycle-aware components.
- Update `CHANGELOG.md` with Every notable change (Keep a Changelog; Conventional Commits).

## 8. Troubleshooting

- If `gradlew` is not recognized, use `./gradlew.bat` on Windows.
- If Android build fails due to SDKs, ensure Android Studio installed the required components and `local.properties`
  points to your SDK.
- If PyInstaller build misses DLLs, use `--add-binary` in a custom spec or ensure dependencies are importable at
  runtime.

## 9. Validation Toolkit Internals

This project includes a hardware-in-the-loop Validation Toolkit to verify temporal synchronization (<5 ms):

- Core module: `pc_controller/src/tools/validate_sync_core.py`
    - `detect_flash_indices_from_brightness(brightness, n_events, min_separation) -> List[int]`:
      Detects flash peaks from per-frame luminance using a combined z-score and derivative score with non-maximum
      suppression.
    - `estimate_T0_ns(aligned_event_ts_ns, rel_times_ns) -> int`:
      Estimates absolute video start time (T0) on the PC master clock by averaging `aligned_ts - rel_times` across
      events.
    - `choose_offset_direction(ref_ts_ns, device_ts_ns, offset_ns) -> int`:
      Chooses whether to add or subtract the NTP-like offset for a device by minimizing the median absolute error to a
      reference device.
    - `compute_validation_report(aligned_events_by_device, detections_by_stream, tolerance_ms) -> ValidationResult`:
      Produces per-event spreads (ms), overall max spread, PASS/FAIL, and stream T0 details.

- CLI: `scripts/validate_sync.py`
    - Inputs: `--session-id`, optional `--base-dir` (default `./pc_controller_data`), `--tolerance-ms` (default 5.0).
    - Loads `session_metadata.json` to get `clock_offsets_ns` and maps them to device folders.
    - Reads each device's `flash_sync_events.csv` and RGB video (`rgb/video.*`), plus the PC webcam video if present (
      `webcam.avi`).
    - Computes per-frame brightness via OpenCV, detects flash indices, and converts indices to relative times using FPS.
    - Aligns device flash timestamps to the PC master clock by applying the offsets with the selected sign.
    - Estimates each stream's absolute T0 and computes per-event time spreads across all streams.
    - Exits 0 on PASS (all events within tolerance) or 1 on FAIL.

Assumptions and Notes:

- Android devices log flash timestamps using a local monotonic clock; the PC stores NTP-like per-device offsets at Stop
  Session.
- The mapping between device names in offsets and session folders is fuzzy-matched (case-insensitive, alphanumeric
  subset).
- Peak detection expects full-screen white flashes or clear scene illumination spikes; ensure camera placement/lighting
  per SOP.
- Extendable to additional streams (e.g., thermal) by adding corresponding video inputs and flash detection logic.

Testing:

- Unit tests in `pc_controller/tests/test_validate_sync_core.py` cover peak detection and the PASS path for report
  computation using synthetic data.
