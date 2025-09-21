# Data Schemas and Storage Layout

This document defines the data structures, file formats, and storage organization for the multi-modal physiological sensing platform.

## Table of Contents

1. [Storage Directory Structure](#storage-directory-structure)
2. [CSV Schema Specifications](#csv-schema-specifications)
3. [Binary Data Formats](#binary-data-formats)
4. [Metadata Specifications](#metadata-specifications)
5. [Data Integrity and Validation](#data-integrity-and-validation)

---

## Storage Directory Structure

### Enhanced Data Management Features (Latest Update)

The system now includes comprehensive data management enhancements for improved reliability and user experience:

#### Enhanced Session Directory Structure
Sessions now use improved naming: `YYYYMMDD_HHMMSS_mmm_DeviceModel_DeviceID`
- Example: `20241218_143052_001_Pixel7_ab12cd34`
- Includes millisecond precision and device identification for uniqueness
- Each session includes comprehensive metadata file: `session_metadata.json`

#### Storage Space Management
- **Pre-recording validation**: Automatic storage space checking before session start
- **Intelligent estimation**: Calculates expected session size based on active sensors
  - RGB: ~902MB per 10 minutes (video + frames + CSV)  
  - Thermal: ~75MB per 10 minutes (CSV data)
  - GSR: ~8MB per 10 minutes (CSV data)
  - Audio: ~100MB per 10 minutes (AAC format)
- **Safety margins**: Requires estimated size + 50MB additional buffer
- **Low storage warnings**: Alerts when remaining space < 20% after session

#### Enhanced Cleanup System
- **Configurable retention**: Default 30-day automatic cleanup (vs previous 7-day)
- **Storage-based cleanup**: Activates when storage > 80% full
- **Smart selection**: Prioritizes oldest and largest sessions for removal
- **Detailed reporting**: Shows space freed and session counts
- **User confirmation**: Preview cleanup actions before execution

#### Data Validation System
- **Session integrity checking**: Validates expected file structure
- **CSV format validation**: Ensures timestamp consistency across sensors
- **Metadata validation**: Confirms required fields and data types
- **Post-session reports**: Generates validation summary for each session

### On-Device Storage Layout (Android)

```
/Android/data/com.yourcompany.sensorspoke/files/
├── sessions/
│   ├── 20241218_143052_001_Pixel7_ab12cd34/    # Enhanced session directory format
│   │   ├── session_metadata.json              # Comprehensive session metadata  
│   │   ├── rgb/
│   │   │   ├── video.mp4                       # H.264 video recording
│   │   │   ├── frames/
│   │   │   │   ├── frame_1703856123456789012.jpg
│   │   │   │   ├── frame_1703856123606789013.jpg
│   │   │   │   └── ...
│   │   │   └── rgb_frames.csv                  # Frame index with timestamps
│   │   ├── thermal/
│   │   │   ├── thermal_data.csv                # Thermal matrix data
│   │   │   ├── thermal_images/                 # PNG thermal images
│   │   │   └── metadata.json                   # Camera settings and calibration
│   │   ├── gsr/
│   │   │   └── gsr.csv                         # GSR and PPG measurements  
│   │   ├── audio/                              # Audio recordings (if enabled)
│   │   │   ├── audio.aac                       # AAC audio file
│   │   │   └── audio_events.csv                # Audio event timestamps
│   │   └── validation_report.json              # Session data validation results
│   ├── 20241218_144105_002_Pixel7_cd34ef56/    # Next session
│   │   └── ...
│   └── ...
├── flash_sync_events.csv                       # Global sync events (all sessions)
├── config/
│   ├── device_config.json                      # Device-specific settings  
│   └── sensor_calibration.json                 # Sensor calibration parameters
└── logs/
    ├── app.log                                  # Application event log
    └── error.log                                # Error and exception log
```

### Legacy On-Device Storage Layout (Pre-Enhancement)

```
/Android/data/com.yourcompany.sensorspoke/files/
├── sessions/
│   ├── 20241218_143052_001/          # Session directory (sessionStartTimestamp_sequence)
│   │   ├── rgb/
│   │   │   ├── video_1703856123456789012.mp4    # MP4 video (timestamped)
│   │   │   ├── frames/
│   │   │   │   ├── frame_1703856123456789012.jpg
│   │   │   │   ├── frame_1703856123606789013.jpg
│   │   │   │   └── ...
│   │   │   └── rgb.csv               # Frame index and timestamps
│   │   ├── thermal/
│   │   │   ├── thermal.csv           # Thermal matrix data
│   │   │   └── metadata.json         # Camera settings and calibration
│   │   └── gsr/
│   │       └── gsr.csv               # GSR and PPG measurements
│   ├── 20241218_144105_002/          # Next session
│   │   └── ...
│   └── ...
├── flash_sync_events.csv            # Global sync events (all sessions)
├── config/
│   ├── device_config.json            # Device-specific settings
│   └── sensor_calibration.json      # Sensor calibration parameters
└── logs/
    ├── app.log                       # Application event log
    └── error.log                     # Error and exception log
```

### PC Hub Storage Layout

```
C:\Users\{username}\Documents\MultiModalSensing\
├── sessions/
│   ├── 20241218_143052_001/          # Aggregated session data
│   │   ├── session_metadata.json    # Session-level metadata
│   │   ├── devices/
│   │   │   ├── pixel7_abc123/        # Device-specific data
│   │   │   │   ├── pixel7_abc123_data.zip    # Raw device archive
│   │   │   │   ├── rgb/              # Extracted RGB data
│   │   │   │   ├── thermal/          # Extracted thermal data
│   │   │   │   └── gsr/              # Extracted GSR data
│   │   │   ├── galaxy_s21_def456/   # Additional device
│   │   │   │   └── ...
│   │   │   └── flash_sync_events.csv # Consolidated sync events
│   │   ├── synchronized/             # Time-aligned data
│   │   │   ├── rgb_sync.csv          # Multi-device RGB index
│   │   │   ├── thermal_sync.csv      # Multi-device thermal data
│   │   │   └── gsr_sync.csv          # Multi-device GSR data
│   │   └── exports/
│   │       ├── session_data.h5       # HDF5 export
│   │       ├── session_data.mat      # MATLAB export
│   │       └── analysis_report.pdf   # Generated report
├── config/
│   ├── system_config.json            # PC system settings
│   └── device_profiles.json          # Known device configurations
└── logs/
    ├── pc_controller.log             # PC application logs
    └── network_events.log            # Network communication logs
```

---

## CSV Schema Specifications

### RGB Camera Data Schema (`rgb.csv`)

**Purpose**: Index of captured still frames with precise timestamps
**Location**: `sessions/{session_id}/rgb/rgb.csv`
**Encoding**: UTF-8
**Line Terminator**: LF (`\n`)

| Column | Type | Unit | Range | Description |
|--------|------|------|-------|-------------|
| `timestamp_ns` | int64 | nanoseconds | 0 to 2^63-1 | Nanosecond timestamp when frame was captured |
| `filename` | string | - | valid filename | JPEG filename in frames/ subdirectory |

**Schema Definition:**
```csv
timestamp_ns,filename
1703856123456789012,frame_1703856123456789012.jpg
1703856123606789013,frame_1703856123606789013.jpg
1703856123756789014,frame_1703856123756789014.jpg
```

**Constraints:**
- Header row must be present
- Timestamps must be monotonically increasing
- Filenames must reference existing JPEG files in frames/ directory
- Expected sampling rate: ~6.67 Hz (150ms intervals)
- No null values permitted

### Thermal Camera Data Schema (`thermal.csv`)

**Purpose**: Raw thermal matrix data from Topdon TC001 camera
**Location**: `sessions/{session_id}/thermal/thermal.csv`
**Encoding**: UTF-8
**Line Terminator**: LF (`\n`)

| Column | Type | Unit | Range | Description |
|--------|------|------|-------|-------------|
| `timestamp_ns` | int64 | nanoseconds | 0 to 2^63-1 | Nanosecond timestamp when thermal frame was captured |
| `w` | int32 | pixels | 256 | Frame width (constant) |
| `h` | int32 | pixels | 192 | Frame height (constant) |
| `v0` to `v49151` | float32 | °C | -40.0 to 550.0 | Temperature values in row-major order |

**Schema Definition:**
```csv
timestamp_ns,w,h,v0,v1,v2,...,v49151
1703856123456789012,256,192,23.5,23.6,23.4,...,24.1
1703856123556789013,256,192,23.6,23.7,23.5,...,24.2
```

**Matrix Layout:**
- Total values per frame: 256 × 192 = 49,152
- Row-major order: v[row * width + col]
- Temperature calibration applied by Topdon SDK
- Expected sampling rate: ~10 Hz (100ms intervals)

**Special Values:**
- `NaN`: Invalid/saturated pixel
- `-999.0`: Sensor error or disconnect
- Values outside range indicate calibration issues

### GSR Sensor Data Schema (`gsr.csv`)

**Purpose**: Galvanic Skin Response and Photoplethysmography measurements
**Location**: `sessions/{session_id}/gsr/gsr.csv`
**Encoding**: UTF-8
**Line Terminator**: LF (`\n`)

| Column | Type | Unit | Range | Description |
|--------|------|------|-------|-------------|
| `timestamp_ns` | int64 | nanoseconds | 0 to 2^63-1 | Nanosecond timestamp when sample was acquired |
| `gsr_microsiemens` | float64 | µS | 0.0 to 100.0 | Galvanic skin response (conductance) |
| `ppg_raw` | int32 | ADC units | 0 to 4095 | Raw photoplethysmography signal (12-bit ADC) |

**Schema Definition:**
```csv
timestamp_ns,gsr_microsiemens,ppg_raw
1703856123456789012,12.34,2048
1703856123464289012,12.35,2051
1703856123472289012,12.33,2045
```

**Constraints:**
- Expected sampling rate: 128 Hz (7.8125ms intervals)
- GSR values filtered and calibrated by Shimmer device
- PPG raw values from 12-bit ADC (0-4095 range)
- Missing samples indicated by duplicate timestamps with NaN values

### Flash Sync Events Schema (`flash_sync_events.csv`)

**Purpose**: Global synchronization markers across all devices
**Location**: Root of files directory (Android) or session directory (PC)
**Encoding**: UTF-8
**Line Terminator**: LF (`\n`)

| Column | Type | Unit | Range | Description |
|--------|------|------|-------|-------------|
| `timestamp_ns` | int64 | nanoseconds | 0 to 2^63-1 | Nanosecond timestamp when flash was triggered |
| `device_id` | string | - | alphanumeric | Device identifier that detected flash |
| `session_id` | string | - | session format | Session ID when flash occurred |
| `confidence` | float32 | - | 0.0 to 1.0 | Detection confidence score |
| `method` | string | - | enum | Detection method: 'manual', 'auto', 'audio' |

**Schema Definition:**
```csv
timestamp_ns,device_id,session_id,confidence,method
1703856125000000000,pixel7_abc123,20241218_143052_001,0.95,manual
1703856125001234567,galaxy_s21_def456,20241218_143052_001,0.92,manual
1703856130500000000,pixel7_abc123,20241218_143052_001,0.88,auto
```

**Synchronization Usage:**
- Flash events provide reference points for cross-device time alignment
- Multiple devices detecting same flash enable clock offset calculation
- Confidence scores help filter unreliable detection events

---

## Binary Data Formats

### MP4 Video Specifications

**RGB Video Files**: `sessions/{session_id}/rgb/video_{timestamp_ns}.mp4`

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Container** | MP4 (H.264/AVC) | Standard container format |
| **Resolution** | 1920×1080 (1080p) | Full HD quality |
| **Frame Rate** | 30 FPS | Constant frame rate |
| **Bitrate** | ~8-12 Mbps | Variable bitrate encoding |
| **Codec** | H.264 High Profile | Hardware acceleration when available |
| **Audio** | None | Video only |
| **Color Space** | YUV 4:2:0 | Standard mobile format |

**File Naming Convention:**
- Timestamp represents session start time in nanoseconds
- Single video file per session per device
- Example: `video_1703856123456789012.mp4`

### JPEG Frame Specifications

**Still Image Files**: `sessions/{session_id}/rgb/frames/frame_{timestamp_ns}.jpg`

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Format** | JPEG | Compressed still images |
| **Quality** | 95% | High quality compression |
| **Resolution** | Camera native | Typically 12MP+ |
| **Color Space** | sRGB | Standard color profile |
| **Orientation** | EXIF corrected | Portrait/landscape handled |
| **Compression** | Optimized | Hardware encoder preferred |

**File Naming Convention:**
- Timestamp represents exact capture time in nanoseconds
- One file per preview frame (~150ms intervals)
- Example: `frame_1703856123456789012.jpg`

---

## Metadata Specifications

### Session Metadata (`session_metadata.json`)

**Location**: `sessions/{session_id}/session_metadata.json` (PC only)
**Purpose**: High-level session information and device inventory

```json
{
  "session_id": "20241218_143052_001",
  "created_at": "2024-12-18T14:30:52.123456789Z",
  "started_at": "2024-12-18T14:31:15.987654321Z",
  "stopped_at": "2024-12-18T14:45:32.456789123Z",
  "duration_seconds": 856.469,
  "researcher": "Dr. Jane Smith",
  "study_protocol": "Stress Response Study v2.1",
  "participant_id": "P001",
  "devices": [
    {
      "device_id": "pixel7_abc123",
      "device_model": "Google Pixel 7",
      "android_version": "Android 13 (API 33)",
      "app_version": "1.0.3",
      "capabilities": ["rgb", "thermal", "gsr"],
      "data_received": true,
      "file_size_bytes": 1048576000
    }
  ],
  "synchronization": {
    "flash_events_count": 3,
    "time_sync_accuracy_ms": 2.3,
    "clock_drift_correction": true
  },
  "data_quality": {
    "rgb_frames_expected": 5734,
    "rgb_frames_received": 5732,
    "thermal_samples_expected": 8560,
    "thermal_samples_received": 8560,
    "gsr_samples_expected": 109568,
    "gsr_samples_received": 109503
  }
}
```

### Thermal Camera Metadata (`metadata.json`)

**Location**: `sessions/{session_id}/thermal/metadata.json`
**Purpose**: Thermal camera configuration and calibration data

```json
{
  "camera_model": "Topdon TC001",
  "firmware_version": "1.2.3",
  "resolution": {
    "width": 256,
    "height": 192
  },
  "calibration": {
    "emissivity": 0.95,
    "ambient_temperature_c": 23.5,
    "calibration_date": "2024-12-15T10:30:00Z",
    "factory_calibration": true
  },
  "measurement_range": {
    "min_temperature_c": -20.0,
    "max_temperature_c": 550.0,
    "accuracy_c": 2.0,
    "resolution_c": 0.1
  },
  "acquisition_settings": {
    "sampling_rate_hz": 10.0,
    "exposure_time_us": 8000,
    "gain": "auto"
  },
  "session_info": {
    "start_timestamp_ns": 1703856123456789012,
    "end_timestamp_ns": 1703856979925123456,
    "frames_captured": 8560,
    "dropped_frames": 0
  }
}
```

### Device Configuration (`device_config.json`)

**Location**: `config/device_config.json` (Android)
**Purpose**: Device-specific settings and capabilities

```json
{
  "device_info": {
    "device_id": "pixel7_abc123",
    "device_model": "Google Pixel 7",
    "manufacturer": "Google",
    "android_version": "13",
    "api_level": 33,
    "app_version": "1.0.3",
    "build_timestamp": "2024-12-01T12:00:00Z"
  },
  "capabilities": {
    "rgb_camera": {
      "available": true,
      "resolution": "4032x3024",
      "video_resolution": "1920x1080",
      "max_fps": 30
    },
    "thermal_camera": {
      "available": true,
      "model": "Topdon TC001",
      "usb_otg_required": true
    },
    "gsr_sensor": {
      "available": true,
      "ble_support": true,
      "shimmer_version": "3+"
    }
  },
  "network_settings": {
    "tcp_port": 8080,
    "udp_sync_port": 3333,
    "mdns_service_name": "SensorSpoke - Pixel_7",
    "reconnect_timeout_ms": 5000,
    "heartbeat_interval_ms": 3000
  },
  "storage_settings": {
    "sessions_path": "/Android/data/com.yourcompany.sensorspoke/files/sessions",
    "max_session_size_mb": 2048,
    "compression_enabled": true,
    "auto_cleanup_days": 30
  }
}
```

---

## Data Integrity and Validation

### Validation Rules

**File-level Validation:**
- All CSV files must have valid headers matching schema
- Timestamp columns must be monotonically increasing
- No missing critical columns
- File sizes within expected ranges
- Character encoding must be UTF-8

**Cross-file Validation:**
- RGB CSV entries must reference existing JPEG files
- Thermal metadata must match CSV dimensions
- Session timestamps must be consistent across modalities
- Flash sync events must have corresponding session data

**Time Synchronization Validation:**
- Clock offset corrections applied consistently
- Flash event timestamps aligned across devices (<5ms variance)
- No significant gaps in timestamp sequences
- Time sync accuracy within NFR requirements

### Data Quality Metrics

**Completeness Scores:**
```python
rgb_completeness = rgb_frames_received / rgb_frames_expected
thermal_completeness = thermal_samples_received / thermal_samples_expected
gsr_completeness = gsr_samples_received / gsr_samples_expected
overall_completeness = min(rgb_completeness, thermal_completeness, gsr_completeness)
```

**Temporal Accuracy:**
```python
time_sync_accuracy = max(abs(device_clock_offset)) for device in devices
frame_timing_jitter = std_dev(inter_frame_intervals)
synchronization_score = 1.0 - (time_sync_accuracy / max_acceptable_offset)
```

**Signal Quality Indicators:**
- RGB: Motion blur detection, exposure analysis
- Thermal: Temperature range validation, dead pixel detection
- GSR: Signal-to-noise ratio, electrode contact quality

### Error Handling

**Missing Data:**
- Timestamp gaps > 2× expected interval logged as warnings
- Missing files referenced in CSV trigger validation errors
- Incomplete sessions marked with data_quality.incomplete = true

**Corrupted Data:**
- Invalid CSV format triggers parsing exceptions
- Out-of-range values flagged in data_quality.anomalies
- Checksum mismatches during file transfer cause re-transmission

**Recovery Procedures:**
- Partial sessions preserved with quality annotations
- Corrupted files isolated in quarantine directories
- Session metadata updated with data integrity status

This comprehensive data specification ensures consistent, high-quality data collection and enables reliable post-processing and analysis workflows.
