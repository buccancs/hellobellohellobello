# Data Formats and Session Directory Structure

**Purpose**: Make file outputs concrete for readers and graders, documenting exact schemas used by RecordingController and sensor recorders.

**Placement**: Chapter 4: Data Management section.

## Session Directory Structure

**Source**: Based on `RecordingController.startSession()` and individual `SensorRecorder` implementations.

### Directory Tree Example

```
sessions/
└── 20250818_173012_123_DeviceX_ab12cd34/
    ├── rgb/
    │   ├── video_1692374212345678901.mp4          # H.264 video (CameraX VideoCapture)
    │   ├── frames/                                # High-res JPEG stills directory
    │   │   ├── frame_1692374212450000000.jpg      # Nanosecond-timestamped frames
    │   │   ├── frame_1692374212466666667.jpg      # Individual JPEG captures
    │   │   ├── frame_1692374212483333333.jpg
    │   │   └── ...                                # Additional frames
    │   └── rgb.csv                                # Frame index with timestamps
    ├── thermal/
    │   ├── thermal.csv                            # Temperature data matrix
    │   └── metadata.json                          # Sensor configuration
    ├── gsr/
    │   └── gsr.csv                                # GSR and PPG measurements
    └── flash_sync_events.csv                      # Synchronization timestamps
```

### Session ID Format

**Pattern**: `YYYYMMDD_HHMMSS_mmm_DeviceModel_DeviceID`

**Example**: `20250818_173012_123_Pixel7_ab12cd34`

**Components**:
- `YYYYMMDD`: Date (ISO 8601 basic format)
- `HHMMSS`: Time (24-hour format)
- `mmm`: Milliseconds (000-999)
- `DeviceModel`: Android Build.MODEL (spaces replaced with underscores)
- `DeviceID`: Last 8 characters of device identifier

**Implementation**: Generated in `RecordingController.generateSessionId()`

## CSV Schema Definitions

### rgb/rgb.csv

**Purpose**: Index mapping JPEG frame files to precise timestamps for synchronization.

**Schema**:
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `timestamp_ns` | int64 | Nanosecond timestamp when frame captured | 1692374212450000000 |
| `filename` | string | Relative path to JPEG file | frames/frame_1692374212450000000.jpg |

**Sample Data**:
```csv
timestamp_ns,filename
1692374212450000000,frames/frame_1692374212450000000.jpg
1692374212466666667,frames/frame_1692374212466666667.jpg
1692374212483333333,frames/frame_1692374212483333333.jpg
```

**Notes**:
- Header written if file is empty
- Timestamps from `TimeManager.nowNanos()` (monotonic)
- MP4 video filename: `video_${sessionStartTs}.mp4` using session start timestamp
- Frame-to-video alignment possible via timestamp correlation

### thermal/thermal.csv

**Purpose**: Temperature data matrix from thermal imaging sensor.

**Schema**:
| Column | Type | Description | Range |
|--------|------|-------------|--------|
| `timestamp_ns` | int64 | Nanosecond timestamp of measurement | - |
| `w` | int32 | Image width in pixels | 256 |
| `h` | int32 | Image height in pixels | 192 |
| `v0` to `v49151` | float32 | Temperature values (row-major order) | -40°C to 1000°C |

**Sample Data**:
```csv
timestamp_ns,w,h,v0,v1,v2,...,v49151
1692374212450000000,256,192,23.5,23.7,23.6,...,24.1
1692374212483333333,256,192,23.6,23.8,23.7,...,24.2
```

**Implementation Notes**:
- Currently a stub implementation in `ThermalCameraRecorder`
- Header dynamically generated for 256×192 = 49,152 pixels
- Placeholder values for development/testing
- Future: Integration with Topdon TC001 or FLIR thermal cameras

### thermal/metadata.json

**Purpose**: Thermal sensor configuration and calibration parameters.

**Schema**:
```json
{
  "sensor": "Topdon TC001",
  "width": 256,
  "height": 192,
  "emissivity": 0.95,
  "format": "temperature_celsius",
  "notes": "Placeholder metadata for thermal camera integration",
  "calibration_date": "2025-08-18",
  "ambient_temperature": 22.5
}
```

### gsr/gsr.csv

**Purpose**: Galvanic skin response and photoplethysmography measurements.

**Schema**:
| Column | Type | Description | Units |
|--------|------|-------------|--------|
| `timestamp_ns` | int64 | Nanosecond timestamp of sample | - |
| `gsr_microsiemens` | float64 | Skin conductance measurement | µS (microsiemens) |
| `ppg_raw` | int32 | Raw photoplethysmography value | ADC counts |

**Sample Data**:
```csv
timestamp_ns,gsr_microsiemens,ppg_raw
1692374212450000000,2.345,2048
1692374212458333333,2.347,2051
1692374212466666667,2.342,2047
```

**Implementation Notes**:
- Data from Shimmer3 GSR+ sensor via BLE
- Sampling rate: 128 Hz (7.8125 ms intervals)
- GSR conversion: 12-bit ADC with multiple gain ranges
- PPG values: Raw 16-bit ADC counts for heart rate analysis

### flash_sync_events.csv

**Purpose**: Record precise timestamps of flash synchronization events for multi-device alignment.

**Schema**:
| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `timestamp_ns` | int64 | Nanosecond timestamp when flash occurred | TimeManager.nowNanos() |

**Sample Data**:
```csv
timestamp_ns
1692374212450000000
1692374218750000000
1692374225125000000
```

**Location**: Written to Android app files directory (`context.filesDir`)
**Transfer**: Added to ZIP stream during file transfer as top-level entry
**Usage**: Cross-device synchronization validation and drift correction

## File Size Estimates

### Typical Session (10 minutes)

| Component | File Size | Description |
|-----------|-----------|-------------|
| **RGB Video** | 500-800 MB | H.264 MP4, 1080p@30fps |
| **RGB Frames** | 200-400 MB | JPEG stills (~2MB each, 30fps) |
| **RGB CSV** | 1-2 MB | Frame index (~18,000 entries) |
| **Thermal CSV** | 50-100 MB | 256×192 pixels, 30fps |
| **GSR CSV** | 5-10 MB | 128Hz sampling, 2 channels |
| **Metadata** | <1 MB | JSON files, sync events |
| **Total** | ~750-1300 MB | Per 10-minute session |

### Network Transfer

- **ZIP Compression**: ~20-30% size reduction
- **Transfer Rate**: 10-50 MB/s (local network)
- **Duration**: 15-60 seconds per session
- **Bandwidth**: Peak ~400 Mbps during transfer

## Data Integrity

### Validation Checks
- CSV header validation on file open
- Timestamp monotonicity verification
- File completeness checks (expected vs. actual record counts)
- ZIP archive integrity validation

### Recovery Mechanisms
- Partial session recovery (incomplete stops)
- CSV file repair (missing headers)
- Timestamp drift detection and correction
- Duplicate session ID handling

### Quality Assurance
- Session metadata correlation across files
- Cross-device timestamp alignment verification
- Data export validation (CSV → HDF5, MAT formats)
- Hardware sensor calibration tracking
