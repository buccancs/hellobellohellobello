# Android Sensor Node Class Diagram

**Purpose**: Make clear boundaries and dependencies for Android application components.

**Placement**: Chapter 4: Android Design section.

## Component/Class Diagram

### Mermaid Class Diagram

```mermaid
classDiagram
    %% Core Controller Classes
    class RecordingController {
        +enum State { IDLE, PREPARING, RECORDING, STOPPING }
        -MutableStateFlow~State~ _state
        -MutableStateFlow~String?~ _currentSessionId
        -List~RecorderEntry~ recorders
        -File? sessionRootDir

        +register(name: String, recorder: SensorRecorder)
        +startSession(sessionId: String?)
        +stopSession()
        -generateSessionId() String
        -safeStopAll()
    }

    class RecorderEntry {
        +String name
        +SensorRecorder recorder
    }

    %% SensorRecorder Interface and Implementations
    class SensorRecorder {
        <<interface>>
        +start(sessionDir: File)
        +stop()
    }

    class RgbCameraRecorder {
        -ProcessCameraProvider? cameraProvider
        -ImageCapture? imageCapture
        -VideoCapture~Recorder~? videoCapture
        -Recording? recording
        -ExecutorService executor
        -CoroutineScope scope
        -Long lastPreviewNs
        -BufferedWriter? csvWriter

        +start(sessionDir: File)
        +stop()
        -startVideoRecording(File videoFile)
        -captureStills()
        -emitPreviewIfThrottled(ByteArray jpegBytes)
    }

    class ThermalCameraRecorder {
        -IRCMD? ircmd
        -UsbManager? usbManager
        -BufferedWriter? csvWriter
        -File? metadataFile
        -CoroutineScope scope
        -Float emissivity

        +start(sessionDir: File)
        +stop()
        -initializeTopdonSDK() IRCMD?
        -scanForTC001Devices() List~UsbDevice~
        -isTopdonTC001Device(Int vendorId, Int productId) Boolean
        -processRealThermalFrame(ByteArray rawData)
        -generateThermalBitmap(FloatArray tempMatrix, Int width, Int height) Bitmap
        -logRealTemperatureData(Float center, Float min, Float max)
        -writeMetadata()
    }

    class ShimmerRecorder {
        -ShimmerBluetooth? shimmerDevice
        -ShimmerConfig sensorConfig
        -BufferedWriter? csvWriter
        -CoroutineScope scope
        -BluetoothDevice? targetDevice
        -Boolean isStreaming

        +start(sessionDir: File)
        +stop()
        -scanForShimmerDevices() List~BluetoothDevice~
        -connectShimmer() Boolean
        -configureGSRSensor(Double samplingRate)
        -startStreaming() Boolean
        -parseDataPacket(ByteArray data)
        -convertGSRValue(Int raw12bit) Float
        -convertPPGValue(Int rawADC) Float
        -logSensorData(Long timestamp, Float gsrMicroSiemens, Float ppgRaw)
    }

    %% Service and Network Classes
    class RecordingService {
        -CoroutineScope scope
        -ServerSocket? serverSocket
        -NetworkClient networkClient
        -FileTransferManager fileTransferManager
        -PreviewBusListener previewListener

        +onStartCommand() Int
        +onDestroy()
        -startTcpServer()
        -handleClientConnection(Socket client)
        -processCommand(JSONObject cmd) JSONObject
        -collectCapabilities() JSONObject
    }

    class NetworkClient {
        -NsdManager nsdManager
        -RegistrationListener? registrationListener
        -Socket? socket
        -AtomicBoolean isConnected

        +register(String type, String name, Int port)
        +unregister()
        +connect(String host, Int port)
        +sendMessage(String json) Boolean
    }

    class FileTransferManager {
        -Context? context
        -File? sessionsRootOverride
        -String? deviceIdOverride

        +transferSession(String sessionId, String host, Int port)
        -sessionRoot() File
        -zipDirectoryContents(ZipOutputStream zos, File dir)
    }

    %% Utility Classes
    class PreviewBus {
        <<object>>
        -CopyOnWriteArrayList~Function~ listeners

        +subscribe(listener: Function)
        +unsubscribe(listener: Function)
        +emit(ByteArray bytes, Long timestampNs)
    }

    class TimeManager {
        <<object>>
        -Long offsetNs

        +nowNanos() Long
        +getSyncedTimestamp() Long
        +sync_with_server(String ip, Int port)
    }

    %% UI Classes
    class MainActivity {
        -ActivityResultLauncher~Array~String~~ permissionLauncher
        -RecordingController recordingController
        -ServiceConnection serviceConnection
        -Boolean isServiceBound

        +onCreate(Bundle? savedInstanceState)
        +onRequestPermissionsResult()
        -bindToRecordingService()
        -updateUI()
    }

    %% Relationships
    RecordingController o-- RecorderEntry
    RecorderEntry --> SensorRecorder
    SensorRecorder <|.. RgbCameraRecorder
    SensorRecorder <|.. ThermalCameraRecorder
    SensorRecorder <|.. ShimmerRecorder

    RecordingService --> NetworkClient
    RecordingService --> FileTransferManager
    RecordingService --> PreviewBus

    RgbCameraRecorder --> PreviewBus
    RgbCameraRecorder --> TimeManager
    ThermalCameraRecorder --> TimeManager
    ShimmerRecorder --> TimeManager

    MainActivity --> RecordingController
    MainActivity --> RecordingService : binds to

    %% Dependencies
    RecordingService ..> RecordingController : controls
    PreviewBus <.. RecordingService : listens
```

### Key Class Relationships

#### Composition and Ownership
- **RecordingController** owns multiple **RecorderEntry** objects
- **RecorderEntry** aggregates **SensorRecorder** implementations
- **RecordingService** owns **NetworkClient** and **FileTransferManager**
- Each sensor recorder manages its own resources independently

#### Interface Implementation
- **SensorRecorder** interface defines common contract
- **RgbCameraRecorder**, **ThermalCameraRecorder**, **ShimmerRecorder** implement interface
- Enables polymorphic handling in **RecordingController**

#### Service Integration
- **MainActivity** binds to **RecordingService** via Android service binding
- **RecordingService** controls **RecordingController** based on network commands
- Broadcast intents coordinate UI updates with service state

#### Utility Dependencies
- All sensor recorders depend on **TimeManager** for synchronized timestamps
- **RgbCameraRecorder** publishes preview frames via **PreviewBus**
- **RecordingService** subscribes to **PreviewBus** for network streaming

## Class Details

### RecordingController
- **State Management**: StateFlow-based reactive state (IDLE → PREPARING → RECORDING → STOPPING)
- **Session Lifecycle**: Directory creation, sensor coordination, cleanup
- **Error Handling**: Exception recovery with `safeStopAll()` mechanism
- **Thread Safety**: Coroutine-based operations with structured concurrency

### SensorRecorder Implementations

#### RgbCameraRecorder
- **CameraX Integration**: Dual-pipeline VideoCapture (MP4) + ImageCapture (JPEG)
- **Preview Pipeline**: Downsampling, throttling, PreviewBus emission
- **File Management**: CSV indexing, timestamp-based naming
- **Performance**: Background threads for capture, coroutines for coordination

#### ThermalCameraRecorder (Production SDK Integration)
- **True Topdon TC001 SDK Integration**: Real IRCMD, LibIRParse, LibIRProcess classes
- **Hardware Detection**: TC001-specific VID/PID identification (0x0525/0xa4a2, 0x0525/0xa4a5)
- **Calibrated Temperature Processing**: ±2°C accuracy with hardware calibration
- **Professional Thermal Imaging**: Iron, Rainbow, and Grayscale color palettes
- **Graceful Fallback**: Simulation mode when hardware unavailable

#### ShimmerRecorder (Production SDK Integration)
- **Real Shimmer Android API**: ShimmerBluetooth, ShimmerConfig classes
- **BLE Communication**: Shimmer3 GSR+ device connection with robust pairing
- **12-bit ADC Precision**: Correct 0-4095 range conversion (not 16-bit)
- **128 Hz Sampling Rate**: Hardware-validated sampling frequency
- **Dual-Sensor Data**: Simultaneous GSR (microsiemens) + PPG (raw ADC) recording

### RecordingService
- **Foreground Service**: Persistent operation with notification
- **TCP Server**: Accept PC connections, process JSON commands
- **Command Handling**: query_capabilities, start/stop_recording, time_sync, etc.
- **NSD Integration**: Service advertisement for automatic discovery

### NetworkClient
- **NSD Management**: Service registration/unregistration wrapper
- **Connection Handling**: TCP socket management with reconnection logic
- **Message Protocol**: Support for both v=1 framed and legacy newline JSON

### FileTransferManager
- **ZIP Streaming**: Direct socket streaming without temporary files
- **Session Packaging**: Recursive directory compression with metadata
- **Error Handling**: Connection timeouts, partial transfer recovery

## Android-Specific Considerations

### Lifecycle Management
- **Service Binding**: MainActivity connects to RecordingService for UI updates
- **Foreground Service**: Prevents Android from killing recording process
- **Permission Handling**: Camera, storage, location permissions with fallback

### Threading Model
- **Main Thread**: UI updates, service binding, permission requests
- **Service Thread**: TCP server, command processing, network operations
- **Background Threads**: Camera operations, file I/O, BLE communication
- **Coroutines**: Structured concurrency for sensor start/stop coordination

### Storage Management
- **External Files**: Session data in getExternalFilesDir() for user access
- **Internal Files**: App configuration and sync events in filesDir
- **Permissions**: MANAGE_EXTERNAL_STORAGE for unrestricted access

### Hardware Abstraction
- **Camera**: CameraX library abstracts Camera2 API complexity
- **Bluetooth**: Standard Android BluetoothAdapter for Shimmer connection
- **Storage**: File API with ContentProvider integration for sharing

## Testing Strategy

### Unit Testing
- **RecordingController**: State transitions with FakeRecorder implementations
- **Sensor Recorders**: Mock hardware dependencies, validate file outputs
- **Utility Classes**: TimeManager sync logic, PreviewBus event delivery

### Integration Testing
- **Service Communication**: TCP server with mock PC client
- **File Transfer**: End-to-end ZIP streaming validation
- **Multi-Sensor**: Parallel sensor operation under realistic load

### Hardware Testing
- **Device Matrix**: Multiple Android versions, hardware configurations
- **Camera Variations**: Different camera implementations, resolutions
- **Performance**: Memory usage, battery consumption, thermal throttling
