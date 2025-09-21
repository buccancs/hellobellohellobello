# RGB Camera Recording Pipeline

This document details the RGB camera recording implementation, including the CameraX pipeline, file management, and preview streaming components.

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [CameraX Integration](#camerax-integration)
3. [Video Recording](#video-recording)
4. [Still Image Capture](#still-image-capture)
5. [Preview Streaming](#preview-streaming)
6. [File Management](#file-management)
7. [Performance Considerations](#performance-considerations)

---

## Pipeline Overview

The RGB camera pipeline handles three concurrent streams: continuous MP4 video recording, periodic high-resolution still captures, and real-time preview frame generation for network streaming.

```mermaid
graph TB
    subgraph "Camera Hardware"
        CS[Camera Sensor<br/>12MP+ Resolution]
    end

    subgraph "CameraX Framework"
        CP[CameraProvider<br/>Hardware Abstraction]
        VC[VideoCapture<br/>H.264 Encoder]
        IC[ImageCapture<br/>JPEG Encoder]
        PA[Preview Analysis<br/>Frame Processing]
    end

    subgraph "RgbCameraRecorder"
        SCL[Still Capture Loop<br/>150ms Timer]
        PB[PreviewBus<br/>Event Emitter]
        CSV[CSV Writer<br/>Frame Index]
    end

    subgraph "Storage Layer"
        VF[Video File<br/>video_{timestamp}.mp4]
        SF[Still Frames<br/>frame_{timestamp}.jpg]
        CF[CSV Index<br/>rgb.csv]
    end

    subgraph "Network Layer"
        RS[RecordingService<br/>TCP Broadcasting]
    end

    CS --> CP
    CP --> VC
    CP --> IC
    CP --> PA

    VC --> VF
    IC --> SCL
    IC --> SF
    PA --> PB

    SCL --> CSV
    CSV --> CF
    PB --> RS

    RS --> |TCP Frames| Network[Connected PC Clients]
```

---

## CameraX Integration

### Camera Initialization

The `RgbCameraRecorder` uses CameraX's modern camera API for robust hardware abstraction:

```kotlin
class RgbCameraRecorder(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
) : SensorRecorder {

    override suspend fun start(sessionDir: File) {
        val provider = ProcessCameraProvider.getInstance(context).get()
        cameraProvider = provider

        // Initialize concurrent use cases
        setupVideoCapture(sessionDir)
        setupImageCapture()
        setupPreviewAnalysis()

        // Bind all use cases to lifecycle
        provider.bindToLifecycle(
            lifecycleOwner,
            cameraSelector,
            videoCapture,
            imageCapture,
            previewAnalysis
        )
    }
}
```

### Use Case Configuration

**VideoCapture Configuration:**
```kotlin
private fun setupVideoCapture(sessionDir: File) {
    val recorder = Recorder.Builder()
        .setQualitySelector(
            QualitySelector.from(
                Quality.FHD,  // 1920x1080
                FallbackStrategy.higherQualityOrLowerThan(Quality.FHD)
            )
        )
        .build()

    videoCapture = VideoCapture.withOutput(recorder)

    // Start recording immediately
    val videoFile = File(sessionDir, "video_${TimeManager.nowNanos()}.mp4")
    val outputOptions = FileOutputOptions.Builder(videoFile).build()

    recording = recorder.prepareRecording(context, outputOptions)
        .start(ContextCompat.getMainExecutor(context)) { event ->
            handleRecordingEvent(event)
        }
}
```

**ImageCapture Configuration:**
```kotlin
private fun setupImageCapture() {
    imageCapture = ImageCapture.Builder()
        .setTargetRotation(Surface.ROTATION_0)
        .setJpegQuality(95)  // High quality for research
        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
        .build()
}
```

---

## Video Recording

### Continuous MP4 Recording

The system records a single continuous MP4 file throughout the session using hardware-accelerated H.264 encoding:

**Key Features:**
- **Resolution**: 1920×1080 (Full HD)
- **Frame Rate**: 30 FPS constant
- **Codec**: H.264 High Profile with hardware acceleration
- **Bitrate**: Variable (8-12 Mbps typical)
- **Container**: MP4 with proper metadata

**File Naming Convention:**
```
video_{session_start_timestamp_ns}.mp4
```

**Recording Management:**
```kotlin
private fun handleRecordingEvent(event: VideoRecordEvent) {
    when (event) {
        is VideoRecordEvent.Start -> {
            Log.i(TAG, "Video recording started")
        }
        is VideoRecordEvent.Finalize -> {
            if (event.hasError()) {
                Log.e(TAG, "Video recording error: ${event.error}")
            } else {
                Log.i(TAG, "Video recording completed: ${event.outputResults.outputUri}")
            }
        }
        is VideoRecordEvent.Status -> {
            // Optional: Monitor recording statistics
            Log.d(TAG, "Recording status: ${event.recordingStats.numBytesRecorded} bytes")
        }
    }
}
```

---

## Still Image Capture

### High-Resolution Frame Capture

Parallel to video recording, the system captures high-resolution JPEG stills at regular intervals for detailed analysis:

**Capture Loop Implementation:**
```kotlin
private fun startStillCaptureLoop(sessionDir: File) {
    val framesDir = File(sessionDir, "frames").apply { mkdirs() }

    scope.launch {
        while (isRecording) {
            try {
                val timestamp = TimeManager.nowNanos()
                val filename = "frame_${timestamp}.jpg"
                val outputFile = File(framesDir, filename)

                val outputOptions = ImageCapture.OutputFileOptions.Builder(outputFile)
                    .setMetadata(ImageCapture.Metadata().apply {
                        isReversedHorizontal = false
                    })
                    .build()

                imageCapture?.takePicture(
                    outputOptions,
                    executor,
                    object : ImageCapture.OnImageSavedCallback {
                        override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                            // Update CSV index
                            csvWriter?.write("$timestamp,$filename\n")
                            csvWriter?.flush()

                            // Generate preview if needed
                            generatePreview(outputFile, timestamp)
                        }

                        override fun onError(exception: ImageCaptureException) {
                            Log.e(TAG, "Still capture failed", exception)
                        }
                    }
                )

                delay(150) // 150ms interval = ~6.67 Hz
            } catch (e: Exception) {
                Log.e(TAG, "Still capture loop error", e)
            }
        }
    }
}
```

### CSV Frame Index

Each captured still frame is indexed in a CSV file for precise temporal alignment:

**CSV Schema:**
```csv
timestamp_ns,filename
1703856123456789012,frame_1703856123456789012.jpg
1703856123606789013,frame_1703856123606789013.jpg
```

**CSV Management:**
```kotlin
private fun initializeCsv(sessionDir: File) {
    csvFile = File(sessionDir, "rgb.csv")
    csvWriter = BufferedWriter(FileWriter(csvFile!!, true))

    // Write header if new file
    if (csvFile!!.length() == 0L) {
        csvWriter!!.write("timestamp_ns,filename\n")
        csvWriter!!.flush()
    }
}
```

---

## Preview Streaming

### Real-time Preview Generation

For live monitoring on the PC interface, the system generates compressed preview frames:

**Preview Analysis Setup:**
```kotlin
private fun setupPreviewAnalysis() {
    previewAnalysis = ImageAnalysis.Builder()
        .setTargetResolution(Size(640, 480))  // Reduced resolution for network
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .build()

    previewAnalysis?.setAnalyzer(executor) { image ->
        processPreviewFrame(image)
    }
}
```

**Frame Processing and Compression:**
```kotlin
private fun processPreviewFrame(image: ImageProxy) {
    val currentTime = System.currentTimeMillis()

    // Throttle to prevent network congestion
    if (currentTime - lastPreviewNs < 150) {
        image.close()
        return
    }
    lastPreviewNs = currentTime

    try {
        // Convert ImageProxy to Bitmap
        val bitmap = imageProxyToBitmap(image)

        // Compress to JPEG
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 75, outputStream)
        val jpegBytes = outputStream.toByteArray()

        // Emit to PreviewBus for network transmission
        PreviewBus.emit(PreviewFrame(
            deviceId = getDeviceId(),
            timestamp = TimeManager.nowNanos(),
            jpegData = jpegBytes
        ))

    } catch (e: Exception) {
        Log.e(TAG, "Preview processing error", e)
    } finally {
        image.close()
    }
}
```

### PreviewBus Event Distribution

The PreviewBus decouples preview generation from network transmission:

```kotlin
object PreviewBus {
    private val _frames = MutableSharedFlow<PreviewFrame>()
    val frames = _frames.asSharedFlow()

    fun emit(frame: PreviewFrame) {
        _frames.tryEmit(frame)
    }
}

data class PreviewFrame(
    val deviceId: String,
    val timestamp: Long,
    val jpegData: ByteArray
)
```

---

## File Management

### Session Directory Structure

Each recording session creates a dedicated RGB subdirectory:

```
sessions/20241218_143052_001/rgb/
├── video_1703856123456789012.mp4    # Continuous video
├── rgb.csv                          # Frame index
└── frames/                          # High-res stills
    ├── frame_1703856123456789012.jpg
    ├── frame_1703856123606789013.jpg
    └── ...
```

### Storage Space Management

**Disk Usage Monitoring:**
```kotlin
private fun checkStorageSpace(): Boolean {
    val sessionRoot = sessionDir ?: return false
    val availableBytes = sessionRoot.usableSpace

    return when {
        availableBytes < 100_000_000 -> { // 100 MB
            Log.e(TAG, "Critical storage space: ${availableBytes / 1_000_000} MB")
            false
        }
        availableBytes < 500_000_000 -> { // 500 MB
            Log.w(TAG, "Low storage space: ${availableBytes / 1_000_000} MB")
            true
        }
        else -> true
    }
}
```

**Cleanup on Stop:**
```kotlin
override suspend fun stop() {
    try {
        // Stop still capture loop
        isRecording = false

        // Stop video recording
        recording?.stop()
        recording = null

        // Finalize CSV
        csvWriter?.flush()
        csvWriter?.close()

        // Release camera resources
        cameraProvider?.unbindAll()
        executor.shutdown()

    } catch (e: Exception) {
        Log.e(TAG, "Error during stop", e)
    }
}
```

---

## Performance Considerations

### Resource Optimization

**Memory Management:**
- **Image Buffers**: Reuse ImageProxy objects via CameraX buffer pools
- **Preview Buffers**: Limit concurrent preview frames to 3-5 instances
- **Bitmap Recycling**: Explicitly recycle Bitmap objects after compression
- **Stream Buffers**: Use appropriate buffer sizes for file I/O (64KB)

**CPU Optimization:**
- **Hardware Encoding**: Prefer hardware H.264 encoder when available
- **Background Threading**: Execute compression on background threads
- **Frame Skipping**: Skip preview frames when CPU usage is high
- **Quality Scaling**: Reduce JPEG quality under thermal pressure

### Error Handling and Resilience

**Camera Errors:**
```kotlin
private fun handleCameraError(exception: Exception) {
    when (exception) {
        is CameraAccessException -> {
            Log.e(TAG, "Camera access denied", exception)
            // Attempt recovery or graceful degradation
        }
        is IllegalStateException -> {
            Log.e(TAG, "Camera in illegal state", exception)
            // Reset camera provider
        }
        else -> {
            Log.e(TAG, "Unexpected camera error", exception)
        }
    }
}
```

**Storage Errors:**
```kotlin
private fun handleStorageError(exception: IOException) {
    Log.e(TAG, "Storage I/O error", exception)

    // Check available space
    if (!checkStorageSpace()) {
        // Trigger emergency stop
        scope.launch { stop() }
    }

    // Notify service of error
    broadcastError("STORAGE_ERROR", exception.message)
}
```

### Performance Metrics

**Key Performance Indicators:**
- **Still Capture Rate**: Target 6.67 Hz (150ms intervals)
- **Video Frame Rate**: Maintain 30 FPS with <1% drops
- **Preview Latency**: <200ms from capture to network transmission
- **CPU Usage**: <40% average during active recording
- **Memory Usage**: <300 MB total allocation
- **Storage Write Rate**: >20 MB/s sustained for thermal + RGB

This RGB camera implementation provides high-quality, synchronized video and still image capture while maintaining real-time preview streaming for remote monitoring.
