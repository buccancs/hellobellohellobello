package com.yourcompany.sensorspoke.sensors.rgb

import android.content.Context
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Recording
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.yourcompany.sensorspoke.network.DeviceConnectionManager
import com.yourcompany.sensorspoke.sensors.SensorRecorder
import com.yourcompany.sensorspoke.utils.TimeManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Refactored RGB camera recorder focused on data logging with improved separation of concerns.
 * Camera management is now handled by RgbCameraManager, data processing by RgbDataProcessor.
 *
 * This recorder is responsible for:
 * - Managing CSV file output for frame metadata
 * - Coordinating with RgbCameraManager for camera state
 * - Reporting status via Flow for UI reactivity
 * - Session-based recording lifecycle
 */
class RgbCameraRecorder(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    initialCameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA,
    private val cameraManager: RgbCameraManager? = null,
    private val deviceConnectionManager: DeviceConnectionManager? = null,
) : SensorRecorder {
    companion object {
        private const val TAG = "RgbCameraRecorder"
    }
    
    /**
     * Recording modes for the camera system
     */
    enum class RecordingMode {
        STANDARD,   // CameraX video + JPEG frames
        RAW_DNG     // CameraX + Camera2 RAW DNG parallel capture
    }

    private var cameraSelector: CameraSelector = initialCameraSelector

    private val rgbCameraManager = cameraManager ?: RgbCameraManager(context, lifecycleOwner, cameraSelector)
    private val dataProcessor = RgbDataProcessor()
    
    // RAW DNG recording state
    private var recordingMode = RecordingMode.STANDARD

    private var recording: Recording? = null
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var captureJob: Job? = null
    private var syncMonitorJob: Job? = null
    private var csvWriter: BufferedWriter? = null
    private var csvFile: File? = null
    private var frameCount = 0
    private var videoStartTime: Long = 0L
    private var actualVideoStartTime: Long = 0L
    private var frameTimestampOffset: Long = 0L

    private val _recordingStatus = MutableStateFlow(RecordingStatus.IDLE)
    val recordingStatus: StateFlow<RecordingStatus> = _recordingStatus.asStateFlow()

    private val _frameRate = MutableStateFlow(0.0)
    val frameRate: StateFlow<Double> = _frameRate.asStateFlow()

    private val _cameraStatus = MutableStateFlow<RgbCameraManager.CameraStatus?>(null)
    val cameraStatus: StateFlow<RgbCameraManager.CameraStatus?> = _cameraStatus.asStateFlow()

    /**
     * Recording status enum for UI state management
     */
    enum class RecordingStatus {
        IDLE,
        STARTING,
        RECORDING,
        STOPPING,
        ERROR,
    }

    override suspend fun start(sessionDir: File) {
        Log.i(TAG, "Starting enhanced RGB camera recording in session: ${sessionDir.absolutePath}")
        _recordingStatus.value = RecordingStatus.STARTING

        try {
            if (!rgbCameraManager.isReady()) {
                if (!rgbCameraManager.initialize()) {
                    throw RuntimeException("Failed to initialize enhanced RGB camera manager")
                }
            }

            _cameraStatus.value = rgbCameraManager.getCameraStatus()
            
            val status = _cameraStatus.value
            Log.i(TAG, "Camera initialized - Model: ${status?.deviceModel}, Quality: ${status?.quality}, 4K Support: ${status?.supports4K}")
            
            deviceConnectionManager?.updateRgbCameraState(
                DeviceConnectionManager.DeviceState.CONNECTING,
                DeviceConnectionManager.DeviceDetails(
                    deviceType = "Enhanced RGB Camera",
                    deviceName = "CameraX ${status?.quality ?: "HD"} (${status?.deviceModel ?: "Unknown"})",
                    connectionState = DeviceConnectionManager.DeviceState.CONNECTING,
                    isRequired = true,
                ),
            )

            val framesDir = File(sessionDir, "frames").apply { mkdirs() }
            
            // Create RAW DNG directory if in RAW mode
            val rawDngDir = if (recordingMode == RecordingMode.RAW_DNG) {
                File(sessionDir, "raw_dng").apply { mkdirs() }
            } else null

            csvFile = File(sessionDir, "rgb_frames.csv")
            csvWriter = BufferedWriter(FileWriter(csvFile!!, true))
            if (csvFile!!.length() == 0L) {
                val header = if (recordingMode == RecordingMode.RAW_DNG) {
                    dataProcessor.getCsvHeader() + ",raw_dng_file"
                } else {
                    dataProcessor.getCsvHeader()
                }
                csvWriter!!.write(header + "\n")
                csvWriter!!.flush()
            }

            startVideoRecording(File(sessionDir, "video.mp4"))
            
            // Initialize RAW DNG capture session if in RAW mode
            if (recordingMode == RecordingMode.RAW_DNG && rawDngDir != null) {
                val rawDngManager = rgbCameraManager.getCamera2RawDngManager()
                if (rawDngManager != null) {
                    val sessionStarted = rawDngManager.startRawCaptureSession()
                    if (sessionStarted) {
                        Log.i(TAG, "RAW DNG capture session started successfully")
                        _recordingStatus.value = RecordingStatus.RECORDING
                    } else {
                        Log.w(TAG, "Failed to start RAW DNG capture session, continuing with standard recording")
                    }
                } else {
                    Log.w(TAG, "RAW DNG manager not available, continuing with standard recording")
                }
            }

            // Start frame capture process
            startFrameCapture(framesDir, rawDngDir)

            startSyncMonitoring()

            _recordingStatus.value = RecordingStatus.RECORDING
            rgbCameraManager.updateRecordingState(true)

            deviceConnectionManager?.updateRgbCameraState(
                DeviceConnectionManager.DeviceState.CONNECTED,
                DeviceConnectionManager.DeviceDetails(
                    deviceType = "RGB Camera",
                    deviceName = "CameraX RGB",
                    connectionState = DeviceConnectionManager.DeviceState.CONNECTED,
                    isRequired = true,
                ),
            )

            Log.i(TAG, "RGB camera recording started successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start RGB camera recording: ${e.message}", e)
            _recordingStatus.value = RecordingStatus.ERROR

            deviceConnectionManager?.updateRgbCameraState(
                DeviceConnectionManager.DeviceState.ERROR,
                DeviceConnectionManager.DeviceDetails(
                    deviceType = "RGB Camera",
                    deviceName = "CameraX RGB",
                    connectionState = DeviceConnectionManager.DeviceState.ERROR,
                    errorMessage = e.message,
                    isRequired = true,
                ),
            )
            throw e
        }
    }

    override suspend fun stop() {
        Log.i(TAG, "Stopping RGB camera recording")
        _recordingStatus.value = RecordingStatus.STOPPING

        try {
            recording?.stop()
            recording = null

            captureJob?.let {
                it.cancel()
                it.join()
            }
            captureJob = null

            syncMonitorJob?.cancel()
            syncMonitorJob = null

            // Stop RAW DNG capture session if active
            if (recordingMode == RecordingMode.RAW_DNG) {
                val rawDngManager = rgbCameraManager.getCamera2RawDngManager()
                rawDngManager?.stopRawCaptureSession()
                Log.i(TAG, "RAW DNG capture session stopped")
            }

            // Close CSV resources
            csvWriter?.flush()
            csvWriter?.close()
            csvWriter = null
            csvFile = null

            rgbCameraManager.updateRecordingState(false)

            deviceConnectionManager?.updateRgbCameraState(DeviceConnectionManager.DeviceState.DISCONNECTED)

            executor.shutdown()

            scope.cancel()

            _recordingStatus.value = RecordingStatus.IDLE
            _frameRate.value = 0.0

            Log.i(TAG, "RGB camera recording stopped, $frameCount frames captured")
        } catch (e: Exception) {
            Log.e(TAG, "Error during RGB camera recording stop: ${e.message}", e)
            _recordingStatus.value = RecordingStatus.ERROR
            throw e
        }
    }

    private fun startVideoRecording(videoFile: File) {
        val sessionStartTs = TimeManager.nowNanos()
        videoStartTime = sessionStartTs
        val videoFileWithTimestamp = File(videoFile.parent, "video_$sessionStartTs.mp4")

        val outputOpts = FileOutputOptions.Builder(videoFileWithTimestamp).build()
        val videoCapture = rgbCameraManager.getVideoCapture()
            ?: throw RuntimeException("Video capture not available from camera manager")

        recording = videoCapture.output
            .prepareRecording(context, outputOpts)
            .start(ContextCompat.getMainExecutor(context)) { event ->
                Log.d(TAG, "Recording event: $event")
                when (event) {
                    is androidx.camera.video.VideoRecordEvent.Start -> {
                        actualVideoStartTime = System.nanoTime()
                        frameTimestampOffset = actualVideoStartTime - videoStartTime
                        Log.i(TAG, "Video recording actually started at: $actualVideoStartTime")
                        Log.i(TAG, "Video start offset: ${frameTimestampOffset / 1_000_000}ms")

                        dataProcessor.logVideoEvent(
                            csvFile, "VIDEO_RECORDING_STARTED", actualVideoStartTime,
                            "offset_ms:${frameTimestampOffset / 1_000_000}",
                        )
                    }
                    is androidx.camera.video.VideoRecordEvent.Finalize -> {
                        val finalizeTime = System.nanoTime()
                        Log.i(TAG, "Video recording finalized: ${event.outputResults}")
                        dataProcessor.logVideoEvent(
                            csvFile, "VIDEO_RECORDING_FINALIZED", finalizeTime,
                            "duration_ms:${(finalizeTime - actualVideoStartTime) / 1_000_000}",
                        )
                    }
                    is androidx.camera.video.VideoRecordEvent.Status -> {
                        if (event.recordingStats.numBytesRecorded > 0) {
                            val statusTime = System.nanoTime()
                            dataProcessor.logVideoEvent(
                                csvFile, "VIDEO_STATUS_UPDATE", statusTime,
                                "bytes:${event.recordingStats.numBytesRecorded}",
                            )
                        }
                    }
                }
            }

        Log.i(TAG, "Video recording started: ${videoFileWithTimestamp.absolutePath} at timestamp $sessionStartTs")
    }

    private fun startFrameCapture(framesDir: File, rawDngDir: File?) {
        captureJob = scope.launch {
            Log.i(TAG, "Starting frame capture loop with mode: $recordingMode")

            while (isActive) {
                try {
                    captureFrame(framesDir, rawDngDir)
                    delay(33) // ~30 FPS for high-quality data capture
                } catch (e: Exception) {
                    Log.e(TAG, "Error capturing frame: ${e.message}", e)
                    delay(1000)
                }
            }
        }
    }

    private fun startSyncMonitoring() {
        syncMonitorJob = scope.launch {
            Log.i(TAG, "Starting synchronization monitoring")

            while (isActive) {
                try {
                    delay(5000)

                    if (actualVideoStartTime > 0 && frameCount > 0) {
                        val currentTime = System.nanoTime()
                        val totalRecordingTime = (currentTime - actualVideoStartTime) / 1_000_000
                        val avgFrameRate = if (totalRecordingTime > 0) {
                            (frameCount * 1000.0) / totalRecordingTime
                        } else {
                            0.0
                        }

                        _frameRate.value = avgFrameRate

                        dataProcessor.logVideoEvent(
                            csvFile, "SYNC_MONITOR_UPDATE", currentTime,
                            "frames:$frameCount,avg_fps:${"%.2f".format(avgFrameRate)},recording_time_ms:$totalRecordingTime",
                        )

                        Log.d(TAG, "Sync monitor - Frames: $frameCount, Avg FPS: ${"%.2f".format(avgFrameRate)}, Recording time: ${totalRecordingTime}ms")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error in sync monitoring: ${e.message}", e)
                    delay(1000)
                }
            }
        }
    }

    private suspend fun captureFrame(framesDir: File, rawDngDir: File?) {
        val timestampNs = TimeManager.nowNanos()
        val timestampMs = System.currentTimeMillis()
        frameCount++

        val filename = "frame_$timestampNs.jpg"
        val outputFile = File(framesDir, filename)
        val outputFileOptions = ImageCapture.OutputFileOptions.Builder(outputFile).build()

        val imageCapture = rgbCameraManager.getImageCapture()
            ?: throw RuntimeException("Image capture not available from camera manager")

        imageCapture.takePicture(
            outputFileOptions,
            executor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exception: ImageCaptureException) {
                    Log.w(TAG, "Frame capture error: ${exception.message}")
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    try {
                        var rawDngFilename: String? = null
                        
                        // Capture RAW DNG if in RAW mode and directory exists
                        if (recordingMode == RecordingMode.RAW_DNG && rawDngDir != null) {
                            rawDngFilename = captureRawDngFrame(timestampNs, rawDngDir)
                        }

                        // Create frame data using data processor
                        val frameData = dataProcessor.createFrameData(
                            timestampNs,
                            timestampMs,
                            frameCount,
                            outputFile,
                            videoStartTime,
                            actualVideoStartTime,
                        )

                        // Write to CSV using data processor with optional RAW DNG filename
                        csvWriter?.apply {
                            val csvLine = if (rawDngFilename != null) {
                                dataProcessor.formatFrameDataForCsv(frameData) + ",$rawDngFilename"
                            } else {
                                dataProcessor.formatFrameDataForCsv(frameData)
                            }
                            write(csvLine + "\n")
                            flush()
                        }

                        dataProcessor.generatePreview(outputFile, timestampNs)

                        val modeInfo = if (recordingMode == RecordingMode.RAW_DNG) " + RAW DNG" else ""
                        Log.d(TAG, "Captured frame$modeInfo: ${frameData.filename} (${frameData.fileSizeBytes} bytes) - Video time: ${frameData.videoRelativeTimeMs}ms, Est. frame: ${frameData.estimatedVideoFrame}, Sync quality: ${"%.3f".format(frameData.syncQuality)}")
                    } catch (e: Exception) {
                        Log.e(TAG, "Error processing captured frame: ${e.message}", e)
                    }
                }
            },
        )
    }

    /**
     * Capture RAW DNG frame using efficient persistent session
     */
    private fun captureRawDngFrame(timestampNs: Long, rawDngDir: File): String? {
        return try {
            val rawDngManager = rgbCameraManager.getCamera2RawDngManager()
            if (rawDngManager == null) {
                Log.w(TAG, "Camera2 RAW DNG manager not available")
                return null
            }
            
            val rawFilename = "raw_${timestampNs}_${frameCount}.dng"
            val rawOutputFile = File(rawDngDir, rawFilename)
            
            // Launch efficient RAW DNG frame capture using persistent session
            scope.launch {
                try {
                    val success = rawDngManager.captureRawDngFrame(rawOutputFile)
                    if (success) {
                        Log.d(TAG, "RAW DNG frame captured: $rawFilename")
                    } else {
                        Log.w(TAG, "RAW DNG frame capture failed: $rawFilename")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error capturing RAW DNG frame: ${e.message}", e)
                }
            }
            
            rawFilename
        } catch (e: Exception) {
            Log.e(TAG, "Error initiating RAW DNG frame capture: ${e.message}", e)
            null
        }
    }

    /**
     * Get current recording statistics for external monitoring
     */
    fun getRecordingStatistics(): RecordingStatistics {
        return RecordingStatistics(
            isRecording = _recordingStatus.value == RecordingStatus.RECORDING,
            framesRecorded = frameCount,
            frameRate = _frameRate.value,
            recordingStatus = _recordingStatus.value,
            timingStatistics = dataProcessor.getTimingStatistics(
                videoStartTime,
                actualVideoStartTime,
                frameTimestampOffset,
                frameCount,
            ),
        )
    }

    /**
     * Recording statistics data class
     */
    data class RecordingStatistics(
        val isRecording: Boolean,
        val framesRecorded: Int,
        val frameRate: Double,
        val recordingStatus: RecordingStatus,
        val timingStatistics: Map<String, Any>,
    )

    /**
     * Switch between front and back camera
     */
    fun switchCamera(): Boolean {
        return try {
            val success = rgbCameraManager.switchCamera()
            if (success) {
                cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
                    CameraSelector.DEFAULT_FRONT_CAMERA
                } else {
                    CameraSelector.DEFAULT_BACK_CAMERA
                }
                
                _cameraStatus.value = rgbCameraManager.getCameraStatus()
                
                Log.i(TAG, "Switched to ${if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) "back" else "front"} camera")
            }
            success
        } catch (e: Exception) {
            Log.e(TAG, "Failed to switch camera: ${e.message}", e)
            false
        }
    }

    /**
     * Set Preview surface provider for on-screen display
     */
    fun setPreviewSurfaceProvider(surfaceProvider: androidx.camera.core.Preview.SurfaceProvider) {
        rgbCameraManager.setPreviewSurfaceProvider(surfaceProvider)
        Log.i(TAG, "Preview surface provider set for on-screen camera display")
    }

    /**
     * Get Preview for UI integration
     */
    fun getPreview(): androidx.camera.core.Preview? {
        return rgbCameraManager.getPreview()
    }

    /**
     * Check if device supports 4K recording
     */
    fun supports4K(): Boolean {
        return rgbCameraManager.supports4K()
    }

    /**
     * Get current camera status for UI feedback
     */
    fun getCurrentCameraStatus(): RgbCameraManager.CameraStatus? {
        return _cameraStatus.value
    }

    /**
     * Check if currently using back camera
     */
    fun isUsingBackCamera(): Boolean {
        return cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA
    }

    /**
     * Enhanced recording statistics with camera information
     */
    fun getEnhancedRecordingStatistics(): EnhancedRecordingStatistics {
        val cameraStatus = _cameraStatus.value
        return EnhancedRecordingStatistics(
            isRecording = _recordingStatus.value == RecordingStatus.RECORDING,
            framesRecorded = frameCount,
            frameRate = _frameRate.value,
            recordingStatus = _recordingStatus.value,
            cameraQuality = cameraStatus?.quality ?: "Unknown",
            cameraResolution = cameraStatus?.resolution ?: "Unknown",
            isBackCamera = cameraStatus?.isBackCamera ?: true,
            supports4K = cameraStatus?.supports4K ?: false,
            deviceModel = cameraStatus?.deviceModel ?: "Unknown",
            timingStatistics = dataProcessor.getTimingStatistics(
                videoStartTime,
                actualVideoStartTime,
                frameTimestampOffset,
                frameCount,
            ),
        )
    }

    /**
     * Enhanced recording statistics data class
     */
    data class EnhancedRecordingStatistics(
        val isRecording: Boolean,
        val framesRecorded: Int,
        val frameRate: Double,
        val recordingStatus: RecordingStatus,
        val cameraQuality: String,
        val cameraResolution: String,
        val isBackCamera: Boolean,
        val supports4K: Boolean,
        val deviceModel: String,
        val timingStatistics: Map<String, Any>,
    )
    
    /**
     * Check if RAW DNG recording is available on this device
     */
    fun supportsRawDng(): Boolean {
        return rgbCameraManager.supportsRawDng()
    }
    
    /**
     * Check if device is Samsung with Camera2 Level 3 support
     */
    fun isSamsungLevel3Device(): Boolean {
        return rgbCameraManager.isSamsungLevel3Device()
    }
    
    /**
     * Set the recording mode (Standard or RAW DNG)
     */
    fun setRecordingMode(mode: RecordingMode) {
        if (_recordingStatus.value != RecordingStatus.IDLE) {
            Log.w(TAG, "Cannot change recording mode while recording")
            return
        }
        
        // Only allow RAW DNG mode on supported devices
        if (mode == RecordingMode.RAW_DNG && !supportsRawDng()) {
            Log.w(TAG, "RAW DNG mode not supported on this device")
            return
        }
        
        recordingMode = mode
        Log.i(TAG, "Recording mode set to: $mode")
    }
    
    /**
     * Get the current recording mode
     */
    fun getRecordingMode(): RecordingMode {
        return recordingMode
    }
    
    /**
     * Get camera manager for advanced operations (internal use)
     */
    internal fun getCameraManager(): RgbCameraManager {
        return rgbCameraManager
    }
}
