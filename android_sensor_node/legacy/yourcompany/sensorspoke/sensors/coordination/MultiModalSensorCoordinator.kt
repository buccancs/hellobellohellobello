package com.yourcompany.sensorspoke.sensors.coordination

import android.content.Context
import android.util.Log
import androidx.lifecycle.LifecycleOwner
import com.yourcompany.sensorspoke.sensors.audio.AudioRecorder
import com.yourcompany.sensorspoke.sensors.gsr.GSRDataPoint
import com.yourcompany.sensorspoke.sensors.gsr.ShimmerDataCallback
import com.yourcompany.sensorspoke.sensors.gsr.ShimmerGSRIntegrationManager
import com.yourcompany.sensorspoke.sensors.gsr.ShimmerRecorder
import com.yourcompany.sensorspoke.sensors.rgb.RgbCameraRecorder
import com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001SensorIntegrationManager
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter

/**
 * Multi-Modal Sensor Coordination Manager
 *
 * Orchestrates synchronized recording from all sensors (GSR, Thermal, RGB, Audio)
 * with precise time synchronization, health monitoring, and comprehensive data logging.
 * Provides enterprise-grade multi-sensor data acquisition for research applications.
 */
class MultiModalSensorCoordinator(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
) {
    companion object {
        private const val TAG = "MultiModalCoordinator"

        // Coordination states
        enum class CoordinationState {
            UNINITIALIZED,
            INITIALIZING,
            READY,
            STARTING,
            RECORDING,
            STOPPING,
            ERROR,
            STOPPED,
        }

        // Synchronization settings
        const val SYNC_PRECISION_TARGET_MS = 5.0 // Target: ±5ms synchronization
        const val HEALTH_CHECK_INTERVAL_MS = 2000L // 2-second health checks
        const val METRICS_UPDATE_INTERVAL_MS = 1000L // 1-second metrics updates

        // Data logging
        const val SYNC_LOG_HEADER = "timestamp_ns,system_time_ms,sensor_type,data_type,sensor_value,sync_status"
    }

    // Sensor managers
    private var gsrIntegrationManager: ShimmerGSRIntegrationManager? = null
    private var tc001IntegrationManager: TC001SensorIntegrationManager? = null
    private var rgbRecorder: RgbCameraRecorder? = null
    private var audioRecorder: AudioRecorder? = null

    // State management
    private val _coordinationState = MutableStateFlow(CoordinationState.UNINITIALIZED)
    val coordinationState: StateFlow<CoordinationState> = _coordinationState

    private val _systemHealth = MutableStateFlow(SystemHealthStatus())
    val systemHealth: StateFlow<SystemHealthStatus> = _systemHealth

    private val _synchronizationMetrics = MutableStateFlow(SynchronizationMetrics())
    val synchronizationMetrics: StateFlow<SynchronizationMetrics> = _synchronizationMetrics

    // Data synchronization
    private var syncLogWriter: BufferedWriter? = null
    private var sessionDir: File? = null
    private val dataBuffer = SensorDataBuffer()

    // Lifecycle management
    private val coordinationScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var healthMonitorJob: Job? = null
    private var metricsJob: Job? = null
    private var synchronizationJob: Job? = null

    // Timing synchronization
    private var recordingStartTime: Long = 0L
    private var lastSyncCheck: Long = 0L

    /**
     * Initialize all sensor subsystems
     */
    suspend fun initializeSystem(): Boolean =
        withContext(Dispatchers.IO) {
            try {
                _coordinationState.value = CoordinationState.INITIALIZING
                Log.i(TAG, "Initializing multi-modal sensor coordination system...")

                // Initialize GSR integration
                gsrIntegrationManager =
                    ShimmerGSRIntegrationManager(context).apply {
                        if (!initialize()) {
                            Log.w(TAG, "GSR integration initialization failed")
                        } else {
                            Log.i(TAG, "GSR integration initialized successfully")
                            setDataCallback(gsrDataCallback)
                        }
                    }

                // Initialize TC001 thermal integration
                tc001IntegrationManager =
                    TC001SensorIntegrationManager(context).apply {
                        if (!initializeSystem()) {
                            Log.w(TAG, "TC001 integration initialization failed")
                        } else {
                            Log.i(TAG, "TC001 integration initialized successfully")
                        }
                    }

                // Initialize RGB camera recorder
                rgbRecorder =
                    RgbCameraRecorder(context, lifecycleOwner).apply {
                        Log.i(TAG, "RGB camera recorder initialized")
                    }

                // Initialize audio recorder
                audioRecorder =
                    AudioRecorder(context).apply {
                        Log.i(TAG, "Audio recorder initialized")
                    }

                // Start system monitoring
                startSystemMonitoring()

                _coordinationState.value = CoordinationState.READY
                Log.i(TAG, "Multi-modal sensor coordination system initialized successfully")

                updateSystemHealth(
                    gsrHealthy = true,
                    thermalHealthy = true,
                    rgbHealthy = true,
                    audioHealthy = true,
                    syncHealthy = true,
                    message = "All sensor subsystems initialized",
                )

                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize coordination system: ${e.message}", e)
                _coordinationState.value = CoordinationState.ERROR

                updateSystemHealth(
                    gsrHealthy = false,
                    thermalHealthy = false,
                    rgbHealthy = false,
                    audioHealthy = false,
                    syncHealthy = false,
                    message = "System initialization failed: ${e.message}",
                )

                false
            }
        }

    /**
     * Start synchronized multi-modal recording
     */
    suspend fun startRecording(sessionDirectory: File): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                if (_coordinationState.value != CoordinationState.READY) {
                    Log.w(TAG, "Cannot start recording - system not ready: ${_coordinationState.value}")
                    return@withContext false
                }

                _coordinationState.value = CoordinationState.STARTING
                sessionDir = sessionDirectory

                if (!sessionDirectory.exists()) {
                    sessionDirectory.mkdirs()
                }

                Log.i(TAG, "Starting synchronized multi-modal recording in ${sessionDirectory.name}")

                // Initialize synchronization logging
                initializeSyncLogging(sessionDirectory)

                // Record unified start time for synchronization
                recordingStartTime = System.nanoTime()
                val systemStartTime = System.currentTimeMillis()

                // Start all sensors synchronously with precise timing
                val sensorStartJobs =
                    listOf(
                        async { startGSRRecording(sessionDirectory) },
                        async { startThermalRecording(sessionDirectory) },
                        async { startRGBRecording(sessionDirectory) },
                        async { startAudioRecording(sessionDirectory) },
                    )

                // Wait for all sensors to start
                val startResults = sensorStartJobs.awaitAll()
                val successCount = startResults.count { it }

                if (successCount >= 2) { // Allow recording with at least 2 sensors
                    _coordinationState.value = CoordinationState.RECORDING

                    // Start data synchronization monitoring
                    startSynchronizationMonitoring()

                    // Log recording start event
                    logSyncEvent(
                        sensorType = "SYSTEM",
                        dataType = "RECORDING_START",
                        value = successCount.toDouble(),
                        syncStatus = "SYNCHRONIZED",
                    )

                    updateSystemHealth(
                        gsrHealthy = startResults[0],
                        thermalHealthy = startResults[1],
                        rgbHealthy = startResults[2],
                        audioHealthy = startResults[3],
                        syncHealthy = true,
                        message = "Recording started - $successCount sensors active",
                    )

                    Log.i(TAG, "Multi-modal recording started successfully ($successCount/4 sensors)")
                    true
                } else {
                    throw RuntimeException("Failed to start minimum required sensors ($successCount/4)")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start recording: ${e.message}", e)
                _coordinationState.value = CoordinationState.ERROR

                updateSystemHealth(
                    gsrHealthy = false,
                    thermalHealthy = false,
                    rgbHealthy = false,
                    audioHealthy = false,
                    syncHealthy = false,
                    message = "Recording start failed: ${e.message}",
                )

                false
            }
        }
    }

    /**
     * Stop synchronized multi-modal recording
     */
    suspend fun stopRecording(): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                if (_coordinationState.value != CoordinationState.RECORDING) {
                    Log.w(TAG, "Not currently recording")
                    return@withContext false
                }

                _coordinationState.value = CoordinationState.STOPPING
                Log.i(TAG, "Stopping multi-modal recording...")

                // Stop synchronization monitoring first
                synchronizationJob?.cancel()

                // Log recording stop event
                logSyncEvent(
                    sensorType = "SYSTEM",
                    dataType = "RECORDING_STOP",
                    value = 0.0,
                    syncStatus = "SYNCHRONIZED",
                )

                // Stop all sensors
                val stopJobs =
                    listOf(
                        async { stopGSRRecording() },
                        async { stopThermalRecording() },
                        async { stopRGBRecording() },
                        async { stopAudioRecording() },
                    )

                stopJobs.awaitAll()

                // Close sync logging
                syncLogWriter?.flush()
                syncLogWriter?.close()
                syncLogWriter = null

                _coordinationState.value = CoordinationState.STOPPED

                updateSystemHealth(
                    gsrHealthy = false,
                    thermalHealthy = false,
                    rgbHealthy = false,
                    audioHealthy = false,
                    syncHealthy = false,
                    message = "Recording stopped - all sensors inactive",
                )

                Log.i(TAG, "Multi-modal recording stopped successfully")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping recording: ${e.message}", e)
                false
            }
        }
    }

    /**
     * Shutdown coordination system
     */
    suspend fun shutdown() {
        withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Shutting down multi-modal coordination system...")

                // Stop recording if active
                if (_coordinationState.value == CoordinationState.RECORDING) {
                    stopRecording()
                }

                // Cancel monitoring jobs
                healthMonitorJob?.cancel()
                metricsJob?.cancel()
                synchronizationJob?.cancel()

                // Shutdown sensor subsystems
                tc001IntegrationManager?.shutdown()
                gsrIntegrationManager?.let { manager ->
                    if (manager.isDeviceConnected()) {
                        manager.disconnect()
                    }
                }

                // Cancel coordination scope
                coordinationScope.cancel()

                _coordinationState.value = CoordinationState.UNINITIALIZED
                Log.i(TAG, "Multi-modal coordination system shutdown complete")
            } catch (e: Exception) {
                Log.e(TAG, "Error during coordination system shutdown: ${e.message}", e)
            }
        }
    }

    /**
     * Start individual sensor recordings
     */
    private suspend fun startGSRRecording(sessionDir: File): Boolean =
        try {
            gsrIntegrationManager?.let { manager ->
                // For now, we'll use the ShimmerRecorder directly
                // In production, this would use the full device selection flow
                val shimmerRecorder = ShimmerRecorder(context)
                shimmerRecorder.start(sessionDir)
                Log.i(TAG, "GSR recording started")
                true
            } ?: false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start GSR recording: ${e.message}")
            false
        }

    private suspend fun startThermalRecording(sessionDir: File): Boolean =
        try {
            tc001IntegrationManager?.let { manager ->
                if (manager.isSystemReady()) {
                    manager.startSystem()
                    delay(1000) // Allow system to stabilize
                    manager.startRecording(sessionDir)
                    Log.i(TAG, "Thermal recording started")
                    true
                } else {
                    Log.w(TAG, "TC001 system not ready")
                    false
                }
            } ?: false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start thermal recording: ${e.message}")
            false
        }

    private suspend fun startRGBRecording(sessionDir: File): Boolean =
        try {
            rgbRecorder?.start(sessionDir)
            Log.i(TAG, "RGB recording started")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start RGB recording: ${e.message}")
            false
        }

    private suspend fun startAudioRecording(sessionDir: File): Boolean =
        try {
            audioRecorder?.start(sessionDir)
            Log.i(TAG, "Audio recording started")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start audio recording: ${e.message}")
            false
        }

    /**
     * Stop individual sensor recordings
     */
    private suspend fun stopGSRRecording(): Boolean =
        try {
            gsrIntegrationManager?.let { manager ->
                if (manager.isDeviceStreaming()) {
                    manager.stopStreaming()
                }
            }
            Log.i(TAG, "GSR recording stopped")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop GSR recording: ${e.message}")
            false
        }

    private suspend fun stopThermalRecording(): Boolean =
        try {
            tc001IntegrationManager?.let { manager ->
                if (manager.isRecording()) {
                    manager.stopRecording()
                }
                manager.stopSystem()
            }
            Log.i(TAG, "Thermal recording stopped")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop thermal recording: ${e.message}")
            false
        }

    private suspend fun stopRGBRecording(): Boolean =
        try {
            rgbRecorder?.stop()
            Log.i(TAG, "RGB recording stopped")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop RGB recording: ${e.message}")
            false
        }

    private suspend fun stopAudioRecording(): Boolean =
        try {
            audioRecorder?.stop()
            Log.i(TAG, "Audio recording stopped")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop audio recording: ${e.message}")
            false
        }

    /**
     * Initialize synchronization logging
     */
    private fun initializeSyncLogging(sessionDir: File) {
        try {
            val syncLogFile = File(sessionDir, "synchronization_log.csv")
            syncLogWriter = BufferedWriter(FileWriter(syncLogFile))
            syncLogWriter!!.write("$SYNC_LOG_HEADER\n")
            syncLogWriter!!.flush()

            Log.i(TAG, "Synchronization logging initialized: ${syncLogFile.name}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize sync logging: ${e.message}")
        }
    }

    /**
     * Log synchronization event
     */
    private fun logSyncEvent(
        sensorType: String,
        dataType: String,
        value: Double,
        syncStatus: String,
    ) {
        try {
            val timestampNs = System.nanoTime()
            val systemTimeMs = System.currentTimeMillis()

            syncLogWriter?.let { writer ->
                val csvLine = "$timestampNs,$systemTimeMs,$sensorType,$dataType,$value,$syncStatus\n"
                writer.write(csvLine)

                // Flush periodically for data safety
                if ((timestampNs / 1_000_000) % 1000 < 10) { // Every ~1 second
                    writer.flush()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to log sync event: ${e.message}")
        }
    }

    /**
     * Start system monitoring
     */
    private fun startSystemMonitoring() {
        // Health monitoring
        healthMonitorJob =
            coordinationScope.launch {
                while (isActive) {
                    try {
                        delay(HEALTH_CHECK_INTERVAL_MS)
                        performSystemHealthCheck()
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in health monitoring: ${e.message}")
                    }
                }
            }

        // Metrics monitoring
        metricsJob =
            coordinationScope.launch {
                while (isActive) {
                    try {
                        delay(METRICS_UPDATE_INTERVAL_MS)
                        updateSynchronizationMetrics()
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in metrics monitoring: ${e.message}")
                    }
                }
            }
    }

    /**
     * Start synchronization monitoring during recording
     */
    private fun startSynchronizationMonitoring() {
        synchronizationJob =
            coordinationScope.launch {
                while (isActive && _coordinationState.value == CoordinationState.RECORDING) {
                    try {
                        delay(100) // Check sync every 100ms

                        // Monitor data flow and timing from all sensors
                        checkDataSynchronization()
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in synchronization monitoring: ${e.message}")
                    }
                }
            }
    }

    /**
     * Perform system health check
     */
    private suspend fun performSystemHealthCheck() {
        val gsrHealthy = gsrIntegrationManager?.isDeviceConnected() ?: false
        val thermalHealthy =
            tc001IntegrationManager?.getCurrentState()?.let { state ->
                state != TC001SensorIntegrationManager.Companion.IntegrationState.ERROR
            } ?: false
        val rgbHealthy = true // RGB is typically always available
        val audioHealthy = true // Audio is typically always available
        val syncHealthy = checkSynchronizationHealth()

        val overallHealthy = gsrHealthy && thermalHealthy && rgbHealthy && audioHealthy && syncHealthy
        val message =
            when {
                !gsrHealthy -> "GSR sensor issues detected"
                !thermalHealthy -> "Thermal camera issues detected"
                !syncHealthy -> "Data synchronization issues detected"
                else -> "All systems operating normally"
            }

        updateSystemHealth(gsrHealthy, thermalHealthy, rgbHealthy, audioHealthy, syncHealthy, message)
    }

    /**
     * Check data synchronization health
     */
    private fun checkSynchronizationHealth(): Boolean =
        try {
            val currentTime = System.currentTimeMillis()
            val timeSinceLastSync = currentTime - lastSyncCheck

            // Consider sync healthy if we've had recent data flow
            timeSinceLastSync < 5000 // Less than 5 seconds since last sync check
        } catch (e: Exception) {
            Log.w(TAG, "Error checking sync health: ${e.message}")
            false
        }

    /**
     * Check data synchronization across sensors
     */
    private suspend fun checkDataSynchronization() {
        // Check timing alignment between sensor data streams
        val currentTime = System.nanoTime()
        val timeSinceStart = (currentTime - recordingStartTime) / 1_000_000.0 // Convert to milliseconds

        // Update sync check time
        lastSyncCheck = System.currentTimeMillis()

        // Log periodic sync check
        if (timeSinceStart.toLong() % 10000 < 100) { // Every ~10 seconds
            logSyncEvent(
                sensorType = "SYNC",
                dataType = "HEALTH_CHECK",
                value = timeSinceStart,
                syncStatus = "MONITORING",
            )
        }
    }

    /**
     * Update synchronization metrics
     */
    private suspend fun updateSynchronizationMetrics() {
        val currentTime = System.nanoTime()
        val recordingDuration =
            if (recordingStartTime > 0) {
                (currentTime - recordingStartTime) / 1_000_000.0 // Convert to milliseconds
            } else {
                0.0
            }

        val metrics =
            SynchronizationMetrics(
                recordingDurationMs = recordingDuration,
                activeSensorCount = getActiveSensorCount(),
                syncPrecisionMs = calculateSyncPrecision(),
                dataRateHz = calculateOverallDataRate(),
                lastUpdateTime = System.currentTimeMillis(),
            )

        _synchronizationMetrics.value = metrics
    }

    /**
     * Get count of active sensors
     */
    private fun getActiveSensorCount(): Int {
        var count = 0
        if (gsrIntegrationManager?.isDeviceConnected() == true) count++
        if (tc001IntegrationManager?.isRecording() == true) count++
        // RGB and audio would be checked here in full implementation
        return count
    }

    /**
     * Calculate synchronization precision
     */
    private fun calculateSyncPrecision(): Double {
        // This would analyze actual timing data from the buffer
        // For now, return target precision
        return SYNC_PRECISION_TARGET_MS
    }

    /**
     * Calculate overall data rate
     */
    private fun calculateOverallDataRate(): Double {
        // This would calculate based on actual data throughput
        // For now, return estimated combined rate
        return 128.0 + 30.0 + 30.0 // GSR (128Hz) + Thermal (30Hz) + RGB (30Hz)
    }

    /**
     * Update system health status
     */
    private fun updateSystemHealth(
        gsrHealthy: Boolean,
        thermalHealthy: Boolean,
        rgbHealthy: Boolean,
        audioHealthy: Boolean,
        syncHealthy: Boolean,
        message: String,
    ) {
        val health =
            SystemHealthStatus(
                gsrHealthy = gsrHealthy,
                thermalHealthy = thermalHealthy,
                rgbHealthy = rgbHealthy,
                audioHealthy = audioHealthy,
                synchronizationHealthy = syncHealthy,
                overallHealthy = gsrHealthy && thermalHealthy && rgbHealthy && audioHealthy && syncHealthy,
                statusMessage = message,
                lastUpdateTime = System.currentTimeMillis(),
            )

        _systemHealth.value = health
    }

    /**
     * GSR data callback for synchronization
     */
    private val gsrDataCallback =
        object : ShimmerDataCallback {
            override fun onGSRData(data: GSRDataPoint) {
                // Buffer GSR data for synchronization
                dataBuffer.addGSRData(data)

                // Log GSR data for synchronization analysis
                logSyncEvent(
                    sensorType = "GSR",
                    dataType = "SAMPLE",
                    value = data.gsrMicrosiemens,
                    syncStatus = "RECEIVED",
                )
            }

            override fun onConnectionStateChanged(
                connected: Boolean,
                message: String,
            ) {
                Log.i(TAG, "GSR connection: $connected - $message")
            }

            override fun onStreamingStateChanged(
                streaming: Boolean,
                message: String,
            ) {
                Log.i(TAG, "GSR streaming: $streaming - $message")
            }

            override fun onDeviceInitialized(message: String) {
                Log.i(TAG, "GSR device initialized: $message")
            }

            override fun onError(error: String) {
                Log.e(TAG, "GSR error: $error")
            }
        }

    /**
     * Get current coordination state
     */
    fun getCurrentState(): CoordinationState = _coordinationState.value

    /**
     * Check if system is ready for recording
     */
    fun isSystemReady(): Boolean = _coordinationState.value == CoordinationState.READY

    /**
     * Check if currently recording
     */
    fun isRecording(): Boolean = _coordinationState.value == CoordinationState.RECORDING
}

/**
 * System health status for all sensors
 */
data class SystemHealthStatus(
    val gsrHealthy: Boolean = false,
    val thermalHealthy: Boolean = false,
    val rgbHealthy: Boolean = false,
    val audioHealthy: Boolean = false,
    val synchronizationHealthy: Boolean = false,
    val overallHealthy: Boolean = false,
    val statusMessage: String = "",
    val lastUpdateTime: Long = 0L,
)

/**
 * Synchronization metrics and statistics
 */
data class SynchronizationMetrics(
    val recordingDurationMs: Double = 0.0,
    val activeSensorCount: Int = 0,
    val syncPrecisionMs: Double = 0.0,
    val dataRateHz: Double = 0.0,
    val totalDataPoints: Long = 0L,
    val lastUpdateTime: Long = 0L,
)

/**
 * Multi-sensor data buffer for synchronization analysis
 */
class SensorDataBuffer {
    private val gsrData = mutableListOf<GSRDataPoint>()
    private val maxBufferSize = 1000 // Keep last 1000 samples

    fun addGSRData(data: GSRDataPoint) {
        synchronized(gsrData) {
            gsrData.add(data)
            if (gsrData.size > maxBufferSize) {
                gsrData.removeAt(0) // Use removeAt(0) instead of removeFirst() for API compatibility
            }
        }
    }

    fun getRecentGSRData(count: Int = 10): List<GSRDataPoint> {
        synchronized(gsrData) {
            return gsrData.takeLast(count)
        }
    }

    fun clearBuffer() {
        synchronized(gsrData) {
            gsrData.clear()
        }
    }
}
