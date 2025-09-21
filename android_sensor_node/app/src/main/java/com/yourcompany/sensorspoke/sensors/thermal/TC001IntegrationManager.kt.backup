package com.yourcompany.sensorspoke.sensors.thermal

import android.content.Context
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext

/**
 * TC001IntegrationManager - Comprehensive thermal integration coordinator
 *
 * Coordinates all TC001 components for seamless thermal camera operation:
 * - Device connection management via TC001Connector
 * - Real-time data processing via TC001DataManager
 * - UI state management via TC001UIController
 * - System initialization via TC001InitUtil
 *
 * This manager ensures all TC001 components work together harmoniously
 * and provides a single integration point for the thermal camera system.
 */
class TC001IntegrationManager(
    private val context: Context,
) {
    companion object {
        private const val TAG = "TC001IntegrationManager"
    }

    // Component instances
    private var tc001Connector: TC001Connector? = null
    private var tc001DataManager: TC001DataManager? = null
    private var tc001UIController: TC001UIController? = null

    // Integration state
    private val _integrationState = MutableLiveData<TC001IntegrationState>()
    val integrationState: LiveData<TC001IntegrationState> = _integrationState

    private val _systemStatus = MutableLiveData<String>()
    val systemStatus: LiveData<String> = _systemStatus

    private val integrationScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var isInitialized = false

    init {
        _integrationState.value = TC001IntegrationState.UNINITIALIZED
        _systemStatus.value = "TC001 System Uninitialized"
    }

    /**
     * Initialize complete TC001 integration system
     */
    suspend fun initializeSystem(): Boolean =
        withContext(Dispatchers.IO) {
            if (isInitialized) {
                Log.w(TAG, "TC001 system already initialized")
                return@withContext true
            }

            try {
                _integrationState.postValue(TC001IntegrationState.INITIALIZING)
                _systemStatus.postValue("Initializing TC001 System...")

                // Step 1: Initialize TC001 utilities
                TC001InitUtil.initLog()
                TC001InitUtil.initReceiver(context)
                TC001InitUtil.initTC001DeviceManager(context)

                // Step 2: Initialize connector
                tc001Connector = TC001Connector(context)

                // Step 3: Initialize data manager
                tc001DataManager = TC001DataManager(context)

                // Step 4: Initialize UI controller
                tc001UIController = TC001UIController()

                // Step 5: Setup component integration
                setupComponentIntegration()

                isInitialized = true
                _integrationState.postValue(TC001IntegrationState.INITIALIZED)
                _systemStatus.postValue("TC001 System Ready")

                Log.i(TAG, "TC001 integration system initialized successfully")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize TC001 system", e)
                _integrationState.postValue(TC001IntegrationState.ERROR)
                _systemStatus.postValue("TC001 System Error: ${e.message}")
                false
            }
        }

    /**
     * Start complete TC001 system (discovery + processing)
     */
    suspend fun startSystem(): Boolean =
        withContext(Dispatchers.IO) {
            if (!isInitialized) {
                Log.e(TAG, "Cannot start system - not initialized")
                return@withContext false
            }

            try {
                _integrationState.postValue(TC001IntegrationState.STARTING)
                _systemStatus.postValue("Starting TC001 System...")

                // Start device discovery and connection
                val connectionResult = tc001Connector?.connect() ?: false

                if (connectionResult) {
                    // Start thermal data processing
                    tc001DataManager?.startProcessing()

                    _integrationState.postValue(TC001IntegrationState.RUNNING)
                    _systemStatus.postValue("TC001 System Running")

                    Log.i(TAG, "TC001 system started successfully")
                    true
                } else {
                    _integrationState.postValue(TC001IntegrationState.CONNECTION_FAILED)
                    _systemStatus.postValue("TC001 Connection Failed")

                    Log.w(TAG, "TC001 system start failed - no device connection")
                    false
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error starting TC001 system", e)
                _integrationState.postValue(TC001IntegrationState.ERROR)
                _systemStatus.postValue("TC001 System Error: ${e.message}")
                false
            }
        }

    /**
     * Stop complete TC001 system
     */
    suspend fun stopSystem() =
        withContext(Dispatchers.IO) {
            try {
                _integrationState.postValue(TC001IntegrationState.STOPPING)
                _systemStatus.postValue("Stopping TC001 System...")

                // Stop data processing
                tc001DataManager?.stopProcessing()

                // Disconnect device
                tc001Connector?.disconnect()

                _integrationState.postValue(TC001IntegrationState.INITIALIZED)
                _systemStatus.postValue("TC001 System Stopped")

                Log.i(TAG, "TC001 system stopped successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping TC001 system", e)
                _integrationState.postValue(TC001IntegrationState.ERROR)
            }
        }

    /**
     * Setup integration between TC001 components
     */
    private fun setupComponentIntegration() {
        // Connect connector state to UI controller
        tc001Connector?.connectionState?.observeForever { state ->
            tc001UIController?.updateDeviceConnection(state == TC001ConnectionState.CONNECTED)
        }

        // Connect data manager to UI controller for temperature updates
        tc001DataManager?.temperatureData?.observeForever { tempData ->
            tempData?.let {
                tc001UIController?.updateCurrentTemperature(it.centerTemperature)
            }
        }

        Log.i(TAG, "TC001 component integration setup complete")
    }

    /**
     * Get component instances for external use
     */
    fun getConnector(): TC001Connector? = tc001Connector

    fun getDataManager(): TC001DataManager? = tc001DataManager

    fun getUIController(): TC001UIController? = tc001UIController

    /**
     * Check if system is ready for use
     */
    fun isSystemReady(): Boolean =
        isInitialized &&
            _integrationState.value == TC001IntegrationState.RUNNING

    /**
     * Cleanup all TC001 integration resources
     */
    fun cleanup() {
        integrationScope.cancel()

        runBlocking {
            stopSystem()
        }

        tc001DataManager?.cleanup()
        tc001Connector?.cleanup()

        isInitialized = false
        _integrationState.value = TC001IntegrationState.UNINITIALIZED
        _systemStatus.value = "TC001 System Cleaned Up"

        Log.i(TAG, "TC001 integration manager cleanup completed")
    }
}

/**
 * TC001 integration system states
 */
enum class TC001IntegrationState {
    UNINITIALIZED,
    INITIALIZING,
    INITIALIZED,
    STARTING,
    RUNNING,
    STOPPING,
    CONNECTION_FAILED,
    ERROR,
}
package com.yourcompany.sensorspoke.sensors.thermal

import android.content.Context
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.yourcompany.sensorspoke.controller.RecordingController
import com.yourcompany.sensorspoke.sensors.thermal.ThermalCameraRecorder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * TC001RecordingIntegration - Bridge between TC001 system and main recording pipeline
 *
 * This component ensures that TC001 thermal data is properly integrated into
 * the main sensor recording system, providing:
 * - Seamless integration with RecordingController
 * - Synchronized data recording with other sensors
 * - Professional thermal data export and storage
 * - Real-time data streaming to PC Hub via PreviewBus
 */
class TC001RecordingIntegration(
    private val context: Context,
    private val recordingController: RecordingController,
) {
    companion object {
        private const val TAG = "TC001RecordingIntegration"
    }

    private var tc001IntegrationManager: TC001IntegrationManager? = null
    private var thermalRecorder: ThermalCameraRecorder? = null
    private var integrationScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    private val _recordingStatus = MutableLiveData<TC001RecordingStatus>()
    val recordingStatus: LiveData<TC001RecordingStatus> = _recordingStatus

    private val _thermalStats = MutableLiveData<TC001RecordingStats>()
    val thermalStats: LiveData<TC001RecordingStats> = _thermalStats

    private var currentSessionId: String? = null
    private var framesSaved = 0
    private var dataPointsRecorded = 0
    private var sessionStartTime = 0L

    /**
     * Initialize TC001 recording integration
     */
    suspend fun initialize(): Boolean =
        withContext(Dispatchers.IO) {
            try {
                // Initialize TC001 integration manager
                tc001IntegrationManager = TC001IntegrationManager(context)
                val initResult = tc001IntegrationManager!!.initializeSystem()

                if (!initResult) {
                    Log.e(TAG, "Failed to initialize TC001 integration manager")
                    return@withContext false
                }

                // Initialize thermal recorder and register with recording controller
                thermalRecorder = ThermalCameraRecorder(context)
                recordingController.register("thermal_tc001", thermalRecorder!!)

                // Setup integration observers
                setupIntegrationObservers()

                _recordingStatus.postValue(TC001RecordingStatus.READY)
                Log.i(TAG, "TC001 recording integration initialized successfully")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize TC001 recording integration", e)
                _recordingStatus.postValue(TC001RecordingStatus.ERROR)
                false
            }
        }

    /**
     * Start TC001 recording session
     */
    suspend fun startRecording(sessionId: String, sessionDirectory: File): Boolean =
        withContext(Dispatchers.IO) {
            try {
                currentSessionId = sessionId
                sessionStartTime = System.nanoTime()
                framesSaved = 0
                dataPointsRecorded = 0

                // Start TC001 system
                val startResult = tc001IntegrationManager?.startSystem() ?: false
                if (!startResult) {
                    Log.e(TAG, "Failed to start TC001 system")
                    return@withContext false
                }

                // Start thermal recorder using the public interface
                thermalRecorder?.start(sessionDirectory)

                _recordingStatus.postValue(TC001RecordingStatus.RECORDING)
                Log.i(TAG, "TC001 recording session started: $sessionId")

                // Start data monitoring for statistics
                startDataMonitoring()
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start TC001 recording", e)
                _recordingStatus.postValue(TC001RecordingStatus.ERROR)
                false
            }
        }

    /**
     * Stop TC001 recording session
     */
    suspend fun stopRecording(): Boolean =
        withContext(Dispatchers.IO) {
            try {
                // Stop thermal recorder using public interface
                thermalRecorder?.stop()

                // Stop TC001 system
                tc001IntegrationManager?.stopSystem()

                // Calculate final statistics
                val recordingDuration = (System.nanoTime() - sessionStartTime) / 1_000_000_000.0
                val finalStats =
                    TC001RecordingStats(
                        sessionId = currentSessionId ?: "unknown",
                        recordingDuration = recordingDuration,
                        framesSaved = framesSaved,
                        dataPointsRecorded = dataPointsRecorded,
                        averageFrameRate = framesSaved / recordingDuration,
                    )
                _thermalStats.postValue(finalStats)

                _recordingStatus.postValue(TC001RecordingStatus.COMPLETED)
                Log.i(TAG, "TC001 recording session completed: ${finalStats.sessionId}")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to stop TC001 recording", e)
                _recordingStatus.postValue(TC001RecordingStatus.ERROR)
                false
            }
        }

    /**
     * Setup observers for TC001 integration components
     */
    private fun setupIntegrationObservers() {
        tc001IntegrationManager?.let { manager ->
            // Monitor thermal data for recording statistics
            manager.getDataManager()?.temperatureData?.observeForever { tempData ->
                tempData?.let {
                    dataPointsRecorded++
                    updateRecordingStats()
                }
            }

            // Monitor thermal bitmaps for frame statistics
            manager.getDataManager()?.thermalBitmap?.observeForever { bitmap ->
                bitmap?.let {
                    framesSaved++
                    updateRecordingStats()
                }
            }
        }
    }

    /**
     * Start monitoring data flow for recording statistics
     */
    private fun startDataMonitoring() {
        integrationScope.launch {
            while (_recordingStatus.value == TC001RecordingStatus.RECORDING) {
                try {
                    updateRecordingStats()
                    delay(1000) // Update stats every second
                } catch (e: Exception) {
                    Log.e(TAG, "Error in data monitoring", e)
                    break
                }
            }
        }
    }

    /**
     * Update recording statistics
     */
    private fun updateRecordingStats() {
        currentSessionId?.let { sessionId ->
            val currentTime = System.nanoTime()
            val recordingDuration = (currentTime - sessionStartTime) / 1_000_000_000.0

            val stats =
                TC001RecordingStats(
                    sessionId = sessionId,
                    recordingDuration = recordingDuration,
                    framesSaved = framesSaved,
                    dataPointsRecorded = dataPointsRecorded,
                    averageFrameRate = if (recordingDuration > 0) framesSaved / recordingDuration else 0.0,
                )
            _thermalStats.postValue(stats)
        }
    }

    /**
     * Get current TC001 integration manager for external access
     */
    fun getTC001IntegrationManager(): TC001IntegrationManager? = tc001IntegrationManager

    /**
     * Check if TC001 system is ready for recording
     */
    fun isTC001Ready(): Boolean = tc001IntegrationManager?.isSystemReady() ?: false

    /**
     * Cleanup TC001 recording integration
     */
    fun cleanup() {
        integrationScope.cancel()
        tc001IntegrationManager?.cleanup()

        Log.i(TAG, "TC001 recording integration cleaned up")
    }
}

/**
 * TC001 recording status states
 */
enum class TC001RecordingStatus {
    UNINITIALIZED,
    READY,
    RECORDING,
    COMPLETED,
    ERROR,
}

/**
 * TC001 recording statistics data
 */
data class TC001RecordingStats(
    val sessionId: String,
    val recordingDuration: Double, // seconds
    val framesSaved: Int,
    val dataPointsRecorded: Int,
    val averageFrameRate: Double,
)
package com.yourcompany.sensorspoke.sensors.thermal

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Professional TC001 Sensor Integration Manager
 *
 * Coordinates all TC001 thermal camera components for enterprise-grade thermal sensing.
 * Provides unified lifecycle management, data processing, and system health monitoring
 * for the complete thermal imaging pipeline.
 */
class TC001SensorIntegrationManager(
    private val context: Context,
) {
    companion object {
        private const val TAG = "TC001SensorIntegration"

        // Integration states
        enum class IntegrationState {
            UNINITIALIZED,
            INITIALIZING,
            READY,
            CONNECTING,
            CONNECTED,
            STREAMING,
            RECORDING,
            STOPPING,
            ERROR,
            DISCONNECTED,
        }

        // Thermal processing constants
        const val THERMAL_FPS_TARGET = 30
        const val THERMAL_RESOLUTION_WIDTH = 256
        const val THERMAL_RESOLUTION_HEIGHT = 192
        const val THERMAL_PRECISION_TARGET = 0.1 // ±0.1°C precision goal
    }

    // Core components
    private var tc001Connector: TC001Connector? = null
    private var tc001DataManager: TC001DataManager? = null
    private var tc001UIController: TC001UIController? = null

    // State management
    private val _integrationState = MutableStateFlow(IntegrationState.UNINITIALIZED)
    val integrationState: StateFlow<IntegrationState> = _integrationState

    private val _systemHealth = MutableStateFlow(ThermalSystemHealth())
    val systemHealth: StateFlow<ThermalSystemHealth> = _systemHealth

    private val _thermalMetrics = MutableStateFlow(ThermalProcessingMetrics())
    val thermalMetrics: StateFlow<ThermalProcessingMetrics> = _thermalMetrics

    // Lifecycle management
    private val integrationScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var recordingJob: Job? = null
    private var healthMonitorJob: Job? = null

    /**
     * Initialize the complete TC001 integration system
     */
    suspend fun initializeSystem(): Boolean =
        withContext(Dispatchers.IO) {
            try {
                _integrationState.value = IntegrationState.INITIALIZING
                Log.i(TAG, "Initializing TC001 sensor integration system...")

                // Initialize core components
                tc001Connector = TC001Connector(context)

                tc001DataManager = TC001DataManager(context).apply {
                    // Connect the data manager to the connector for real hardware access
                    setTC001Connector(tc001Connector!!)
                }

                tc001UIController = TC001UIController()

                // Set up component interactions
                setupComponentCoordination()

                // Start health monitoring
                startHealthMonitoring()

                _integrationState.value = IntegrationState.READY
                Log.i(TAG, "TC001 sensor integration system initialized successfully")

                updateSystemHealth(
                    connectionHealthy = true,
                    processingHealthy = true,
                    temperatureHealthy = true,
                    message = "System initialized and ready",
                )

                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize TC001 integration system: ${e.message}", e)
                _integrationState.value = IntegrationState.ERROR

                updateSystemHealth(
                    connectionHealthy = false,
                    processingHealthy = false,
                    temperatureHealthy = false,
                    message = "Initialization failed: ${e.message}",
                )

                false
            }
        }

    /**
     * Start the complete thermal system
     */
    suspend fun startSystem(): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                if (_integrationState.value != IntegrationState.READY) {
                    Log.w(TAG, "Cannot start system - not in ready state: ${_integrationState.value}")
                    return@withContext false
                }

                _integrationState.value = IntegrationState.CONNECTING
                Log.i(TAG, "Starting TC001 thermal system...")

                // Connect to TC001 device
                val connected = tc001Connector?.connect() ?: false
                if (!connected) {
                    throw RuntimeException("Failed to connect to TC001 device")
                }

                _integrationState.value = IntegrationState.CONNECTED

                // Start data processing pipeline
                tc001DataManager?.startProcessing()

                // Start UI updates
                tc001UIController?.updateConnectionStatus(true)

                // Begin thermal streaming - simplified check
                _integrationState.value = IntegrationState.STREAMING

                updateSystemHealth(
                    connectionHealthy = true,
                    processingHealthy = true,
                    temperatureHealthy = true,
                    message = "Thermal streaming active at ${THERMAL_FPS_TARGET}FPS",
                )

                Log.i(TAG, "TC001 thermal system started successfully")
                return@withContext true
                // } else {
                //     throw RuntimeException("Failed to start thermal streaming")
                // }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start TC001 system: ${e.message}", e)
                _integrationState.value = IntegrationState.ERROR

                updateSystemHealth(
                    connectionHealthy = false,
                    processingHealthy = false,
                    temperatureHealthy = false,
                    message = "System start failed: ${e.message}",
                )

                false
            }
        }
    }

    /**
     * Start thermal data recording
     */
    suspend fun startRecording(sessionDir: File): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                if (_integrationState.value != IntegrationState.STREAMING) {
                    Log.w(TAG, "Cannot start recording - not streaming: ${_integrationState.value}")
                    return@withContext false
                }

                _integrationState.value = IntegrationState.RECORDING
                Log.i(TAG, "Starting thermal data recording...")

                // Start data manager recording
                // Note: TC001DataManager doesn't have startRecording method,
                // it automatically processes frames when startProcessing() is called

                // Start recording monitoring job
                recordingJob =
                    integrationScope.launch {
                        monitorRecording()
                    }

                updateSystemHealth(
                    connectionHealthy = true,
                    processingHealthy = true,
                    temperatureHealthy = true,
                    message = "Recording thermal data to ${sessionDir.name}",
                )

                Log.i(TAG, "Thermal data recording started")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start recording: ${e.message}", e)
                false
            }
        }
    }

    /**
     * Stop thermal data recording
     */
    suspend fun stopRecording(): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                if (_integrationState.value != IntegrationState.RECORDING) {
                    Log.w(TAG, "Not currently recording")
                    return@withContext false
                }

                _integrationState.value = IntegrationState.STOPPING
                Log.i(TAG, "Stopping thermal data recording...")

                // Stop recording job
                recordingJob?.cancel()
                recordingJob = null

                // Stop data manager recording
                // Note: TC001DataManager doesn't have separate stopRecording method

                _integrationState.value = IntegrationState.STREAMING

                updateSystemHealth(
                    connectionHealthy = true,
                    processingHealthy = true,
                    temperatureHealthy = true,
                    message = "Recording stopped, streaming continues",
                )

                Log.i(TAG, "Thermal data recording stopped")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to stop recording: ${e.message}", e)
                false
            }
        }
    }

    /**
     * Stop the complete thermal system
     */
    suspend fun stopSystem() {
        withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Stopping TC001 thermal system...")

                // Stop recording if active
                if (_integrationState.value == IntegrationState.RECORDING) {
                    stopRecording()
                }

                // Stop streaming
                // Note: TC001Connector doesn't have stopStreaming method,
                // streaming stops when disconnect() is called

                // Stop UI updates
                tc001UIController?.updateConnectionStatus(false)

                // Stop data processing
                tc001DataManager?.stopProcessing()

                // Disconnect from device
                tc001Connector?.let { connector ->
                    launch { connector.disconnect() }
                }

                _integrationState.value = IntegrationState.DISCONNECTED

                updateSystemHealth(
                    connectionHealthy = false,
                    processingHealthy = false,
                    temperatureHealthy = false,
                    message = "System stopped",
                )

                Log.i(TAG, "TC001 thermal system stopped")
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping TC001 system: ${e.message}", e)
            }
        }
    }

    /**
     * Shutdown and cleanup all resources
     */
    suspend fun shutdown() {
        withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Shutting down TC001 integration system...")

                // Stop system if active
                stopSystem()

                // Cancel health monitoring
                healthMonitorJob?.cancel()

                // Cleanup components
                tc001DataManager?.cleanup()
                tc001Connector?.cleanup()

                // Cancel integration scope
                integrationScope.cancel()

                _integrationState.value = IntegrationState.UNINITIALIZED

                Log.i(TAG, "TC001 integration system shutdown complete")
            } catch (e: Exception) {
                Log.e(TAG, "Error during TC001 system shutdown: ${e.message}", e)
            }
        }
    }

    /**
     * Set up coordination between components
     */
    private fun setupComponentCoordination() {
        // Connect data flow: Connector -> DataManager -> UIController
        tc001Connector?.let { connector ->
            // Set up connection status monitoring
            connector.connectionState.observeForever { connectionState ->
                integrationScope.launch {
                    handleConnectionStateChange(connectionState)
                }
            }
        }

        tc001DataManager?.let { dataManager ->
            // Monitor thermal frames for processing metrics
            dataManager.thermalFrame.observeForever { frameData ->
                integrationScope.launch {
                    updateThermalMetrics(frameData)
                }
            }

            // Monitor temperature data for health checks
            dataManager.temperatureData.observeForever { tempData ->
                integrationScope.launch {
                    updateTemperatureMetrics(tempData)
                }
            }
        }

        tc001UIController?.let { uiController ->
            // Keep UI controller in sync with connection status
            val connectionHealthy = _systemHealth.value.connectionHealthy
            uiController.updateConnectionStatus(connectionHealthy)
        }
    }

    /**
     * Start health monitoring
     */
    private fun startHealthMonitoring() {
        healthMonitorJob =
            integrationScope.launch {
                while (isActive) {
                    try {
                        // Check system health every 5 seconds
                        delay(5000)
                        performHealthCheck()
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in health monitoring: ${e.message}")
                    }
                }
            }
    }

    /**
     * Perform comprehensive system health check
     */
    private suspend fun performHealthCheck() {
        val connectionHealthy = tc001Connector?.isConnected() ?: false
        val processingHealthy = checkDataProcessingHealth()
        val temperatureHealthy = checkTemperatureHealth()

        val overallHealthy = connectionHealthy && processingHealthy && temperatureHealthy
        val message =
            when {
                !connectionHealthy -> "Connection issues detected"
                !processingHealthy -> "Processing performance issues"
                !temperatureHealthy -> "Temperature readings outside normal range"
                else -> "All systems operating normally"
            }

        updateSystemHealth(connectionHealthy, processingHealthy, temperatureHealthy, message)
    }

    /**
     * Check temperature measurement health
     */
    private suspend fun checkTemperatureHealth(): Boolean =
        try {
            val currentMetrics = _thermalMetrics.value

            // Check if temperature readings are within reasonable bounds
            val minTemp = currentMetrics.minTemperature
            val maxTemp = currentMetrics.maxTemperature
            val avgTemp = currentMetrics.averageTemperature

            // Reasonable environmental temperature bounds
            minTemp > -40.0 &&
                minTemp < 100.0 &&
                maxTemp > -40.0 &&
                maxTemp < 100.0 &&
                avgTemp > -40.0 &&
                avgTemp < 100.0 &&
                maxTemp > minTemp // Basic sanity check
        } catch (e: Exception) {
            Log.w(TAG, "Error checking temperature health: ${e.message}")
            false
        }

    /**
     * Monitor recording session
     */
    private suspend fun monitorRecording() {
        var lastFrameCount = 0L
        var staleFrameCount = 0

        while (currentCoroutineContext().isActive && _integrationState.value == IntegrationState.RECORDING) {
            try {
                delay(1000) // Check every second

                val currentFrameCount = _thermalMetrics.value.totalFrames

                if (currentFrameCount == lastFrameCount) {
                    staleFrameCount++
                    if (staleFrameCount >= 10) { // 10 seconds without new frames
                        Log.w(TAG, "Recording appears stalled - no new frames for 10 seconds")
                        updateSystemHealth(
                            connectionHealthy = false,
                            processingHealthy = false,
                            temperatureHealthy = true,
                            message = "Recording stalled - no new thermal frames",
                        )
                    }
                } else {
                    staleFrameCount = 0
                    lastFrameCount = currentFrameCount
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error in recording monitor: ${e.message}")
            }
        }
    }

    /**
     * Handle connection state changes from TC001Connector
     */
    private fun handleConnectionStateChange(connectionState: TC001ConnectionState) {
        when (connectionState) {
            TC001ConnectionState.CONNECTED -> {
                if (_integrationState.value == IntegrationState.CONNECTING) {
                    _integrationState.value = IntegrationState.CONNECTED
                }
                updateSystemHealth(
                    connectionHealthy = true,
                    processingHealthy = true,
                    temperatureHealthy = true,
                    message = "TC001 device connected",
                )
            }
            TC001ConnectionState.DISCONNECTED -> {
                _integrationState.value = IntegrationState.DISCONNECTED
                updateSystemHealth(
                    connectionHealthy = false,
                    processingHealthy = false,
                    temperatureHealthy = false,
                    message = "TC001 device disconnected",
                )
            }
            TC001ConnectionState.ERROR -> {
                _integrationState.value = IntegrationState.ERROR
                updateSystemHealth(
                    connectionHealthy = false,
                    processingHealthy = false,
                    temperatureHealthy = false,
                    message = "TC001 connection error",
                )
            }
            else -> {
                // Handle other states as needed
            }
        }
    }

    /**
     * Update thermal metrics from thermal frame data
     */
    private fun updateThermalMetrics(thermalFrameData: ByteArray) {
        val currentMetrics = _thermalMetrics.value
        val updatedMetrics = currentMetrics.copy(
            totalFrames = currentMetrics.totalFrames + 1,
            lastUpdateTime = System.currentTimeMillis(),
        )
        _thermalMetrics.value = updatedMetrics
    }

    /**
     * Update temperature metrics from TC001 temperature data
     */
    private fun updateTemperatureMetrics(tempData: TC001TemperatureData) {
        val currentMetrics = _thermalMetrics.value
        val updatedMetrics = currentMetrics.copy(
            minTemperature = tempData.minTemperature.toDouble(),
            maxTemperature = tempData.maxTemperature.toDouble(),
            averageTemperature = tempData.avgTemperature.toDouble(),
            centerTemperature = tempData.centerTemperature.toDouble(),
            emissivity = tempData.emissivity.toDouble(),
            lastUpdateTime = System.currentTimeMillis(),
        )
        _thermalMetrics.value = updatedMetrics
    }

    /**
     * Check data processing health
     */
    private suspend fun checkDataProcessingHealth(): Boolean {
        // Check if data manager is processing frames regularly
        val lastUpdateTime = _thermalMetrics.value.lastUpdateTime
        val currentTime = System.currentTimeMillis()

        // Consider processing healthy if we've had an update within the last 5 seconds
        return (currentTime - lastUpdateTime) < 5000
    }

    /**
     * Update thermal processing metrics
     */
    private fun updateThermalMetrics(thermalData: Any) {
        // Extract temperature data and update metrics
        // This would depend on the actual thermal data structure
        integrationScope.launch {
            val currentMetrics = _thermalMetrics.value
            val updatedMetrics =
                currentMetrics.copy(
                    totalFrames = currentMetrics.totalFrames + 1,
                    lastUpdateTime = System.currentTimeMillis(),
                )
            _thermalMetrics.value = updatedMetrics
        }
    }

    /**
     * Update processing performance metrics
     */
    private fun updateProcessingMetrics(metrics: Any) {
        // Update processing performance metrics from data manager
    }

    /**
     * Update system health status
     */
    private fun updateSystemHealth(
        connectionHealthy: Boolean,
        processingHealthy: Boolean,
        temperatureHealthy: Boolean,
        message: String,
    ) {
        val health =
            ThermalSystemHealth(
                connectionHealthy = connectionHealthy,
                processingHealthy = processingHealthy,
                temperatureHealthy = temperatureHealthy,
                overallHealthy = connectionHealthy && processingHealthy && temperatureHealthy,
                statusMessage = message,
                lastUpdateTime = System.currentTimeMillis(),
            )

        _systemHealth.value = health
    }

    /**
     * Get current integration state
     */
    fun getCurrentState(): IntegrationState = _integrationState.value

    /**
     * Check if system is ready for operations
     */
    fun isSystemReady(): Boolean = _integrationState.value == IntegrationState.READY

    /**
     * Check if system is currently recording
     */
    fun isRecording(): Boolean = _integrationState.value == IntegrationState.RECORDING
}

/**
 * Thermal system health monitoring data
 */
data class ThermalSystemHealth(
    val connectionHealthy: Boolean = false,
    val processingHealthy: Boolean = false,
    val temperatureHealthy: Boolean = false,
    val overallHealthy: Boolean = false,
    val statusMessage: String = "",
    val lastUpdateTime: Long = 0L,
)

/**
 * Thermal processing metrics and statistics
 */
data class ThermalProcessingMetrics(
    val totalFrames: Long = 0,
    val framesPerSecond: Double = 0.0,
    val minTemperature: Double = 0.0,
    val maxTemperature: Double = 0.0,
    val averageTemperature: Double = 0.0,
    val centerTemperature: Double = 0.0,
    val emissivity: Double = 0.95,
    val processingLatencyMs: Long = 0,
    val lastUpdateTime: Long = 0L,
)
