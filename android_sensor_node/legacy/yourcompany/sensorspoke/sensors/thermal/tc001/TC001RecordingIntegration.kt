package com.yourcompany.sensorspoke.sensors.thermal.tc001

import android.content.Context
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.yourcompany.sensorspoke.controller.RecordingController
import com.yourcompany.sensorspoke.sensors.thermal.ThermalCameraRecorder
import kotlinx.coroutines.*

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
    suspend fun startRecording(sessionId: String): Boolean =
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

                // Start thermal recorder
                thermalRecorder?.startRecording()

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
                // Stop thermal recorder
                thermalRecorder?.stopRecording()

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
