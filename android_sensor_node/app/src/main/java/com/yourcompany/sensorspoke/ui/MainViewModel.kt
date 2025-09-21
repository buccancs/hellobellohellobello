package com.yourcompany.sensorspoke.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.yourcompany.sensorspoke.controller.SessionOrchestrator
import com.yourcompany.sensorspoke.network.ConnectionManager
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.concurrent.TimeUnit

/**
 * Enhanced MainViewModel with comprehensive UI feedback and sensor status tracking.
 * Coordinates with RecordingController and provides real-time status updates for all sensors.
 *
 * This ViewModel implements the UI feedback requirements including:
 * - Real-time sensor status indicators
 * - Recording timer and status updates
 * - Error handling and notifications
 * - Thermal camera simulation feedback
 */
class MainViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(MainUiState())
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    private val _currentSessionId = MutableStateFlow<String?>(null)
    val currentSessionId: StateFlow<String?> = _currentSessionId.asStateFlow()

    private val _recordingState = MutableStateFlow(RecordingState.IDLE)
    val recordingState: StateFlow<RecordingState> = _recordingState.asStateFlow()

    private val _recordingStartTime = MutableStateFlow(0L)
    val recordingStartTime: StateFlow<Long> = _recordingStartTime.asStateFlow()

    private val _recordingElapsedTime = MutableStateFlow(0L)
    val recordingElapsedTime: StateFlow<Long> = _recordingElapsedTime.asStateFlow()

    private val _statusMessage = MutableStateFlow("Ready")
    val statusMessage: StateFlow<String> = _statusMessage.asStateFlow()

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    private val _showErrorDialog = MutableStateFlow(false)
    val showErrorDialog: StateFlow<Boolean> = _showErrorDialog.asStateFlow()

    private val _sensorStatus = MutableStateFlow<Map<String, SensorStatus>>(emptyMap())
    val sensorStatus: StateFlow<Map<String, SensorStatus>> = _sensorStatus.asStateFlow()

    private var connectionManager: ConnectionManager? = null

    private var sessionOrchestrator: SessionOrchestrator? = null

    enum class RecordingState {
        IDLE,
        PREPARING,
        RECORDING,
        STOPPING,
        ERROR,
    }

    /**
     * Individual sensor status for UI display
     */
    data class SensorStatus(
        val name: String,
        val isActive: Boolean,
        val isHealthy: Boolean,
        val isSimulated: Boolean = false,
        val lastUpdate: Long = System.currentTimeMillis(),
        val statusMessage: String = "",
        val errorMessage: String? = null,
    )

    /**
     * Comprehensive UI state data class
     */
    data class MainUiState(
        val isCameraConnected: Boolean = false,
        val isThermalConnected: Boolean = false,
        val isShimmerConnected: Boolean = false,
        val isPcConnected: Boolean = false,

        val isRecording: Boolean = false,
        val isInitialized: Boolean = false,
        val recordingElapsedTime: String = "00:00",

        val statusText: String = "Initializing...",
        val errorMessage: String? = null,
        val showErrorDialog: Boolean = false,

        val thermalStatus: ThermalStatus = ThermalStatus(),
        val sensorStatus: Map<String, SensorStatus> = emptyMap(),

        val startButtonEnabled: Boolean = false,
        val stopButtonEnabled: Boolean = false,

        val rgbPreviewAvailable: Boolean = false,
        val thermalPreviewAvailable: Boolean = false,
    )

    /**
     * Thermal camera status details
     */
    data class ThermalStatus(
        val isAvailable: Boolean = false,
        val isSimulated: Boolean = false,
        val deviceName: String? = null,
        val statusMessage: String = "Not connected",
    )

    /**
     * Initialize the ViewModel with a SessionOrchestrator instance (typically RecordingController)
     */
    fun initialize(orchestrator: SessionOrchestrator, connManager: ConnectionManager? = null) {
        sessionOrchestrator = orchestrator
        connectionManager = connManager

        updateUiState { copy(isInitialized = true, statusText = "Ready to connect") }

        viewModelScope.launch {
            orchestrator.state.collect { orchestratorState ->
                val newRecordingState = when (orchestratorState) {
                    SessionOrchestrator.State.IDLE -> RecordingState.IDLE
                    SessionOrchestrator.State.PREPARING -> RecordingState.PREPARING
                    SessionOrchestrator.State.RECORDING -> RecordingState.RECORDING
                    SessionOrchestrator.State.STOPPING -> RecordingState.STOPPING
                }

                _recordingState.value = newRecordingState
                _isRecording.value = orchestratorState == SessionOrchestrator.State.RECORDING

                updateUiState {
                    copy(
                        isRecording = orchestratorState == SessionOrchestrator.State.RECORDING,
                        startButtonEnabled = orchestratorState == SessionOrchestrator.State.IDLE && isInitialized,
                        stopButtonEnabled = orchestratorState == SessionOrchestrator.State.RECORDING,
                        statusText = when (orchestratorState) {
                            SessionOrchestrator.State.IDLE -> "Ready to record"
                            SessionOrchestrator.State.PREPARING -> "Preparing to record..."
                            SessionOrchestrator.State.RECORDING -> "Recording in progress..."
                            SessionOrchestrator.State.STOPPING -> "Stopping recording..."
                        },
                    )
                }

                if (orchestratorState == SessionOrchestrator.State.RECORDING) {
                    startRecordingTimer()
                } else {
                    stopRecordingTimer()
                }
            }
        }

        viewModelScope.launch {
            orchestrator.currentSessionId.collect { sessionId ->
                updateUiState { copy(statusText = if (sessionId != null) "Session: $sessionId" else "Ready") }
            }
        }

        // If orchestrator is RecordingController, observe additional features
        if (orchestrator is com.yourcompany.sensorspoke.controller.RecordingController) {
            viewModelScope.launch {
                orchestrator.sensorStates.collect { sensorStates ->
                    val statusMap = sensorStates.mapValues { (sensorName, state) ->
                        val isActive = state == com.yourcompany.sensorspoke.controller.RecordingController.RecorderState.RECORDING
                        val isHealthy = state != com.yourcompany.sensorspoke.controller.RecordingController.RecorderState.ERROR
                        val statusMessage = when (state) {
                            com.yourcompany.sensorspoke.controller.RecordingController.RecorderState.IDLE -> "Ready"
                            com.yourcompany.sensorspoke.controller.RecordingController.RecorderState.STARTING -> "Starting..."
                            com.yourcompany.sensorspoke.controller.RecordingController.RecorderState.RECORDING -> "Recording"
                            com.yourcompany.sensorspoke.controller.RecordingController.RecorderState.STOPPING -> "Stopping..."
                            com.yourcompany.sensorspoke.controller.RecordingController.RecorderState.STOPPED -> "Stopped"
                            com.yourcompany.sensorspoke.controller.RecordingController.RecorderState.ERROR -> "Error"
                            com.yourcompany.sensorspoke.controller.RecordingController.RecorderState.RECOVERING -> "Recovering..."
                        }
                        // Determine if sensor is simulated based on recorder type and state
                        val isSimulated = when (sensorName.lowercase()) {
                            "thermal" -> {
                                // Check if thermal is simulated from thermal status
                                _uiState.value.thermalStatus.isSimulated
                            }
                            else -> false // Other sensors are not typically simulated in this system
                        }
                        
                        SensorStatus(sensorName, isActive, isHealthy, isSimulated, System.currentTimeMillis(), statusMessage)
                    }
                    _sensorStatus.value = statusMap
                    
                    // Update UI state with sensor status
                    updateUiState { copy(sensorStatus = statusMap) }
                }
            }

            viewModelScope.launch {
                orchestrator.lastSessionResult.collect { result ->
                    result?.let {
                        if (it.isPartialSuccess) {
                            val failedSensors = it.failedSensors.keys.joinToString(", ")
                            _statusMessage.value = "Recording started ($failedSensors failed)"
                        }
                    }
                }
            }
        }

        connManager?.let { setupConnectionMonitoring(it) }

        initializeSensorStatus()
    }

    /**
     * Setup connection monitoring for PC Hub
     */
    private fun setupConnectionMonitoring(connManager: ConnectionManager) {
        viewModelScope.launch {
            while (true) {
                val status = connManager.getConnectionStatus()
                updateUiState {
                    copy(
                        isPcConnected = status.isConnected,
                        statusText = if (status.isConnected) {
                            "Connected to PC Hub"
                        } else if (status.isReconnecting) {
                            "Reconnecting to PC Hub..."
                        } else {
                            "Not connected to PC Hub"
                        },
                    )
                }
                kotlinx.coroutines.delay(2000)
            }
        }
    }

    /**
     * Initialize sensor status tracking
     */
    private fun initializeSensorStatus() {
        val initialSensors = mapOf(
            "rgb" to SensorStatus("RGB Camera", false, false),
            "thermal" to SensorStatus("Thermal Camera", false, false),
            "gsr" to SensorStatus("GSR Sensor", false, false),
            "audio" to SensorStatus("Audio", false, false),
        )
        _sensorStatus.value = initialSensors

        updateUiState {
            copy(
                isCameraConnected = false,
                isThermalConnected = false,
                isShimmerConnected = false,
                isInitialized = false,
            )
        }
    }

    /**
     * Start recording timer for elapsed time tracking
     */
    private fun startRecordingTimer() {
        _recordingStartTime.value = System.currentTimeMillis()

        viewModelScope.launch {
            while (_isRecording.value) {
                val elapsed = System.currentTimeMillis() - _recordingStartTime.value
                _recordingElapsedTime.value = elapsed

                val minutes = TimeUnit.MILLISECONDS.toMinutes(elapsed)
                val seconds = TimeUnit.MILLISECONDS.toSeconds(elapsed) % 60
                val timeString = "%02d:%02d".format(minutes, seconds)

                updateUiState {
                    copy(recordingElapsedTime = timeString)
                }

                kotlinx.coroutines.delay(1000)
            }
        }
    }

    /**
     * Stop recording timer
     */
    private fun stopRecordingTimer() {
        _recordingElapsedTime.value = 0L
        updateUiState {
            copy(recordingElapsedTime = "00:00")
        }
    }

    /**
     * Update UI state helper function
     */
    fun updateUiState(update: MainUiState.() -> MainUiState) {
        _uiState.value = _uiState.value.update()
    }

    /**
     * Start a new recording session
     */
    fun startRecording(sessionId: String? = null) {
        viewModelScope.launch {
            try {
                clearError()
                updateUiState {
                    copy(statusText = "Starting recording...")
                }
                sessionOrchestrator?.startSession(sessionId)
            } catch (e: Exception) {
                _recordingState.value = RecordingState.ERROR
                showError("Failed to start recording: ${e.message}")
            }
        }
    }

    /**
     * Stop the current recording session
     */
    fun stopRecording() {
        viewModelScope.launch {
            try {
                clearError()
                updateUiState {
                    copy(statusText = "Stopping recording...")
                }
                sessionOrchestrator?.stopSession()
            } catch (e: Exception) {
                _recordingState.value = RecordingState.ERROR
                showError("Failed to stop recording: ${e.message}")
            }
        }
    }

    /**
     * Update sensor status for UI display
     */
    fun updateSensorStatus(sensorName: String, status: SensorStatus) {
        val currentStatus = _sensorStatus.value.toMutableMap()
        currentStatus[sensorName] = status
        _sensorStatus.value = currentStatus

        updateUiState {
            copy(
                isCameraConnected = currentStatus["rgb"]?.isActive == true,
                isThermalConnected = currentStatus["thermal"]?.isActive == true,
                isShimmerConnected = currentStatus["gsr"]?.isActive == true,
                thermalStatus = ThermalStatus(
                    isAvailable = currentStatus["thermal"]?.isActive == true,
                    isSimulated = currentStatus["thermal"]?.isSimulated == true,
                    statusMessage = currentStatus["thermal"]?.statusMessage ?: "Not connected",
                ),
                isInitialized = currentStatus.values.any { it.isActive },
                startButtonEnabled = !isRecording && currentStatus.values.any { it.isActive },
            )
        }
    }

    /**
     * Show error message with dialog
     */
    fun showError(message: String) {
        _errorMessage.value = message
        _showErrorDialog.value = true
        updateUiState {
            copy(
                errorMessage = message,
                showErrorDialog = true,
            )
        }
    }

    /**
     * Show toast message (handled by Activity)
     */
    fun showToast(message: String) {
        updateUiState {
            copy(statusText = message)
        }
    }

    /**
     * Clear error message and dialog
     */
    fun clearError() {
        _errorMessage.value = null
        _showErrorDialog.value = false
        updateUiState {
            copy(
                errorMessage = null,
                showErrorDialog = false,
            )
        }
    }

    fun updateThermalStatus(isAvailable: Boolean, isSimulated: Boolean, deviceName: String? = null) {
        updateUiState {
            copy(
                isThermalConnected = isAvailable,
                thermalStatus = ThermalStatus(
                    isAvailable = isAvailable,
                    isSimulated = isSimulated,
                    deviceName = deviceName,
                    statusMessage = when {
                        !isAvailable -> "Not connected"
                        isSimulated -> "Simulation mode (hardware not found)"
                        else -> "Connected to ${deviceName ?: "thermal camera"}"
                    },
                ),
                thermalPreviewAvailable = isAvailable,
            )
        }

        updateSensorStatus(
            "thermal",
            SensorStatus(
                name = "Thermal Camera",
                isActive = isAvailable,
                isHealthy = isAvailable,
                isSimulated = isSimulated,
                statusMessage = if (isSimulated) "Simulation mode" else "Connected",
            ),
        )
    }

    /**
     * Update RGB camera status
     */
    fun updateRgbCameraStatus(isConnected: Boolean, previewAvailable: Boolean = false) {
        updateUiState {
            copy(
                isCameraConnected = isConnected,
                rgbPreviewAvailable = previewAvailable,
            )
        }

        updateSensorStatus(
            "rgb",
            SensorStatus(
                name = "RGB Camera",
                isActive = isConnected,
                isHealthy = isConnected,
                statusMessage = if (isConnected) "Connected" else "Not connected",
            ),
        )
    }

    /**
     * Update GSR sensor status
     */
    fun updateGsrStatus(isConnected: Boolean, deviceName: String? = null) {
        updateUiState {
            copy(isShimmerConnected = isConnected)
        }

        updateSensorStatus(
            "gsr",
            SensorStatus(
                name = "GSR Sensor",
                isActive = isConnected,
                isHealthy = isConnected,
                statusMessage = if (isConnected) "Connected to ${deviceName ?: "Shimmer"}" else "Not connected",
            ),
        )
    }

    /**
     * Update PC connection status
     */
    fun updatePcConnectionStatus(isConnected: Boolean, serverInfo: String? = null) {
        updateUiState {
            copy(
                isPcConnected = isConnected,
                statusText = if (isConnected) {
                    "Connected to PC Hub${serverInfo?.let { " ($it)" } ?: ""}"
                } else {
                    "Not connected to PC Hub"
                },
            )
        }
    }

    /**
     * Handle recording completion
     */
    fun onRecordingCompleted(sessionInfo: String) {
        updateUiState {
            copy(
                statusText = "Recording completed. $sessionInfo",
                isRecording = false,
                startButtonEnabled = isInitialized,
                stopButtonEnabled = false,
                recordingElapsedTime = "00:00",
            )
        }
    }

    override fun onCleared() {
        super.onCleared()
    }
}
