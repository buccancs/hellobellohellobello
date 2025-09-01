package com.topdon.tc001.ui.viewmodel

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.topdon.tc001.controller.RecordingController
import com.topdon.tc001.sensors.SensorInfo
import com.topdon.tc001.sensors.SensorType
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.io.File
import java.util.*

/**
 * Main ViewModel for GSR recording interface
 * 
 * Follows MVVM architecture pattern with proper lifecycle management
 * All UI state is exposed through StateFlow for lifecycle-aware observation
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {
    
    companion object {
        private const val TAG = "MainViewModel"
    }
    
    private val recordingController: RecordingController
    
    init {
        // Initialize output directory
        val outputDir = File(application.getExternalFilesDir(null), "sensor_data")
        recordingController = RecordingController(application, outputDir)
        
        // Observe recording controller state
        viewModelScope.launch {
            recordingController.isRecording.collect { recording ->
                _isRecording.value = recording
            }
        }
        
        viewModelScope.launch {
            recordingController.connectedSensors.collect { sensors ->
                _connectedSensors.value = sensors
            }
        }
        
        viewModelScope.launch {
            recordingController.sensorInfoMap.collect { infoMap ->
                _sensorInfoMap.value = infoMap
            }
        }
    }
    
    // UI State
    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()
    
    private val _connectedSensors = MutableStateFlow<Set<SensorType>>(emptySet())
    val connectedSensors: StateFlow<Set<SensorType>> = _connectedSensors.asStateFlow()
    
    private val _sensorInfoMap = MutableStateFlow<Map<SensorType, SensorInfo>>(emptyMap())
    val sensorInfoMap: StateFlow<Map<SensorType, SensorInfo>> = _sensorInfoMap.asStateFlow()
    
    private val _isConnectingGsr = MutableStateFlow(false)
    val isConnectingGsr: StateFlow<Boolean> = _isConnectingGsr.asStateFlow()
    
    private val _errorMessage = MutableSharedFlow<String>()
    val errorMessage: SharedFlow<String> = _errorMessage.asSharedFlow()
    
    private val _statusMessage = MutableSharedFlow<String>()
    val statusMessage: SharedFlow<String> = _statusMessage.asSharedFlow()
    
    // Current session info
    private val _currentSession = MutableStateFlow<SessionInfo?>(null)
    val currentSession: StateFlow<SessionInfo?> = _currentSession.asStateFlow()
    
    /**
     * Connect to a Shimmer GSR sensor
     * @param deviceAddress Bluetooth MAC address of the device
     */
    fun connectGsrSensor(deviceAddress: String) {
        if (_isConnectingGsr.value) {
            Log.w(TAG, "GSR connection already in progress")
            return
        }
        
        viewModelScope.launch {
            try {
                _isConnectingGsr.value = true
                _statusMessage.emit("Connecting to GSR sensor...")
                
                val success = recordingController.addGsrSensor(deviceAddress)
                
                if (success) {
                    _statusMessage.emit("GSR sensor connected successfully")
                    Log.d(TAG, "GSR sensor connected: $deviceAddress")
                } else {
                    _errorMessage.emit("Failed to connect GSR sensor")
                    Log.e(TAG, "Failed to connect GSR sensor: $deviceAddress")
                }
                
            } catch (e: Exception) {
                _errorMessage.emit("Error connecting GSR sensor: ${e.message}")
                Log.e(TAG, "Exception connecting GSR sensor", e)
            } finally {
                _isConnectingGsr.value = false
            }
        }
    }
    
    /**
     * Disconnect a specific sensor
     */
    fun disconnectSensor(sensorType: SensorType) {
        viewModelScope.launch {
            try {
                _statusMessage.emit("Disconnecting ${sensorType.name} sensor...")
                recordingController.removeSensor(sensorType)
                _statusMessage.emit("${sensorType.name} sensor disconnected")
                Log.d(TAG, "Sensor disconnected: $sensorType")
                
            } catch (e: Exception) {
                _errorMessage.emit("Error disconnecting sensor: ${e.message}")
                Log.e(TAG, "Exception disconnecting sensor: $sensorType", e)
            }
        }
    }
    
    /**
     * Start recording session on all connected sensors
     */
    fun startRecording() {
        if (_isRecording.value) {
            Log.w(TAG, "Recording already in progress")
            return
        }
        
        if (_connectedSensors.value.isEmpty()) {
            viewModelScope.launch {
                _errorMessage.emit("No sensors connected for recording")
            }
            return
        }
        
        viewModelScope.launch {
            try {
                val sessionId = generateSessionId()
                _statusMessage.emit("Starting recording session: $sessionId")
                
                val success = recordingController.startRecording(sessionId)
                
                if (success) {
                    _currentSession.value = SessionInfo(
                        sessionId = sessionId,
                        startTime = System.currentTimeMillis(),
                        sensorTypes = _connectedSensors.value
                    )
                    _statusMessage.emit("Recording started successfully")
                    Log.d(TAG, "Recording started: $sessionId")
                } else {
                    _errorMessage.emit("Failed to start recording")
                    Log.e(TAG, "Failed to start recording")
                }
                
            } catch (e: Exception) {
                _errorMessage.emit("Error starting recording: ${e.message}")
                Log.e(TAG, "Exception starting recording", e)
            }
        }
    }
    
    /**
     * Stop the current recording session
     */
    fun stopRecording() {
        if (!_isRecording.value) {
            Log.w(TAG, "No recording in progress")
            return
        }
        
        viewModelScope.launch {
            try {
                _statusMessage.emit("Stopping recording...")
                
                val success = recordingController.stopRecording()
                
                if (success) {
                    _currentSession.value = null
                    _statusMessage.emit("Recording stopped successfully")
                    Log.d(TAG, "Recording stopped")
                } else {
                    _errorMessage.emit("Some sensors failed to stop recording")
                    Log.w(TAG, "Some sensors failed to stop recording")
                }
                
            } catch (e: Exception) {
                _errorMessage.emit("Error stopping recording: ${e.message}")
                Log.e(TAG, "Exception stopping recording", e)
            }
        }
    }
    
    /**
     * Get GSR sensor information
     */
    fun getGsrSensorInfo(): SensorInfo? {
        return _sensorInfoMap.value[SensorType.GSR_SHIMMER]
    }
    
    /**
     * Check if GSR sensor is connected
     */
    fun isGsrConnected(): Boolean {
        return _connectedSensors.value.contains(SensorType.GSR_SHIMMER)
    }
    
    /**
     * Get current session duration in milliseconds
     */
    fun getSessionDuration(): Long {
        val session = _currentSession.value
        return if (session != null && _isRecording.value) {
            System.currentTimeMillis() - session.startTime
        } else {
            0L
        }
    }
    
    /**
     * Generate a unique session ID
     */
    private fun generateSessionId(): String {
        val timestamp = System.currentTimeMillis()
        val uuid = UUID.randomUUID().toString().take(8)
        return "session_${timestamp}_${uuid}"
    }
    
    override fun onCleared() {
        super.onCleared()
        viewModelScope.launch {
            try {
                recordingController.disconnectAll()
                Log.d(TAG, "ViewModel cleared - all sensors disconnected")
            } catch (e: Exception) {
                Log.e(TAG, "Error during ViewModel cleanup", e)
            }
        }
    }
}

/**
 * Information about the current recording session
 */
data class SessionInfo(
    val sessionId: String,
    val startTime: Long,
    val sensorTypes: Set<SensorType>
)