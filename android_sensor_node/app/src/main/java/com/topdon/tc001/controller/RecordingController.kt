package com.topdon.tc001.controller

import android.content.Context
import android.util.Log
import com.topdon.tc001.sensors.SensorInfo
import com.topdon.tc001.sensors.SensorRecorder
import com.topdon.tc001.sensors.SensorType
import com.topdon.tc001.sensors.gsr.ShimmerRecorder
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * Central controller for managing multiple sensor recordings
 * Coordinates GSR, thermal camera, and RGB camera sensors
 * 
 * Follows MVVM architecture pattern and uses Kotlin Coroutines
 * for all asynchronous operations
 */
class RecordingController(
    private val context: Context,
    private val outputDirectory: File
) {
    
    companion object {
        private const val TAG = "RecordingController"
    }
    
    private val coroutineScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    
    // Active sensor recorders
    private val sensorRecorders = ConcurrentHashMap<SensorType, SensorRecorder>()
    
    // Recording session state
    private val _currentSessionId = MutableStateFlow<String?>(null)
    val currentSessionId: StateFlow<String?> = _currentSessionId.asStateFlow()
    
    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()
    
    private val _recordingStartTime = MutableStateFlow<Long?>(null)
    val recordingStartTime: StateFlow<Long?> = _recordingStartTime.asStateFlow()
    
    // Sensor connection states
    private val _connectedSensors = MutableStateFlow<Set<SensorType>>(emptySet())
    val connectedSensors: StateFlow<Set<SensorType>> = _connectedSensors.asStateFlow()
    
    // Sensor information
    private val _sensorInfoMap = MutableStateFlow<Map<SensorType, SensorInfo>>(emptyMap())
    val sensorInfoMap: StateFlow<Map<SensorType, SensorInfo>> = _sensorInfoMap.asStateFlow()
    
    init {
        // Initialize output directory
        if (!outputDirectory.exists()) {
            outputDirectory.mkdirs()
        }
        
        // Start sensor info monitoring
        startSensorInfoMonitoring()
    }
    
    /**
     * Add a GSR sensor to the recording setup
     * @param deviceAddress Bluetooth address of Shimmer3 GSR+ device
     * @return true if sensor was added successfully
     */
    suspend fun addGsrSensor(deviceAddress: String): Boolean = withContext(Dispatchers.IO) {
        try {
            if (sensorRecorders.containsKey(SensorType.GSR_SHIMMER)) {
                Log.w(TAG, "GSR sensor already added")
                return@withContext false
            }
            
            val gsrRecorder = ShimmerRecorder(context, deviceAddress, outputDirectory)
            val initialized = gsrRecorder.initialize()
            
            if (initialized) {
                sensorRecorders[SensorType.GSR_SHIMMER] = gsrRecorder
                Log.d(TAG, "GSR sensor added successfully: $deviceAddress")
                
                // Monitor connection state
                coroutineScope.launch {
                    gsrRecorder.isConnected.collect { connected ->
                        updateSensorConnection(SensorType.GSR_SHIMMER, connected)
                    }
                }
                
                true
            } else {
                Log.e(TAG, "Failed to initialize GSR sensor: $deviceAddress")
                false
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error adding GSR sensor", e)
            false
        }
    }
    
    /**
     * Remove a sensor from the recording setup
     * @param sensorType Type of sensor to remove
     */
    suspend fun removeSensor(sensorType: SensorType) {
        withContext(Dispatchers.IO) {
            try {
                val recorder = sensorRecorders.remove(sensorType)
                recorder?.let {
                    it.disconnect()
                    updateSensorConnection(sensorType, false)
                    Log.d(TAG, "Sensor removed: $sensorType")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error removing sensor: $sensorType", e)
            }
            Unit
        }
    }
    
    /**
     * Start recording on all connected sensors
     * @param sessionId Unique identifier for this recording session
     * @return true if recording started successfully on all sensors
     */
    suspend fun startRecording(sessionId: String): Boolean = withContext(Dispatchers.IO) {
        try {
            if (_isRecording.value) {
                Log.w(TAG, "Recording already in progress")
                return@withContext false
            }
            
            if (sensorRecorders.isEmpty()) {
                Log.w(TAG, "No sensors available for recording")
                return@withContext false
            }
            
            Log.d(TAG, "Starting recording session: $sessionId")
            _currentSessionId.value = sessionId
            _recordingStartTime.value = System.currentTimeMillis()
            
            // Start recording on all sensors
            val startResults = sensorRecorders.map { (sensorType, recorder) ->
                async {
                    try {
                        val result = recorder.startRecording(sessionId)
                        if (!result) {
                            Log.e(TAG, "Failed to start recording on sensor: $sensorType")
                        }
                        sensorType to result
                    } catch (e: Exception) {
                        Log.e(TAG, "Exception starting recording on sensor: $sensorType", e)
                        sensorType to false
                    }
                }
            }.awaitAll()
            
            val allStarted = startResults.all { it.second }
            
            if (allStarted) {
                _isRecording.value = true
                Log.d(TAG, "Recording started successfully on all sensors")
            } else {
                Log.e(TAG, "Failed to start recording on some sensors")
                // Stop recording on sensors that started successfully
                stopRecording()
            }
            
            allStarted
            
        } catch (e: Exception) {
            Log.e(TAG, "Error starting recording", e)
            false
        }
    }
    
    /**
     * Stop recording on all sensors
     * @return true if recording stopped successfully on all sensors
     */
    suspend fun stopRecording(): Boolean = withContext(Dispatchers.IO) {
        try {
            if (!_isRecording.value) {
                Log.w(TAG, "No recording in progress")
                return@withContext false
            }
            
            Log.d(TAG, "Stopping recording session: ${_currentSessionId.value}")
            
            // Stop recording on all sensors
            val stopResults = sensorRecorders.map { (sensorType, recorder) ->
                async {
                    try {
                        val result = recorder.stopRecording()
                        if (!result) {
                            Log.e(TAG, "Failed to stop recording on sensor: $sensorType")
                        }
                        sensorType to result
                    } catch (e: Exception) {
                        Log.e(TAG, "Exception stopping recording on sensor: $sensorType", e)
                        sensorType to false
                    }
                }
            }.awaitAll()
            
            val allStopped = stopResults.all { it.second }
            
            _isRecording.value = false
            _currentSessionId.value = null
            _recordingStartTime.value = null
            
            if (allStopped) {
                Log.d(TAG, "Recording stopped successfully on all sensors")
            } else {
                Log.w(TAG, "Some sensors failed to stop recording properly")
            }
            
            allStopped
            
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping recording", e)
            false
        }
    }
    
    /**
     * Get information about a specific sensor
     */
    suspend fun getSensorInfo(sensorType: SensorType): SensorInfo? {
        return sensorRecorders[sensorType]?.getSensorInfo()
    }
    
    /**
     * Get information about all sensors
     */
    suspend fun getAllSensorInfo(): Map<SensorType, SensorInfo> = withContext(Dispatchers.IO) {
        val infoMap = mutableMapOf<SensorType, SensorInfo>()
        
        sensorRecorders.map { (sensorType, recorder) ->
            async {
                try {
                    val info = recorder.getSensorInfo()
                    sensorType to info
                } catch (e: Exception) {
                    Log.e(TAG, "Error getting sensor info: $sensorType", e)
                    null
                }
            }
        }.awaitAll().forEach { result ->
            if (result != null) {
                infoMap[result.first] = result.second
            }
        }
        
        infoMap
    }
    
    /**
     * Disconnect all sensors and cleanup resources
     */
    suspend fun disconnectAll() = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Disconnecting all sensors")
            
            // Stop recording if in progress
            if (_isRecording.value) {
                stopRecording()
            }
            
            // Disconnect all sensors
            sensorRecorders.map { (sensorType, recorder) ->
                async {
                    try {
                        recorder.disconnect()
                        Log.d(TAG, "Disconnected sensor: $sensorType")
                    } catch (e: Exception) {
                        Log.e(TAG, "Error disconnecting sensor: $sensorType", e)
                    }
                }
            }.awaitAll()
            
            sensorRecorders.clear()
            _connectedSensors.value = emptySet()
            _sensorInfoMap.value = emptyMap()
            
            coroutineScope.cancel()
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during disconnect all", e)
        }
    }
    
    private fun updateSensorConnection(sensorType: SensorType, connected: Boolean) {
        val currentConnected = _connectedSensors.value.toMutableSet()
        if (connected) {
            currentConnected.add(sensorType)
        } else {
            currentConnected.remove(sensorType)
        }
        _connectedSensors.value = currentConnected
    }
    
    private fun startSensorInfoMonitoring() {
        coroutineScope.launch {
            while (isActive) {
                try {
                    val infoMap = getAllSensorInfo()
                    _sensorInfoMap.value = infoMap
                    
                    // Update every 5 seconds
                    delay(5000)
                    
                } catch (e: Exception) {
                    Log.e(TAG, "Error in sensor info monitoring", e)
                    delay(10000) // Longer delay on error
                }
            }
        }
    }
}