package com.topdon.tc001.sensors

import kotlinx.coroutines.flow.Flow

/**
 * Common interface for all sensor recording implementations.
 * Ensures consistent API across different sensor types (GSR, Thermal, RGB, etc.)
 * 
 * All implementations must be lifecycle-aware and handle cleanup properly.
 */
interface SensorRecorder {
    
    /**
     * Unique identifier for this sensor type
     */
    val sensorType: SensorType
    
    /**
     * Current connection state of the sensor
     */
    val isConnected: Flow<Boolean>
    
    /**
     * Current recording state of the sensor
     */
    val isRecording: Flow<Boolean>
    
    /**
     * Initialize the sensor connection
     * @return true if initialization successful, false otherwise
     */
    suspend fun initialize(): Boolean
    
    /**
     * Start recording sensor data
     * @param sessionId Unique identifier for this recording session
     * @return true if recording started successfully, false otherwise
     */
    suspend fun startRecording(sessionId: String): Boolean
    
    /**
     * Stop recording sensor data
     * @return true if recording stopped successfully, false otherwise
     */
    suspend fun stopRecording(): Boolean
    
    /**
     * Disconnect from the sensor and cleanup resources
     */
    suspend fun disconnect()
    
    /**
     * Get the current sensor status/info
     */
    suspend fun getSensorInfo(): SensorInfo
}

/**
 * Supported sensor types
 */
enum class SensorType {
    GSR_SHIMMER,
    THERMAL_TOPDON,
    RGB_CAMERA
}

/**
 * General sensor information
 */
data class SensorInfo(
    val type: SensorType,
    val deviceName: String,
    val deviceAddress: String? = null,
    val batteryLevel: Int? = null,
    val signalStrength: Int? = null,
    val lastDataTimestamp: Long? = null
)