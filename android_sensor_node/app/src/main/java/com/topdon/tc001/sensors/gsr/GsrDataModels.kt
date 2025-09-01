package com.topdon.tc001.sensors.gsr

/**
 * Data model for a single GSR measurement from Shimmer3 GSR+ sensor
 * 
 * According to specifications:
 * - GSR value must be calculated using 12-bit ADC resolution (0-4095 range), NOT 16-bit
 * - Values are converted to microsiemens (μS)
 * - All timestamps use nanosecond precision for synchronization
 */
data class GsrReading(
    /** High-precision monotonic timestamp in nanoseconds */
    val timestampNanos: Long,
    
    /** Raw GSR sensor value from ADC (12-bit: 0-4095) */
    val rawGsrValue: Int,
    
    /** Converted GSR value in microsiemens (μS) */
    val gsrMicrosiemens: Double,
    
    /** Raw PPG (photoplethysmography) sensor value */
    val rawPpgValue: Int,
    
    /** Session ID for this recording session */
    val sessionId: String
) {
    companion object {
        /**
         * Convert raw 12-bit GSR value to microsiemens
         * @param rawValue Raw ADC value (0-4095)
         * @return GSR value in microsiemens
         */
        fun convertToMicrosiemens(rawValue: Int): Double {
            // Shimmer3 GSR+ conversion formula for 12-bit ADC
            // Reference voltage: 3.0V, Gain: 49, ADC resolution: 12-bit (4096 levels)
            val voltage = (rawValue.toDouble() / 4095.0) * 3.0
            val resistance = (3.0 * 1000000.0) / voltage - 1000000.0 // in ohms
            return if (resistance > 0) 1000000.0 / resistance else 0.0 // convert to microsiemens
        }
        
        /**
         * Validate that raw GSR value is within 12-bit range
         */
        fun isValidGsrValue(rawValue: Int): Boolean = rawValue in 0..4095
    }
    
    init {
        require(isValidGsrValue(rawGsrValue)) { 
            "Raw GSR value $rawGsrValue is out of 12-bit range (0-4095)" 
        }
    }
}

/**
 * Shimmer device information and status
 */
data class ShimmerDeviceInfo(
    val deviceName: String,
    val deviceAddress: String,
    val batteryLevel: Int? = null,
    val connectionState: ShimmerConnectionState,
    val streamingState: ShimmerStreamingState,
    val sampleRate: Double? = null,
    val enabledSensors: Set<ShimmerSensorType> = emptySet()
)

/**
 * Shimmer connection states
 */
enum class ShimmerConnectionState {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    LOST_CONNECTION,
    CONNECTION_FAILED
}

/**
 * Shimmer streaming/recording states
 */
enum class ShimmerStreamingState {
    STOPPED,
    STARTING,
    STREAMING,
    STOPPING,
    FAILED
}

/**
 * Shimmer sensor types that can be enabled
 */
enum class ShimmerSensorType {
    GSR,
    PPG,
    ACCELEROMETER,
    GYROSCOPE,
    MAGNETOMETER
}

/**
 * Configuration for GSR recording session
 */
data class GsrRecordingConfig(
    val sessionId: String,
    val sampleRate: Double = 51.2, // Hz - Default Shimmer GSR sampling rate
    val enabledSensors: Set<ShimmerSensorType> = setOf(ShimmerSensorType.GSR, ShimmerSensorType.PPG),
    val outputFileName: String? = null,
    val enableRealTimeStreaming: Boolean = false
)