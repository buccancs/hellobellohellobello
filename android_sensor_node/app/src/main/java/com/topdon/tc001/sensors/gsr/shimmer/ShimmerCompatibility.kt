package com.topdon.tc001.sensors.gsr.shimmer

/**
 * Shimmer BLE state enumeration
 * Compatible with ShimmerAndroidAPI BT_STATE enum
 */
enum class ShimmerBTState {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    STREAMING,
    STREAMING_AND_SDLOGGING,
    CONNECTION_LOST,
    SDLOGGING
}

/**
 * Shimmer device hardware types
 * Compatible with ShimmerAndroidAPI HW_ID enum
 */
enum class ShimmerHardwareType {
    SHIMMER_3,
    SHIMMER_3R
}

/**
 * Shimmer notification message types
 * Compatible with ShimmerAndroidAPI notification constants
 */
object ShimmerNotifications {
    const val SHIMMER_FULLY_INITIALIZED = 101
    const val SHIMMER_START_STREAMING = 102
    const val SHIMMER_STOP_STREAMING = 103
}

/**
 * Shimmer message identifiers
 * Compatible with ShimmerAndroidAPI message constants
 */
object ShimmerMessages {
    const val MSG_IDENTIFIER_STATE_CHANGE = 201
    const val MSG_IDENTIFIER_DATA_PACKET = 202
    const val MSG_IDENTIFIER_NOTIFICATION_MESSAGE = 203
    const val MSG_IDENTIFIER_PACKET_RECEPTION_RATE_OVERALL = 204
    const val MESSAGE_TOAST = 205
}

/**
 * Shimmer configuration constants
 */
object ShimmerConfiguration {
    // Shimmer commands
    const val COMMAND_START_STREAMING: Byte = 0x07
    const val COMMAND_STOP_STREAMING: Byte = 0x20
    const val COMMAND_SET_SAMPLE_RATE: Byte = 0x05
    const val COMMAND_SET_SENSORS: Byte = 0x08
    
    // Default settings
    const val DEFAULT_SAMPLE_RATE = 51.2 // Hz
    const val GSR_SENSOR_MASK = 0x04
    const val PPG_SENSOR_MASK = 0x01
    
    // GSR calibration constants
    const val GSR_UNCAL_TO_CAL_CONSTANT = 1.0 / (4096 * 40.5e-9)
    const val GSR_REF_VOLTAGE = 3.0
}

/**
 * Object cluster for Shimmer sensor data
 * Simplified version compatible with ShimmerAndroidAPI ObjectCluster
 */
data class ShimmerObjectCluster(
    val timestamp: Double = 0.0,
    val gsrRaw: Int = 0,
    val gsrCalibrated: Double = 0.0,
    val ppgRaw: Int = 0,
    val deviceAddress: String = "",
    val state: ShimmerBTState = ShimmerBTState.DISCONNECTED
)

/**
 * Callback object for Shimmer events
 * Compatible with ShimmerAndroidAPI CallbackObject
 */
data class ShimmerCallbackObject(
    val state: ShimmerBTState = ShimmerBTState.DISCONNECTED,
    val bluetoothAddress: String = "",
    val indicator: Int = 0
)

/**
 * Format cluster for sensor data
 * Simplified version compatible with ShimmerAndroidAPI FormatCluster
 */
data class ShimmerFormatCluster(
    val data: Double,
    val format: String,
    val unit: String
)

/**
 * Shimmer message wrapper
 * Compatible with ShimmerAndroidAPI ShimmerMsg
 */
data class ShimmerMsg(
    val identifier: Int,
    val objectData: Any?
)