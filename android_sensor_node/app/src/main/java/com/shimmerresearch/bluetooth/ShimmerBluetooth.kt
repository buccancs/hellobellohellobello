package com.shimmerresearch.bluetooth

/**
 * Shimmer Bluetooth communication interface and state definitions.
 * Based on official ShimmerAndroidAPI.
 */
interface ShimmerBluetooth {
    companion object {
        const val MSG_IDENTIFIER_DATA_PACKET = 0x10
        const val MSG_IDENTIFIER_STATE_CHANGE = 0x11
        const val MSG_IDENTIFIER_NOTIFICATION_MESSAGE = 0x12

        const val START_STREAMING_COMMAND = 0x07.toByte()
        const val STOP_STREAMING_COMMAND = 0x20.toByte()
        const val GET_SAMPLING_RATE_COMMAND = 0x03.toByte()
        const val SET_SAMPLING_RATE_COMMAND = 0x05.toByte()
    }

    /**
     * Bluetooth connection states matching official ShimmerAndroidAPI
     */
    enum class BtState {
        DISCONNECTED,
        CONNECTING,
        CONNECTED,
        STREAMING,
        STREAMING_AND_SDLOGGING,
        SDLOGGING,
        CONNECTION_LOST,
        NONE,
    }
}
