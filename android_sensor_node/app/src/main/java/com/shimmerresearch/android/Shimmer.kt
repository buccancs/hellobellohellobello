package com.shimmerresearch.android

import android.content.Context
import android.os.Handler
import android.os.Message
import android.util.Log
import com.shimmerresearch.bluetooth.ShimmerBluetooth
import com.shimmerresearch.driver.ObjectCluster
import com.shimmerresearch.exceptions.ShimmerException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import java.util.concurrent.ConcurrentHashMap

/**
 * Core Shimmer sensor management class integrating ShimmerAndroidAPI.
 * Handles BLE connection, configuration, and real-time data streaming
 * for Shimmer3 GSR+ and other Shimmer sensors.
 */
abstract class Shimmer(
    protected val context: Context,
    private val bluetoothAddressParam: String,
    protected val handler: Handler,
) {
    companion object {
        const val MSG_IDENTIFIER_DATA_PACKET = 0x10
        const val MSG_IDENTIFIER_STATE_CHANGE = 0x11
        const val MSG_IDENTIFIER_NOTIFICATION_MESSAGE = 0x12
        const val MSG_IDENTIFIER_PACKET_RECEPTION_RATE_OVERALL = 0x13
        const val MESSAGE_TOAST = 0x20
        const val TOAST = "toast"

        const val NOTIFICATION_SHIMMER_FULLY_INITIALIZED = 0x01
        const val NOTIFICATION_SHIMMER_START_STREAMING = 0x02
        const val NOTIFICATION_SHIMMER_STOP_STREAMING = 0x03

        // Sensor bit flags (matching official API)
        const val SENSOR_GSR = 0x04
        const val SENSOR_INT_A13 = 0x08
        const val SENSOR_TIMESTAMP = 0x01
        const val SENSOR_ACCEL = 0x80

        // GSR range settings
        const val GSR_RANGE_AUTO = 0
        const val GSR_RANGE_4_7M = 1
        const val GSR_RANGE_2_3M = 2
        const val GSR_RANGE_1_2M = 3
        const val GSR_RANGE_560K = 4

        private const val TAG = "Shimmer"
    }

    protected var mIsConnected = false
    protected var mIsStreaming = false
    protected var mIsInitialized = false
    protected val mBluetoothAddress = bluetoothAddressParam

    // Configuration
    protected var mSamplingRate = 128.0
    protected var mEnabledSensors = 0L
    protected var mGSRRange = GSR_RANGE_4_7M

    protected val mCallbackObjects = ConcurrentHashMap<String, Any>()
    protected var mDataProcessingScope: CoroutineScope? = null

    /**
     * Connect to Shimmer device via Bluetooth
     */
    abstract fun connect(
        bluetoothAddress: String,
        deviceName: String,
    )

    /**
     * Disconnect from Shimmer device
     */
    @Throws(ShimmerException::class)
    abstract fun disconnect()

    /**
     * Start data streaming
     */
    @Throws(ShimmerException::class)
    abstract fun startStreaming()

    /**
     * Stop data streaming
     */
    abstract fun stopStreaming()

    /**
     * Configure enabled sensors
     */
    open fun setEnabledSensors(enabledSensors: Long) {
        mEnabledSensors = enabledSensors
        Log.d(TAG, "Enabled sensors: 0x${enabledSensors.toString(16)}")
    }

    /**
     * Set sampling rate in Hz
     */
    open fun setSamplingRateShimmer(rate: Double) {
        mSamplingRate = rate
        Log.d(TAG, "Set sampling rate: ${rate}Hz")
    }

    /**
     * Set GSR range
     */
    open fun setGSRRange(range: Int) {
        mGSRRange = range
        Log.d(TAG, "Set GSR range: $range")
    }

    /**
     * Get current connection state
     */
    fun isConnected(): Boolean = mIsConnected

    /**
     * Get current streaming state
     */
    fun isStreaming(): Boolean = mIsStreaming

    /**
     * Get Bluetooth address
     */
    fun getBluetoothAddress(): String = mBluetoothAddress

    /**
     * Send message to handler
     */
    protected fun sendMessageToHandler(
        what: Int,
        obj: Any? = null,
    ) {
        try {
            val message = Message.obtain(handler, what, obj)
            handler.sendMessage(message)
        } catch (e: Exception) {
            Log.e(TAG, "Error sending message to handler: ${e.message}")
        }
    }

    /**
     * Send toast message
     */
    protected fun sendToastMessage(text: String) {
        try {
            val message = Message.obtain(handler, MESSAGE_TOAST)
            val bundle = android.os.Bundle()
            bundle.putString(TOAST, text)
            message.data = bundle
            handler.sendMessage(message)
        } catch (e: Exception) {
            Log.e(TAG, "Error sending toast message: ${e.message}")
        }
    }

    /**
     * Process incoming data packet
     */
    protected open fun processDataPacket(objectCluster: ObjectCluster) {
        sendMessageToHandler(MSG_IDENTIFIER_DATA_PACKET, objectCluster)
    }

    /**
     * Process state change
     */
    protected open fun processStateChange(state: ShimmerBluetooth.BtState) {
        val objectCluster = ObjectCluster()
        objectCluster.mState = state
        objectCluster.setBluetoothAddress(mBluetoothAddress)
        sendMessageToHandler(MSG_IDENTIFIER_STATE_CHANGE, objectCluster)
    }

    /**
     * Process notification message
     */
    protected open fun processNotificationMessage(notification: Int) {
        sendMessageToHandler(MSG_IDENTIFIER_NOTIFICATION_MESSAGE, notification)
    }

    /**
     * Initialize data processing coroutine scope
     */
    protected fun initializeDataProcessing() {
        mDataProcessingScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    }

    /**
     * Cleanup resources
     */
    protected open fun cleanup() {
        mDataProcessingScope?.cancel()
        mDataProcessingScope = null
        mCallbackObjects.clear()
    }
}
