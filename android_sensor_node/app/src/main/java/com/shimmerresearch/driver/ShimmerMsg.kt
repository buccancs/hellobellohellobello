package com.shimmerresearch.driver

/**
 * ShimmerMsg data container for callback communication
 */
data class ShimmerMsg(
    val mIdentifier: Int,
    val mB: Any?,
)

/**
 * Basic process with callback interface from ShimmerAndroidAPI
 */
abstract class BasicProcessWithCallBack {
    private var shimmerDevice: Any? = null

    /**
     * Set the Shimmer device to wait for data from
     */
    open fun setWaitForData(device: Any) {
        shimmerDevice = device
    }

    /**
     * Process messages from callback - must be implemented by subclass
     */
    protected abstract fun processMsgFromCallback(shimmerMSG: ShimmerMsg)

    /**
     * Handle incoming callback message
     */
    fun handleCallback(
        identifier: Int,
        payload: Any?,
    ) {
        val shimmerMsg = ShimmerMsg(identifier, payload)
        processMsgFromCallback(shimmerMsg)
    }
}

/**
 * Callback object for state and data communication
 */
class CallbackObject {
    var mState: com.shimmerresearch.bluetooth.ShimmerBluetooth.BtState =
        com.shimmerresearch.bluetooth.ShimmerBluetooth.BtState.DISCONNECTED
    var mBluetoothAddress: String = ""
    var mIndicator: Int = 0
    var mPayload: Any? = null

    constructor()

    constructor(state: com.shimmerresearch.bluetooth.ShimmerBluetooth.BtState, address: String) {
        mState = state
        mBluetoothAddress = address
    }

    constructor(indicator: Int, payload: Any?) {
        mIndicator = indicator
        mPayload = payload
    }
}
