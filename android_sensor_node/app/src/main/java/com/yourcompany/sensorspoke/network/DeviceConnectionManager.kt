package com.yourcompany.sensorspoke.network

import android.content.Context
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine

/**
 * DeviceConnectionManager provides unified device status management across all sensor types.
 * This addresses the separation of concerns requirement by centralizing device state management.
 *
 * This manager coordinates the connection status of:
 * - Shimmer GSR sensors
 * - RGB cameras
 * - Thermal cameras
 * - Audio devices
 * - Network connections to PC hub
 */
class DeviceConnectionManager(
    private val context: Context?,
) {
    companion object {
        private const val TAG = "DeviceConnectionManager"
    }

    private val _shimmerState = MutableStateFlow(DeviceState.DISCONNECTED)
    val shimmerState: StateFlow<DeviceState> = _shimmerState.asStateFlow()

    private val _rgbCameraState = MutableStateFlow(DeviceState.DISCONNECTED)
    val rgbCameraState: StateFlow<DeviceState> = _rgbCameraState.asStateFlow()

    private val _thermalCameraState = MutableStateFlow(DeviceState.DISCONNECTED)
    val thermalCameraState: StateFlow<DeviceState> = _thermalCameraState.asStateFlow()

    private val _audioDeviceState = MutableStateFlow(DeviceState.DISCONNECTED)
    val audioDeviceState: StateFlow<DeviceState> = _audioDeviceState.asStateFlow()

    private val _networkState = MutableStateFlow(DeviceState.DISCONNECTED)
    val networkState: StateFlow<DeviceState> = _networkState.asStateFlow()

    private val _overallState = MutableStateFlow(OverallSystemState.NOT_READY)
    val overallState: StateFlow<OverallSystemState> = _overallState.asStateFlow()

    private val _deviceDetails = MutableStateFlow<Map<String, DeviceDetails>>(emptyMap())
    val deviceDetails: StateFlow<Map<String, DeviceDetails>> = _deviceDetails.asStateFlow()

    /**
     * Device connection states
     */
    enum class DeviceState {
        DISCONNECTED,
        CONNECTING,
        CONNECTED,
        ERROR,
        NOT_AVAILABLE,
    }

    /**
     * Overall system readiness states
     */
    enum class OverallSystemState {
        NOT_READY,
        PARTIALLY_READY,
        READY,
        ERROR,
    }

    /**
     * Detailed device information
     */
    data class DeviceDetails(
        val deviceType: String,
        val deviceName: String,
        val connectionState: DeviceState,
        val lastUpdate: Long = System.currentTimeMillis(),
        val errorMessage: String? = null,
        val batteryLevel: Int? = null,
        val dataRate: Double? = null,
        val isRequired: Boolean = true,
    )

    init {
        combine(
            shimmerState,
            rgbCameraState,
            thermalCameraState,
            audioDeviceState,
            networkState,
        ) { shimmer, rgb, thermal, audio, network ->
            calculateOverallState(shimmer, rgb, thermal, audio, network)
        }.also { flow ->
        }
    }

    /**
     * Update Shimmer device state
     */
    fun updateShimmerState(state: DeviceState, details: DeviceDetails? = null) {
        Log.d(TAG, "Updating Shimmer state: $state")
        _shimmerState.value = state
        details?.let { updateDeviceDetails("shimmer", it) }
        updateOverallState()
    }

    /**
     * Update RGB camera state
     */
    fun updateRgbCameraState(state: DeviceState, details: DeviceDetails? = null) {
        Log.d(TAG, "Updating RGB camera state: $state")
        _rgbCameraState.value = state
        details?.let { updateDeviceDetails("rgb_camera", it) }
        updateOverallState()
    }

    /**
     * Update thermal camera state
     */
    fun updateThermalCameraState(state: DeviceState, details: DeviceDetails? = null) {
        Log.d(TAG, "Updating thermal camera state: $state")
        _thermalCameraState.value = state
        details?.let { updateDeviceDetails("thermal_camera", it) }
        updateOverallState()
    }

    /**
     * Update audio device state
     */
    fun updateAudioDeviceState(state: DeviceState, details: DeviceDetails? = null) {
        Log.d(TAG, "Updating audio device state: $state")
        _audioDeviceState.value = state
        details?.let { updateDeviceDetails("audio_device", it) }
        updateOverallState()
    }

    /**
     * Update network connection state
     */
    fun updateNetworkState(state: DeviceState, details: DeviceDetails? = null) {
        Log.d(TAG, "Updating network state: $state")
        _networkState.value = state
        details?.let { updateDeviceDetails("network", it) }
        updateOverallState()
    }

    /**
     * Update device details
     */
    private fun updateDeviceDetails(deviceId: String, details: DeviceDetails) {
        val currentDetails = _deviceDetails.value.toMutableMap()
        currentDetails[deviceId] = details
        _deviceDetails.value = currentDetails
    }

    /**
     * Calculate overall system state based on individual device states
     */
    private fun calculateOverallState(
        shimmer: DeviceState,
        rgb: DeviceState,
        thermal: DeviceState,
        audio: DeviceState,
        network: DeviceState,
    ): OverallSystemState {
        val states = listOf(shimmer, rgb, thermal, audio, network)

        return when {
            states.any { it == DeviceState.ERROR } -> OverallSystemState.ERROR
            states.all { it == DeviceState.CONNECTED } -> OverallSystemState.READY
            states.count { it == DeviceState.CONNECTED } >= 2 -> OverallSystemState.PARTIALLY_READY
            else -> OverallSystemState.NOT_READY
        }
    }

    /**
     * Update overall state based on current device states
     */
    private fun updateOverallState() {
        val newState = calculateOverallState(
            _shimmerState.value,
            _rgbCameraState.value,
            _thermalCameraState.value,
            _audioDeviceState.value,
            _networkState.value,
        )

        if (_overallState.value != newState) {
            Log.i(TAG, "Overall system state changed: ${_overallState.value} -> $newState")
            _overallState.value = newState
        }
    }

    /**
     * Get count of connected devices
     */
    fun getConnectedDeviceCount(): Int {
        return listOf(
            _shimmerState.value,
            _rgbCameraState.value,
            _thermalCameraState.value,
            _audioDeviceState.value,
            _networkState.value,
        ).count { it == DeviceState.CONNECTED }
    }

    /**
     * Get count of required devices
     */
    fun getRequiredDeviceCount(): Int {
        return _deviceDetails.value.values.count { it.isRequired }
    }

    /**
     * Check if system is ready for recording
     */
    fun isReadyForRecording(): Boolean {
        return _overallState.value == OverallSystemState.READY ||
            _overallState.value == OverallSystemState.PARTIALLY_READY
    }

    /**
     * Get summary of device states for UI display
     */
    fun getDeviceStateSummary(): Map<String, DeviceState> {
        return mapOf(
            "shimmer" to _shimmerState.value,
            "rgb_camera" to _rgbCameraState.value,
            "thermal_camera" to _thermalCameraState.value,
            "audio_device" to _audioDeviceState.value,
            "network" to _networkState.value,
        )
    }

    /**
     * Reset all device states
     */
    fun resetAllStates() {
        Log.i(TAG, "Resetting all device states")
        _shimmerState.value = DeviceState.DISCONNECTED
        _rgbCameraState.value = DeviceState.DISCONNECTED
        _thermalCameraState.value = DeviceState.DISCONNECTED
        _audioDeviceState.value = DeviceState.DISCONNECTED
        _networkState.value = DeviceState.DISCONNECTED
        _deviceDetails.value = emptyMap()
        updateOverallState()
    }

    /**
     * Clean up resources
     */
    fun cleanup() {
        Log.i(TAG, "Cleaning up DeviceConnectionManager")
        resetAllStates()
    }
}
