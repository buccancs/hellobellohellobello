package com.yourcompany.sensorspoke.sensors.thermal.tc001

import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.yourcompany.sensorspoke.sensors.thermal.TopdonThermalPalette
import kotlinx.coroutines.launch

/**
 * TC001UIController - Enhanced thermal camera UI management
 *
 * Manages thermal camera UI state and controls based on IRCamera's
 * comprehensive interface design
 */
class TC001UIController : ViewModel() {
    companion object {
        private const val TAG = "TC001UIController"
    }

    // Device management
    private val _hasConnectLine = MutableLiveData<Boolean>(false)
    val hasConnectLine: LiveData<Boolean> = _hasConnectLine

    private val _deviceConnectionStatus = MutableLiveData<String>("offline")
    val deviceConnectionStatus: LiveData<String> = _deviceConnectionStatus

    // Thermal control states
    private val _currentPalette = MutableLiveData<TopdonThermalPalette>(TopdonThermalPalette.IRON)
    val currentPalette: LiveData<TopdonThermalPalette> = _currentPalette

    private val _emissivity = MutableLiveData<Float>(0.95f)
    val emissivity: LiveData<Float> = _emissivity

    private val _temperatureRange = MutableLiveData<Pair<Float, Float>>(-20f to 150f)
    val temperatureRange: LiveData<Pair<Float, Float>> = _temperatureRange

    private val _autoGainEnabled = MutableLiveData<Boolean>(true)
    val autoGainEnabled: LiveData<Boolean> = _autoGainEnabled

    private val _temperatureCompensationEnabled = MutableLiveData<Boolean>(true)
    val temperatureCompensationEnabled: LiveData<Boolean> = _temperatureCompensationEnabled

    private val _currentTemperature = MutableLiveData<Float>(0f)
    val currentTemperature: LiveData<Float> = _currentTemperature

    // UI callback interfaces inspired by IRCamera
    var onItemClickListener: ((type: TC001ConnectType) -> Unit)? = null
    var onItemLongClickListener: ((type: TC001ConnectType) -> Unit)? = null

    /**
     * Update connection status from TC001Connector
     */
    fun updateConnectionStatus(isConnected: Boolean) {
        viewModelScope.launch {
            _hasConnectLine.value = isConnected
            _deviceConnectionStatus.value = if (isConnected) "online" else "offline"
            Log.i(TAG, "Connection status updated: $isConnected")
        }
    }

    /**
     * Handle thermal palette change
     */
    fun onPaletteChanged(palette: TopdonThermalPalette) {
        viewModelScope.launch {
            _currentPalette.value = palette
            Log.i(TAG, "Thermal palette changed: $palette")
        }
    }

    /**
     * Handle emissivity change
     */
    fun onEmissivityChanged(emissivity: Float) {
        viewModelScope.launch {
            val clampedEmissivity = emissivity.coerceIn(0.1f, 1.0f)
            _emissivity.value = clampedEmissivity
            Log.i(TAG, "Emissivity changed: $clampedEmissivity")
        }
    }

    /**
     * Handle temperature range change
     */
    fun onTemperatureRangeChanged(
        minTemp: Float,
        maxTemp: Float,
    ) {
        viewModelScope.launch {
            if (minTemp < maxTemp) {
                _temperatureRange.value = minTemp to maxTemp
                Log.i(TAG, "Temperature range changed: $minTemp to $maxTemp")
            }
        }
    }

    /**
     * Handle auto gain toggle
     */
    fun onAutoGainToggled(enabled: Boolean) {
        viewModelScope.launch {
            _autoGainEnabled.value = enabled
            Log.i(TAG, "Auto gain toggled: $enabled")
        }
    }

    /**
     * Handle temperature compensation toggle
     */
    fun onTemperatureCompensationToggled(enabled: Boolean) {
        viewModelScope.launch {
            _temperatureCompensationEnabled.value = enabled
            Log.i(TAG, "Temperature compensation toggled: $enabled")
        }
    }

    /**
     * Update current temperature reading
     */
    fun updateCurrentTemperature(temp: Float) {
        viewModelScope.launch {
            _currentTemperature.value = temp
        }
    }

    /**
     * Handle device item click (based on IRCamera MainFragment logic)
     */
    fun handleDeviceClick(type: TC001ConnectType) {
        onItemClickListener?.invoke(type)
        Log.i(TAG, "Device clicked: $type")
    }

    /**
     * Handle device item long click for deletion
     */
    fun handleDeviceLongClick(type: TC001ConnectType) {
        if (!_hasConnectLine.value!!) {
            // Only allow deletion when device is offline
            onItemLongClickListener?.invoke(type)
            Log.i(TAG, "Device long clicked for deletion: $type")
        }
    }

    /**
     * Refresh device connection state
     */
    fun refresh() {
        viewModelScope.launch {
            // Trigger refresh of connection states
            Log.i(TAG, "Refreshing device states")
        }
    }

    /**
     * Update device connection status
     */
    fun updateDeviceConnection(isConnected: Boolean) {
        viewModelScope.launch {
            _hasConnectLine.value = isConnected
            _deviceConnectionStatus.value = if (isConnected) "online" else "offline"
            Log.i(TAG, "Device connection updated: $isConnected")
        }
    }
}

/**
 * TC001 connection types (adapted from IRCamera)
 */
enum class TC001ConnectType {
    LINE, // TC001 via USB
    WIFI, // TC001 via WiFi (if supported)
    BLE, // TC001 via Bluetooth (if supported)
}

/**
 * Device information for connected TC001
 */
data class TC001DeviceStatus(
    val isConnected: Boolean,
    val deviceName: String,
    val connectionType: TC001ConnectType,
    val batteryLevel: Int? = null,
    val temperature: Float? = null,
)
