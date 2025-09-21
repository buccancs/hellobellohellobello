package com.yourcompany.sensorspoke.sensors.thermal

import android.content.Context
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.yourcompany.sensorspoke.utils.PermissionManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * ThermalCameraManager handles thermal camera lifecycle management and configuration.
 * Separates thermal camera management concerns from data recording logic.
 *
 * This class is responsible for:
 * - USB device discovery and permission management
 * - Thermal camera initialization and configuration
 * - Camera state monitoring
 * - Error handling and recovery
 */
class ThermalCameraManager(
    private val context: Context,
    private val permissionManager: PermissionManager? = null,
) {
    companion object {
        private const val TAG = "ThermalCameraManager"
        private const val TOPDON_VENDOR_ID = 0x4d54
        private const val TC001_PRODUCT_ID_1 = 0x0100
        private const val TC001_PRODUCT_ID_2 = 0x0200
        private const val DEFAULT_FPS = 10
        private const val MAX_FPS = 25
    }

    private val _cameraState = MutableStateFlow(CameraState.UNINITIALIZED)
    val cameraState: StateFlow<CameraState> = _cameraState.asStateFlow()

    private val _cameraInfo = MutableStateFlow<ThermalCameraInfo?>(null)
    val cameraInfo: StateFlow<ThermalCameraInfo?> = _cameraInfo.asStateFlow()

    private val _frameRate = MutableStateFlow(DEFAULT_FPS.toDouble())
    val frameRate: StateFlow<Double> = _frameRate.asStateFlow()

    private var realTopdonIntegration: RealTopdonIntegration? = null
    private var topdonIntegration: TopdonThermalIntegration? = null
    private var targetFps = DEFAULT_FPS
    
    private val managerScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    /**
     * Camera states for thermal camera management
     */
    enum class CameraState {
        UNINITIALIZED,
        INITIALIZING,
        READY,
        RECORDING,
        ERROR,
    }

    /**
     * Thermal camera information container
     */
    data class ThermalCameraInfo(
        val deviceId: String,
        val deviceName: String,
        val isRealDevice: Boolean = false,
        val vendorId: Int = TOPDON_VENDOR_ID,
        val productId: Int = TC001_PRODUCT_ID_1,
        val maxFps: Int = MAX_FPS,
        val currentFps: Int = DEFAULT_FPS,
    )

    /**
     * Initialize the thermal camera system
     */
    suspend fun initialize(): Boolean {
        return try {
            Log.i(TAG, "Initializing thermal camera manager")
            _cameraState.value = CameraState.INITIALIZING

            initializeThermalCamera()

            _cameraState.value = CameraState.READY
            Log.i(TAG, "Thermal camera manager initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize thermal camera manager: ${e.message}", e)
            _cameraState.value = CameraState.ERROR
            false
        }
    }

    /**
     * Initialize thermal camera integration
     */
    private fun initializeThermalCamera() {
        try {
            val usbManager = context.getSystemService(Context.USB_SERVICE) as UsbManager
            val deviceList = usbManager.deviceList

            val topdonDevice = deviceList.values.find { device ->
                isTopdonTC001Device(device)
            }

            if (topdonDevice != null) {
                Log.i(TAG, "Topdon TC001 device found: ${topdonDevice.deviceName}")

                val hasPermission = checkUsbPermission(topdonDevice, usbManager)
                if (hasPermission) {
                    Log.i(TAG, "USB permission granted, initializing real Topdon integration")
                    initializeRealThermalIntegration()
                } else {
                    Log.w(TAG, "USB permission not granted, using simulation mode")
                    initializeSimulationIntegration()
                }
            } else {
                Log.w(TAG, "No Topdon TC001 device found - using simulation mode")
                initializeSimulationIntegration()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing thermal camera integration: ${e.message}", e)
            initializeSimulationIntegration()
        }
    }

    /**
     * Initialize real Topdon integration
     */
    private fun initializeRealThermalIntegration() {
        try {
            realTopdonIntegration = RealTopdonIntegration(context)
            
            managerScope.launch {
                val success = realTopdonIntegration!!.initialize()
                
                if (success) {
                    Log.i(TAG, "Real Topdon TC001 integration initialized successfully")
                    updateCameraInfo(isReal = true)
                } else {
                    Log.w(TAG, "Real Topdon integration failed, falling back to simulation")
                    realTopdonIntegration = null
                    initializeSimulationIntegration()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing real Topdon integration: ${e.message}", e)
            realTopdonIntegration = null
            initializeSimulationIntegration()
        }
    }

    /**
     * Initialize simulation integration
     */
    private fun initializeSimulationIntegration() {
        topdonIntegration = TopdonThermalIntegration(context)

        if (topdonIntegration!!.initialize() == TopdonResult.SUCCESS) {
            configureDevice()
            Log.i(TAG, "Simulation thermal integration initialized")
            updateCameraInfo(isReal = false)
        } else {
            Log.w(TAG, "Failed to initialize simulation thermal integration")
            topdonIntegration = null
        }
    }

    /**
     * Update camera information
     */
    private fun updateCameraInfo(isReal: Boolean) {
        _cameraInfo.value = ThermalCameraInfo(
            deviceId = if (isReal) "TOPDON_TC001" else "THERMAL_SIM",
            deviceName = if (isReal) "Topdon TC001 Thermal Camera" else "Simulated Thermal Camera",
            isRealDevice = isReal,
            currentFps = targetFps,
        )
    }

    /**
     * Configure thermal camera frame rate (1-25 FPS)
     */
    fun setFrameRate(fps: Int): Boolean {
        return try {
            targetFps = fps.coerceIn(1, MAX_FPS)
            _frameRate.value = targetFps.toDouble()
            Log.i(TAG, "Thermal camera frame rate set to $targetFps FPS")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set frame rate: ${e.message}", e)
            false
        }
    }

    /**
     * Configure device settings
     */
    private fun configureDevice() {
        topdonIntegration?.apply {
            setEmissivity(0.95f)
            setTemperatureRange(-20f, 150f)
            setThermalPalette(TopdonThermalPalette.IRON)
            enableAutoGainControl(true)
            enableTemperatureCompensation(true)
            configureDevice()
        }
    }

    /**
     * Check USB permission for Topdon device
     */
    private fun checkUsbPermission(device: UsbDevice, usbManager: UsbManager): Boolean {
        return if (usbManager.hasPermission(device)) {
            Log.d(TAG, "USB permission already granted for ${device.deviceName}")
            true
        } else {
            Log.i(TAG, "USB permission required for Topdon TC001")
            false
        }
    }

    /**
     * Check if device is Topdon TC001
     */
    private fun isTopdonTC001Device(device: UsbDevice): Boolean {
        return device.vendorId == TOPDON_VENDOR_ID &&
            (device.productId == TC001_PRODUCT_ID_1 || device.productId == TC001_PRODUCT_ID_2)
    }

    /**
     * Get thermal integration for data capture
     */
    fun getThermalIntegration(): TopdonThermalIntegration? = topdonIntegration

    /**
     * Get real thermal integration for data capture
     */
    fun getRealThermalIntegration(): RealTopdonIntegration? = realTopdonIntegration

    /**
     * Check if camera is ready for recording
     */
    fun isReady(): Boolean = _cameraState.value == CameraState.READY

    /**
     * Check if using real thermal camera
     */
    fun isRealCamera(): Boolean = realTopdonIntegration != null

    /**
     * Update recording state
     */
    fun updateRecordingState(isRecording: Boolean) {
        _cameraState.value = if (isRecording) CameraState.RECORDING else CameraState.READY
    }

    /**
     * Get current frame rate
     */
    fun getCurrentFrameRate(): Int = targetFps

    /**
     * Clean up thermal camera resources
     */
    fun cleanup() {
        Log.i(TAG, "Cleaning up thermal camera manager")

        realTopdonIntegration?.let { integration ->
            managerScope.launch {
                integration.stopStreaming()
                integration.disconnect()
            }
            realTopdonIntegration = null
        }

        topdonIntegration?.let {
            it.disconnect()
            it.cleanup()
            topdonIntegration = null
        }

        _cameraState.value = CameraState.UNINITIALIZED
        _cameraInfo.value = null
        _frameRate.value = DEFAULT_FPS.toDouble()
    }
}

/**
 * TC001UIController - Enhanced thermal camera UI management
 * 
 * Integrated into ThermalCameraManager for centralized thermal camera control
 * Manages thermal camera UI state and controls based on IRCamera's comprehensive interface design
 */
class TC001UIController : ViewModel() {
    companion object {
        private const val TAG = "TC001UIController"
    }

    private val _isRecording = MutableLiveData(false)
    val isRecording: LiveData<Boolean> = _isRecording

    private val _currentPalette = MutableLiveData(TopdonThermalPalette.GRAYSCALE)
    val currentPalette: LiveData<TopdonThermalPalette> = _currentPalette

    private val _deviceStatus = MutableLiveData<TC001DeviceStatus?>(null)
    val deviceStatus: LiveData<TC001DeviceStatus?> = _deviceStatus

    private val _deviceConnectionStatus = MutableLiveData(false)
    val deviceConnectionStatus: LiveData<Boolean> = _deviceConnectionStatus

    private val _currentTemperature = MutableLiveData<Float?>(null)
    val currentTemperature: LiveData<Float?> = _currentTemperature

    /**
     * Update device connection status
     */
    fun updateDeviceConnection(connected: Boolean) {
        _deviceConnectionStatus.value = connected
        Log.i(TAG, "Device connection updated: $connected")
    }

    /**
     * Update current temperature reading
     */
    fun updateCurrentTemperature(temperature: Float) {
        _currentTemperature.value = temperature
        Log.d(TAG, "Temperature updated: $temperature°C")
    }

    /**
     * Start thermal recording
     */
    fun startRecording() {
        viewModelScope.launch {
            try {
                Log.i(TAG, "Starting thermal recording")
                _isRecording.value = true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start recording", e)
            }
        }
    }

    /**
     * Stop thermal recording
     */
    fun stopRecording() {
        viewModelScope.launch {
            try {
                Log.i(TAG, "Stopping thermal recording")
                _isRecording.value = false
            } catch (e: Exception) {
                Log.e(TAG, "Failed to stop recording", e)
            }
        }
    }

    /**
     * Update thermal palette
     */
    fun updatePalette(palette: TopdonThermalPalette) {
        _currentPalette.value = palette
        Log.i(TAG, "Updated thermal palette to: $palette")
    }

    /**
     * Update device connection status
     */
    fun updateDeviceStatus(status: TC001DeviceStatus) {
        _deviceStatus.value = status
        Log.i(TAG, "Updated device status: $status")
    }

    /**
     * Handle auto gain toggle
     */
    fun onAutoGainToggled(enabled: Boolean) {
        Log.i(TAG, "Auto gain toggled: $enabled")
        // Implementation for auto gain control
    }

    /**
     * Handle temperature compensation toggle
     */
    fun onTemperatureCompensationToggled(enabled: Boolean) {
        Log.i(TAG, "Temperature compensation toggled: $enabled")
        // Implementation for temperature compensation control
    }

    // Additional UI controller methods for EnhancedMainFragment integration
    
    private val _hasConnectLine = MutableLiveData(false)
    val hasConnectLine: LiveData<Boolean> = _hasConnectLine

    /**
     * Handle device click events
     */
    fun handleDeviceClick(deviceType: Any) {
        Log.i(TAG, "Device clicked: $deviceType")
        // Handle device selection/interaction
    }

    /**
     * Refresh UI state
     */
    fun refresh() {
        Log.d(TAG, "Refreshing UI state")
        // Refresh device list and connection status
    }

    /**
     * Update connection status
     */
    fun updateConnectionStatus(connected: Boolean) {
        _deviceConnectionStatus.value = connected
        Log.i(TAG, "Connection status updated: $connected")
    }
}

/**
 * TC001 connection types (adapted from IRCamera)
 */
enum class TC001ConnectType {
    LINE,
    WIFI,
    BLE,
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
