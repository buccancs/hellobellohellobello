package com.yourcompany.sensorspoke.sensors.thermal

import android.content.Context
import android.hardware.usb.UsbDevice
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.infisense.iruvc.usb.USBMonitor
import com.infisense.iruvc.uvc.UVCCamera
import com.infisense.iruvc.uvc.UVCType
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * TC001Connector - Core thermal camera connection handler using IRCamera integration
 *
 * Manages connection to Topdon TC001 thermal camera using the proven IRCamera
 * USBMonitor and IRCMD classes for reliable hardware communication
 */
class TC001Connector(
    private val context: Context,
) : USBMonitor.OnDeviceConnectListener {
    companion object {
        private const val TAG = "TC001Connector"
        private const val TC001_VENDOR_ID = 0x4d54
        private const val TC001_PRODUCT_ID = 0x0100
    }

    private val _connectionState = MutableLiveData<TC001ConnectionState>()
    val connectionState: LiveData<TC001ConnectionState> = _connectionState

    private val _deviceInfo = MutableLiveData<TC001DeviceInfo?>()
    val deviceInfo: LiveData<TC001DeviceInfo?> = _deviceInfo

    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var isConnected = false
    private var currentDevice: UsbDevice? = null

    private var usbMonitor: USBMonitor? = null
    private var uvcCamera: UVCCamera? = null

    init {
        _connectionState.value = TC001ConnectionState.DISCONNECTED
        initIRCameraUSBMonitoring()
    }

    /**
     * Initialize IRCamera USB monitoring for TC001 devices
     */
    private fun initIRCameraUSBMonitoring() {
        scope.launch {
            try {
                usbMonitor = USBMonitor(context, this@TC001Connector)
                usbMonitor?.register()

                Log.i(TAG, "IRCamera USBMonitor initialized for TC001")
                scanForTC001Devices()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize IRCamera USB monitoring", e)
                _connectionState.value = TC001ConnectionState.ERROR
            }
        }
    }

    /**
     * Scan for connected TC001 devices using IRCamera USBMonitor
     */
    private suspend fun scanForTC001Devices() =
        withContext(Dispatchers.IO) {
            try {
                usbMonitor?.let { monitor ->
                    val deviceList = monitor.deviceList
                    for (device in deviceList) {
                        if (isTC001Device(device)) {
                            Log.i(TAG, "TC001 device found: ${device.deviceName}")
                            withContext(Dispatchers.Main) {
                                handleDeviceFound(device)
                            }
                            return@withContext
                        }
                    }
                }

                withContext(Dispatchers.Main) {
                    _connectionState.value = TC001ConnectionState.NOT_FOUND
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error scanning for TC001 devices", e)
                withContext(Dispatchers.Main) {
                    _connectionState.value = TC001ConnectionState.ERROR
                }
            }
        }

    /**
     * USBMonitor.OnDeviceConnectListener implementation for IRCamera integration
     */
    override fun onAttach(device: UsbDevice?) {
        device?.let {
            if (isTC001Device(it)) {
                Log.i(TAG, "TC001 device attached: ${it.deviceName}")
                handleDeviceFound(it)
            }
        }
    }

    override fun onDettach(device: UsbDevice?) {
        device?.let {
            if (isTC001Device(it) && it == currentDevice) {
                Log.i(TAG, "TC001 device detached: ${it.deviceName}")
                scope.launch {
                    disconnect()
                }
            }
        }
    }

    override fun onConnect(
        device: UsbDevice?,
        controlBlock: USBMonitor.UsbControlBlock?,
        createNew: Boolean,
    ) {
        if (device != null && isTC001Device(device)) {
            Log.i(TAG, "TC001 device connected via IRCamera USBMonitor")
            scope.launch {
                initializeIRCameraConnection(controlBlock)
            }
        }
    }

    override fun onDisconnect(
        device: UsbDevice?,
        controlBlock: USBMonitor.UsbControlBlock?,
    ) {
        if (device != null && isTC001Device(device) && device == currentDevice) {
            Log.i(TAG, "TC001 device disconnected via IRCamera USBMonitor")
            scope.launch {
                handleIRCameraDisconnection()
            }
        }
    }

    override fun onCancel(device: UsbDevice?) {
        Log.w(TAG, "TC001 connection cancelled: ${device?.deviceName}")
    }

    override fun onGranted(device: UsbDevice?, granted: Boolean) {
        if (device != null && isTC001Device(device)) {
            if (granted) {
                Log.i(TAG, "USB permission granted for TC001: ${device.deviceName}")
            } else {
                Log.w(TAG, "USB permission denied for TC001: ${device.deviceName}")
                _connectionState.value = TC001ConnectionState.ERROR
            }
        }
    }

    /**
     * Check if USB device is a TC001 thermal camera
     */
    private fun isTC001Device(device: UsbDevice): Boolean = device.vendorId == TC001_VENDOR_ID && device.productId == TC001_PRODUCT_ID

    /**
     * Initialize IRCamera connection using proven UVCCamera approach
     */
    private suspend fun initializeIRCameraConnection(controlBlock: USBMonitor.UsbControlBlock?) =
        withContext(Dispatchers.IO) {
            try {
                if (controlBlock == null) {
                    Log.e(TAG, "Invalid USB control block for TC001")
                    _connectionState.postValue(TC001ConnectionState.ERROR)
                    return@withContext
                }

                uvcCamera = UVCCamera().apply {
                    uvcType = UVCType.USB_UVC
                }

                uvcCamera?.let { camera ->
                    val openResult = camera.openUVCCamera(controlBlock)
                    if (openResult == 0) {
                        isConnected = true
                        _connectionState.postValue(TC001ConnectionState.CONNECTED)
                        Log.i(TAG, "TC001 connected successfully using IRCamera UVCCamera")
                    } else {
                        Log.e(TAG, "Failed to open TC001 via IRCamera UVCCamera: $openResult")
                        _connectionState.postValue(TC001ConnectionState.ERROR)
                    }
                } ?: run {
                    Log.e(TAG, "Failed to create UVCCamera for TC001")
                    _connectionState.postValue(TC001ConnectionState.ERROR)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Exception during IRCamera UVCCamera initialization", e)
                _connectionState.postValue(TC001ConnectionState.ERROR)
            }
        }

    /**
     * Handle IRCamera disconnection
     */
    private suspend fun handleIRCameraDisconnection() =
        withContext(Dispatchers.IO) {
            try {
                uvcCamera?.closeUVCCamera()
                uvcCamera = null
                isConnected = false
                currentDevice = null
                _connectionState.postValue(TC001ConnectionState.DISCONNECTED)
                _deviceInfo.postValue(null)
                Log.i(TAG, "TC001 disconnected via IRCamera UVCCamera")
            } catch (e: Exception) {
                Log.e(TAG, "Error during IRCamera disconnection", e)
            }
        }

    /**
     * Handle TC001 device discovery
     */
    private fun handleDeviceFound(device: UsbDevice) {
        currentDevice = device
        _connectionState.value = TC001ConnectionState.FOUND

        val deviceInfo =
            TC001DeviceInfo(
                deviceName = device.deviceName,
                vendorId = device.vendorId,
                productId = device.productId,
                serialNumber = device.serialNumber ?: "Unknown",
            )
        _deviceInfo.value = deviceInfo

        Log.i(TAG, "TC001 device info: $deviceInfo")

        usbMonitor?.requestPermission(device)
    }

    /**
     * Attempt to connect to the TC001 device using IRCamera approach
     */
    suspend fun connect(): Boolean =
        withContext(Dispatchers.IO) {
            try {
                _connectionState.postValue(TC001ConnectionState.CONNECTING)

                currentDevice?.let { device ->
                    usbMonitor?.openDevice(device)
                    return@withContext true
                }

                _connectionState.postValue(TC001ConnectionState.ERROR)
                return@withContext false
            } catch (e: Exception) {
                Log.e(TAG, "Failed to connect to TC001 via IRCamera", e)
                _connectionState.postValue(TC001ConnectionState.ERROR)
                return@withContext false
            }
        }

    /**
     * Disconnect from TC001 device using IRCamera approach
     */
    suspend fun disconnect() =
        withContext(Dispatchers.IO) {
            try {
                _connectionState.postValue(TC001ConnectionState.DISCONNECTING)

                uvcCamera?.closeUVCCamera()
                uvcCamera = null

                Log.i(TAG, "Closing device connection via IRCamera USBMonitor")

                isConnected = false
                currentDevice = null
                _connectionState.postValue(TC001ConnectionState.DISCONNECTED)
                _deviceInfo.postValue(null)

                Log.i(TAG, "Disconnected from TC001 via IRCamera")
            } catch (e: Exception) {
                Log.e(TAG, "Error during TC001 IRCamera disconnection", e)
                _connectionState.postValue(TC001ConnectionState.ERROR)
            }
        }

    /**
     * Check if TC001 is currently connected
     */
    fun isConnected(): Boolean = isConnected

    /**
     * Refresh device scan using IRCamera USBMonitor
     */
    suspend fun refreshDeviceScan() {
        usbMonitor?.let { monitor ->
            scanForTC001Devices()
        }
    }

    /**
     * Get IRCamera UVCCamera instance for thermal data operations
     */
    fun getUVCCamera(): UVCCamera? = uvcCamera

    /**
     * Start thermal data streaming using IRCamera UVCCamera
     */
    fun startThermalStream(): Boolean {
        return try {
            uvcCamera?.let { camera ->
                val result = camera.onStartPreview()
                result == 0
            } ?: false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start thermal stream via IRCamera UVCCamera", e)
            false
        }
    }

    /**
     * Stop thermal data streaming using IRCamera UVCCamera
     */
    fun stopThermalStream(): Boolean {
        return try {
            uvcCamera?.let { camera ->
                val result = camera.onStopPreview()
                result == 0
            } ?: false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop thermal stream via IRCamera UVCCamera", e)
            false
        }
    }

    /**
     * Clean up IRCamera resources
     */
    fun cleanup() {
        scope.cancel()
        uvcCamera?.closeUVCCamera()
        usbMonitor?.unregister()
        usbMonitor?.destroy()
    }
}

/**
 * TC001 connection states
 */
enum class TC001ConnectionState {
    DISCONNECTED,
    NOT_FOUND,
    FOUND,
    CONNECTING,
    CONNECTED,
    DISCONNECTING,
    ERROR,
}

/**
 * TC001 device information
 */
data class TC001DeviceInfo(
    val deviceName: String,
    val vendorId: Int,
    val productId: Int,
    val serialNumber: String,
)
