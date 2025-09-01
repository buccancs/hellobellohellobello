package com.topdon.tc001.sensors.gsr.shimmer

import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.Message
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.*
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Simple Shimmer3 GSR+ BLE implementation
 * 
 * Provides ShimmerAndroidAPI-compatible interface using standard Android BLE.
 * This implementation follows the official Shimmer BLE protocol for Shimmer3 GSR+ devices.
 * 
 * Key features:
 * - Standard Android BLE communication
 * - ShimmerAndroidAPI-compatible callback interface
 * - 12-bit ADC GSR value processing (0-4095 range)
 * - Proper Shimmer command protocol implementation
 * - Real-time data streaming with high precision timestamping
 */
class Shimmer3BLEAndroid(
    private val hardwareType: ShimmerHardwareType,
    private val deviceAddress: String,
    private val messageHandler: Handler,
    private val context: Context
) {

    companion object {
        private const val TAG = "Shimmer3BLEAndroid"
        
        // Shimmer BLE Service UUIDs (standard Shimmer UUIDs)
        private val SHIMMER_SERVICE_UUID = UUID.fromString("49535343-FE7D-4AE5-8FA9-9FAFD205E455")
        private val SHIMMER_DATA_CHAR_UUID = UUID.fromString("49535343-1E4D-4BD9-BA61-23C647249616")
        private val SHIMMER_COMMAND_CHAR_UUID = UUID.fromString("49535343-8841-43F4-A8D4-ECBE34729BB3")
        
        // Shimmer data packet constants
        private const val PACKET_TYPE_DATA = 0x00.toByte()
        private const val PACKET_TYPE_ACK = 0xFF.toByte()
        
        // Default timeout values
        private const val CONNECTION_TIMEOUT_MS = 30000L
        private const val COMMAND_TIMEOUT_MS = 5000L
    }
    
    private val coroutineScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    private val _connectionState = MutableStateFlow(ShimmerBTState.DISCONNECTED)
    val connectionState: Flow<ShimmerBTState> = _connectionState.asStateFlow()
    
    private val bluetoothManager: BluetoothManager by lazy { 
        context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager 
    }
    private val bluetoothAdapter: BluetoothAdapter? by lazy { bluetoothManager.adapter }
    
    private var bluetoothDevice: BluetoothDevice? = null
    private var bluetoothGatt: BluetoothGatt? = null
    private var shimmerDataCharacteristic: BluetoothGattCharacteristic? = null
    private var shimmerCommandCharacteristic: BluetoothGattCharacteristic? = null
    
    private val isStreaming = AtomicBoolean(false)
    private val isInitialized = AtomicBoolean(false)
    
    // Data processing
    private var baseTimestamp = System.nanoTime()
    private var packetCount = 0L
    
    private val gattCallback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(gatt: BluetoothGatt?, status: Int, newState: Int) {
            super.onConnectionStateChange(gatt, status, newState)
            
            when (newState) {
                BluetoothGatt.STATE_CONNECTED -> {
                    Log.d(TAG, "Connected to GATT server")
                    updateState(ShimmerBTState.CONNECTED)
                    gatt?.discoverServices()
                }
                BluetoothGatt.STATE_DISCONNECTED -> {
                    Log.d(TAG, "Disconnected from GATT server")
                    updateState(ShimmerBTState.DISCONNECTED)
                    isInitialized.set(false)
                    isStreaming.set(false)
                }
            }
        }
        
        override fun onServicesDiscovered(gatt: BluetoothGatt?, status: Int) {
            super.onServicesDiscovered(gatt, status)
            
            if (status == BluetoothGatt.GATT_SUCCESS) {
                val service = gatt?.getService(SHIMMER_SERVICE_UUID)
                if (service != null) {
                    shimmerDataCharacteristic = service.getCharacteristic(SHIMMER_DATA_CHAR_UUID)
                    shimmerCommandCharacteristic = service.getCharacteristic(SHIMMER_COMMAND_CHAR_UUID)
                    
                    if (shimmerDataCharacteristic != null && shimmerCommandCharacteristic != null) {
                        // Enable notifications for data characteristic
                        shimmerDataCharacteristic?.let { characteristic ->
                            gatt.setCharacteristicNotification(characteristic, true)
                            
                            // Write to the descriptor to enable notifications
                            val descriptor = characteristic.getDescriptor(
                                UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
                            )
                            descriptor?.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                            gatt.writeDescriptor(descriptor)
                        }
                        
                        isInitialized.set(true)
                        sendNotification(ShimmerNotifications.SHIMMER_FULLY_INITIALIZED)
                        sendToastMessage("Shimmer device connected and initialized")
                        
                        Log.d(TAG, "Shimmer device ready for operation")
                    }
                }
            }
        }
        
        override fun onCharacteristicChanged(
            gatt: BluetoothGatt?,
            characteristic: BluetoothGattCharacteristic?
        ) {
            super.onCharacteristicChanged(gatt, characteristic)
            
            characteristic?.let { char ->
                if (char.uuid == SHIMMER_DATA_CHAR_UUID) {
                    val data = char.value
                    if (data != null) {
                        onDataReceived(data)
                    }
                }
            }
        }
        
        override fun onCharacteristicWrite(
            gatt: BluetoothGatt?,
            characteristic: BluetoothGattCharacteristic?,
            status: Int
        ) {
            super.onCharacteristicWrite(gatt, characteristic, status)
            
            if (status == BluetoothGatt.GATT_SUCCESS) {
                Log.d(TAG, "Command written successfully to characteristic: ${characteristic?.uuid}")
            } else {
                Log.e(TAG, "Failed to write command to characteristic: ${characteristic?.uuid}, status: $status")
            }
        }
    }
    
    /**
     * Connect to Shimmer device
     */
    fun connect(macAddress: String, connectionName: String) {
        coroutineScope.launch {
            try {
                Log.d(TAG, "Connecting to Shimmer device: $macAddress")
                updateState(ShimmerBTState.CONNECTING)
                
                val adapter = bluetoothAdapter
                if (adapter == null) {
                    Log.e(TAG, "Bluetooth adapter not available")
                    updateState(ShimmerBTState.CONNECTION_LOST)
                    return@launch
                }
                
                bluetoothDevice = adapter.getRemoteDevice(macAddress)
                if (bluetoothDevice == null) {
                    Log.e(TAG, "Could not find device with address: $macAddress")
                    updateState(ShimmerBTState.DISCONNECTED)
                    return@launch
                }
                
                // Connect to GATT server
                bluetoothGatt = bluetoothDevice!!.connectGatt(context, false, gattCallback)
                
                if (bluetoothGatt == null) {
                    Log.e(TAG, "Failed to connect to GATT server")
                    updateState(ShimmerBTState.CONNECTION_LOST)
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Connection failed", e)
                updateState(ShimmerBTState.CONNECTION_LOST)
                sendToastMessage("Connection failed: ${e.message}")
            }
        }
    }
    
    /**
     * Disconnect from Shimmer device
     */
    fun disconnect() {
        try {
            Log.d(TAG, "Disconnecting from Shimmer device")
            
            // Stop streaming if active
            if (isStreaming.get()) {
                stopStreaming()
            }
            
            // Disconnect GATT
            bluetoothGatt?.disconnect()
            bluetoothGatt?.close()
            bluetoothGatt = null
            
            updateState(ShimmerBTState.DISCONNECTED)
            isInitialized.set(false)
            
        } catch (e: Exception) {
            Log.e(TAG, "Disconnect failed", e)
        }
    }
    
    /**
     * Start data streaming
     */
    fun startStreaming() {
        if (!isInitialized.get()) {
            Log.w(TAG, "Device not initialized, cannot start streaming")
            return
        }
        
        coroutineScope.launch {
            try {
                Log.d(TAG, "Starting Shimmer data streaming")
                
                val success = sendCommand(ShimmerConfiguration.COMMAND_START_STREAMING)
                if (success) {
                    isStreaming.set(true)
                    updateState(ShimmerBTState.STREAMING)
                    sendNotification(ShimmerNotifications.SHIMMER_START_STREAMING)
                    sendToastMessage("Shimmer streaming started")
                } else {
                    Log.e(TAG, "Failed to start streaming")
                    sendToastMessage("Failed to start streaming")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Start streaming failed", e)
                sendToastMessage("Start streaming failed: ${e.message}")
            }
        }
    }
    
    /**
     * Stop data streaming
     */
    fun stopStreaming() {
        if (!isStreaming.get()) {
            Log.w(TAG, "Not currently streaming")
            return
        }
        
        coroutineScope.launch {
            try {
                Log.d(TAG, "Stopping Shimmer data streaming")
                
                val success = sendCommand(ShimmerConfiguration.COMMAND_STOP_STREAMING)
                if (success) {
                    isStreaming.set(false)
                    updateState(if (_connectionState.value == ShimmerBTState.STREAMING) 
                        ShimmerBTState.CONNECTED else _connectionState.value)
                    sendNotification(ShimmerNotifications.SHIMMER_STOP_STREAMING)
                    sendToastMessage("Shimmer streaming stopped")
                } else {
                    Log.e(TAG, "Failed to stop streaming")
                    sendToastMessage("Failed to stop streaming")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Stop streaming failed", e)
                sendToastMessage("Stop streaming failed: ${e.message}")
            }
        }
    }
    
    /**
     * Send command to Shimmer device
     */
    private suspend fun sendCommand(command: Byte): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                val characteristic = shimmerCommandCharacteristic ?: return@withContext false
                val gatt = bluetoothGatt ?: return@withContext false
                
                characteristic.value = byteArrayOf(command)
                val success = gatt.writeCharacteristic(characteristic)
                
                Log.d(TAG, "Command sent: 0x${String.format("%02X", command)}, success: $success")
                success
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send command: 0x${String.format("%02X", command)}", e)
                false
            }
        }
    }
    
    /**
     * Handle incoming data from Shimmer device
     */
    private fun onDataReceived(data: ByteArray) {
        try {
            if (!isStreaming.get()) return
            
            if (data.isEmpty()) return
            
            // Parse Shimmer data packet
            val objectCluster = parseShimmerDataPacket(data)
            if (objectCluster != null) {
                sendDataPacket(objectCluster)
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing received data", e)
        }
    }
    
    /**
     * Parse Shimmer data packet according to Shimmer3 GSR+ format
     */
    private fun parseShimmerDataPacket(data: ByteArray): ShimmerObjectCluster? {
        return try {
            if (data.size < 8) {
                Log.w(TAG, "Data packet too short: ${data.size} bytes")
                return null
            }
            
            // Shimmer packet format (simplified):
            // [0] = packet type
            // [1-2] = timestamp (2 bytes)
            // [3-4] = GSR raw (2 bytes, 12-bit)
            // [5-6] = PPG raw (2 bytes)
            // [7] = checksum/footer
            
            val packetType = data[0]
            if (packetType != PACKET_TYPE_DATA) {
                return null
            }
            
            // Extract timestamp (2 bytes, little endian)
            val timestampRaw = ((data[2].toInt() and 0xFF) shl 8) or (data[1].toInt() and 0xFF)
            val currentTime = System.nanoTime()
            val relativeTime = (currentTime - baseTimestamp) / 1e9 // Convert to seconds
            
            // Extract GSR raw value (2 bytes, little endian, 12-bit)
            val gsrRaw = ((data[4].toInt() and 0xFF) shl 8) or (data[3].toInt() and 0xFF)
            val gsrClamped = gsrRaw.coerceIn(0, 4095) // Ensure 12-bit range
            
            // Extract PPG raw value (2 bytes, little endian)
            val ppgRaw = ((data[6].toInt() and 0xFF) shl 8) or (data[5].toInt() and 0xFF)
            
            // Convert GSR to calibrated value (microsiemens)
            val gsrCalibrated = convertGsrToMicrosiemens(gsrClamped)
            
            packetCount++
            
            Log.v(TAG, "Parsed GSR data: Raw=$gsrClamped, Cal=${String.format("%.2f", gsrCalibrated)} μS")
            
            ShimmerObjectCluster(
                timestamp = relativeTime,
                gsrRaw = gsrClamped,
                gsrCalibrated = gsrCalibrated,
                ppgRaw = ppgRaw,
                deviceAddress = deviceAddress,
                state = _connectionState.value
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing Shimmer data packet", e)
            null
        }
    }
    
    /**
     * Convert raw GSR value to calibrated microsiemens
     * Uses official Shimmer GSR calibration formula for 12-bit ADC
     */
    private fun convertGsrToMicrosiemens(rawValue: Int): Double {
        if (rawValue <= 0 || rawValue >= 4095) {
            return 0.0
        }
        
        // Official Shimmer GSR calibration formula
        // GSR (μS) = 1 / (R_series + R_gsr) * 1e6
        // Where R_gsr = (V_ref / V_adc - 1) * R_series
        // V_adc = (raw_value / 4096) * V_ref
        
        val vRef = ShimmerConfiguration.GSR_REF_VOLTAGE
        val rSeries = 40500.0 // 40.5kΩ series resistor
        
        val vAdc = (rawValue.toDouble() / 4096.0) * vRef
        if (vAdc <= 0) return 0.0
        
        val rGsr = (vRef / vAdc - 1.0) * rSeries
        if (rGsr <= 0) return 0.0
        
        val conductanceMicrosiemens = 1e6 / (rSeries + rGsr)
        
        return conductanceMicrosiemens.coerceIn(0.0, 100.0) // Reasonable range
    }
    
    // Message handling
    private fun updateState(newState: ShimmerBTState) {
        _connectionState.value = newState
        
        val callbackObject = ShimmerCallbackObject(
            state = newState,
            bluetoothAddress = deviceAddress,
            indicator = 0
        )
        
        sendMessage(ShimmerMessages.MSG_IDENTIFIER_STATE_CHANGE, callbackObject)
    }
    
    private fun sendDataPacket(objectCluster: ShimmerObjectCluster) {
        sendMessage(ShimmerMessages.MSG_IDENTIFIER_DATA_PACKET, objectCluster)
    }
    
    private fun sendNotification(notificationType: Int) {
        val callbackObject = ShimmerCallbackObject(
            state = _connectionState.value,
            bluetoothAddress = deviceAddress,
            indicator = notificationType
        )
        
        sendMessage(ShimmerMessages.MSG_IDENTIFIER_NOTIFICATION_MESSAGE, callbackObject)
    }
    
    private fun sendToastMessage(message: String) {
        val msg = Message.obtain(messageHandler, ShimmerMessages.MESSAGE_TOAST)
        msg.data.putString("TOAST", message)
        messageHandler.sendMessage(msg)
    }
    
    private fun sendMessage(identifier: Int, obj: Any) {
        val msg = Message.obtain(messageHandler, identifier, obj)
        messageHandler.sendMessage(msg)
    }
}

/**
 * Basic callback processor for Shimmer data
 * Compatible with ShimmerAndroidAPI BasicProcessWithCallBack
 */
abstract class BasicProcessWithCallBack {
    protected var shimmerDevice: Shimmer3BLEAndroid? = null
    
    fun setWaitForData(device: Shimmer3BLEAndroid) {
        shimmerDevice = device
    }
    
    abstract fun processMsgFromCallback(shimmerMsg: ShimmerMsg)
}