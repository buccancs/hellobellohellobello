package com.topdon.tc001.sensors.gsr

import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.content.Context
import android.util.Log
import com.topdon.tc001.sensors.SensorInfo
import com.topdon.tc001.sensors.SensorRecorder
import com.topdon.tc001.sensors.SensorType
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.io.File
import java.io.FileWriter
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * Shimmer3 GSR+ sensor recorder implementation using Nordic BLE library
 * 
 * Key requirements from specifications:
 * - Use Nordic BLE library for robust communication
 * - Send start (0x07) and stop (0x20) commands
 * - Parse incoming data packets in notification callback
 * - Calculate GSR using 12-bit ADC resolution (0-4095 range)
 * - Log converted GSR (microsiemens) and raw PPG to timestamped CSV
 */
class ShimmerRecorder(
    private val context: Context,
    private val deviceAddress: String,
    private val outputDirectory: File
) : SensorRecorder {
    
    companion object {
        private const val TAG = "ShimmerRecorder"
        
        // Shimmer BLE Commands
        private const val COMMAND_START_STREAMING = 0x07.toByte()
        private const val COMMAND_STOP_STREAMING = 0x20.toByte()
        private const val COMMAND_SET_SAMPLE_RATE = 0x05.toByte()
        private const val COMMAND_SET_SENSORS = 0x08.toByte()
        
        // Default configuration
        private const val DEFAULT_SAMPLE_RATE = 51.2 // Hz
        private const val GSR_SENSOR_MASK = 0x04 // GSR sensor bit mask
        private const val PPG_SENSOR_MASK = 0x01 // PPG sensor bit mask
    }
    
    override val sensorType: SensorType = SensorType.GSR_SHIMMER
    
    private val _isConnected = MutableStateFlow(false)
    override val isConnected: Flow<Boolean> = _isConnected.asStateFlow()
    
    private val _isRecording = MutableStateFlow(false)
    override val isRecording: Flow<Boolean> = _isRecording.asStateFlow()
    
    private val coroutineScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val bluetoothManager: BluetoothManager by lazy { 
        context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager 
    }
    private val bluetoothAdapter: BluetoothAdapter? by lazy { bluetoothManager.adapter }
    
    private var bluetoothDevice: BluetoothDevice? = null
    private var currentSessionId: String? = null
    private var csvWriter: FileWriter? = null
    private var recordingStartTime: Long = 0L
    
    // Thread-safe queue for incoming sensor data
    private val dataQueue = ConcurrentLinkedQueue<GsrReading>()
    
    // Device info tracking
    private val _deviceInfo = MutableStateFlow(
        ShimmerDeviceInfo(
            deviceName = "Shimmer3 GSR+",
            deviceAddress = deviceAddress,
            connectionState = ShimmerConnectionState.DISCONNECTED,
            streamingState = ShimmerStreamingState.STOPPED
        )
    )
    
    override suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Initializing Shimmer recorder for device: $deviceAddress")
            
            // Check if Bluetooth is enabled
            val adapter = bluetoothAdapter ?: return@withContext false
            if (!adapter.isEnabled) {
                Log.e(TAG, "Bluetooth is not enabled")
                return@withContext false
            }
            
            // Get the Bluetooth device
            bluetoothDevice = adapter.getRemoteDevice(deviceAddress)
            if (bluetoothDevice == null) {
                Log.e(TAG, "Could not find Bluetooth device with address: $deviceAddress")
                return@withContext false
            }
            
            // Start connection process
            return@withContext connectToDevice()
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Shimmer recorder", e)
            updateConnectionState(ShimmerConnectionState.CONNECTION_FAILED)
            false
        }
    }
    
    private suspend fun connectToDevice(): Boolean = withContext(Dispatchers.IO) {
        try {
            updateConnectionState(ShimmerConnectionState.CONNECTING)
            
            // TODO: Implement Nordic BLE connection
            // This is a simplified version - full implementation would use Nordic BLE library
            delay(1000) // Simulate connection time
            
            updateConnectionState(ShimmerConnectionState.CONNECTED)
            _isConnected.value = true
            
            Log.d(TAG, "Successfully connected to Shimmer device")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to connect to Shimmer device", e)
            updateConnectionState(ShimmerConnectionState.CONNECTION_FAILED)
            false
        }
    }
    
    override suspend fun startRecording(sessionId: String): Boolean = withContext(Dispatchers.IO) {
        try {
            if (!_isConnected.value) {
                Log.w(TAG, "Cannot start recording - device not connected")
                return@withContext false
            }
            
            if (_isRecording.value) {
                Log.w(TAG, "Recording already in progress")
                return@withContext false
            }
            
            Log.d(TAG, "Starting GSR recording session: $sessionId")
            currentSessionId = sessionId
            recordingStartTime = System.nanoTime()
            
            // Create output CSV file
            val outputFile = File(outputDirectory, "gsr_${sessionId}_${System.currentTimeMillis()}.csv")
            csvWriter = FileWriter(outputFile).apply {
                // Write CSV header
                append("timestamp_nanos,raw_gsr_value,gsr_microsiemens,raw_ppg_value,session_id\n")
                flush()
            }
            
            // Configure sensor settings
            if (!configureShimmerSettings()) {
                Log.e(TAG, "Failed to configure Shimmer settings")
                return@withContext false
            }
            
            // Send start streaming command
            if (!sendStartStreamingCommand()) {
                Log.e(TAG, "Failed to send start streaming command")
                return@withContext false
            }
            
            updateStreamingState(ShimmerStreamingState.STREAMING)
            _isRecording.value = true
            
            // Start data processing coroutine
            startDataProcessing()
            
            Log.d(TAG, "GSR recording started successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start GSR recording", e)
            updateStreamingState(ShimmerStreamingState.FAILED)
            false
        }
    }
    
    override suspend fun stopRecording(): Boolean = withContext(Dispatchers.IO) {
        try {
            if (!_isRecording.value) {
                Log.w(TAG, "No recording in progress")
                return@withContext false
            }
            
            Log.d(TAG, "Stopping GSR recording")
            updateStreamingState(ShimmerStreamingState.STOPPING)
            
            // Send stop streaming command
            if (!sendStopStreamingCommand()) {
                Log.e(TAG, "Failed to send stop streaming command")
                return@withContext false
            }
            
            // Process remaining data in queue
            processRemainingData()
            
            // Close CSV file
            csvWriter?.close()
            csvWriter = null
            
            updateStreamingState(ShimmerStreamingState.STOPPED)
            _isRecording.value = false
            currentSessionId = null
            
            Log.d(TAG, "GSR recording stopped successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop GSR recording", e)
            updateStreamingState(ShimmerStreamingState.FAILED)
            false
        }
    }
    
    override suspend fun disconnect() {
        try {
            Log.d(TAG, "Disconnecting from Shimmer device")
            
            // Stop recording if in progress
            if (_isRecording.value) {
                stopRecording()
            }
            
            // TODO: Disconnect BLE connection using Nordic BLE library
            
            updateConnectionState(ShimmerConnectionState.DISCONNECTED)
            _isConnected.value = false
            
            // Clean up resources
            csvWriter?.close()
            csvWriter = null
            
            coroutineScope.cancel()
            
            Log.d(TAG, "Successfully disconnected from Shimmer device")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during disconnect", e)
        }
    }
    
    override suspend fun getSensorInfo(): SensorInfo {
        val deviceInfo = _deviceInfo.value
        return SensorInfo(
            type = sensorType,
            deviceName = deviceInfo.deviceName,
            deviceAddress = deviceInfo.deviceAddress,
            batteryLevel = deviceInfo.batteryLevel,
            lastDataTimestamp = if (_isRecording.value) System.nanoTime() else null
        )
    }
    
    private fun configureShimmerSettings(): Boolean {
        return try {
            Log.d(TAG, "Configuring Shimmer sensor settings")
            
            // TODO: Send sensor configuration commands via Nordic BLE
            // - Set sample rate to DEFAULT_SAMPLE_RATE
            // - Enable GSR and PPG sensors
            // - Configure other settings as needed
            
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to configure Shimmer settings", e)
            false
        }
    }
    
    private fun sendStartStreamingCommand(): Boolean {
        return try {
            Log.d(TAG, "Sending start streaming command (0x07)")
            
            // TODO: Send COMMAND_START_STREAMING via Nordic BLE characteristic write
            
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send start streaming command", e)
            false
        }
    }
    
    private fun sendStopStreamingCommand(): Boolean {
        return try {
            Log.d(TAG, "Sending stop streaming command (0x20)")
            
            // TODO: Send COMMAND_STOP_STREAMING via Nordic BLE characteristic write
            
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send stop streaming command", e)
            false
        }
    }
    
    private fun startDataProcessing() {
        coroutineScope.launch {
            while (_isRecording.value) {
                try {
                    // Process data from queue
                    while (dataQueue.isNotEmpty()) {
                        val reading = dataQueue.poll()
                        if (reading != null) {
                            writeToCSV(reading)
                        }
                    }
                    
                    // Small delay to prevent excessive CPU usage
                    delay(10)
                    
                } catch (e: Exception) {
                    Log.e(TAG, "Error in data processing", e)
                }
            }
        }
    }
    
    private fun processRemainingData() {
        try {
            Log.d(TAG, "Processing remaining data in queue: ${dataQueue.size} items")
            
            while (dataQueue.isNotEmpty()) {
                val reading = dataQueue.poll()
                if (reading != null) {
                    writeToCSV(reading)
                }
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing remaining data", e)
        }
    }
    
    private fun writeToCSV(reading: GsrReading) {
        try {
            csvWriter?.apply {
                append("${reading.timestampNanos},")
                append("${reading.rawGsrValue},")
                append("${reading.gsrMicrosiemens},")
                append("${reading.rawPpgValue},")
                append("${reading.sessionId}\n")
                flush()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error writing to CSV", e)
        }
    }
    
    /**
     * This method would be called from Nordic BLE notification callback
     * when new data packets are received from Shimmer device
     */
    fun onDataPacketReceived(data: ByteArray) {
        try {
            // Parse the incoming data packet
            val parsedData = parseShimmerDataPacket(data)
            if (parsedData != null) {
                // Add to processing queue
                dataQueue.offer(parsedData)
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing received data packet", e)
        }
    }
    
    private fun parseShimmerDataPacket(data: ByteArray): GsrReading? {
        return try {
            // TODO: Implement actual Shimmer data packet parsing
            // This is a simplified version - actual implementation would parse
            // the binary data format according to Shimmer protocol
            
            val sessionId = currentSessionId ?: return null
            val timestamp = System.nanoTime()
            
            // Simulated data parsing (replace with actual implementation)
            val rawGsrValue = ((data.getOrNull(0)?.toInt() ?: 0) and 0xFF) * 16 // 12-bit simulation
            val rawPpgValue = ((data.getOrNull(1)?.toInt() ?: 0) and 0xFF) * 16
            
            // Ensure 12-bit range
            val clampedGsrValue = rawGsrValue.coerceIn(0, 4095)
            val gsrMicrosiemens = GsrReading.convertToMicrosiemens(clampedGsrValue)
            
            GsrReading(
                timestampNanos = timestamp,
                rawGsrValue = clampedGsrValue,
                gsrMicrosiemens = gsrMicrosiemens,
                rawPpgValue = rawPpgValue,
                sessionId = sessionId
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing Shimmer data packet", e)
            null
        }
    }
    
    private fun updateConnectionState(state: ShimmerConnectionState) {
        _deviceInfo.value = _deviceInfo.value.copy(connectionState = state)
    }
    
    private fun updateStreamingState(state: ShimmerStreamingState) {
        _deviceInfo.value = _deviceInfo.value.copy(streamingState = state)
    }
}