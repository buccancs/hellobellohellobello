package com.topdon.tc001.sensors.gsr

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.Message
import android.util.Log
import com.topdon.tc001.sensors.SensorInfo
import com.topdon.tc001.sensors.SensorRecorder
import com.topdon.tc001.sensors.SensorType
import com.topdon.tc001.sensors.gsr.shimmer.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.io.File
import java.io.FileWriter

/**
 * Enhanced ShimmerAndroidAPI-compatible Shimmer3 GSR+ sensor recorder implementation
 * 
 * Uses Nordic BLE library with ShimmerAndroidAPI-compatible interfaces:
 * - Shimmer3BLEAndroid for robust BLE communication
 * - Official GSR sensor data processing with proper calibration
 * - 12-bit ADC resolution GSR conversion (0-4095 range)
 * - High-precision timestamping and CSV data logging
 * 
 * Based on official Shimmer protocol specifications with Nordic BLE backend
 */
class ShimmerRecorder(
    private val context: Context,
    private val deviceAddress: String,
    private val outputDirectory: File
) : SensorRecorder {
    
    companion object {
        private const val TAG = "ShimmerRecorder"
    }
    
    override val sensorType: SensorType = SensorType.GSR_SHIMMER
    
    private val _isConnected = MutableStateFlow(false)
    override val isConnected: Flow<Boolean> = _isConnected.asStateFlow()
    
    private val _isRecording = MutableStateFlow(false)
    override val isRecording: Flow<Boolean> = _isRecording.asStateFlow()
    
    private val coroutineScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    private var shimmerDevice: Shimmer3BLEAndroid? = null
    private var currentSessionId: String? = null
    private var csvWriter: FileWriter? = null
    private var recordingStartTime: Long = 0L
    
    // Handler for receiving messages from Shimmer device
    private val shimmerHandler = object : Handler(Looper.getMainLooper()) {
        override fun handleMessage(msg: Message) {
            when (msg.what) {
                ShimmerMessages.MSG_IDENTIFIER_DATA_PACKET -> {
                    if (msg.obj is ShimmerObjectCluster) {
                        processDataPacket(msg.obj as ShimmerObjectCluster)
                    }
                }
                ShimmerMessages.MESSAGE_TOAST -> {
                    val toastMsg = msg.data.getString("TOAST") ?: ""
                    Log.d(TAG, "Shimmer message: $toastMsg")
                }
                ShimmerMessages.MSG_IDENTIFIER_STATE_CHANGE -> {
                    handleStateChange(msg.obj)
                }
            }
            super.handleMessage(msg)
        }
    }
    
    // Custom callback processor for enhanced data handling
    private inner class ShimmerDataProcessor : BasicProcessWithCallBack() {
        override fun processMsgFromCallback(shimmerMsg: ShimmerMsg) {
            when (shimmerMsg.identifier) {
                ShimmerMessages.MSG_IDENTIFIER_STATE_CHANGE -> {
                    val callbackObject = shimmerMsg.objectData as? ShimmerCallbackObject
                    callbackObject?.let { 
                        handleStateChange(it)
                    }
                }
                ShimmerMessages.MSG_IDENTIFIER_NOTIFICATION_MESSAGE -> {
                    val callbackObject = shimmerMsg.objectData as? ShimmerCallbackObject
                    callbackObject?.let {
                        handleNotificationMessage(it)
                    }
                }
                ShimmerMessages.MSG_IDENTIFIER_DATA_PACKET -> {
                    val objectCluster = shimmerMsg.objectData as? ShimmerObjectCluster
                    objectCluster?.let {
                        processDataPacket(it)
                    }
                }
                ShimmerMessages.MSG_IDENTIFIER_PACKET_RECEPTION_RATE_OVERALL -> {
                    // Handle packet reception rate if needed
                    Log.v(TAG, "Packet reception rate updated")
                }
            }
        }
    }
    
    private val dataProcessor = ShimmerDataProcessor()
    
    override suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Initializing Shimmer recorder with enhanced Nordic BLE implementation for device: $deviceAddress")
            
            // Create Shimmer3BLEAndroid instance
            shimmerDevice = Shimmer3BLEAndroid(
                ShimmerHardwareType.SHIMMER_3, 
                deviceAddress, 
                shimmerHandler,
                context
            )
            
            // Set up data processor for enhanced callback handling
            dataProcessor.setWaitForData(shimmerDevice!!)
            
            return@withContext true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Shimmer recorder", e)
            updateConnectionState(ShimmerConnectionState.CONNECTION_FAILED)
            false
        }
    }
    
    override suspend fun startRecording(sessionId: String): Boolean = withContext(Dispatchers.IO) {
        try {
            val device = shimmerDevice ?: return@withContext false
            
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
                // Write CSV header with all relevant GSR data fields
                append("timestamp_nanos,timestamp_shimmer,raw_gsr_value,gsr_microsiemens,gsr_resistance_kohms,session_id\n")
                flush()
            }
            
            // Connect to device if not already connected
            if (!_isConnected.value) {
                Log.d(TAG, "Connecting to Shimmer device...")
                val connected = connectToDevice()
                if (!connected) {
                    Log.e(TAG, "Failed to connect to device before starting recording")
                    csvWriter?.close()
                    csvWriter = null
                    return@withContext false
                }
            }
            
            // Start streaming
            Log.d(TAG, "Starting GSR data streaming...")
            startStreamingInBackground(device)
            
            _isRecording.value = true
            updateStreamingState(ShimmerStreamingState.STREAMING)
            
            Log.d(TAG, "GSR recording started successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start GSR recording", e)
            updateStreamingState(ShimmerStreamingState.FAILED)
            csvWriter?.close()
            csvWriter = null
            false
        }
    }
    
    override suspend fun stopRecording(): Boolean = withContext(Dispatchers.IO) {
        try {
            val device = shimmerDevice ?: return@withContext false
            
            if (!_isRecording.value) {
                Log.w(TAG, "No recording in progress")
                return@withContext false
            }
            
            Log.d(TAG, "Stopping GSR recording")
            updateStreamingState(ShimmerStreamingState.STOPPING)
            
            // Stop streaming
            stopStreamingInBackground(device)
            
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
            
            // Disconnect device
            shimmerDevice?.let { device ->
                disconnectInBackground(device)
            }
            
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
        return SensorInfo(
            type = sensorType,
            deviceName = "Shimmer3 GSR+",
            deviceAddress = deviceAddress,
            batteryLevel = null, // Can be enhanced with battery info from ObjectCluster
            lastDataTimestamp = if (_isRecording.value) System.nanoTime() else null
        )
    }
    
    private suspend fun connectToDevice(): Boolean = withContext(Dispatchers.IO) {
        try {
            val device = shimmerDevice ?: return@withContext false
            
            updateConnectionState(ShimmerConnectionState.CONNECTING)
            
            // Connect in background thread (required by Nordic BLE)
            val connectionJob = async {
                try {
                    device.connect(deviceAddress, "default")
                    true
                } catch (e: Exception) {
                    Log.e(TAG, "Connection failed", e)
                    false
                }
            }
            
            // Wait for connection with timeout
            val connected = withTimeoutOrNull(30000) {
                connectionJob.await()
            } ?: false
            
            if (connected) {
                // Wait for full initialization
                delay(2000)
                updateConnectionState(ShimmerConnectionState.CONNECTED)
                _isConnected.value = true
                Log.d(TAG, "Successfully connected to Shimmer device")
            } else {
                updateConnectionState(ShimmerConnectionState.CONNECTION_FAILED)
                Log.e(TAG, "Failed to connect to Shimmer device")
            }
            
            connected
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during connection", e)
            updateConnectionState(ShimmerConnectionState.CONNECTION_FAILED)
            false
        }
    }
    
    private fun startStreamingInBackground(device: Shimmer3BLEAndroid) {
        Thread {
            try {
                device.startStreaming()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start streaming", e)
                updateStreamingState(ShimmerStreamingState.FAILED)
            }
        }.start()
    }
    
    private fun stopStreamingInBackground(device: Shimmer3BLEAndroid) {
        Thread {
            try {
                device.stopStreaming()
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping streaming", e)
            }
        }.start()
    }
    
    private fun disconnectInBackground(device: Shimmer3BLEAndroid) {
        Thread {
            try {
                device.disconnect()
            } catch (e: Exception) {
                Log.e(TAG, "Error during disconnect", e)
            }
        }.start()
    }
    
    private fun processDataPacket(objectCluster: ShimmerObjectCluster) {
        try {
            if (!_isRecording.value || csvWriter == null) return
            
            val currentTime = System.nanoTime()
            val sessionId = currentSessionId ?: "unknown"
            
            // Calculate resistance in kOhms (alternative representation)
            val gsrResistanceKohms = if (objectCluster.gsrCalibrated > 0) {
                1000.0 / objectCluster.gsrCalibrated // Convert from microsiemens to kOhms
            } else {
                Double.MAX_VALUE
            }
            
            // Create GSR reading
            val gsrReading = GsrReading(
                timestampNanos = currentTime,
                rawGsrValue = objectCluster.gsrRaw,
                gsrMicrosiemens = objectCluster.gsrCalibrated,
                rawPpgValue = objectCluster.ppgRaw,
                sessionId = sessionId
            )
            
            // Write to CSV file
            writeToCSV(gsrReading, objectCluster.timestamp, gsrResistanceKohms)
            
            Log.v(TAG, "GSR data: ${String.format("%.2f", objectCluster.gsrCalibrated)} μS, " +
                    "Raw: ${objectCluster.gsrRaw}, Resistance: ${String.format("%.2f", gsrResistanceKohms)} kΩ")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing data packet", e)
        }
    }
    
    private fun writeToCSV(reading: GsrReading, shimmerTimestamp: Double, resistanceKohms: Double) {
        try {
            csvWriter?.apply {
                append("${reading.timestampNanos},")
                append("$shimmerTimestamp,")
                append("${reading.rawGsrValue},")
                append("${reading.gsrMicrosiemens},")
                append("$resistanceKohms,")
                append("${reading.sessionId}\n")
                flush()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error writing to CSV", e)
        }
    }
    
    private fun handleStateChange(stateObj: Any?) {
        try {
            val state = when {
                stateObj is ShimmerObjectCluster -> stateObj.state
                stateObj is ShimmerCallbackObject -> stateObj.state
                else -> null
            } ?: return
            
            when (state) {
                ShimmerBTState.CONNECTED -> {
                    Log.d(TAG, "Shimmer device connected")
                    updateConnectionState(ShimmerConnectionState.CONNECTED)
                    _isConnected.value = true
                }
                ShimmerBTState.CONNECTING -> {
                    Log.d(TAG, "Shimmer device connecting")
                    updateConnectionState(ShimmerConnectionState.CONNECTING)
                }
                ShimmerBTState.STREAMING -> {
                    Log.d(TAG, "Shimmer device streaming")
                    updateStreamingState(ShimmerStreamingState.STREAMING)
                }
                ShimmerBTState.STREAMING_AND_SDLOGGING -> {
                    Log.d(TAG, "Shimmer device streaming and SD logging")
                    updateStreamingState(ShimmerStreamingState.STREAMING)
                }
                ShimmerBTState.DISCONNECTED,
                ShimmerBTState.CONNECTION_LOST -> {
                    Log.d(TAG, "Shimmer device disconnected")
                    updateConnectionState(ShimmerConnectionState.DISCONNECTED)
                    _isConnected.value = false
                    if (_isRecording.value) {
                        _isRecording.value = false
                        updateStreamingState(ShimmerStreamingState.STOPPED)
                    }
                }
                else -> {
                    // Handle other states
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error handling state change", e)
        }
    }
    
    private fun handleNotificationMessage(callbackObject: ShimmerCallbackObject) {
        when (callbackObject.indicator) {
            ShimmerNotifications.SHIMMER_FULLY_INITIALIZED -> {
                Log.d(TAG, "Shimmer device fully initialized")
            }
            ShimmerNotifications.SHIMMER_START_STREAMING -> {
                Log.d(TAG, "Shimmer streaming started")
                updateStreamingState(ShimmerStreamingState.STREAMING)
            }
            ShimmerNotifications.SHIMMER_STOP_STREAMING -> {
                Log.d(TAG, "Shimmer streaming stopped")
                updateStreamingState(ShimmerStreamingState.STOPPED)
            }
        }
    }
    
    // Device info tracking
    private val _deviceInfo = MutableStateFlow(
        ShimmerDeviceInfo(
            deviceName = "Shimmer3 GSR+",
            deviceAddress = deviceAddress,
            connectionState = ShimmerConnectionState.DISCONNECTED,
            streamingState = ShimmerStreamingState.STOPPED
        )
    )
    
    private fun updateConnectionState(state: ShimmerConnectionState) {
        _deviceInfo.value = _deviceInfo.value.copy(connectionState = state)
    }
    
    private fun updateStreamingState(state: ShimmerStreamingState) {
        _deviceInfo.value = _deviceInfo.value.copy(streamingState = state)
    }
}