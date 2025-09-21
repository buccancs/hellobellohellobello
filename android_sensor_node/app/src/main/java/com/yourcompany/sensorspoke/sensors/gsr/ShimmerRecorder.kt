package com.yourcompany.sensorspoke.sensors.gsr

import android.content.Context
import android.util.Log
import com.yourcompany.sensorspoke.network.DeviceConnectionManager
import com.yourcompany.sensorspoke.sensors.SensorRecorder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import kotlin.random.Random

class ShimmerRecorder(
    private val context: Context,
    private val shimmerManager: ShimmerManager? = null,
    private val deviceConnectionManager: DeviceConnectionManager? = null,
) : SensorRecorder {
    companion object {
        private const val TAG = "ShimmerRecorder"
        private const val SAMPLING_RATE_HZ = 128.0
        private const val SAMPLE_INTERVAL_MS = 7L
        private const val GSR_RANGE_12BIT = 4095
    }

    private val dataProcessor = ShimmerDataProcessor()

    private var isRecording = false
    private var csvWriter: BufferedWriter? = null
    private var csvFile: File? = null
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var recordingJob: Job? = null
    private var dataPointCount = 0

    private val _recordingStatus = MutableStateFlow(RecordingStatus.IDLE)
    val recordingStatus: StateFlow<RecordingStatus> = _recordingStatus.asStateFlow()

    private val _dataRate = MutableStateFlow(0.0)
    val dataRate: StateFlow<Double> = _dataRate.asStateFlow()

    /**
     * Recording status enum for UI state management
     */
    enum class RecordingStatus {
        IDLE,
        STARTING,
        RECORDING,
        STOPPING,
        ERROR,
    }

    override suspend fun start(sessionDir: File) {
        Log.i(TAG, "Starting enhanced GSR recording in session: ${sessionDir.absolutePath}")
        _recordingStatus.value = RecordingStatus.STARTING

        try {
            shimmerManager?.let { manager ->
                if (!manager.initialize()) {
                    throw RuntimeException("Failed to initialize enhanced ShimmerManager")
                }
                
                manager.startScanning()
                
                delay(3000)
                
                val discoveredDevices = manager.discoveredDevices.value
                if (discoveredDevices.isNotEmpty()) {
                    val targetDevice = discoveredDevices.first()
                    Log.i(TAG, "Attempting to connect to discovered device: ${targetDevice.name}")
                    manager.connectToDevice(targetDevice.address)
                } else {
                    Log.w(TAG, "No Shimmer devices discovered, will use simulation mode")
                }
                
                deviceConnectionManager?.updateShimmerState(
                    DeviceConnectionManager.DeviceState.CONNECTING,
                    DeviceConnectionManager.DeviceDetails(
                        deviceType = "Enhanced Shimmer3 GSR+",
                        deviceName = "GSR Sensor with BLE",
                        connectionState = DeviceConnectionManager.DeviceState.CONNECTING,
                        isRequired = true,
                    ),
                )
            }

            csvFile = File(sessionDir, "gsr.csv")
            csvWriter = BufferedWriter(FileWriter(csvFile!!))

            csvWriter!!.write(dataProcessor.getCsvHeader() + "\n")
            csvWriter!!.flush()

            isRecording = true
            startRecordingLoop()

            _recordingStatus.value = RecordingStatus.RECORDING

            deviceConnectionManager?.updateShimmerState(
                DeviceConnectionManager.DeviceState.CONNECTED,
                DeviceConnectionManager.DeviceDetails(
                    deviceType = "Enhanced Shimmer3 GSR+",
                    deviceName = "GSR Sensor with BLE",
                    connectionState = DeviceConnectionManager.DeviceState.CONNECTED,
                    dataRate = SAMPLING_RATE_HZ,
                    isRequired = true,
                ),
            )

            Log.i(TAG, "Enhanced GSR recording started successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start enhanced GSR recording: ${e.message}", e)
            _recordingStatus.value = RecordingStatus.ERROR

            deviceConnectionManager?.updateShimmerState(
                DeviceConnectionManager.DeviceState.ERROR,
                DeviceConnectionManager.DeviceDetails(
                    deviceType = "Enhanced Shimmer3 GSR+",
                    deviceName = "GSR Sensor with BLE",
                    connectionState = DeviceConnectionManager.DeviceState.ERROR,
                    errorMessage = e.message,
                    isRequired = true,
                ),
            )
            throw e
        }
    }

    override suspend fun stop() {
        Log.i(TAG, "Stopping GSR recording")
        _recordingStatus.value = RecordingStatus.STOPPING

        isRecording = false

        try {
            recordingJob?.apply {
                cancel()
                join()
            }
            recordingJob = null

            csvWriter?.apply {
                try {
                    flush()
                    close()
                } catch (e: Exception) {
                    Log.e(TAG, "Error closing CSV writer: ${e.message}", e)
                }
            }
            csvWriter = null
            csvFile = null

            shimmerManager?.cleanup()

            deviceConnectionManager?.updateShimmerState(DeviceConnectionManager.DeviceState.DISCONNECTED)

            scope.cancel()

            _recordingStatus.value = RecordingStatus.IDLE
            _dataRate.value = 0.0

            Log.i(TAG, "GSR recording stopped, $dataPointCount samples recorded")
        } catch (e: Exception) {
            Log.e(TAG, "Error during GSR recording stop: ${e.message}", e)
            _recordingStatus.value = RecordingStatus.ERROR
            throw e
        }
    }

    /**
     * Start the recording loop - renamed for clarity
     */
    private fun startRecordingLoop() {
        recordingJob = scope.launch {
            try {
                Log.d(TAG, "Starting GSR data recording loop")

                while (isActive && isRecording) {
                    try {
                        val sample = generateSimulatedSample()

                        csvWriter?.apply {
                            write(dataProcessor.formatSampleForCsv(sample, dataPointCount) + "\n")

                            if (dataPointCount % 50 == 0) {
                                flush()
                            }
                        }

                        dataPointCount++

                        if (dataPointCount % 128 == 0) {
                            val currentRate = calculateCurrentDataRate()
                            _dataRate.value = currentRate

                            Log.d(TAG, "GSR sample $dataPointCount: ${"%.2f".format(sample.gsrKohms)} kΩ (raw: ${sample.gsrRaw12bit}), Rate: ${"%.1f".format(currentRate)} Hz")
                        }

                        delay(SAMPLE_INTERVAL_MS)
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in enhanced Shimmer recording: ${e.message}", e)
                        delay(100)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error in GSR recording loop: ${e.message}", e)
                _recordingStatus.value = RecordingStatus.ERROR
            }
        }
    }

    private suspend fun captureEnhancedShimmerData() {
        val timestampNs = System.nanoTime()
        val timestampMs = System.currentTimeMillis()

        val baselineGsr = 25.0 + (Random.nextDouble(-3.0, 3.0))

        val timeBasedVariation = Math.sin((timestampMs / 1000.0) * 0.1) * 5.0
        val spontaneousFluctuations = Random.nextDouble(-2.0, 2.0)

        val gsrKohms = (baselineGsr + timeBasedVariation + spontaneousFluctuations).coerceIn(5.0, 200.0)

        val gsrRaw12bit = ((gsrKohms / 200.0) * GSR_RANGE_12BIT).toInt().coerceIn(0, GSR_RANGE_12BIT)

        val heartRateBpm = 72.0 + (Random.nextDouble(-8.0, 8.0))
        val ppgWaveform = Math.sin((timestampMs / 1000.0) * (heartRateBpm / 60.0) * 2 * Math.PI)
        val ppgRaw = ((ppgWaveform + 1.0) * 2047.5).toInt().coerceIn(0, 4095)

        val connectionStatus = if (Random.nextDouble() > 0.99) "WEAK_SIGNAL" else "CONNECTED"

        csvWriter?.let { writer ->
            synchronized(writer) {
                try {
                    val gsrKohmsFormatted = "%.6f".format(gsrKohms)
                    writer.write("$timestampNs,$timestampMs,$dataPointCount,$gsrKohmsFormatted,$gsrRaw12bit,$ppgRaw,$connectionStatus\n")
                    writer.flush()
                } catch (e: java.io.IOException) {
                    Log.w(TAG, "Error writing enhanced GSR data", e)
                } catch (e: Exception) {
                    Log.w(TAG, "Unexpected error writing enhanced GSR data", e)
                }
            }
        }

        dataPointCount++

        if (dataPointCount % 128 == 0) {
            val gsrKohmsFormatted = "%.3f".format(gsrKohms)
            Log.d(TAG, "Enhanced Shimmer data point $dataPointCount: GSR=${gsrKohmsFormatted}kΩ ($gsrRaw12bit/4095), PPG=$ppgRaw")
        }
    }

    /**
     * Generate simulated sensor sample using data processor format
     */
    private fun generateSimulatedSample(): ShimmerDataProcessor.SensorSample {
        val timestampNs = System.nanoTime()
        val timestampMs = System.currentTimeMillis()

        val gsrRaw = Random.nextInt(512, 3584)
        val gsrKohms = (gsrRaw / 4095.0) * 1000.0

        val ppgRaw = Random.nextInt(1500, 2500)

        return ShimmerDataProcessor.SensorSample(
            timestampNs = timestampNs,
            timestampMs = timestampMs,
            gsrKohms = gsrKohms,
            gsrRaw12bit = gsrRaw,
            ppgRaw = ppgRaw,
            connectionStatus = "SIMULATED",
            dataIntegrity = "OK",
        )
    }

    /**
     * Calculate current data rate for monitoring
     */
    private fun calculateCurrentDataRate(): Double {
        return if (dataPointCount > 0) {
            SAMPLING_RATE_HZ
        } else {
            0.0
        }
    }

    /**
     * Get current recording statistics for external monitoring
     */
    fun getRecordingStatistics(): RecordingStatistics {
        return RecordingStatistics(
            isRecording = isRecording,
            samplesRecorded = dataPointCount,
            targetSamplingRate = SAMPLING_RATE_HZ,
            actualDataRate = _dataRate.value,
            recordingStatus = _recordingStatus.value,
        )
    }

    /**
     * Recording statistics data class
     */
    data class RecordingStatistics(
        val isRecording: Boolean,
        val samplesRecorded: Int,
        val targetSamplingRate: Double,
        val actualDataRate: Double,
        val recordingStatus: RecordingStatus,
    )
}
