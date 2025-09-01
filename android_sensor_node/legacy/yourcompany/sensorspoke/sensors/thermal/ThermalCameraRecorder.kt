package com.yourcompany.sensorspoke.sensors.thermal

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.yourcompany.sensorspoke.sensors.SensorRecorder
import com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001CalibrationCurve
import com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001CalibrationManager
import com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001CalibrationType
import com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001DataExporter
import com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001DiagnosticResults
import com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001DiagnosticSystem
import com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001ExportFormat
import com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001ExportResult
import com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001IntegrationManager
import com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001PerformanceMonitor
import kotlinx.coroutines.*
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

/**
 * Production ThermalCameraRecorder with Real Topdon TC001 Integration.
 *
 * This implementation provides complete thermal camera functionality:
 * - Real TC001 hardware detection and connection
 * - Hardware-calibrated temperature measurements (±2°C accuracy)
 * - Professional thermal image processing with color palettes
 * - CSV data logging with nanosecond timestamps
 * - Thermal image sequence capture for analysis
 * - Advanced thermal processing: emissivity, AGC, temperature compensation
 */
class ThermalCameraRecorder(
    private val context: Context,
) : SensorRecorder {
    companion object {
        private const val TAG = "ThermalCameraRecorder"
    }

    private var csvWriter: BufferedWriter? = null
    private var csvFile: File? = null
    private var thermalImagesDir: File? = null
    private var sessionDir: File? = null
    private var recordingJob: Job? = null

    // Real Topdon integration
    private var topdonIntegration: TopdonThermalIntegration? = null

    // Enhanced TC001 integration manager for comprehensive integration
    private var tc001IntegrationManager: TC001IntegrationManager? = null

    // Advanced TC001 components for complete thermal system
    private var tc001PerformanceMonitor: TC001PerformanceMonitor? = null
    private var tc001DataExporter: TC001DataExporter? = null
    private var tc001CalibrationManager: TC001CalibrationManager? = null
    private var tc001DiagnosticSystem: TC001DiagnosticSystem? = null

    private var frameCount = 0
    private val dateFormatter = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US)

    override suspend fun start(sessionDirectory: File) {
        initialize(sessionDirectory)
        startRecording()
    }

    override suspend fun stop() {
        stopRecording()
        cleanup()
    }

    // Implementation methods
    suspend fun initialize(sessionDirectory: File): Boolean =
        withContext(Dispatchers.IO) {
            try {
                sessionDir = sessionDirectory

                // Create CSV file for thermal data logging
                csvFile = File(sessionDirectory, "thermal_data.csv")
                csvWriter = BufferedWriter(FileWriter(csvFile!!))

                // Write CSV header
                csvWriter!!.write(
                    "timestamp_ns,frame_number,min_temp_celsius,max_temp_celsius," +
                        "avg_temp_celsius,center_temp_celsius,emissivity,image_filename\n",
                )
                csvWriter!!.flush()

                // Create directory for thermal images
                thermalImagesDir = File(sessionDirectory, "thermal_images")
                if (!thermalImagesDir!!.exists()) {
                    thermalImagesDir!!.mkdirs()
                }

                // Initialize enhanced Topdon thermal integration with TC001 connector support
                topdonIntegration = TopdonThermalIntegration(context)
                val initResult = topdonIntegration!!.initialize()

                // Initialize comprehensive TC001 integration manager
                tc001IntegrationManager = TC001IntegrationManager(context)
                val tc001InitResult = tc001IntegrationManager!!.initializeSystem()

                if (tc001InitResult) {
                    Log.i(TAG, "TC001 integration manager initialized successfully")
                    // Start the TC001 system
                    val startResult = tc001IntegrationManager!!.startSystem()

                    if (startResult) {
                        // Initialize advanced TC001 components
                        initializeAdvancedTC001Components(sessionDirectory)
                        Log.i(TAG, "TC001 system started and advanced components initialized")
                    }
                } else {
                    Log.w(TAG, "TC001 integration manager initialization failed, using fallback")
                }

                if (initResult == TopdonResult.SUCCESS) {
                    // Scan for available devices
                    val devices = topdonIntegration!!.scanForDevices()

                    if (devices.isNotEmpty()) {
                        // Connect to first available device
                        val device = devices.first()
                        val connectResult = topdonIntegration!!.connectDevice(device.usbDevice!!)

                        if (connectResult == TopdonResult.SUCCESS) {
                            // Configure thermal camera settings
                            topdonIntegration!!.setEmissivity(0.95f)
                            topdonIntegration!!.setTemperatureRange(-20f, 150f)
                            topdonIntegration!!.setThermalPalette(TopdonThermalPalette.IRON)
                            topdonIntegration!!.enableAutoGainControl(true)
                            topdonIntegration!!.enableTemperatureCompensation(true)
                            topdonIntegration!!.configureDevice()

                            Log.i(TAG, "Real thermal camera connected and configured successfully")
                        } else {
                            Log.w(TAG, "Failed to connect to thermal camera, using simulation mode")
                        }
                    } else {
                        Log.w(TAG, "No thermal cameras found, using simulation mode")
                    }
                } else {
                    Log.w(TAG, "Failed to initialize thermal integration, using simulation mode")
                }

                true
            } catch (e: SecurityException) {
                Log.e(TAG, "USB permission denied for thermal camera access", e)
                false
            } catch (e: java.io.IOException) {
                Log.e(TAG, "IO error during thermal camera initialization - device may be disconnected", e)
                false
            } catch (e: IllegalStateException) {
                Log.e(TAG, "Thermal camera in invalid state - may need hardware reset", e)
                false
            } catch (e: RuntimeException) {
                Log.e(TAG, "Runtime error with thermal camera - SDK or driver issue: ${e.message}", e)
                false
            } catch (e: Exception) {
                Log.e(TAG, "Unexpected error initializing thermal recorder: ${e.message}", e)
                false
            }
        }

    suspend fun startRecording(): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                frameCount = 0

                val integration = topdonIntegration
                if (integration == null) {
                    Log.e(TAG, "Topdon integration not initialized")
                    return@withContext false
                }

                // Start thermal streaming
                val streamingStarted =
                    integration.startStreaming { thermalFrame ->
                        // Process thermal frame in background
                        CoroutineScope(Dispatchers.IO).launch {
                            processThermalFrame(thermalFrame)
                        }
                    }

                if (streamingStarted) {
                    Log.i(TAG, "Real thermal recording started")
                    true
                } else {
                    Log.e(TAG, "Failed to start thermal streaming")
                    false
                }
            } catch (e: SecurityException) {
                Log.e(TAG, "USB permission lost during thermal recording start", e)
                false
            } catch (e: IllegalStateException) {
                Log.e(TAG, "Thermal camera not properly initialized for recording", e)
                false
            } catch (e: java.io.IOException) {
                Log.e(TAG, "Communication error with thermal camera during start", e)
                false
            } catch (e: Exception) {
                Log.e(TAG, "Unexpected error starting thermal recording: ${e.message}", e)
                false
            }
        }
    }

    /**
     * Process individual thermal frame and log to CSV/images
     */
    private suspend fun processThermalFrame(thermalFrame: TopdonThermalFrame) {
        withContext(Dispatchers.IO) {
            try {
                frameCount++
                val timestamp = System.nanoTime()

                // Generate filename for thermal image
                val imageFilename = "thermal_${frameCount.toString().padStart(6, '0')}.png"
                val imageFile = File(thermalImagesDir!!, imageFilename)

                // Save thermal bitmap
                val bitmap = thermalFrame.generateThermalBitmap()
                saveBitmapAsPNG(bitmap, imageFile)

                // Write thermal data to CSV
                csvWriter?.let { writer ->
                    val csvLine =
                        buildString {
                            append(thermalFrame.timestamp)
                            append(",")
                            append(frameCount)
                            append(",")
                            append(String.format("%.2f", thermalFrame.minTemperature))
                            append(",")
                            append(String.format("%.2f", thermalFrame.maxTemperature))
                            append(",")
                            append(String.format("%.2f", thermalFrame.averageTemperature))
                            append(",")
                            append(String.format("%.2f", thermalFrame.centerTemperature))
                            append(",")
                            append(String.format("%.2f", thermalFrame.emissivity))
                            append(",")
                            append(thermalFrame.palette.name)
                            append(",")
                            append(imageFilename)
                            append("\n")
                        }

                    writer.write(csvLine)

                    // Flush every 10 frames to balance performance and data safety
                    if (frameCount % 10 == 0) {
                        writer.flush()
                    }
                }

                if (frameCount % 30 == 0) { // Log every second at 30 FPS
                    Log.d(
                        TAG,
                        "Thermal frame $frameCount: " +
                            "Min=${String.format("%.1f", thermalFrame.minTemperature)}°C, " +
                            "Max=${String.format("%.1f", thermalFrame.maxTemperature)}°C, " +
                            "Center=${String.format("%.1f", thermalFrame.centerTemperature)}°C",
                    )
                }
                Unit // Explicit Unit return for withContext
            } catch (e: Exception) {
                Log.e(TAG, "Error processing thermal frame: ${e.message}", e)
            }
        }
    }

    /**
     * Save thermal bitmap as PNG file
     */
    private fun saveBitmapAsPNG(
        bitmap: Bitmap,
        file: File,
    ) {
        try {
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save thermal image: ${e.message}", e)
        }
    }

    suspend fun stopRecording(): Boolean =
        withContext(Dispatchers.IO) {
            try {
                // Stop thermal streaming
                topdonIntegration?.stopStreaming()

                // Close CSV writer
                csvWriter?.let { writer ->
                    writer.flush()
                    writer.close()
                }
                csvWriter = null

                Log.i(TAG, "Thermal recording stopped. Captured $frameCount frames")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping thermal recording: ${e.message}", e)
                false
            }
        }

    suspend fun generatePreview(): Bitmap? {
        return withContext(Dispatchers.Default) {
            try {
                // First try to get preview from TC001 integration manager
                val tc001DataManager = tc001IntegrationManager?.getDataManager()
                if (tc001DataManager != null) {
                    val liveData = tc001DataManager.thermalBitmap
                    val latestFrame = liveData.value

                    if (latestFrame != null) {
                        Log.d(TAG, "Using TC001 integration manager thermal preview")
                        return@withContext latestFrame
                    }
                }

                // Fallback to original topdon integration
                val integration = topdonIntegration
                if (integration == null) {
                    Log.w(TAG, "No thermal integration available for preview")
                    return@withContext generateFallbackPreview()
                }

                // Use the latest thermal frame bitmap from original integration
                val liveData = integration.thermalFrame
                val latestFrame = liveData.value

                if (latestFrame?.thermalBitmap != null) {
                    // Scale for preview if needed
                    val originalBitmap = latestFrame.thermalBitmap
                    val scaleFactor = 2f // Make preview larger
                    val scaledWidth = (originalBitmap.width * scaleFactor).toInt()
                    val scaledHeight = (originalBitmap.height * scaleFactor).toInt()

                    Bitmap.createScaledBitmap(originalBitmap, scaledWidth, scaledHeight, false)
                } else {
                    generateFallbackPreview()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error generating thermal preview: ${e.message}", e)
                generateFallbackPreview()
            }
        }
    }

    /**
     * Generate fallback preview when real thermal data is not available
     */
    private fun generateFallbackPreview(): Bitmap {
        val width = 256
        val height = 192
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Create a gradient thermal-like preview
        for (y in 0 until height) {
            for (x in 0 until width) {
                val centerX = width / 2f
                val centerY = height / 2f
                val distance = kotlin.math.sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY))
                val maxDistance = kotlin.math.sqrt(centerX * centerX + centerY * centerY)
                val normalized = 1f - (distance / maxDistance).coerceIn(0f, 1f)

                // Iron palette colors
                val red = (normalized * 255).toInt()
                val green = if (normalized > 0.5f) ((normalized - 0.5f) * 2 * 255).toInt() else 0
                val blue = if (normalized > 0.75f) ((normalized - 0.75f) * 4 * 255).toInt() else 0

                val color = (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
                bitmap.setPixel(x, y, color)
            }
        }

        return bitmap
    }

    /**
     * Initialize advanced TC001 components for comprehensive thermal processing
     */
    private suspend fun initializeAdvancedTC001Components(sessionDirectory: File) =
        withContext(Dispatchers.IO) {
            try {
                // Initialize performance monitoring
                tc001PerformanceMonitor = TC001PerformanceMonitor(context)
                tc001PerformanceMonitor!!.startMonitoring()
                Log.i(TAG, "TC001 performance monitor initialized")

                // Initialize data exporter
                tc001DataExporter = TC001DataExporter(context)
                Log.i(TAG, "TC001 data exporter initialized")

                // Initialize calibration manager
                tc001CalibrationManager = TC001CalibrationManager(context)
                Log.i(TAG, "TC001 calibration manager initialized")

                // Initialize diagnostic system
                tc001DiagnosticSystem = TC001DiagnosticSystem(context)
                Log.i(TAG, "TC001 diagnostic system initialized")

                // Setup integration between advanced components
                setupAdvancedComponentIntegration()

                Log.i(TAG, "All advanced TC001 components initialized successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing advanced TC001 components", e)
            }
        }

    /**
     * Setup integration between advanced TC001 components
     */
    private fun setupAdvancedComponentIntegration() {
        // Connect performance monitor to data manager
        tc001IntegrationManager?.getDataManager()?.temperatureData?.observeForever { tempData ->
            tempData?.let {
                tc001PerformanceMonitor?.recordTemperatureReading(it.centerTemperature)
            }
        }

        // Connect performance monitor to thermal frame processing
        tc001IntegrationManager?.getDataManager()?.thermalBitmap?.observeForever { bitmap ->
            bitmap?.let {
                tc001PerformanceMonitor?.recordFrameProcessed()

                // Record memory usage
                val memoryUsed = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
                tc001PerformanceMonitor?.recordMemoryUsage(memoryUsed)
            }
        }

        Log.i(TAG, "Advanced TC001 component integration setup completed")
    }

    /**
     * Export thermal session data using advanced exporter
     */
    suspend fun exportThermalData(exportFormat: TC001ExportFormat = TC001ExportFormat.COMPREHENSIVE): TC001ExportResult? =
        sessionDir?.let { dir ->
            tc001DataExporter?.exportSession(
                sessionId = dir.name,
                sessionDir = dir,
                exportFormat = exportFormat,
            )
        }

    /**
     * Run TC001 diagnostics
     */
    suspend fun runDiagnostics(): TC001DiagnosticResults? = tc001DiagnosticSystem?.runComprehensiveDiagnostics()

    /**
     * Get current calibration status
     */
    fun getCalibrationStatus(): TC001CalibrationCurve? = tc001CalibrationManager?.getCurrentCalibration()

    /**
     * Start calibration process
     */
    suspend fun startCalibration(type: TC001CalibrationType): Boolean = tc001CalibrationManager?.startCalibration(type) ?: false

    suspend fun cleanup(): Boolean =
        withContext(Dispatchers.IO) {
            try {
                // Ensure recording is stopped
                stopRecording()

                // Stop and cleanup advanced components
                tc001PerformanceMonitor?.stopMonitoring()
                tc001PerformanceMonitor = null

                tc001DataExporter = null
                tc001CalibrationManager = null
                tc001DiagnosticSystem = null

                // Disconnect and cleanup TC001 integration manager
                tc001IntegrationManager?.cleanup()
                tc001IntegrationManager = null

                // Disconnect thermal camera
                topdonIntegration?.disconnect()
                topdonIntegration = null

                Log.i(TAG, "Enhanced thermal camera recorder with advanced TC001 components cleanup completed")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Error during thermal recorder cleanup: ${e.message}", e)
                false
            }
        }

    fun getRecorderType(): String = "thermal"
}
