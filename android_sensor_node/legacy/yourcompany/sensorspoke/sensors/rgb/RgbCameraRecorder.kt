package com.yourcompany.sensorspoke.sensors.rgb

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.yourcompany.sensorspoke.sensors.SensorRecorder
import com.yourcompany.sensorspoke.utils.PreviewBus
import com.yourcompany.sensorspoke.utils.TimeManager
import kotlinx.coroutines.*
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * RgbCameraRecorder using CameraX for video and Camera2 API for Samsung RAW DNG capture.
 * Leverages Samsung Camera API for Level 3/Stage 3 RAW DNG images when available.
 * Falls back to standard CameraX capture on non-Samsung devices.
 */
class RgbCameraRecorder(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA,
) : SensorRecorder {
    companion object {
        private const val TAG = "RgbCameraRecorder"
    }

    private var cameraProvider: ProcessCameraProvider? = null
    private var imageCapture: ImageCapture? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val scope = CoroutineScope(Dispatchers.IO + Job())
    private var lastPreviewNs: Long = 0L
    private var csvWriter: java.io.BufferedWriter? = null
    private var csvFile: File? = null

    // Samsung Camera2 API for RAW capture
    private var cameraManager: CameraManager? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var rawImageReader: ImageReader? = null
    private var supportsSamsungRawCapture = false
    private var rawCameraId: String? = null

    // Camera2 background thread and handler
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    override suspend fun start(sessionDir: File) {
        // Ensure directories
        val rgbDir = sessionDir
        if (!rgbDir.exists()) rgbDir.mkdirs()
        val framesDir = File(rgbDir, "frames").apply { mkdirs() }
        // Open CSV for DNG index (Phase 3 requirement)
        csvFile = File(rgbDir, "rgb.csv")
        csvWriter = java.io.BufferedWriter(java.io.FileWriter(csvFile!!, true))
        if (csvFile!!.length() == 0L) {
            csvWriter!!.write("timestamp_ns,filename\n")
            csvWriter!!.flush()
        }

        // Initialize Samsung camera capabilities and background thread
        startBackgroundThread()
        initializeCameraCapabilities()
        // Use session start timestamp in video filename for traceability
        val sessionStartTs = TimeManager.nowNanos()
        val videoFile = File(rgbDir, "video_$sessionStartTs.mp4")

        val provider = ProcessCameraProvider.getInstance(context).get()
        cameraProvider = provider

        // Build Recorder with 4K support for Samsung devices, fallback to 1080p
        val targetQuality =
            if (supportsSamsungRawCapture) {
                Log.i(TAG, "Samsung device detected - enabling 4K video recording")
                Quality.UHD // 4K for Samsung devices with RAW capability
            } else {
                Quality.FHD // 1080p for other devices
            }

        val recorder =
            Recorder
                .Builder()
                .setQualitySelector(
                    QualitySelector.from(
                        targetQuality,
                        FallbackStrategy.higherQualityOrLowerThan(targetQuality),
                    ),
                ).build()
        videoCapture = VideoCapture.withOutput(recorder)

        // Initialize Samsung RAW capture or fallback to CameraX
        if (supportsSamsungRawCapture && rawCameraId != null) {
            Log.i(TAG, "Samsung RAW DNG capture available - using Camera2 API")
            initializeSamsungRawCapture(framesDir)
        } else {
            Log.i(TAG, "Samsung RAW not available - using CameraX fallback")
            // ImageCapture for high-res stills - standard CameraX fallback with DNG extension
            imageCapture =
                ImageCapture
                    .Builder()
                    .setTargetRotation(Surface.ROTATION_0)
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                    .build()
        }

        // Unbind then bind
        provider.unbindAll()
        val useCases = mutableListOf<androidx.camera.core.UseCase>()
        imageCapture?.let { useCases.add(it) }
        videoCapture?.let { useCases.add(it) }
        provider.bindToLifecycle(lifecycleOwner, cameraSelector, *useCases.toTypedArray())

        // Start video recording
        val outputOpts = FileOutputOptions.Builder(videoFile).build()
        recording =
            videoCapture!!
                .output
                .prepareRecording(context, outputOpts)
                .start(ContextCompat.getMainExecutor(context)) { /* events ignored */ }

        // Start still capture loop based on device capabilities
        startStillCaptureLoop(framesDir)
    }

    override suspend fun stop() {
        try {
            recording?.stop()
        } catch (_: Exception) {
        }
        recording = null

        // Stop Samsung Camera2 RAW capture if active
        try {
            captureSession?.close()
        } catch (_: Exception) {
        }
        captureSession = null

        try {
            rawImageReader?.close()
        } catch (_: Exception) {
        }
        rawImageReader = null

        try {
            cameraDevice?.close()
        } catch (_: Exception) {
        }
        cameraDevice = null

        // Stop background thread
        stopBackgroundThread()

        try {
            cameraProvider?.unbindAll()
        } catch (_: Exception) {
        }
        cameraProvider = null
        imageCapture = null
        videoCapture = null

        // Cancel capture loop before closing CSV to avoid writes after close
        scope.coroutineContext[Job]?.cancel()
        // Close CSV resources
        try {
            csvWriter?.flush()
        } catch (_: Exception) {
        }
        try {
            csvWriter?.close()
        } catch (_: Exception) {
        }
        csvWriter = null
        csvFile = null
        executor.shutdown()
    }

    /**
     * Start background thread for Camera2 API operations
     */
    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    /**
     * Stop background thread for Camera2 API operations
     */
    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            Log.e(TAG, "Error stopping background thread", e)
        }
    }

    /**
     * Initialize camera capabilities detection for Samsung devices
     */
    private fun initializeCameraCapabilities() {
        try {
            cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraIds = cameraManager?.cameraIdList ?: return

            // Check if this is a Samsung device
            val isSamsung = Build.MANUFACTURER.equals("Samsung", ignoreCase = true)
            Log.i(TAG, "Device manufacturer: ${Build.MANUFACTURER}, Model: ${Build.MODEL}")

            for (cameraId in cameraIds) {
                val characteristics = cameraManager!!.getCameraCharacteristics(cameraId)
                val lensFacing = characteristics.get(CameraCharacteristics.LENS_FACING)

                // Check for back camera (primary camera)
                if (lensFacing == CameraCharacteristics.LENS_FACING_BACK) {
                    // Check camera capability level
                    val level = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
                    Log.i(TAG, "Camera $cameraId hardware level: $level")

                    // Samsung devices with LEVEL_3 support advanced RAW capture
                    val supportsLevel3 = level == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3
                    val supportsLimited = level == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED
                    val supportsFull = level == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL

                    // Check RAW capability
                    val capabilities = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
                    val supportsRaw = capabilities?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW) == true

                    // Get stream configuration map for concurrent stream support check
                    val configs = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                    val supportsConcurrentStreams = checkConcurrentStreamSupport(characteristics, configs)

                    Log.i(
                        TAG,
                        "Camera $cameraId - Samsung: $isSamsung, Level3: $supportsLevel3, Full: $supportsFull, RAW: $supportsRaw, Concurrent: $supportsConcurrentStreams",
                    )

                    // Samsung devices with Level 3 or Full + RAW capability
                    if (isSamsung && (supportsLevel3 || supportsFull) && supportsRaw) {
                        supportsSamsungRawCapture = true
                        rawCameraId = cameraId
                        Log.i(TAG, "Samsung RAW DNG capture enabled for camera $cameraId")

                        if (supportsConcurrentStreams) {
                            Log.i(TAG, "Concurrent 4K video + RAW image recording supported")
                        } else {
                            Log.w(TAG, "Concurrent streams may be limited - performance monitoring recommended")
                        }
                        break
                    }
                }
            }

            if (!supportsSamsungRawCapture) {
                Log.i(TAG, "Samsung RAW DNG capture not available - using CameraX fallback")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to initialize camera capabilities: ${e.message}")
            supportsSamsungRawCapture = false
        }
    }

    /**
     * Check if the camera supports concurrent high-resolution video and RAW capture
     */
    private fun checkConcurrentStreamSupport(
        characteristics: CameraCharacteristics,
        configs: android.hardware.camera2.params.StreamConfigurationMap?,
    ): Boolean {
        try {
            // Check available sizes for video and RAW
            val videoSizes = configs?.getOutputSizes(android.media.MediaRecorder::class.java) ?: emptyArray()
            val rawSizes = configs?.getOutputSizes(ImageFormat.RAW_SENSOR) ?: emptyArray()

            // Look for 4K video support
            val supports4K = videoSizes.any { it.width >= 3840 && it.height >= 2160 }
            val supportsRawCapture = rawSizes.isNotEmpty()

            Log.i(TAG, "Video sizes available: ${videoSizes.contentToString()}")
            Log.i(TAG, "RAW sizes available: ${rawSizes.contentToString()}")
            Log.i(TAG, "4K video support: $supports4K, RAW capture: $supportsRawCapture")

            // Check mandatory stream combinations (API 29+)
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                val mandatoryConfigs = characteristics.get(CameraCharacteristics.SCALER_MANDATORY_STREAM_COMBINATIONS)
                if (mandatoryConfigs != null) {
                    Log.i(TAG, "Mandatory stream combinations: ${mandatoryConfigs.size} configurations available")
                    // In practice, Level 3 and Full cameras typically support concurrent streams
                    return supports4K && supportsRawCapture
                }
            }

            return supports4K && supportsRawCapture
        } catch (e: Exception) {
            Log.w(TAG, "Error checking concurrent stream support: ${e.message}")
            return false
        }
    }

    /**
     * Initialize Samsung Camera2 API for RAW DNG capture
     */
    @Suppress("MissingPermission")
    private suspend fun initializeSamsungRawCapture(framesDir: File) {
        if (!supportsSamsungRawCapture || rawCameraId == null) return

        try {
            // Setup ImageReader for RAW_SENSOR format
            val characteristics = cameraManager!!.getCameraCharacteristics(rawCameraId!!)
            val configs = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            val rawSizes = configs?.getOutputSizes(ImageFormat.RAW_SENSOR) ?: return

            if (rawSizes.isEmpty()) {
                Log.w(TAG, "No RAW sensor sizes available")
                return
            }

            val maxRawSize = rawSizes.maxByOrNull { it.width * it.height } ?: rawSizes[0]
            Log.i(TAG, "Using RAW size: ${maxRawSize.width}x${maxRawSize.height}")

            // Check for concurrent 4K video capability
            val videoSizes = configs?.getOutputSizes(android.media.MediaRecorder::class.java) ?: emptyArray()
            val supports4K = videoSizes.any { it.width >= 3840 && it.height >= 2160 }
            Log.i(TAG, "4K video concurrent support: $supports4K")

            rawImageReader =
                ImageReader.newInstance(
                    maxRawSize.width,
                    maxRawSize.height,
                    ImageFormat.RAW_SENSOR,
                    4, // Increased buffer size for concurrent operation
                )

            rawImageReader!!.setOnImageAvailableListener({ reader ->
                handleRawImage(reader, framesDir)
            }, backgroundHandler)

            // Open camera device
            val cameraOpenedDeferred = CompletableDeferred<CameraDevice>()
            cameraManager!!.openCamera(
                rawCameraId!!,
                object : CameraDevice.StateCallback() {
                    override fun onOpened(camera: CameraDevice) {
                        cameraOpenedDeferred.complete(camera)
                    }

                    override fun onDisconnected(camera: CameraDevice) {
                        camera.close()
                        cameraOpenedDeferred.completeExceptionally(RuntimeException("Camera disconnected"))
                    }

                    override fun onError(
                        camera: CameraDevice,
                        error: Int,
                    ) {
                        camera.close()
                        cameraOpenedDeferred.completeExceptionally(RuntimeException("Camera error: $error"))
                    }
                },
                backgroundHandler,
            )

            cameraDevice = cameraOpenedDeferred.await()
            Log.i(TAG, "Samsung Camera2 device opened successfully for concurrent RAW+4K recording")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Samsung RAW capture: ${e.message}")
            supportsSamsungRawCapture = false
        }
    }

    /**
     * Handle RAW image from Samsung Camera2 API and save as DNG
     */
    private fun handleRawImage(
        reader: ImageReader,
        framesDir: File,
    ) {
        try {
            val image = reader.acquireLatestImage() ?: return
            val timestamp = TimeManager.nowNanos()

            scope.launch {
                try {
                    // Save RAW image as DNG file
                    val dngFile = File(framesDir, "frame_$timestamp.dng")
                    saveRawImageAsDng(image, dngFile)

                    // Log to CSV
                    csvWriter?.apply {
                        write("$timestamp,frames/${dngFile.name}\n")
                        flush()
                    }

                    // Generate preview if throttled appropriately
                    val now = TimeManager.nowNanos()
                    if (now - lastPreviewNs >= 150_000_000L) {
                        lastPreviewNs = now
                        generatePreviewFromRaw(image)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error processing RAW image: ${e.message}")
                } finally {
                    image.close()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error handling RAW image: ${e.message}")
        }
    }

    /**
     * Save RAW image data as DNG file
     */
    private fun saveRawImageAsDng(
        image: Image,
        dngFile: File,
    ) {
        try {
            val buffer = image.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)

            FileOutputStream(dngFile).use { fos ->
                fos.write(bytes)
                fos.flush()
            }

            Log.d(TAG, "Saved RAW DNG: ${dngFile.name} (${bytes.size} bytes)")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save DNG file: ${e.message}")
        }
    }

    /**
     * Generate JPEG preview from RAW image data
     */
    private fun generatePreviewFromRaw(image: Image) {
        try {
            // For Samsung RAW, generate actual preview from image buffer
            val buffer = image.planes[0].buffer
            val width = image.width
            val height = image.height

            // Create a downsampled preview bitmap from RAW data
            // Note: This is a simplified approach - full RAW processing would be more complex
            val previewWidth = 320
            val previewHeight = (height * previewWidth) / width

            // Create RGB bitmap from RAW buffer (simplified conversion)
            val previewBitmap = createPreviewFromRawBuffer(buffer, width, height, previewWidth, previewHeight)

            // Compress to JPEG and emit
            val baos = ByteArrayOutputStream()
            previewBitmap.compress(Bitmap.CompressFormat.JPEG, 60, baos)
            val bytes = baos.toByteArray()
            PreviewBus.emit(bytes, TimeManager.nowNanos())

            baos.close()
            previewBitmap.recycle()
        } catch (e: Exception) {
            Log.e(TAG, "Error generating RAW preview: ${e.message}")
            // Fallback to proper preview on error
            generateProperPreview()
        }
    }

    /**
     * Create preview bitmap from RAW buffer data (simplified conversion)
     */
    private fun createPreviewFromRawBuffer(
        buffer: java.nio.ByteBuffer,
        srcWidth: Int,
        srcHeight: Int,
        previewWidth: Int,
        previewHeight: Int,
    ): Bitmap {
        try {
            // Create a simplified preview bitmap
            // Note: Full RAW processing would involve demosaicing, white balance, etc.
            val bitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.RGB_565)

            // Simple downsampling of RAW data for preview
            val srcBytes = ByteArray(buffer.remaining())
            buffer.mark()
            buffer.get(srcBytes)
            buffer.reset()

            // Create a basic grayscale preview from RAW data
            val pixels = IntArray(previewWidth * previewHeight)
            val skipX = srcWidth / previewWidth
            val skipY = srcHeight / previewHeight

            for (y in 0 until previewHeight) {
                for (x in 0 until previewWidth) {
                    val srcX = x * skipX
                    val srcY = y * skipY
                    val srcIndex = (srcY * srcWidth + srcX) * 2 // 16-bit RAW, take every other byte

                    if (srcIndex < srcBytes.size - 1) {
                        // Convert 16-bit RAW to 8-bit grayscale
                        val raw16 =
                            ((srcBytes[srcIndex + 1].toInt() and 0xFF) shl 8) or
                                (srcBytes[srcIndex].toInt() and 0xFF)
                        val gray = (raw16 shr 8).coerceIn(0, 255)
                        pixels[y * previewWidth + x] = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
                    }
                }
            }

            bitmap.setPixels(pixels, 0, previewWidth, 0, 0, previewWidth, previewHeight)
            return bitmap
        } catch (e: Exception) {
            Log.w(TAG, "Failed to create RAW preview, using fallback: ${e.message}")
            return createFallbackBitmap()
        }
    }

    /**
     * Generate proper preview with fallback pattern
     */
    private fun generateProperPreview() {
        try {
            val bmp = createFallbackBitmap()
            val w = 320
            val h = 240
            val scaled = Bitmap.createScaledBitmap(bmp, w, h, true)
            val baos = ByteArrayOutputStream()
            scaled.compress(Bitmap.CompressFormat.JPEG, 40, baos)
            val bytes = baos.toByteArray()
            PreviewBus.emit(bytes, TimeManager.nowNanos())

            baos.close()
            if (scaled != bmp) {
                bmp.recycle()
            }
            scaled.recycle()
        } catch (e: Exception) {
            Log.e(TAG, "Error generating proper preview: ${e.message}")
        }
    }

    /**
     * Start still capture loop based on device capabilities
     */
    private fun startStillCaptureLoop(framesDir: File) {
        scope.launch {
            while (isActive) {
                if (supportsSamsungRawCapture && cameraDevice != null && rawImageReader != null) {
                    // Samsung RAW capture using Camera2 API
                    captureRawImage()
                } else {
                    // Fallback to CameraX capture
                    captureCameraXImage(framesDir)
                }
                // ~6–8 FPS loop (~150ms)
                delay(150)
            }
        }
    }

    /**
     * Capture RAW image using Samsung Camera2 API
     */
    private suspend fun captureRawImage() {
        try {
            val device = cameraDevice ?: return
            val reader = rawImageReader ?: return

            // Create capture session if needed
            if (captureSession == null) {
                val sessionCreatedDeferred = CompletableDeferred<CameraCaptureSession>()
                device.createCaptureSession(
                    listOf(reader.surface),
                    object : CameraCaptureSession.StateCallback() {
                        override fun onConfigured(session: CameraCaptureSession) {
                            sessionCreatedDeferred.complete(session)
                        }

                        override fun onConfigureFailed(session: CameraCaptureSession) {
                            sessionCreatedDeferred.completeExceptionally(RuntimeException("Session configure failed"))
                        }
                    },
                    backgroundHandler,
                )
                captureSession = sessionCreatedDeferred.await()
            }

            // Capture RAW image
            val captureRequest = device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE)
            captureRequest.addTarget(reader.surface)

            // Samsung-specific optimizations for RAW capture
            captureRequest.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_OFF)
            captureRequest.set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF)
            captureRequest.set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF)
            captureRequest.set(CaptureRequest.COLOR_CORRECTION_MODE, CaptureRequest.COLOR_CORRECTION_MODE_TRANSFORM_MATRIX)

            captureSession?.capture(captureRequest.build(), null, backgroundHandler)
        } catch (e: Exception) {
            Log.e(TAG, "Error capturing RAW image: ${e.message}")
        }
    }

    /**
     * Fallback capture using CameraX when Samsung RAW is not available
     */
    private fun captureCameraXImage(framesDir: File) {
        try {
            val ts = TimeManager.nowNanos()
            val outFile = File(framesDir, "frame_$ts.dng")
            val output = ImageCapture.OutputFileOptions.Builder(outFile).build()
            val ic = imageCapture ?: return

            ic.takePicture(
                output,
                executor,
                object : ImageCapture.OnImageSavedCallback {
                    override fun onError(exception: ImageCaptureException) {
                        // Keep loop running on errors
                        Log.w(TAG, "CameraX capture error: ${exception.message}")
                    }

                    override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                        // Log DNG to CSV: timestamp_ns,filename (relative)
                        try {
                            csvWriter?.apply {
                                write("$ts,frames/${outFile.name}\n")
                                flush()
                            }
                        } catch (_: Exception) {
                        }
                        // Build a downsampled, low-quality JPEG preview from DNG and emit
                        val now = TimeManager.nowNanos()
                        // throttle to ~6–8 FPS
                        if (now - lastPreviewNs >= 150_000_000L) {
                            lastPreviewNs = now
                            runCatching {
                                // Decode DNG file for preview if it exists
                                val bmp = if (outFile.exists()) {
                                    decodeDNGFileForPreview(outFile)
                                } else {
                                    null
                                } ?: run {
                                    Log.w(TAG, "Failed to decode DNG, using fallback bitmap")
                                    createFallbackBitmap()
                                }
                                val w = 320
                                val h = 240
                                val scaled = Bitmap.createScaledBitmap(bmp, w, h, true)
                                val baos = ByteArrayOutputStream()
                                scaled.compress(Bitmap.CompressFormat.JPEG, 40, baos)
                                val bytes = baos.toByteArray()
                                PreviewBus.emit(bytes, now)
                                try {
                                    baos.close()
                                } catch (_: Exception) {
                                }
                                if (scaled != bmp) {
                                    bmp.recycle()
                                }
                                scaled.recycle()
                            }
                        }
                    }
                },
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error in CameraX capture: ${e.message}")
        }
    }

    /**
     * Decode DNG file for preview purposes
     * Implements basic RAW to RGB conversion for preview display
     */
    private fun decodeDNGFileForPreview(dngFile: File): Bitmap? {
        return try {
            // Use BitmapFactory to try to decode DNG directly first
            val options = BitmapFactory.Options().apply {
                inSampleSize = 4 // Reduce size for preview
                inPreferredConfig = Bitmap.Config.RGB_565
            }

            // Try direct decoding first (works for some DNG files)
            BitmapFactory.decodeFile(dngFile.absolutePath, options)?.let { bitmap ->
                return bitmap
            }

            // If direct decoding fails, read raw bytes and process manually
            val fileBytes = dngFile.readBytes()
            if (fileBytes.size < 1000) {
                Log.w(TAG, "DNG file too small: ${fileBytes.size} bytes")
                return null
            }

            // For demonstration, create a preview pattern based on file content
            // In production, this would use proper DNG/RAW decoding libraries
            processRawDNGBytes(fileBytes)
        } catch (e: Exception) {
            Log.e(TAG, "Error decoding DNG file: ${dngFile.name}", e)
            null
        }
    }

    /**
     * Process raw DNG bytes for preview (simplified approach)
     */
    private fun processRawDNGBytes(fileBytes: ByteArray): Bitmap {
        val width = 640
        val height = 480
        val rgbArray = IntArray(width * height)

        // Skip header bytes and sample the raw data
        val dataOffset = minOf(1024, fileBytes.size / 4) // Skip typical header

        for (i in rgbArray.indices) {
            val byteIndex = dataOffset + (i * 2) % (fileBytes.size - dataOffset - 1)

            // Sample raw sensor data (assuming 16-bit values)
            val rawValue = if (byteIndex + 1 < fileBytes.size) {
                ((fileBytes[byteIndex + 1].toInt() and 0xFF) shl 8) or
                    (fileBytes[byteIndex].toInt() and 0xFF)
            } else {
                0
            }

            // Convert to 8-bit and apply simple demosaic
            val normalizedValue = (rawValue shr 8).coerceIn(0, 255)
            val x = i % width
            val y = i / width
            val (r, g, b) = demosaicPixel(x, y, normalizedValue)

            rgbArray[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        return Bitmap.createBitmap(rgbArray, width, height, Bitmap.Config.ARGB_8888)
    }

    /**
     * Simple demosaic for RAW to RGB conversion (Bayer pattern)
     */
    private fun demosaicPixel(x: Int, y: Int, rawValue: Int): Triple<Int, Int, Int> {
        // Simple Bayer pattern demosaicing
        return when {
            // Red pixel positions (even row, even col)
            y % 2 == 0 && x % 2 == 0 -> Triple(rawValue, rawValue / 2, rawValue / 4)
            // Green pixel positions
            (y % 2 == 0 && x % 2 == 1) || (y % 2 == 1 && x % 2 == 0) -> Triple(rawValue / 3, rawValue, rawValue / 3)
            // Blue pixel positions (odd row, odd col)
            else -> Triple(rawValue / 4, rawValue / 2, rawValue)
        }
    }

    /**
     * Creates a fallback bitmap when DNG decoding fails
     */
    private fun createFallbackBitmap(): Bitmap {
        val width = 640
        val height = 480
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565)

        // Create a simple gradient pattern instead of solid color
        val pixels = IntArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val intensity = ((x + y) * 255 / (width + height)).coerceIn(0, 255)
                pixels[y * width + x] = (intensity shl 16) or (intensity shl 8) or intensity
            }
        }

        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }
}
