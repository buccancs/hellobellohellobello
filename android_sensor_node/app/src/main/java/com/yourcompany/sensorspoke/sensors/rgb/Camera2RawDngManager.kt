package com.yourcompany.sensorspoke.sensors.rgb

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.hardware.camera2.params.StreamConfigurationMap
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.Surface
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume

/**
 * Camera2RawDngManager provides RAW DNG capture functionality using Camera2 API
 * specifically for Samsung devices supporting Camera2 Level 3 (Stage3/Level3 processing).
 *
 * This manager handles:
 * - Samsung device detection and Camera2 Level 3 support verification
 * - RAW DNG capture with Stage3/Level3 processing pipeline
 * - High-quality professional RAW image capture
 * - Integration with existing RGB camera system
 */
class Camera2RawDngManager(
    private val context: Context
) {
    companion object {
        private const val TAG = "Camera2RawDngManager"
        
        // Samsung devices known to support Camera2 Level 3 with excellent RAW DNG
        private val SAMSUNG_CAMERA2_LEVEL3_DEVICES = setOf(
            // Galaxy S22 series
            "SM-S901", "SM-S906", "SM-S908",  // S22, S22+, S22 Ultra
            "SM-S901B", "SM-S901E", "SM-S901U", "SM-S901W",
            "SM-S906B", "SM-S906E", "SM-S906U", "SM-S906W", 
            "SM-S908B", "SM-S908E", "SM-S908U", "SM-S908W",
            
            // Galaxy S23 series
            "SM-S911", "SM-S916", "SM-S918",  // S23, S23+, S23 Ultra
            
            // Galaxy S24 series
            "SM-S921", "SM-S926", "SM-S928",  // S24, S24+, S24 Ultra
            
            // Galaxy Note series (recent)
            "SM-N981", "SM-N986",  // Note 20, Note 20 Ultra
            
            // Galaxy S21 series (Level 3 capable)
            "SM-G991", "SM-G996", "SM-G998"   // S21, S21+, S21 Ultra
        )
        
        // Maximum wait time for camera operations
        private const val CAMERA_OPEN_TIMEOUT_MS = 2500L
    }
    
    // Camera state management
    private val _cameraState = MutableStateFlow(Camera2State.UNINITIALIZED)
    val cameraState: StateFlow<Camera2State> = _cameraState.asStateFlow()
    
    private val _deviceInfo = MutableStateFlow<Camera2DeviceInfo?>(null)
    val deviceInfo: StateFlow<Camera2DeviceInfo?> = _deviceInfo.asStateFlow()
    
    // Camera2 components
    private var cameraManager: CameraManager? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null
    private var imageReader: ImageReader? = null
    
    // Synchronization primitives
    private val cameraOpenCloseLock = Semaphore(1)
    
    /**
     * Camera2 states for RAW DNG management
     */
    enum class Camera2State {
        UNINITIALIZED,
        INITIALIZING, 
        READY,
        CAPTURING,
        ERROR
    }
    
    /**
     * Camera2 device information with Samsung-specific capabilities
     */
    data class Camera2DeviceInfo(
        val deviceModel: String,
        val isSamsungDevice: Boolean,
        val supportsCamera2Level3: Boolean,
        val supportsRawDng: Boolean,
        val backCameraId: String?,
        val frontCameraId: String?,
        val maxRawSize: Size?,
        val availableRawFormats: List<Int>,
        val hardwareLevel: Int,
        val capabilities: Set<Int>
    )
    
    /**
     * Initialize Camera2 RAW DNG manager with Samsung device detection
     */
    suspend fun initialize(): Boolean {
        return try {
            Log.i(TAG, "Initializing Camera2 RAW DNG manager")
            _cameraState.value = Camera2State.INITIALIZING
            
            // Check camera permission
            if (!hasRequiredPermissions()) {
                Log.e(TAG, "Missing required camera permissions")
                _cameraState.value = Camera2State.ERROR
                return false
            }
            
            // Initialize Camera2 manager
            cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            
            // Start background thread for camera operations
            startBackgroundThread()
            
            // Detect Samsung device and Camera2 Level 3 support
            val deviceInfo = detectCamera2Capabilities()
            _deviceInfo.value = deviceInfo
            
            if (!deviceInfo.supportsRawDng) {
                Log.w(TAG, "Device does not support RAW DNG capture")
                _cameraState.value = Camera2State.ERROR
                return false
            }
            
            _cameraState.value = Camera2State.READY
            Log.i(TAG, "Camera2 RAW DNG manager initialized successfully")
            Log.i(TAG, "Device: ${deviceInfo.deviceModel}, Camera2 Level 3: ${deviceInfo.supportsCamera2Level3}")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Camera2 RAW DNG manager: ${e.message}", e)
            _cameraState.value = Camera2State.ERROR
            false
        }
    }
    
    /**
     * Detect Camera2 capabilities with focus on Samsung Level 3 support
     */
    private fun detectCamera2Capabilities(): Camera2DeviceInfo {
        val deviceModel = Build.MODEL.uppercase()
        val isSamsungDevice = Build.MANUFACTURER.uppercase() == "SAMSUNG"
        val isSamsungLevel3Device = SAMSUNG_CAMERA2_LEVEL3_DEVICES.any { model ->
            deviceModel.startsWith(model.uppercase())
        }
        
        val manager = cameraManager ?: throw RuntimeException("CameraManager not initialized")
        
        var backCameraId: String? = null
        var frontCameraId: String? = null
        var maxRawSize: Size? = null
        var availableRawFormats = emptyList<Int>()
        var hardwareLevel = CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY
        var capabilities = emptySet<Int>()
        var supportsRawDng = false
        var supportsCamera2Level3 = false
        
        try {
            // Enumerate cameras and find back camera with best RAW support
            for (cameraId in manager.cameraIdList) {
                val characteristics = manager.getCameraCharacteristics(cameraId)
                val lensFacing = characteristics.get(CameraCharacteristics.LENS_FACING)
                val hwLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL) ?: 
                    CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY
                val caps = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf()
                
                // Check for back camera
                if (lensFacing == CameraMetadata.LENS_FACING_BACK) {
                    backCameraId = cameraId
                    hardwareLevel = hwLevel
                    capabilities = caps.toSet()
                    
                    // Check Camera2 Level 3 support
                    supportsCamera2Level3 = hwLevel == CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_3 ||
                        caps.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) &&
                        caps.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_POST_PROCESSING) &&
                        caps.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_RAW)
                    
                    // Get stream configuration map for RAW formats
                    val streamMap = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                    if (streamMap != null) {
                        // Check RAW format support
                        val rawFormats = mutableListOf<Int>()
                        
                        if (streamMap.isOutputSupportedFor(ImageFormat.RAW_SENSOR)) {
                            rawFormats.add(ImageFormat.RAW_SENSOR)
                            val rawSizes = streamMap.getOutputSizes(ImageFormat.RAW_SENSOR)
                            if (rawSizes.isNotEmpty()) {
                                maxRawSize = rawSizes.maxByOrNull { it.width * it.height }
                            }
                        }
                        
                        if (streamMap.isOutputSupportedFor(ImageFormat.RAW10)) {
                            rawFormats.add(ImageFormat.RAW10)
                        }
                        
                        if (streamMap.isOutputSupportedFor(ImageFormat.RAW12)) {
                            rawFormats.add(ImageFormat.RAW12)
                        }
                        
                        availableRawFormats = rawFormats
                        supportsRawDng = rawFormats.isNotEmpty() && maxRawSize != null
                    }
                    
                    break // Use first back camera found
                }
                
                // Also check for front camera
                if (lensFacing == CameraMetadata.LENS_FACING_FRONT && frontCameraId == null) {
                    frontCameraId = cameraId
                }
            }
            
            Log.i(TAG, "Camera2 detection complete:")
            Log.i(TAG, "  Device: $deviceModel, Samsung: $isSamsungDevice")
            Log.i(TAG, "  Hardware Level: $hardwareLevel (Level 3: $supportsCamera2Level3)")
            Log.i(TAG, "  RAW Support: $supportsRawDng, Max RAW Size: $maxRawSize")
            Log.i(TAG, "  RAW Formats: $availableRawFormats")
            Log.i(TAG, "  Capabilities: ${capabilities.joinToString()}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error detecting Camera2 capabilities: ${e.message}", e)
        }
        
        return Camera2DeviceInfo(
            deviceModel = deviceModel,
            isSamsungDevice = isSamsungDevice,
            supportsCamera2Level3 = supportsCamera2Level3 && isSamsungLevel3Device,
            supportsRawDng = supportsRawDng,
            backCameraId = backCameraId,
            frontCameraId = frontCameraId,
            maxRawSize = maxRawSize,
            availableRawFormats = availableRawFormats,
            hardwareLevel = hardwareLevel,
            capabilities = capabilities
        )
    }
    
    /**
     * Start RAW DNG capture session (persistent camera connection)
     */
    suspend fun startRawCaptureSession(): Boolean {
        val deviceInfo = _deviceInfo.value ?: return false
        
        if (!deviceInfo.supportsRawDng || deviceInfo.backCameraId == null) {
            Log.e(TAG, "Device does not support RAW DNG capture")
            return false
        }
        
        if (_cameraState.value != Camera2State.READY && _cameraState.value != Camera2State.UNINITIALIZED) {
            Log.w(TAG, "RAW capture session already active or in error state")
            return false
        }
        
        return try {
            _cameraState.value = Camera2State.INITIALIZING
            Log.i(TAG, "Starting persistent RAW DNG capture session")
            
            // Open camera for persistent session
            if (!openCamera(deviceInfo.backCameraId)) {
                _cameraState.value = Camera2State.ERROR
                return false
            }
            
            // Setup ImageReader for RAW capture (persistent)
            val rawSize = deviceInfo.maxRawSize ?: return false
            val rawFormat = deviceInfo.availableRawFormats.firstOrNull() ?: return false
            
            imageReader = ImageReader.newInstance(
                rawSize.width, 
                rawSize.height,
                rawFormat,
                2  // Buffer for 2 images to prevent blocking
            )
            
            // Create persistent capture session
            val reader = imageReader
            val device = cameraDevice
            
            if (reader == null || device == null) {
                Log.e(TAG, "ImageReader or CameraDevice is null")
                _cameraState.value = Camera2State.ERROR
                return false
            }
            
            val surfaces = listOf(reader.surface)
            createCaptureSession(surfaces) { session ->
                captureSession = session
                _cameraState.value = Camera2State.READY
                Log.i(TAG, "RAW DNG capture session established successfully")
            }
            
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Error starting RAW DNG capture session: ${e.message}", e)
            _cameraState.value = Camera2State.ERROR
            cleanup()
            false
        }
    }
    
    /**
     * Stop RAW DNG capture session and cleanup resources
     */
    fun stopRawCaptureSession() {
        Log.i(TAG, "Stopping RAW DNG capture session")
        cleanup()
        _cameraState.value = Camera2State.UNINITIALIZED
    }
    
    /**
     * Capture single RAW DNG frame using persistent session (efficient)
     */
    suspend fun captureRawDngFrame(outputFile: File): Boolean {
        val session = captureSession
        val device = cameraDevice
        val reader = imageReader
        
        if (session == null || device == null || reader == null) {
            Log.e(TAG, "RAW capture session not active")
            return false
        }
        
        if (_cameraState.value != Camera2State.READY) {
            Log.w(TAG, "Camera not ready for capture")
            return false
        }
        
        return try {
            _cameraState.value = Camera2State.CAPTURING
            
            // Use suspendCancellableCoroutine to properly wait for capture completion
            suspendCancellableCoroutine { continuation ->
                // Setup image listener for this specific capture
                reader.setOnImageAvailableListener({ imageReader ->
                    imageReader.acquireLatestImage()?.use { image ->
                        try {
                            // Save RAW DNG data
                            val buffer = image.planes[0].buffer
                            val bytes = ByteArray(buffer.remaining())
                            buffer.get(bytes)
                            
                            FileOutputStream(outputFile).use { fos ->
                                fos.write(bytes)
                            }
                            
                            Log.d(TAG, "RAW DNG frame saved: ${outputFile.name} (${bytes.size} bytes)")
                        } catch (e: Exception) {
                            Log.e(TAG, "Error saving RAW DNG frame: ${e.message}", e)
                        }
                    }
                }, backgroundHandler)
                
                // Build capture request with Samsung Stage3/Level3 optimizations
                val captureRequestBuilder = device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE)
                captureRequestBuilder.addTarget(reader.surface)
                
                captureRequestBuilder.apply {
                    // Manual sensor controls for highest quality
                    set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
                    set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
                    set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_OFF)
                    
                    // RAW format processing - disable all post-processing
                    set(CaptureRequest.EDGE_MODE, CameraMetadata.EDGE_MODE_OFF)
                    set(CaptureRequest.NOISE_REDUCTION_MODE, CameraMetadata.NOISE_REDUCTION_MODE_OFF)
                    set(CaptureRequest.HOT_PIXEL_MODE, CameraMetadata.HOT_PIXEL_MODE_OFF)
                }
                
                // Handle cancellation
                continuation.invokeOnCancellation {
                    Log.w(TAG, "RAW DNG capture cancelled")
                }
                
                // Capture the RAW DNG frame
                session.capture(
                    captureRequestBuilder.build(),
                    object : CameraCaptureSession.CaptureCallback() {
                        override fun onCaptureCompleted(
                            session: CameraCaptureSession,
                            request: CaptureRequest,
                            result: TotalCaptureResult
                        ) {
                            _cameraState.value = Camera2State.READY
                            if (continuation.isActive) {
                                continuation.resume(true)
                            }
                        }
                        
                        override fun onCaptureFailed(
                            session: CameraCaptureSession,
                            request: CaptureRequest,
                            failure: CaptureFailure
                        ) {
                            Log.e(TAG, "RAW DNG capture failed: ${failure.reason}")
                            _cameraState.value = Camera2State.READY
                            if (continuation.isActive) {
                                continuation.resume(false)
                            }
                        }
                    },
                    backgroundHandler
                )
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during RAW DNG frame capture: ${e.message}", e)
            _cameraState.value = Camera2State.ERROR
            false
        }
    }
    
    /**
     * Legacy method for backward compatibility (now uses persistent session internally)
     */
    suspend fun captureRawDng(outputFile: File): Boolean {
        Log.w(TAG, "Using legacy captureRawDng - consider using persistent session for better performance")
        
        // Start session, capture frame, then stop (inefficient but compatible)
        return if (startRawCaptureSession()) {
            val result = captureRawDngFrame(outputFile)
            stopRawCaptureSession()
            result
        } else {
            false
        }
    }
    
    /**
     * Check if device has required permissions
     */
    private fun hasRequiredPermissions(): Boolean {
        return ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    /**
     * Start background thread for camera operations
     */
    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("Camera2Background").also { thread ->
            thread.start()
            backgroundHandler = Handler(thread.looper)
        }
    }
    
    /**
     * Stop background thread
     */
    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            Log.e(TAG, "Error stopping background thread: ${e.message}", e)
        }
    }
    
    /**
     * Open camera device using suspendCancellableCoroutine to properly handle async callbacks
     */
    private suspend fun openCamera(cameraId: String): Boolean {
        val manager = cameraManager ?: return false
        
        return try {
            if (!cameraOpenCloseLock.tryAcquire(CAMERA_OPEN_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
                throw RuntimeException("Camera open timeout")
            }
            
            suspendCancellableCoroutine { continuation ->
                val callback = object : CameraDevice.StateCallback() {
                    override fun onOpened(device: CameraDevice) {
                        cameraDevice = device
                        cameraOpenCloseLock.release()
                        Log.i(TAG, "Camera opened successfully: $cameraId")
                        continuation.resume(true)
                    }
                    
                    override fun onDisconnected(device: CameraDevice) {
                        cameraDevice = null
                        device.close()
                        cameraOpenCloseLock.release()
                        Log.w(TAG, "Camera disconnected: $cameraId")
                        if (continuation.isActive) {
                            continuation.resume(false)
                        }
                    }
                    
                    override fun onError(device: CameraDevice, error: Int) {
                        cameraDevice = null
                        device.close()
                        cameraOpenCloseLock.release()
                        Log.e(TAG, "Camera error: $error for camera: $cameraId")
                        if (continuation.isActive) {
                            continuation.resume(false)
                        }
                    }
                }
                
                continuation.invokeOnCancellation {
                    cameraDevice?.close()
                    cameraDevice = null
                    cameraOpenCloseLock.release()
                }
                
                manager.openCamera(cameraId, callback, backgroundHandler)
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error opening camera: ${e.message}", e)
            cameraOpenCloseLock.release()
            false
        }
    }
    
    /**
     * Create capture session
     */
    private fun createCaptureSession(
        surfaces: List<Surface>,
        onSessionConfigured: (CameraCaptureSession) -> Unit
    ) {
        val device = cameraDevice ?: return
        
        device.createCaptureSession(
            surfaces,
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    onSessionConfigured(session)
                }
                
                override fun onConfigureFailed(session: CameraCaptureSession) {
                    Log.e(TAG, "Failed to configure capture session")
                }
            },
            backgroundHandler
        )
    }
    
    /**
     * Close camera and cleanup resources
     */
    private fun closeCamera() {
        try {
            cameraOpenCloseLock.acquire()
            captureSession?.close()
            captureSession = null
            cameraDevice?.close()
            cameraDevice = null
            imageReader?.close()
            imageReader = null
        } catch (e: InterruptedException) {
            Log.e(TAG, "Error closing camera: ${e.message}", e)
        } finally {
            cameraOpenCloseLock.release()
        }
    }
    
    /**
     * Check if device supports RAW DNG capture
     */
    fun supportsRawDng(): Boolean {
        return _deviceInfo.value?.supportsRawDng == true
    }
    
    /**
     * Check if device is Samsung with Camera2 Level 3 support
     */
    fun isSamsungLevel3Device(): Boolean {
        val info = _deviceInfo.value ?: return false
        return info.isSamsungDevice && info.supportsCamera2Level3
    }
    
    /**
     * Get current camera status
     */
    fun getCamera2Status(): Camera2Status {
        val info = _deviceInfo.value
        return Camera2Status(
            state = _cameraState.value,
            deviceModel = info?.deviceModel ?: "Unknown",
            isSamsungDevice = info?.isSamsungDevice ?: false,
            supportsCamera2Level3 = info?.supportsCamera2Level3 ?: false,
            supportsRawDng = info?.supportsRawDng ?: false,
            maxRawSize = info?.maxRawSize,
            hardwareLevel = info?.hardwareLevel ?: CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY
        )
    }
    
    /**
     * Camera2 status data class
     */
    data class Camera2Status(
        val state: Camera2State,
        val deviceModel: String,
        val isSamsungDevice: Boolean,
        val supportsCamera2Level3: Boolean,
        val supportsRawDng: Boolean,
        val maxRawSize: Size?,
        val hardwareLevel: Int
    )
    
    /**
     * Cleanup resources
     */
    fun cleanup() {
        Log.i(TAG, "Cleaning up Camera2 RAW DNG manager")
        
        closeCamera()
        stopBackgroundThread()
        
        _cameraState.value = Camera2State.UNINITIALIZED
        _deviceInfo.value = null
        
        Log.i(TAG, "Camera2 RAW DNG manager cleanup completed")
    }
}