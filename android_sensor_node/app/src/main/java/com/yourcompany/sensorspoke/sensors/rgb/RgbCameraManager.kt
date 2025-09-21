package com.yourcompany.sensorspoke.sensors.rgb

import android.content.Context
import android.util.Log
import android.util.Size
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceOrientedMeteringPointFactory
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.VideoCapture
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Enhanced RgbCameraManager with Preview support, Samsung S22 4K recording, and improved lifecycle management.
 * Implements all requirements from the ASD issue for RGB camera integration.
 *
 * Key improvements:
 * - Preview use case for on-screen display
 * - Samsung S22 4K recording support with fallback
 * - Front/back camera selection
 * - Enhanced camera state reporting
 * - Proper lifecycle and resource management
 */
class RgbCameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private var cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA,
) {
    companion object {
        private const val TAG = "RgbCameraManager"
        
        private val PREFERRED_QUALITIES = listOf(
            Quality.UHD,
            Quality.FHD,
            Quality.HD,
            Quality.SD
        )
    }

    private val _cameraState = MutableStateFlow(CameraState.UNINITIALIZED)
    val cameraState: StateFlow<CameraState> = _cameraState.asStateFlow()

    private val _cameraInfo = MutableStateFlow<CameraInfo?>(null)
    val cameraInfo: StateFlow<CameraInfo?> = _cameraInfo.asStateFlow()

    private val _previewSurface = MutableStateFlow<Preview.SurfaceProvider?>(null)
    val previewSurface: StateFlow<Preview.SurfaceProvider?> = _previewSurface.asStateFlow()

    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var imageCapture: ImageCapture? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var preview: Preview? = null
    private var imageAnalysis: ImageAnalysis? = null
    
    // RAW DNG support via Camera2 API
    private var camera2RawDngManager: Camera2RawDngManager? = null

    /**
     * Camera states for RGB camera management
     */
    enum class CameraState {
        UNINITIALIZED,
        INITIALIZING,
        READY,
        RECORDING,
        ERROR,
    }

    /**
     * Enhanced camera information container with RAW DNG support
     */
    data class CameraInfo(
        val cameraId: String,
        val supportedQualities: List<Quality>,
        val actualQuality: Quality,
        val resolution: Size,
        val hasFlash: Boolean = false,
        val isBackCamera: Boolean = true,
        val maxZoomRatio: Float = 1.0f,
        val deviceModel: String = "",
        val supportsRawDng: Boolean = false,
        val isSamsungLevel3Device: Boolean = false,
    )

    /**
     * Enhanced camera system initialization with Samsung S22 optimization and RAW DNG support
     */
    suspend fun initialize(): Boolean {
        return try {
            Log.i(TAG, "Initializing enhanced RGB camera manager with RAW DNG support")
            _cameraState.value = CameraState.INITIALIZING

            // Initialize CameraX for standard video/photo capture
            initializeCameraX()
            
            // Initialize Camera2 RAW DNG manager for Samsung devices
            initializeCamera2RawDng()

            _cameraState.value = CameraState.READY
            Log.i(TAG, "Enhanced RGB camera manager initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize enhanced RGB camera manager: ${e.message}", e)
            _cameraState.value = CameraState.ERROR
            false
        }
    }

    /**
     * Enhanced CameraX initialization with Preview support and 4K recording
     */
    private suspend fun initializeCameraX() {
        val provider = ProcessCameraProvider.getInstance(context).get()
        cameraProvider = provider

        // Detect device capabilities for Samsung S22 optimization
        val deviceModel = android.os.Build.MODEL
        Log.i(TAG, "Configuring camera for device: $deviceModel")

        val qualitySelector = QualitySelector.fromOrderedList(
            PREFERRED_QUALITIES,
            FallbackStrategy.higherQualityOrLowerThan(Quality.FHD)
        )

        val recorder = Recorder.Builder()
            .setQualitySelector(qualitySelector)
            .build()

        videoCapture = VideoCapture.withOutput(recorder)

        imageCapture = ImageCapture.Builder()
            .setJpegQuality(95)
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .build()

        preview = Preview.Builder()
            .build()

        imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        bindUseCases()

        updateCameraInfo(provider, deviceModel)
    }
    
    /**
     * Initialize Camera2 RAW DNG manager for Samsung devices
     */
    private suspend fun initializeCamera2RawDng() {
        try {
            camera2RawDngManager = Camera2RawDngManager(context)
            val rawInitialized = camera2RawDngManager?.initialize() == true
            
            if (rawInitialized) {
                val rawStatus = camera2RawDngManager?.getCamera2Status()
                rawStatus?.let { status ->
                    Log.i(TAG, "Camera2 RAW DNG initialized: Samsung Level 3: ${status.supportsCamera2Level3}, RAW DNG: ${status.supportsRawDng}")
                } ?: Log.i(TAG, "Camera2 RAW DNG status not available")
            } else {
                Log.i(TAG, "Camera2 RAW DNG not available on this device")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Camera2 RAW DNG initialization failed: ${e.message}")
            camera2RawDngManager = null
        }
    }

    /**
     * Bind camera use cases including Preview for on-screen display
     */
    private fun bindUseCases() {
        val provider = cameraProvider ?: return

        try {
            provider.unbindAll()

            camera = provider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                videoCapture,
                imageCapture,
                imageAnalysis
            )

            Log.i(TAG, "Camera use cases bound successfully with Preview support")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to bind camera use cases: ${e.message}", e)
            throw e
        }
    }

    /**
     * Update camera information with detected capabilities including RAW DNG support
     */
    private fun updateCameraInfo(provider: ProcessCameraProvider, deviceModel: String) {
        try {
            val availableCameraInfos = provider.availableCameraInfos
            val cameraInfo = availableCameraInfos.find { info ->
                val lensFacing = (info as? androidx.camera.core.impl.CameraInfoInternal)?.lensFacing
                lensFacing == if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
                    androidx.camera.core.CameraSelector.LENS_FACING_BACK
                } else {
                    androidx.camera.core.CameraSelector.LENS_FACING_FRONT
                }
            } ?: availableCameraInfos.firstOrNull()
            
            val supportedQualities = cameraInfo?.let { QualitySelector.getSupportedQualities(it) } ?: emptyList()
            val actualQuality = getActualQuality(supportedQualities)
            val resolution = getResolutionForQuality(actualQuality)
            
            // Get RAW DNG capabilities from Camera2 manager
            val camera2Status = camera2RawDngManager?.getCamera2Status()
            val supportsRawDng = camera2Status?.supportsRawDng ?: false
            val isSamsungLevel3Device = camera2Status?.supportsCamera2Level3 ?: false

            _cameraInfo.value = CameraInfo(
                cameraId = cameraInfo.toString(),
                supportedQualities = supportedQualities,
                actualQuality = actualQuality,
                resolution = resolution,
                hasFlash = cameraInfo?.hasFlashUnit() ?: false,
                isBackCamera = cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA,
                maxZoomRatio = cameraInfo?.zoomState?.value?.maxZoomRatio ?: 1.0f,
                deviceModel = deviceModel,
                supportsRawDng = supportsRawDng,
                isSamsungLevel3Device = isSamsungLevel3Device
            )

            Log.i(TAG, "Camera capabilities - Qualities: $supportedQualities, Using: $actualQuality, Resolution: $resolution")
            Log.i(TAG, "RAW DNG support - Enabled: $supportsRawDng, Samsung Level 3: $isSamsungLevel3Device")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to update camera info: ${e.message}", e)
        }
    }

    /**
     * Get the actual quality being used based on device support
     */
    private fun getActualQuality(supportedQualities: List<Quality>): Quality {
        return PREFERRED_QUALITIES.firstOrNull { it in supportedQualities } ?: Quality.FHD
    }

    /**
     * Get resolution for quality setting
     */
    private fun getResolutionForQuality(quality: Quality): Size {
        return when (quality) {
            Quality.UHD -> Size(3840, 2160)
            Quality.FHD -> Size(1920, 1080)
            Quality.HD -> Size(1280, 720)
            Quality.SD -> Size(720, 480)
            else -> Size(1920, 1080)
        }
    }

    /**
     * Switch between front and back camera
     */
    fun switchCamera(): Boolean {
        return try {
            cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
                CameraSelector.DEFAULT_FRONT_CAMERA
            } else {
                CameraSelector.DEFAULT_BACK_CAMERA
            }

            bindUseCases()
            
            cameraProvider?.let { provider ->
                updateCameraInfo(provider, android.os.Build.MODEL)
            }

            Log.i(TAG, "Switched to ${if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) "back" else "front"} camera")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to switch camera: ${e.message}", e)
            false
        }
    }

    /**
     * Set Preview surface provider for on-screen display
     */
    fun setPreviewSurfaceProvider(surfaceProvider: Preview.SurfaceProvider) {
        preview?.setSurfaceProvider(surfaceProvider)
        _previewSurface.value = surfaceProvider
        Log.i(TAG, "Preview surface provider set for on-screen display")
    }

    /**
     * Check if camera is ready for recording
     */
    fun isReady(): Boolean {
        return _cameraState.value == CameraState.READY
    }

    /**
     * Get VideoCapture for recording
     */
    fun getVideoCapture(): VideoCapture<Recorder>? = videoCapture

    /**
     * Get ImageCapture for still photos
     */
    fun getImageCapture(): ImageCapture? = imageCapture

    /**
     * Get Preview for on-screen display
     */
    fun getPreview(): Preview? = preview
    
    /**
     * Get Camera2 RAW DNG manager
     */
    fun getCamera2RawDngManager(): Camera2RawDngManager? = camera2RawDngManager
    
    /**
     * Check if device supports RAW DNG capture
     */
    fun supportsRawDng(): Boolean {
        return _cameraInfo.value?.supportsRawDng == true
    }
    
    /**
     * Check if device is Samsung with Camera2 Level 3 support
     */
    fun isSamsungLevel3Device(): Boolean {
        return _cameraInfo.value?.isSamsungLevel3Device == true
    }

    /**
     * Update recording state
     */
    fun updateRecordingState(isRecording: Boolean) {
        _cameraState.value = if (isRecording) CameraState.RECORDING else CameraState.READY
    }

    /**
     * Check if device likely supports 4K recording (Samsung S22 detection)
     */
    fun supports4K(): Boolean {
        val model = android.os.Build.MODEL.lowercase()
        val supportedDevices = listOf("sm-s901", "sm-s906", "sm-s908", "galaxy s22")
        
        return supportedDevices.any { model.contains(it) } ||
               _cameraInfo.value?.supportedQualities?.contains(Quality.UHD) == true
    }

    /**
     * Get current camera status for UI feedback including RAW DNG support
     */
    fun getCameraStatus(): CameraStatus {
        val info = _cameraInfo.value
        return CameraStatus(
            state = _cameraState.value,
            quality = info?.actualQuality?.toString() ?: "Unknown",
            resolution = info?.resolution?.let { "${it.width}x${it.height}" } ?: "Unknown",
            isBackCamera = info?.isBackCamera ?: true,
            hasFlash = info?.hasFlash ?: false,
            supports4K = supports4K(),
            deviceModel = info?.deviceModel ?: android.os.Build.MODEL,
            supportsRawDng = info?.supportsRawDng ?: false,
            isSamsungLevel3Device = info?.isSamsungLevel3Device ?: false
        )
    }

    /**
     * Camera status data class for UI feedback with RAW DNG support
     */
    data class CameraStatus(
        val state: CameraState,
        val quality: String,
        val resolution: String,
        val isBackCamera: Boolean,
        val hasFlash: Boolean,
        val supports4K: Boolean,
        val deviceModel: String,
        val supportsRawDng: Boolean,
        val isSamsungLevel3Device: Boolean,
    )

    /**
     * Enhanced cleanup with proper resource management including Camera2 RAW DNG
     */
    fun cleanup() {
        Log.i(TAG, "Cleaning up RGB camera manager resources")
        
        try {
            // Cleanup CameraX resources
            cameraProvider?.unbindAll()
            camera = null
            videoCapture = null
            imageCapture = null
            preview = null
            imageAnalysis = null
            cameraProvider = null
            
            // Cleanup Camera2 RAW DNG manager
            camera2RawDngManager?.cleanup()
            camera2RawDngManager = null
            
            _cameraState.value = CameraState.UNINITIALIZED
            _cameraInfo.value = null
            _previewSurface.value = null
            
            Log.i(TAG, "RGB camera manager cleanup completed")
        } catch (e: Exception) {
            Log.e(TAG, "Error during camera cleanup: ${e.message}", e)
        }
    }
}
