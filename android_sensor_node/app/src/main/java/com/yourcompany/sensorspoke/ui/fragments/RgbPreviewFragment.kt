package com.yourcompany.sensorspoke.ui.fragments

import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.Switch
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.yourcompany.sensorspoke.R
import com.yourcompany.sensorspoke.sensors.rgb.RgbCameraRecorder
import com.yourcompany.sensorspoke.utils.PreviewBus
import kotlinx.coroutines.launch

/**
 * Fragment for displaying RGB camera preview with RAW DNG support
 */
class RgbPreviewFragment : Fragment() {
    private var previewImageView: ImageView? = null
    private var statusText: TextView? = null
    private var resolutionText: TextView? = null
    private var framerateText: TextView? = null
    private var deviceModelText: TextView? = null
    private var rawDngControls: LinearLayout? = null
    private var rawDngToggle: Switch? = null
    private var rawDngStatusText: TextView? = null

    private var frameCount = 0
    private var lastFrameTime = 0L
    
    // Camera recorder instance (would be injected in real implementation)
    private var rgbCameraRecorder: RgbCameraRecorder? = null

    private val previewListener: (ByteArray, Long) -> Unit = { bytes, timestampNs ->
        activity?.runOnUiThread {
            updatePreview(bytes, timestampNs)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View? = inflater.inflate(R.layout.fragment_rgb_preview, container, false)

    override fun onViewCreated(
        view: View,
        savedInstanceState: Bundle?,
    ) {
        super.onViewCreated(view, savedInstanceState)

        previewImageView = view.findViewById(R.id.previewImageView)
        statusText = view.findViewById(R.id.statusText)
        resolutionText = view.findViewById(R.id.resolutionText)
        framerateText = view.findViewById(R.id.framerateText)
        deviceModelText = view.findViewById(R.id.deviceModelText)
        rawDngControls = view.findViewById(R.id.rawDngControls)
        rawDngToggle = view.findViewById(R.id.rawDngToggle)
        rawDngStatusText = view.findViewById(R.id.rawDngStatusText)

        setupRawDngControls()
    }
    
    /**
     * Setup RAW DNG controls and Samsung device detection
     */
    private fun setupRawDngControls() {
        // This would typically be injected via dependency injection
        // For now, we'll check if the device supports RAW DNG
        lifecycleScope.launch {
            checkRawDngSupport()
        }
        
        rawDngToggle?.setOnCheckedChangeListener { _, isChecked ->
            handleRawDngToggle(isChecked)
        }
    }
    
    /**
     * Check if device supports RAW DNG and update UI accordingly by querying the camera recorder
     */
    private suspend fun checkRawDngSupport() {
        try {
            // Get RAW DNG capabilities from the camera recorder
            val recorder = rgbCameraRecorder
            
            if (recorder == null) {
                rawDngControls?.visibility = View.GONE
                return
            }
            
            // Get capabilities from recorder without initializing camera directly
            val supportsRawDng = recorder.supportsRawDng()
            val isSamsungLevel3 = recorder.isSamsungLevel3Device()
            
            // Get camera status (this will initialize camera if needed)
            val cameraManager = recorder.getCameraManager()
            val cameraStatus = cameraManager.getCameraStatus()
            
            activity?.runOnUiThread {
                if (supportsRawDng && isSamsungLevel3) {
                    rawDngControls?.visibility = View.VISIBLE
                    rawDngStatusText?.text = "Samsung Level 3 supported"
                    rawDngStatusText?.setTextColor(ContextCompat.getColor(requireContext(), android.R.color.holo_green_light))
                    rawDngToggle?.isEnabled = true
                } else if (cameraStatus.isSamsungLevel3Device) {
                    rawDngControls?.visibility = View.VISIBLE
                    rawDngStatusText?.text = "Samsung device (Level 3 not supported)"
                    rawDngStatusText?.setTextColor(ContextCompat.getColor(requireContext(), android.R.color.holo_orange_light))
                    rawDngToggle?.isEnabled = false
                } else {
                    rawDngControls?.visibility = View.GONE
                }
                
                // Update device model display from camera status
                deviceModelText?.text = "Device: ${cameraStatus.deviceModel}"
            }
        } catch (e: Exception) {
            // Handle error silently in UI context
            activity?.runOnUiThread {
                rawDngControls?.visibility = View.GONE
                deviceModelText?.text = "Device: ${android.os.Build.MODEL}"
            }
        }
    }
    
    /**
     * Handle RAW DNG toggle state change
     */
    private fun handleRawDngToggle(enabled: Boolean) {
        try {
            val recorder = rgbCameraRecorder
            if (recorder == null) {
                rawDngToggle?.isChecked = false
                statusText?.text = "Error: Camera recorder not available"
                return
            }
            
            val mode = if (enabled) {
                RgbCameraRecorder.RecordingMode.RAW_DNG
            } else {
                RgbCameraRecorder.RecordingMode.STANDARD
            }
            
            recorder.setRecordingMode(mode)
            
            val statusMessage = if (enabled) "RAW DNG mode enabled" else "Standard mode enabled"
            statusText?.text = statusMessage
            
        } catch (e: Exception) {
            // Reset toggle on error
            rawDngToggle?.isChecked = false
            statusText?.text = "Error: ${e.message}"
        }
    }
    
    /**
     * Set the RGB camera recorder instance (for dependency injection)
     */
    fun setRgbCameraRecorder(recorder: RgbCameraRecorder) {
        this.rgbCameraRecorder = recorder
        // Refresh RAW DNG support check
        lifecycleScope.launch {
            checkRawDngSupport()
        }
    }

    override fun onResume() {
        super.onResume()
        PreviewBus.subscribe(previewListener)
        statusText?.text = "Waiting for preview..."
    }

    override fun onPause() {
        super.onPause()
        PreviewBus.unsubscribe(previewListener)
    }

    private fun updatePreview(
        jpegBytes: ByteArray,
        timestampNs: Long,
    ) {
        try {
            val bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
            if (bitmap != null) {
                previewImageView?.setImageBitmap(bitmap)
            statusText?.text = "Live preview active"
            
            // Show RAW DNG status if toggle is enabled
            rawDngToggle?.let { toggle ->
                if (toggle.isChecked) {
                    statusText?.text = "Live preview active + RAW DNG"
                }
            }
                resolutionText?.text = "Resolution: ${bitmap.width}x${bitmap.height}"

                frameCount++
                val currentTime = System.currentTimeMillis()
                if (lastFrameTime > 0) {
                    val deltaMs = currentTime - lastFrameTime
                    if (deltaMs > 1000) {
                        val fps = (frameCount * 1000.0 / deltaMs).toInt()
                        framerateText?.text = "Frame Rate: $fps fps"
                        frameCount = 0
                        lastFrameTime = currentTime
                    }
                } else {
                    lastFrameTime = currentTime
                }
            }
        } catch (e: Exception) {
            statusText?.text = "Preview error: ${e.message}"
        }
    }

    companion object {
        fun newInstance(): RgbPreviewFragment = RgbPreviewFragment()
    }
}
