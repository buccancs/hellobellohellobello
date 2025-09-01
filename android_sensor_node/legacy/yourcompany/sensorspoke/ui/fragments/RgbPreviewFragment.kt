package com.yourcompany.sensorspoke.ui.fragments

import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.fragment.app.Fragment
import com.yourcompany.sensorspoke.R
import com.yourcompany.sensorspoke.utils.PreviewBus

/**
 * Fragment for displaying RGB camera preview
 */
class RgbPreviewFragment : Fragment() {
    private var previewImageView: ImageView? = null
    private var statusText: TextView? = null
    private var resolutionText: TextView? = null
    private var framerateText: TextView? = null

    private var frameCount = 0
    private var lastFrameTime = 0L

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
                resolutionText?.text = "Resolution: ${bitmap.width}x${bitmap.height}"

                // Calculate frame rate
                frameCount++
                val currentTime = System.currentTimeMillis()
                if (lastFrameTime > 0) {
                    val deltaMs = currentTime - lastFrameTime
                    if (deltaMs > 1000) { // Update every second
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
