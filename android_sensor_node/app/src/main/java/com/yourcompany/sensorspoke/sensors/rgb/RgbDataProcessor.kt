package com.yourcompany.sensorspoke.sensors.rgb

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import com.yourcompany.sensorspoke.utils.PreviewBus
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * RgbDataProcessor handles RGB camera data processing and formatting.
 * Extracted from RgbCameraRecorder to improve modularity and testability.
 *
 * This utility class is responsible for:
 * - Processing captured frames and video timing data
 * - Generating preview images for UI
 * - CSV formatting for frame metadata
 * - Synchronization quality calculations
 */
class RgbDataProcessor {
    companion object {
        private const val TAG = "RgbDataProcessor"
    }

    /**
     * Standardized frame data class
     */
    data class FrameData(
        val timestampNs: Long,
        val timestampMs: Long,
        val frameNumber: Int,
        val filename: String,
        val fileSizeBytes: Long,
        val videoRelativeTimeMs: Int,
        val estimatedVideoFrame: Int,
        val syncQuality: Double,
        val actualVideoOffsetMs: Int,
    )

    /**
     * Get CSV header for frame data
     */
    fun getCsvHeader(): String {
        return "timestamp_ns,timestamp_ms,frame_number,filename,file_size_bytes,video_relative_time_ms,video_frame_estimate,sync_quality,actual_video_offset_ms"
    }

    /**
     * Create frame data from captured image
     */
    fun createFrameData(
        timestampNs: Long,
        timestampMs: Long,
        frameNumber: Int,
        imageFile: File,
        videoStartTime: Long,
        actualVideoStartTime: Long,
    ): FrameData {
        val fileSize = if (imageFile.exists()) imageFile.length() else 0
        val filename = imageFile.name

        val baseTime = if (actualVideoStartTime > 0) actualVideoStartTime else videoStartTime
        val videoRelativeTimeMs = if (baseTime > 0) {
            ((timestampNs - baseTime) / 1_000_000).toInt()
        } else {
            0
        }

        val estimatedVideoFrame = if (videoRelativeTimeMs > 0) {
            (videoRelativeTimeMs * 30.0 / 1000.0).toInt()
        } else {
            0
        }

        val syncQuality = calculateSyncQuality(timestampNs, baseTime)

        val actualVideoOffsetMs = if (actualVideoStartTime > 0) {
            ((timestampNs - actualVideoStartTime) / 1_000_000).toInt()
        } else {
            -1
        }

        return FrameData(
            timestampNs = timestampNs,
            timestampMs = timestampMs,
            frameNumber = frameNumber,
            filename = filename,
            fileSizeBytes = fileSize,
            videoRelativeTimeMs = videoRelativeTimeMs,
            estimatedVideoFrame = estimatedVideoFrame,
            syncQuality = syncQuality,
            actualVideoOffsetMs = actualVideoOffsetMs,
        )
    }

    /**
     * Format frame data for CSV output
     */
    fun formatFrameDataForCsv(frameData: FrameData): String {
        return "${frameData.timestampNs},${frameData.timestampMs},${frameData.frameNumber},${frameData.filename},${frameData.fileSizeBytes},${frameData.videoRelativeTimeMs},${frameData.estimatedVideoFrame},${"%.3f".format(frameData.syncQuality)},${frameData.actualVideoOffsetMs}"
    }

    /**
     * Generate preview image for UI
     */
    fun generatePreview(imageFile: File, timestamp: Long): Boolean {
        return try {
            if (!imageFile.exists()) return false

            val options = BitmapFactory.Options().apply {
                inSampleSize = 4
                inPreferredConfig = Bitmap.Config.RGB_565
            }

            val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath, options)
            if (bitmap != null) {
                val previewBitmap = Bitmap.createScaledBitmap(bitmap, 320, 240, true)

                val baos = ByteArrayOutputStream()
                previewBitmap.compress(Bitmap.CompressFormat.JPEG, 60, baos)
                val previewBytes = baos.toByteArray()

                PreviewBus.emit(previewBytes, timestamp)

                baos.close()
                if (previewBitmap != bitmap) {
                    bitmap.recycle()
                }
                previewBitmap.recycle()

                true
            } else {
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error generating preview: ${e.message}", e)
            false
        }
    }

    /**
     * Calculate synchronization quality metric based on timing alignment
     * Returns a score from 0.0 (poor) to 1.0 (excellent) synchronization
     */
    fun calculateSyncQuality(frameTimestamp: Long, videoBaseTime: Long): Double {
        return if (videoBaseTime <= 0) {
            0.0 // No video timing reference yet
        } else {
            val relativeTime = frameTimestamp - videoBaseTime
            val expectedFrameInterval = 33_333_333L

            val timingDeviation = relativeTime % expectedFrameInterval
            val deviationRatio = timingDeviation.toDouble() / expectedFrameInterval.toDouble()

            1.0 - kotlin.math.min(deviationRatio, 0.5) * 2.0
        }
    }

    /**
     * Log video-related events for comprehensive timing analysis
     */
    fun logVideoEvent(csvFile: File?, event: String, timestamp: Long, details: String = "") {
        try {
            val videoEventsFile = File(csvFile?.parent, "video_events.csv")

            if (!videoEventsFile.exists()) {
                videoEventsFile.writeText("timestamp_ns,timestamp_ms,event,details\n")
            }

            val timestampMs = timestamp / 1_000_000
            videoEventsFile.appendText("$timestamp,$timestampMs,$event,$details\n")

            Log.d(TAG, "Video event logged: $event at $timestamp ($details)")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to log video event: ${e.message}", e)
        }
    }

    /**
     * Get comprehensive timing statistics for post-processing analysis
     */
    fun getTimingStatistics(
        videoStartTime: Long,
        actualVideoStartTime: Long,
        frameTimestampOffset: Long,
        frameCount: Int,
    ): Map<String, Any> {
        return mapOf(
            "videoStartTime" to videoStartTime,
            "actualVideoStartTime" to actualVideoStartTime,
            "frameTimestampOffset" to frameTimestampOffset,
            "totalFramesCaptured" to frameCount,
            "avgFrameInterval" to if (frameCount > 1) {
                (System.nanoTime() - videoStartTime) / frameCount / 1_000_000
            } else {
                0
            },
            "syncQualityMetricAvailable" to (actualVideoStartTime > 0),
        )
    }
}
