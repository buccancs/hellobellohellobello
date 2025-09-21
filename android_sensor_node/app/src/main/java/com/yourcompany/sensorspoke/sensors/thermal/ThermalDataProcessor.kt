package com.yourcompany.sensorspoke.sensors.thermal

import android.graphics.Bitmap
import android.util.Log
import com.yourcompany.sensorspoke.sensors.thermal.ThermalFrame
import java.io.File
import java.io.FileOutputStream

/**
 * ThermalDataProcessor handles thermal camera data processing and formatting.
 * Extracted from ThermalCameraRecorder to improve modularity and testability.
 *
 * This utility class is responsible for:
 * - Processing thermal frame data and temperature calculations
 * - Saving thermal images to files
 * - CSV formatting for thermal metadata
 * - Performance monitoring and statistics
 */
class ThermalDataProcessor {
    companion object {
        private const val TAG = "ThermalDataProcessor"
    }

    /**
     * Standardized thermal frame data class
     */
    data class ThermalFrameData(
        val timestampNs: Long,
        val timestampMs: Long,
        val frameNumber: Int,
        val centerTemperature: Float,
        val minTemperature: Float,
        val maxTemperature: Float,
        val averageTemperature: Float,
        val imageFilename: String,
    )

    /**
     * Performance metrics data class
     */
    data class ThermalPerformanceMetrics(
        val frameCount: Int,
        val targetFps: Int,
        val averageProcessingTimeMs: Double,
        val actualFps: Double,
    )

    /**
     * Get CSV header for thermal data
     */
    fun getCsvHeader(): String {
        return "timestamp_ns,timestamp_ms,frame_number,temperature_celsius,min_temp,max_temp,avg_temp,filename"
    }

    /**
     * Create thermal frame data from real thermal camera
     */
    fun createThermalFrameData(
        thermalFrame: ThermalFrame,
        frameNumber: Int,
    ): ThermalFrameData {
        val imageFilename = "thermal_real_${thermalFrame.timestamp}.png"

        return ThermalFrameData(
            timestampNs = thermalFrame.timestamp,
            timestampMs = System.currentTimeMillis(),
            frameNumber = frameNumber,
            centerTemperature = calculateCenterTemperature(thermalFrame),
            minTemperature = thermalFrame.minTemp,
            maxTemperature = thermalFrame.maxTemp,
            averageTemperature = thermalFrame.avgTemp,
            imageFilename = imageFilename,
        )
    }

    /**
     * Calculate center temperature from thermal frame
     */
    private fun calculateCenterTemperature(frame: ThermalFrame): Float {
        val centerX = frame.width / 2
        val centerY = frame.height / 2
        val centerIndex = centerY * frame.width + centerX
        return if (centerIndex < frame.temperatureMatrix.size) {
            frame.temperatureMatrix[centerIndex]
        } else {
            frame.avgTemp
        }
    }

    /**
     * Create thermal frame data from simulated camera
     */
    fun createSimulatedThermalFrameData(
        timestampNs: Long,
        timestampMs: Long,
        frameNumber: Int,
        baseTemp: Float,
    ): ThermalFrameData {
        val minTemp = baseTemp - 5.0f
        val maxTemp = baseTemp + 15.0f
        val avgTemp = (minTemp + maxTemp) / 2
        val imageFilename = "thermal_sim_$timestampNs.png"

        return ThermalFrameData(
            timestampNs = timestampNs,
            timestampMs = timestampMs,
            frameNumber = frameNumber,
            centerTemperature = baseTemp,
            minTemperature = minTemp,
            maxTemperature = maxTemp,
            averageTemperature = avgTemp,
            imageFilename = imageFilename,
        )
    }

    /**
     * Create thermal frame data from TopdonThermalIntegration
     */
    fun createThermalFrameDataFromIntegration(
        thermalData: ThermalFrameData,
        timestampNs: Long,
        timestampMs: Long,
        frameNumber: Int,
    ): ThermalFrameData {
        val imageFilename = "thermal_$timestampNs.png"

        return ThermalFrameData(
            timestampNs = timestampNs,
            timestampMs = timestampMs,
            frameNumber = frameNumber,
            centerTemperature = thermalData.centerTemperature,
            minTemperature = thermalData.minTemperature,
            maxTemperature = thermalData.maxTemperature,
            averageTemperature = thermalData.averageTemperature,
            imageFilename = imageFilename,
        )
    }

    /**
     * Format thermal frame data for CSV output
     */
    fun formatThermalDataForCsv(frameData: ThermalFrameData): String {
        return "${frameData.timestampNs},${frameData.timestampMs},${frameData.frameNumber},${"%.2f".format(frameData.centerTemperature)},${"%.2f".format(frameData.minTemperature)},${"%.2f".format(frameData.maxTemperature)},${"%.2f".format(frameData.averageTemperature)},${frameData.imageFilename}"
    }

    /**
     * Save thermal image to file
     */
    fun saveThermalImage(bitmap: Bitmap, imageFile: File): Boolean {
        return try {
            FileOutputStream(imageFile).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            Log.d(TAG, "Saved thermal image: ${imageFile.name}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save thermal image: ${e.message}", e)
            false
        }
    }

    /**
     * Generate simulated thermal image
     */
    fun generateSimulatedThermalImage(width: Int, height: Int, baseTemp: Float): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val centerX = width / 2f
                val centerY = height / 2f
                val distance = kotlin.math.sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY))
                val maxDistance = kotlin.math.sqrt(centerX * centerX + centerY * centerY)

                val temp = baseTemp * (1 - distance / maxDistance * 0.3f)
                val normalizedTemp = ((temp - 20f) / 40f).coerceIn(0f, 1f)

                val color = when {
                    normalizedTemp < 0.33f -> {
                        val blue = (255 * (normalizedTemp / 0.33f)).toInt()
                        (0xFF shl 24) or blue
                    }
                    normalizedTemp < 0.66f -> {
                        val green = (255 * ((normalizedTemp - 0.33f) / 0.33f)).toInt()
                        (0xFF shl 24) or (green shl 8)
                    }
                    else -> {
                        val red = (255 * ((normalizedTemp - 0.66f) / 0.34f)).toInt()
                        (0xFF shl 24) or (red shl 16)
                    }
                }

                pixels[y * width + x] = color
            }
        }

        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    /**
     * Calculate performance metrics
     */
    fun calculatePerformanceMetrics(
        frameCount: Int,
        targetFps: Int,
        frameProcessingTimeSum: Long,
        frameTimeWindow: List<Long>,
    ): ThermalPerformanceMetrics {
        val avgProcessingTime = if (frameCount > 0) {
            frameProcessingTimeSum / frameCount.toDouble() / 1_000_000.0
        } else {
            0.0
        }

        val actualFps = if (frameTimeWindow.size >= 2) {
            val timeSpan = (frameTimeWindow.last() - frameTimeWindow.first()) / 1_000_000_000.0
            (frameTimeWindow.size - 1) / timeSpan
        } else {
            0.0
        }

        return ThermalPerformanceMetrics(
            frameCount = frameCount,
            targetFps = targetFps,
            averageProcessingTimeMs = avgProcessingTime,
            actualFps = actualFps,
        )
    }

    /**
     * Log performance metrics
     */
    fun logPerformanceMetrics(metrics: ThermalPerformanceMetrics) {
        Log.i(TAG, "Thermal performance: Frame ${metrics.frameCount}, Avg processing: ${"%.2f".format(metrics.averageProcessingTimeMs)}ms, Actual FPS: ${"%.1f".format(metrics.actualFps)}")
    }

    /**
     * Validate temperature data ranges
     */
    fun validateTemperatureData(frameData: ThermalFrameData): String {
        return when {
            frameData.centerTemperature < -50f || frameData.centerTemperature > 200f -> "TEMP_OUT_OF_RANGE"
            frameData.minTemperature > frameData.maxTemperature -> "INVALID_TEMP_RANGE"
            frameData.centerTemperature < frameData.minTemperature || frameData.centerTemperature > frameData.maxTemperature -> "CENTER_TEMP_INCONSISTENT"
            else -> "OK"
        }
    }

    /**
     * Get thermal statistics for monitoring
     */
    fun getThermalStatistics(
        frameCount: Int,
        totalProcessingTime: Long,
        frameTimeWindow: List<Long>,
    ): Map<String, Any> {
        return mapOf(
            "totalFramesProcessed" to frameCount,
            "totalProcessingTimeMs" to totalProcessingTime / 1_000_000.0,
            "averageProcessingTimeMs" to if (frameCount > 0) {
                totalProcessingTime / frameCount.toDouble() / 1_000_000.0
            } else {
                0.0
            },
            "currentFps" to if (frameTimeWindow.size >= 2) {
                val timeSpan = (frameTimeWindow.last() - frameTimeWindow.first()) / 1_000_000_000.0
                (frameTimeWindow.size - 1) / timeSpan
            } else {
                0.0
            },
        )
    }
}
