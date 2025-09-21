package com.yourcompany.sensorspoke.sensors.thermal

import android.graphics.Bitmap

/**
 * Connection status for thermal camera
 */
enum class ConnectionStatus {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    STREAMING,
    ERROR
}

/**
 * Thermal frame data structure based on IRCamera format
 */
data class ThermalFrame(
    val timestamp: Long,
    val width: Int,
    val height: Int,
    val temperatureMatrix: FloatArray,
    val minTemp: Float,
    val maxTemp: Float,
    val avgTemp: Float,
    val rotation: Int = 0,
    val isValid: Boolean = true,
    val frameNumber: Int = 0,
    val isRealHardware: Boolean = false
) {
    val minTemperature: Float get() = minTemp
    val maxTemperature: Float get() = maxTemp
    val averageTemperature: Float get() = avgTemp
    var bitmap: Bitmap? = null
        private set

    /**
     * Generate thermal visualization bitmap
     */
    fun generateBitmap(): Bitmap {
        if (bitmap != null) return bitmap!!
        
        val pixels = IntArray(width * height)
        val tempRange = maxTemp - minTemp
        
        for (i in temperatureMatrix.indices) {
            val normalizedTemp = if (tempRange > 0) {
                (temperatureMatrix[i] - minTemp) / tempRange
            } else {
                0.5f
            }
            
            val red = (normalizedTemp * 255).toInt().coerceIn(0, 255)
            val blue = ((1f - normalizedTemp) * 255).toInt().coerceIn(0, 255)
            val green = 0
            
            pixels[i] = android.graphics.Color.rgb(red, green, blue)
        }
        
        bitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
        return bitmap!!
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as ThermalFrame

        if (timestamp != other.timestamp) return false
        if (width != other.width) return false
        if (height != other.height) return false
        if (!temperatureMatrix.contentEquals(other.temperatureMatrix)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = timestamp.hashCode()
        result = 31 * result + width
        result = 31 * result + height
        result = 31 * result + temperatureMatrix.contentHashCode()
        return result
    }
}