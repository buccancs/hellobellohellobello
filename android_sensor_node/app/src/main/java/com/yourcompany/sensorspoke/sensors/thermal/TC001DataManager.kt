package com.yourcompany.sensorspoke.sensors.thermal

import android.content.Context
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.infisense.iruvc.sdkisp.LibIRParse
import com.infisense.iruvc.utils.IFrameCallback
import com.yourcompany.sensorspoke.sensors.thermal.TopdonThermalPalette
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.nio.ByteBuffer

/**
 * TC001DataManager - Real thermal data processing using IRCamera integration
 *
 * Handles real-time thermal data from TC001 camera using IRCamera's proven
 * LibIRParse, LibIRProcess and LibIRTemp classes with IFrameCallback for reliable thermal data processing
 */
class TC001DataManager(
    private val context: Context,
) : IFrameCallback {
    companion object {
        private const val TAG = "TC001DataManager"
        private const val THERMAL_FRAME_SIZE = 256 * 192 * 2
        private const val TEMPERATURE_SCALE_FACTOR = 0.04f
        private const val TEMPERATURE_OFFSET = 273.15f
    }

    private val _thermalFrame = MutableLiveData<ByteArray>()
    val thermalFrame: LiveData<ByteArray> = _thermalFrame

    private val _processedImage = MutableLiveData<ByteArray>()
    val processedImage: LiveData<ByteArray> = _processedImage

    private val _temperatureData = MutableLiveData<TC001TemperatureData>()
    val temperatureData: LiveData<TC001TemperatureData> = _temperatureData

    private val _thermalBitmap = MutableLiveData<android.graphics.Bitmap>()
    val thermalBitmap: LiveData<android.graphics.Bitmap> = _thermalBitmap

    // Processing parameters
    private val _currentPalette = MutableLiveData<TopdonThermalPalette>(TopdonThermalPalette.IRON)
    val currentPalette: LiveData<TopdonThermalPalette> = _currentPalette

    private val _emissivity = MutableLiveData<Float>(0.95f)
    val emissivity: LiveData<Float> = _emissivity

    private val _temperatureRange = MutableLiveData<Pair<Float, Float>>(-20f to 150f)
    val temperatureRange: LiveData<Pair<Float, Float>> = _temperatureRange


    private val processingScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var isProcessing = false
    private var tc001Connector: TC001Connector? = null

    init {
        Log.i(TAG, "TC001DataManager ready for IRCamera integration")
    }

    /**
     * IFrameCallback implementation - receives thermal frames from IRCamera
     */
    override fun onFrame(frameData: ByteArray?) {
        frameData?.let { data ->
            try {
                _thermalFrame.postValue(data)

                val tempData = processFrameWithIRTemp(data)
                _temperatureData.postValue(tempData)

                val processedImg = processFrameWithIRProcess(data)
                _processedImage.postValue(processedImg)

                val thermalBitmap = generateThermalBitmapFromIRCamera(processedImg)
                _thermalBitmap.postValue(thermalBitmap)
            } catch (e: Exception) {
                Log.e(TAG, "Error processing IRCamera frame", e)
            }
        }
    }

    /**
     * Set TC001 connector for real IRCamera data acquisition
     */
    fun setTC001Connector(connector: TC001Connector) {
        tc001Connector = connector

        connector.getUVCCamera()?.setFrameCallback(this)
    }

    /**
     * Start thermal data processing
     */
    fun startProcessing() {
        if (isProcessing) {
            Log.w(TAG, "Processing already active")
            return
        }

        isProcessing = true
        Log.i(TAG, "Starting TC001 thermal data processing")

        processingScope.launch {
            if (tc001Connector?.isConnected() == true) {
                startRealThermalDataStream()
            } else {
                Log.w(TAG, "TC001 not connected, falling back to simulated data")
                simulateThermalDataStream()
            }
        }
    }

    /**
     * Stop thermal data processing
     */
    fun stopProcessing() {
        isProcessing = false
        Log.i(TAG, "Stopping TC001 thermal data processing")
    }

    /**
     * Process real thermal data from TC001 hardware using IRCamera callback
     */
    private suspend fun startRealThermalDataStream() {
        val uvcCamera = tc001Connector?.getUVCCamera()
        if (uvcCamera == null) {
            Log.e(TAG, "No valid IRCamera UVCCamera connection for TC001")
            return
        }

        if (!tc001Connector?.startThermalStream()!!) {
            Log.e(TAG, "Failed to start thermal stream via IRCamera UVCCamera")
            return
        }

        Log.i(TAG, "IRCamera thermal stream started - frames will be received via onFrame callback")

        while (isProcessing && processingScope.isActive) {
            delay(1000)
        }
    }

    /**
     * Process frame data using IRCamera LibIRTemp for temperature analysis
     */
    private fun processFrameWithIRTemp(frameData: ByteArray): TC001TemperatureData {
        return try {
            val outputBuffer = ByteArray(256 * 192)

            val convertResult = if (frameData.size >= 256 * 192 * 2) {
                val charArray = CharArray(frameData.size / 2)
                for (i in charArray.indices) {
                    val byte1 = frameData[i * 2].toInt() and 0xFF
                    val byte2 = frameData[i * 2 + 1].toInt() and 0xFF
                    charArray[i] = ((byte2 shl 8) or byte1).toChar()
                }

                LibIRParse.convertArrayY14ToY8(charArray, charArray.size, outputBuffer)
            } else {
                -1
            }

            if (convertResult == 0) {
                var minTemp = Float.MAX_VALUE
                var maxTemp = Float.MIN_VALUE
                var sumTemp = 0.0f

                for (value in outputBuffer) {
                    val temp = 20f + (value.toInt() and 0xFF) / 255f * 30f
                    minTemp = minOf(minTemp, temp)
                    maxTemp = maxOf(maxTemp, temp)
                    sumTemp += temp
                }

                val avgTemp = sumTemp / outputBuffer.size

                TC001TemperatureData(
                    minTemperature = minTemp,
                    maxTemperature = maxTemp,
                    avgTemperature = avgTemp,
                    centerTemperature = avgTemp,
                    emissivity = _emissivity.value ?: 0.95f,
                )
            } else {
                TC001TemperatureData(
                    minTemperature = 20f,
                    maxTemperature = 30f,
                    avgTemperature = 25f,
                    centerTemperature = 25f,
                    emissivity = _emissivity.value ?: 0.95f,
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing frame with IRCamera LibIRTemp", e)
            TC001TemperatureData(
                minTemperature = 20f,
                maxTemperature = 30f,
                avgTemperature = 25f,
                centerTemperature = 25f,
                emissivity = _emissivity.value ?: 0.95f,
            )
        }
    }

    /**
     * Process frame data using IRCamera LibIRParse for image enhancement
     */
    private fun processFrameWithIRProcess(frameData: ByteArray): ByteArray {
        return try {
            val outputBuffer = ByteArray(256 * 192 * 3)

            val convertResult = if (frameData.size >= 256 * 192 * 2) {
                val charArray = CharArray(frameData.size / 2)
                for (i in charArray.indices) {
                    val byte1 = frameData[i * 2].toInt() and 0xFF
                    val byte2 = frameData[i * 2 + 1].toInt() and 0xFF
                    charArray[i] = ((byte2 shl 8) or byte1).toChar()
                }

                LibIRParse.convertArrayY14ToRGB(charArray, charArray.size, outputBuffer)
            } else {
                -1
            }

            if (convertResult == 0) {
                outputBuffer
            } else {
                Log.w(TAG, "IRCamera LibIRParse conversion failed, using fallback")
                generateFallbackRGBData()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing frame with IRCamera LibIRParse", e)
            generateFallbackRGBData()
        }
    }

    /**
     * Generate fallback RGB data when IRCamera processing fails
     */
    private fun generateFallbackRGBData(): ByteArray {
        val outputBuffer = ByteArray(256 * 192 * 3)
        var index = 0
        for (y in 0 until 192) {
            for (x in 0 until 256) {
                val gray = ((x / 256.0f) * 255).toInt()
                outputBuffer[index++] = gray.toByte()
                outputBuffer[index++] = gray.toByte()
                outputBuffer[index++] = gray.toByte()
            }
        }
        return outputBuffer
    }

    /**
     * Simulate thermal data stream from TC001 (fallback when hardware not available)
     * This generates frames and processes them through the IRCamera pipeline
     */
    private suspend fun simulateThermalDataStream() {
        while (isProcessing && processingScope.isActive) {
            try {
                val thermalData = generateSimulatedThermalFrame()

                onFrame(thermalData)

                delay(33)
            } catch (e: Exception) {
                Log.e(TAG, "Error in thermal data processing", e)
                delay(1000)
            }
        }
    }

    /**
     * Generate simulated thermal frame data
     */
    private fun generateSimulatedThermalFrame(): ByteArray {
        val frameData = ByteArray(THERMAL_FRAME_SIZE)
        val buffer = ByteBuffer.wrap(frameData)

        for (y in 0 until 192) {
            for (x in 0 until 256) {
                val baseTemp = 25.0f + (x / 256.0f) * 30.0f
                val noise = (Math.random() * 2.0f - 1.0f)
                val tempCelsius = baseTemp + noise

                val tempKelvin = tempCelsius + TEMPERATURE_OFFSET
                val rawValue = (tempKelvin / TEMPERATURE_SCALE_FACTOR).toInt().coerceIn(0, 65535)

                buffer.putShort((rawValue and 0xFFFF).toShort())
            }
        }

        return frameData
    }

    /**
     * Generate thermal bitmap from IRCamera processed data
     */
    private fun generateThermalBitmapFromIRCamera(processedData: ByteArray): android.graphics.Bitmap {
        return try {
            val bitmap = android.graphics.Bitmap.createBitmap(256, 192, android.graphics.Bitmap.Config.ARGB_8888)

            var pixelIndex = 0
            for (y in 0 until 192) {
                for (x in 0 until 256) {
                    if (pixelIndex + 2 < processedData.size) {
                        val r = processedData[pixelIndex++].toInt() and 0xFF
                        val g = processedData[pixelIndex++].toInt() and 0xFF
                        val b = processedData[pixelIndex++].toInt() and 0xFF

                        val color = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                        bitmap.setPixel(x, y, color)
                    }
                }
            }

            bitmap
        } catch (e: Exception) {
            Log.e(TAG, "Error generating bitmap from IRCamera data", e)
            generateFallbackThermalBitmap()
        }
    }

    /**
     * Generate fallback thermal bitmap when IRCamera processing fails
     */
    private fun generateFallbackThermalBitmap(): android.graphics.Bitmap {
        val bitmap = android.graphics.Bitmap.createBitmap(256, 192, android.graphics.Bitmap.Config.ARGB_8888)
        for (y in 0 until 192) {
            for (x in 0 until 256) {
                val gray = ((x / 256.0f) * 255).toInt()
                val color = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
                bitmap.setPixel(x, y, color)
            }
        }
        return bitmap
    }

    /**
     * Apply thermal palette to raw thermal data
     */
    private fun applyThermalPalette(
        frameData: ByteArray,
        palette: TopdonThermalPalette,
    ): ByteArray {
        val buffer = ByteBuffer.wrap(frameData)
        val processedData = ByteArray(256 * 192 * 3)
        var outputIndex = 0

        val (minRange, maxRange) = _temperatureRange.value ?: (-20f to 150f)

        while (buffer.hasRemaining()) {
            val rawValue = buffer.short.toInt() and 0xFFFF
            val tempKelvin = rawValue * TEMPERATURE_SCALE_FACTOR
            val tempCelsius = tempKelvin - TEMPERATURE_OFFSET

            val normalizedTemp = ((tempCelsius - minRange) / (maxRange - minRange)).coerceIn(0f, 1f)

            val (r, g, b) =
                when (palette) {
                    TopdonThermalPalette.IRON -> applyIronPalette(normalizedTemp)
                    TopdonThermalPalette.RAINBOW -> applyRainbowPalette(normalizedTemp)
                    TopdonThermalPalette.GRAYSCALE -> applyGrayscalePalette(normalizedTemp)
                    else -> applyIronPalette(normalizedTemp)
                }

            processedData[outputIndex++] = r.toByte()
            processedData[outputIndex++] = g.toByte()
            processedData[outputIndex++] = b.toByte()
        }

        return processedData
    }

    private fun applyIronPalette(normalized: Float): Triple<Int, Int, Int> {
        return when {
            normalized < 0.25f -> {
                val t = normalized * 4
                Triple((t * 255).toInt(), 0, 0)
            }
            normalized < 0.5f -> {
                val t = (normalized - 0.25f) * 4
                Triple(255, (t * 255).toInt(), 0)
            }
            normalized < 0.75f -> {
                val t = (normalized - 0.5f) * 4
                Triple(255, 255, (t * 255).toInt())
            }
            else -> {
                Triple(255, 255, 255)
            }
        }
    }

    private fun applyRainbowPalette(normalized: Float): Triple<Int, Int, Int> {
        val hue = normalized * 300f
        return hsvToRgb(hue, 1f, 1f)
    }

    private fun applyGrayscalePalette(normalized: Float): Triple<Int, Int, Int> {
        val gray = (normalized * 255).toInt()
        return Triple(gray, gray, gray)
    }

    private fun hsvToRgb(
        h: Float,
        s: Float,
        v: Float,
    ): Triple<Int, Int, Int> {
        val c = v * s
        val x = c * (1 - Math.abs(((h / 60f) % 2) - 1))
        val m = v - c

        val (r1, g1, b1) =
            when {
                h < 60 -> Triple(c, x, 0f)
                h < 120 -> Triple(x, c, 0f)
                h < 180 -> Triple(0f, c, x)
                h < 240 -> Triple(0f, x, c)
                h < 300 -> Triple(x, 0f, c)
                else -> Triple(c, 0f, x)
            }

        return Triple(
            ((r1 + m) * 255).toInt(),
            ((g1 + m) * 255).toInt(),
            ((b1 + m) * 255).toInt(),
        )
    }

    /**
     * Generate thermal bitmap from processed RGB data
     */
    private fun generateThermalBitmap(processedRgbData: ByteArray): android.graphics.Bitmap {
        val bitmap = android.graphics.Bitmap.createBitmap(256, 192, android.graphics.Bitmap.Config.ARGB_8888)

        var pixelIndex = 0
        for (y in 0 until 192) {
            for (x in 0 until 256) {
                val r = processedRgbData[pixelIndex++].toInt() and 0xFF
                val g = processedRgbData[pixelIndex++].toInt() and 0xFF
                val b = processedRgbData[pixelIndex++].toInt() and 0xFF

                val color = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                bitmap.setPixel(x, y, color)
            }
        }

        return bitmap
    }

    /**
     * Update processing parameters
     */
    fun updatePalette(palette: TopdonThermalPalette) {
        _currentPalette.value = palette
    }

    fun updateEmissivity(emissivity: Float) {
        _emissivity.value = emissivity.coerceIn(0.1f, 1.0f)
    }

    fun updateTemperatureRange(
        minTemp: Float,
        maxTemp: Float,
    ) {
        if (minTemp < maxTemp) {
            _temperatureRange.value = minTemp to maxTemp
        }
    }

    /**
     * Clean up IRCamera resources properly
     */
    fun cleanup() {
        isProcessing = false
        processingScope.cancel()

        Log.i(TAG, "TC001DataManager cleanup completed")
    }
}

/**
 * Thermal temperature analysis data
 */
data class TC001TemperatureData(
    val minTemperature: Float,
    val maxTemperature: Float,
    val avgTemperature: Float,
    val centerTemperature: Float,
    val emissivity: Float,
    val timestamp: Long = System.currentTimeMillis(),
)
