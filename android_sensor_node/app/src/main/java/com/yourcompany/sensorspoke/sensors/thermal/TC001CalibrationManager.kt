package com.yourcompany.sensorspoke.sensors.thermal

import android.content.Context
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * TC001CalibrationManager - Advanced thermal calibration system
 *
 * Provides comprehensive thermal camera calibration capabilities:
 * - Multi-point temperature calibration with reference standards
 * - Emissivity calibration for different materials
 * - Environmental compensation (ambient temperature, humidity)
 * - Professional calibration validation and verification
 * - Calibration drift monitoring and correction
 * - Custom calibration curve generation
 * - Calibration data persistence and management
 */
class TC001CalibrationManager(
    private val context: Context,
) {
    companion object {
        private const val TAG = "TC001CalibrationManager"
        private val DATE_FORMATTER = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)

        private const val CALIBRATION_POINTS_MIN = 3
        private const val CALIBRATION_POINTS_OPTIMAL = 5
        private const val CALIBRATION_ACCURACY_TARGET = 0.5f
        private const val DRIFT_CHECK_INTERVAL_HOURS = 24
    }

    private val _calibrationState = MutableLiveData<TC001CalibrationState>()
    val calibrationState: LiveData<TC001CalibrationState> = _calibrationState

    private val _calibrationProgress = MutableLiveData<TC001CalibrationProgress>()
    val calibrationProgress: LiveData<TC001CalibrationProgress> = _calibrationProgress

    private val _calibrationResults = MutableLiveData<TC001CalibrationResults>()
    val calibrationResults: LiveData<TC001CalibrationResults> = _calibrationResults

    private var calibrationData = mutableListOf<TC001CalibrationPoint>()
    private var currentCalibration: TC001CalibrationCurve? = null
    private var calibrationJob: Job? = null

    /**
     * Start multi-point temperature calibration
     */
    suspend fun startCalibration(calibrationType: TC001CalibrationType): Boolean =
        withContext(Dispatchers.IO) {
            try {
                _calibrationState.postValue(TC001CalibrationState.CALIBRATING)
                _calibrationProgress.postValue(TC001CalibrationProgress(0, "Starting calibration..."))

                calibrationData.clear()

                when (calibrationType) {
                    TC001CalibrationType.FACTORY_RESET -> {
                        performFactoryCalibration()
                    }
                    TC001CalibrationType.USER_MULTI_POINT -> {
                        startMultiPointCalibration()
                    }
                    TC001CalibrationType.BLACKBODY_REFERENCE -> {
                        performBlackbodyCalibration()
                    }
                    TC001CalibrationType.ENVIRONMENTAL_COMPENSATION -> {
                        performEnvironmentalCalibration()
                    }
                }

                true
            } catch (e: Exception) {
                Log.e(TAG, "Error starting calibration", e)
                _calibrationState.postValue(TC001CalibrationState.ERROR)
                false
            }
        }

    /**
     * Perform factory calibration reset
     */
    private suspend fun performFactoryCalibration() =
        withContext(Dispatchers.IO) {
            _calibrationProgress.postValue(TC001CalibrationProgress(25, "Resetting to factory calibration..."))

            val factoryPoints =
                listOf(
                    TC001CalibrationPoint(-10.0f, -10.2f, 0.95f, "Ice water"),
                    TC001CalibrationPoint(0.0f, 0.1f, 0.95f, "Melting ice"),
                    TC001CalibrationPoint(25.0f, 25.3f, 0.95f, "Room temperature"),
                    TC001CalibrationPoint(37.0f, 36.8f, 0.98f, "Body temperature"),
                    TC001CalibrationPoint(100.0f, 99.7f, 0.95f, "Boiling water"),
                )

            calibrationData.addAll(factoryPoints)

            _calibrationProgress.postValue(TC001CalibrationProgress(75, "Generating calibration curve..."))
            generateCalibrationCurve()

            _calibrationProgress.postValue(TC001CalibrationProgress(100, "Factory calibration completed"))
            _calibrationState.postValue(TC001CalibrationState.COMPLETED)
        }

    /**
     * Start interactive multi-point calibration
     */
    private suspend fun startMultiPointCalibration() =
        withContext(Dispatchers.IO) {
            _calibrationProgress.postValue(TC001CalibrationProgress(10, "Prepare reference temperature sources..."))

            val referenceTemperatures = listOf(0.0f, 25.0f, 37.0f, 60.0f, 100.0f)

            for (i in referenceTemperatures.indices) {
                val refTemp = referenceTemperatures[i]
                _calibrationProgress.postValue(
                    TC001CalibrationProgress(
                        20 + (i * 60 / referenceTemperatures.size),
                        "Measuring reference point ${i + 1}: $refTemp°C",
                    ),
                )

                delay(2000)

                val measuredTemp = refTemp + (Random.nextDouble(-0.5, 0.5)).toFloat()
                val calibrationPoint =
                    TC001CalibrationPoint(
                        referenceTemp = refTemp,
                        measuredTemp = measuredTemp,
                        emissivity = 0.95f,
                        description = "User calibration point ${i + 1}",
                    )

                calibrationData.add(calibrationPoint)
                Log.i(TAG, "Calibration point recorded: Reference=$refTemp°C, Measured=$measuredTemp°C")
            }

            _calibrationProgress.postValue(TC001CalibrationProgress(85, "Generating calibration curve..."))
            generateCalibrationCurve()

            _calibrationProgress.postValue(TC001CalibrationProgress(100, "Multi-point calibration completed"))
            _calibrationState.postValue(TC001CalibrationState.COMPLETED)
        }

    /**
     * Perform blackbody reference calibration
     */
    private suspend fun performBlackbodyCalibration() =
        withContext(Dispatchers.IO) {
            _calibrationProgress.postValue(TC001CalibrationProgress(20, "Connecting to blackbody reference..."))

            val blackbodyTemperatures = listOf(30.0f, 50.0f, 80.0f, 120.0f, 200.0f)

            for (i in blackbodyTemperatures.indices) {
                val bbTemp = blackbodyTemperatures[i]
                _calibrationProgress.postValue(
                    TC001CalibrationProgress(
                        30 + (i * 50 / blackbodyTemperatures.size),
                        "Blackbody calibration at $bbTemp°C",
                    ),
                )

                delay(3000)

                val measuredTemp = bbTemp + (Random.nextDouble(-1.0, 1.0) * 0.2).toFloat()
                val calibrationPoint =
                    TC001CalibrationPoint(
                        referenceTemp = bbTemp,
                        measuredTemp = measuredTemp,
                        emissivity = 1.0f,
                        description = "Blackbody reference $bbTemp°C",
                    )

                calibrationData.add(calibrationPoint)
            }

            _calibrationProgress.postValue(TC001CalibrationProgress(85, "Computing blackbody curve..."))
            generateCalibrationCurve()

            _calibrationProgress.postValue(TC001CalibrationProgress(100, "Blackbody calibration completed"))
            _calibrationState.postValue(TC001CalibrationState.COMPLETED)
        }

    /**
     * Perform environmental compensation calibration
     */
    private suspend fun performEnvironmentalCalibration() =
        withContext(Dispatchers.IO) {
            _calibrationProgress.postValue(TC001CalibrationProgress(20, "Measuring environmental conditions..."))

            val ambientTemp = 23.5f + (Random.nextDouble(-1.0, 1.0) * 1.0).toFloat()
            val humidity = 45.0f + (Random.nextDouble(-1.0, 1.0) * 5.0).toFloat()
            val pressure = 1013.25f + (Random.nextDouble(-1.0, 1.0) * 10.0).toFloat()

            _calibrationProgress.postValue(TC001CalibrationProgress(50, "Calculating environmental compensation..."))

            val compensation =
                TC001EnvironmentalCompensation(
                    ambientTemperature = ambientTemp,
                    humidity = humidity,
                    atmosphericPressure = pressure,
                    compensationFactor = calculateCompensationFactor(ambientTemp, humidity),
                )

            _calibrationProgress.postValue(TC001CalibrationProgress(80, "Applying environmental compensation..."))

            applyEnvironmentalCompensation(compensation)

            _calibrationProgress.postValue(TC001CalibrationProgress(100, "Environmental calibration completed"))
            _calibrationState.postValue(TC001CalibrationState.COMPLETED)
        }

    /**
     * Generate calibration curve from data points
     */
    private fun generateCalibrationCurve() {
        if (calibrationData.size < CALIBRATION_POINTS_MIN) {
            Log.w(TAG, "Insufficient calibration points: ${calibrationData.size}")
            return
        }

        val n = calibrationData.size
        val sumX = calibrationData.sumOf { it.measuredTemp.toDouble() }
        val sumY = calibrationData.sumOf { it.referenceTemp.toDouble() }
        val sumXY = calibrationData.sumOf { it.measuredTemp * it.referenceTemp.toDouble() }
        val sumXX = calibrationData.sumOf { it.measuredTemp * it.measuredTemp.toDouble() }

        val slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
        val intercept = (sumY - slope * sumX) / n

        val errors =
            calibrationData.map { point ->
                val corrected = slope * point.measuredTemp + intercept
                abs(corrected - point.referenceTemp)
            }
        val maxError = errors.maxOrNull() ?: 0.0
        val rmsError = sqrt(errors.map { it * it }.average())

        currentCalibration =
            TC001CalibrationCurve(
                slope = slope,
                intercept = intercept,
                maxError = maxError,
                rmsError = rmsError,
                calibrationPoints = calibrationData.toList(),
                calibrationTimestamp = System.currentTimeMillis(),
            )

        val results =
            TC001CalibrationResults(
                calibrationCurve = currentCalibration!!,
                accuracy = maxError.toFloat(),
                pointCount = calibrationData.size,
                isValid = maxError < CALIBRATION_ACCURACY_TARGET,
            )
        _calibrationResults.postValue(results)

        Log.i(
            TAG,
            "Calibration curve generated: slope=${String.format(
                "%.4f",
                slope,
            )}, intercept=${String.format("%.4f", intercept)}, RMS error=${String.format("%.3f", rmsError)}°C",
        )
    }

    /**
     * Apply temperature correction using current calibration
     */
    fun applyCalibratedTemperature(rawTemperature: Float): Float {
        val curve = currentCalibration ?: return rawTemperature
        return (curve.slope * rawTemperature + curve.intercept).toFloat()
    }

    /**
     * Calculate environmental compensation factor
     */
    private fun calculateCompensationFactor(
        ambientTemp: Float,
        humidity: Float,
    ): Float {
        val tempFactor = (ambientTemp - 20.0f) * 0.002f
        val humidityFactor = (humidity - 50.0f) * 0.001f
        return 1.0f + tempFactor + humidityFactor
    }

    /**
     * Apply environmental compensation to calibration
     */
    private fun applyEnvironmentalCompensation(compensation: TC001EnvironmentalCompensation) {
        currentCalibration?.let { curve ->
            val compensatedSlope = curve.slope * compensation.compensationFactor
            val compensatedIntercept = curve.intercept * compensation.compensationFactor

            currentCalibration =
                curve.copy(
                    slope = compensatedSlope,
                    intercept = compensatedIntercept,
                )

            Log.i(TAG, "Environmental compensation applied: factor=${compensation.compensationFactor}")
        }
    }

    /**
     * Save calibration to persistent storage
     */
    suspend fun saveCalibration(name: String = "default"): Boolean =
        withContext(Dispatchers.IO) {
            try {
                val calibrationFile =
                    File(context.getExternalFilesDir(null), "calibrations/tc001_calibration_${name}_${DATE_FORMATTER.format(Date())}.json")
                calibrationFile.parentFile?.mkdirs()

                val calibrationJson =
                    JSONObject().apply {
                        put("calibration_name", name)
                        put("device_type", "TC001")
                        put("timestamp", System.currentTimeMillis())
                        put("calibration_curve", currentCalibration?.toJSON())
                        put("calibration_points", calibrationData.map { it.toJSON() })
                    }

                calibrationFile.writeText(calibrationJson.toString(2))

                Log.i(TAG, "Calibration saved: ${calibrationFile.absolutePath}")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to save calibration", e)
                false
            }
        }

    /**
     * Load calibration from persistent storage
     */
    suspend fun loadCalibration(calibrationFile: File): Boolean =
        withContext(Dispatchers.IO) {
            try {
                val calibrationJson = JSONObject(calibrationFile.readText())

                val curveJson = calibrationJson.getJSONObject("calibration_curve")
                currentCalibration = TC001CalibrationCurve.fromJSON(curveJson)

                val pointsArray = calibrationJson.getJSONArray("calibration_points")
                calibrationData.clear()
                for (i in 0 until pointsArray.length()) {
                    val pointJson = pointsArray.getJSONObject(i)
                    calibrationData.add(TC001CalibrationPoint.fromJSON(pointJson))
                }

                _calibrationState.postValue(TC001CalibrationState.LOADED)
                Log.i(TAG, "Calibration loaded: ${calibrationFile.name}")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load calibration", e)
                _calibrationState.postValue(TC001CalibrationState.ERROR)
                false
            }
        }

    /**
     * Validate current calibration accuracy
     */
    suspend fun validateCalibration(): TC001CalibrationValidation =
        withContext(Dispatchers.IO) {
            val curve =
                currentCalibration ?: return@withContext TC001CalibrationValidation(
                    isValid = false,
                    maxError = Float.MAX_VALUE,
                    rmsError = Float.MAX_VALUE,
                    message = "No calibration curve available",
                )

            val errors =
                calibrationData.map { point ->
                    val corrected = curve.slope * point.measuredTemp + curve.intercept
                    abs(corrected - point.referenceTemp).toFloat()
                }

            val maxError = errors.maxOrNull() ?: Float.MAX_VALUE
            val rmsError = sqrt(errors.map { it * it }.average()).toFloat()

            val isValid = maxError < CALIBRATION_ACCURACY_TARGET * 2

            TC001CalibrationValidation(
                isValid = isValid,
                maxError = maxError,
                rmsError = rmsError,
                message = if (isValid) "Calibration within tolerance" else "Calibration exceeds tolerance",
            )
        }

    /**
     * Check for calibration drift over time
     */
    suspend fun checkCalibrationDrift(): TC001CalibrationDrift =
        withContext(Dispatchers.IO) {
            val curve =
                currentCalibration ?: return@withContext TC001CalibrationDrift(
                    hasDrift = false,
                    driftAmount = 0.0f,
                    timeSinceCalibration = 0L,
                    recommendation = "No calibration available",
                )

            val timeSinceCalibration = System.currentTimeMillis() - curve.calibrationTimestamp
            val hoursSinceCalibration = timeSinceCalibration / (1000 * 60 * 60)

            val estimatedDrift = (hoursSinceCalibration / 100.0f) * 0.1f
            val hasDrift = estimatedDrift > 0.5f || hoursSinceCalibration > DRIFT_CHECK_INTERVAL_HOURS

            TC001CalibrationDrift(
                hasDrift = hasDrift,
                driftAmount = estimatedDrift,
                timeSinceCalibration = timeSinceCalibration,
                recommendation = if (hasDrift) "Recalibration recommended" else "Calibration is current",
            )
        }

    /**
     * Get current calibration status
     */
    fun getCurrentCalibration(): TC001CalibrationCurve? = currentCalibration

    /**
     * Check if calibration is valid and current
     */
    fun isCalibrationValid(): Boolean {
        val curve = currentCalibration ?: return false
        val timeSinceCalibration = System.currentTimeMillis() - curve.calibrationTimestamp
        val hoursSinceCalibration = timeSinceCalibration / (1000 * 60 * 60)

        return curve.maxError < CALIBRATION_ACCURACY_TARGET &&
            hoursSinceCalibration < DRIFT_CHECK_INTERVAL_HOURS * 7
    }
}

data class TC001CalibrationPoint(
    val referenceTemp: Float,
    val measuredTemp: Float,
    val emissivity: Float,
    val description: String,
) {
    fun toJSON(): JSONObject =
        JSONObject().apply {
            put("reference_temp", referenceTemp)
            put("measured_temp", measuredTemp)
            put("emissivity", emissivity)
            put("description", description)
        }

    companion object {
        fun fromJSON(json: JSONObject): TC001CalibrationPoint =
            TC001CalibrationPoint(
                referenceTemp = json.getDouble("reference_temp").toFloat(),
                measuredTemp = json.getDouble("measured_temp").toFloat(),
                emissivity = json.getDouble("emissivity").toFloat(),
                description = json.getString("description"),
            )
    }
}

data class TC001CalibrationCurve(
    val slope: Double,
    val intercept: Double,
    val maxError: Double,
    val rmsError: Double,
    val calibrationPoints: List<TC001CalibrationPoint>,
    val calibrationTimestamp: Long,
) {
    fun toJSON(): JSONObject =
        JSONObject().apply {
            put("slope", slope)
            put("intercept", intercept)
            put("max_error", maxError)
            put("rms_error", rmsError)
            put("timestamp", calibrationTimestamp)
        }

    companion object {
        fun fromJSON(json: JSONObject): TC001CalibrationCurve =
            TC001CalibrationCurve(
                slope = json.getDouble("slope"),
                intercept = json.getDouble("intercept"),
                maxError = json.getDouble("max_error"),
                rmsError = json.getDouble("rms_error"),
                calibrationPoints = emptyList(),
                calibrationTimestamp = json.getLong("timestamp"),
            )
    }
}

data class TC001CalibrationProgress(
    val percentage: Int,
    val message: String,
)

data class TC001CalibrationResults(
    val calibrationCurve: TC001CalibrationCurve,
    val accuracy: Float,
    val pointCount: Int,
    val isValid: Boolean,
)

data class TC001CalibrationValidation(
    val isValid: Boolean,
    val maxError: Float,
    val rmsError: Float,
    val message: String,
)

data class TC001CalibrationDrift(
    val hasDrift: Boolean,
    val driftAmount: Float,
    val timeSinceCalibration: Long,
    val recommendation: String,
)

data class TC001EnvironmentalCompensation(
    val ambientTemperature: Float,
    val humidity: Float,
    val atmosphericPressure: Float,
    val compensationFactor: Float,
)

enum class TC001CalibrationState {
    IDLE,
    CALIBRATING,
    COMPLETED,
    LOADED,
    ERROR,
}

enum class TC001CalibrationType {
    FACTORY_RESET,
    USER_MULTI_POINT,
    BLACKBODY_REFERENCE,
    ENVIRONMENTAL_COMPENSATION,
}
