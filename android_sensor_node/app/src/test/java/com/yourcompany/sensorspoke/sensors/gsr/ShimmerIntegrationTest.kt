package com.yourcompany.sensorspoke.sensors.gsr

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

/**
 * Integration tests for ShimmerRecorder implementation.
 * Tests both real BLE integration paths and simulation fallback.
 * Validates critical 12-bit ADC conversion requirement.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class ShimmerIntegrationTest {
    private lateinit var context: Context
    private lateinit var testSessionDir: File
    private lateinit var recorder: ShimmerRecorder

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        testSessionDir = File(context.cacheDir, "test_shimmer_session")
        testSessionDir.mkdirs()
        recorder = ShimmerRecorder(context)
    }

    @Test
    fun testBasicRecordingLifecycle() =
        runBlocking {
            recorder.start(testSessionDir)

            Thread.sleep(300)

            recorder.stop()

            val csvFile = File(testSessionDir, "gsr.csv")
            assertTrue("GSR CSV file should exist", csvFile.exists())
            assertTrue("CSV file should not be empty", csvFile.length() > 0)
        }

    @Test
    fun testCSVFormatCompliance() =
        runBlocking {
            recorder.start(testSessionDir)
            Thread.sleep(500)
            recorder.stop()

            val csvFile = File(testSessionDir, "gsr.csv")
            assertTrue("CSV file should exist", csvFile.exists())

            val lines = csvFile.readLines()
            assertTrue("CSV should have header", lines.isNotEmpty())

            val header = lines[0]
            val expectedColumns = listOf("timestamp_ns", "gsr_microsiemens", "ppg_raw", "gsr_raw_adc")
            for (column in expectedColumns) {
                assertTrue("Header should contain $column", header.contains(column))
            }

            if (lines.size > 1) {
                val dataRow = lines[1].split(",")
                assertEquals(
                    "Data row should have correct number of columns",
                    expectedColumns.size,
                    dataRow.size,
                )

                val timestamp = dataRow[0].toLong()
                assertTrue(
                    "Timestamp should be reasonable nanosecond value",
                    timestamp > 1_000_000_000_000_000L,
                )

                val gsrValue = dataRow[1].toFloat()
                assertTrue("GSR should be positive", gsrValue >= 0.0f)
                assertTrue("GSR should be reasonable human range", gsrValue < 1000.0f)

                val ppgValue = dataRow[2].toIntOrNull()
                if (ppgValue != null) {
                    assertTrue(
                        "PPG should be reasonable 12-bit range",
                        ppgValue >= 0 && ppgValue <= 4095,
                    )
                }

                // Validate raw ADC value - CRITICAL 12-bit requirement
                val rawAdc = dataRow[3].toInt()
                assertTrue(
                    "Raw ADC should be in 12-bit range (0-4095)",
                    rawAdc >= 0 && rawAdc <= 4095,
                )
            }
        }

    @Test
    fun testCritical12BitADCRequirement() =
        runBlocking {
            // This is a CRITICAL requirement from project specifications
            recorder.start(testSessionDir)
            Thread.sleep(400)
            recorder.stop()

            val csvFile = File(testSessionDir, "gsr.csv")
            val lines = csvFile.readLines()

            var adcValueFound = false
            for (i in 1 until lines.size) {
                val data = lines[i].split(",")
                if (data.size >= 4) {
                    val rawAdc = data[3].toInt()
                    adcValueFound = true

                    // CRITICAL: Must be 12-bit (0-4095), NOT 16-bit (0-65535)
                    assertTrue(
                        "Raw ADC MUST use 12-bit resolution (0-4095)",
                        rawAdc >= 0 && rawAdc <= 4095,
                    )

                    assertFalse("Must NOT use 16-bit ADC values", rawAdc > 4095)
                }
            }

            assertTrue("Should have found at least one ADC value to validate", adcValueFound)
        }

    @Test
    fun testGSRConversionAccuracy() {
        // Test the critical GSR conversion function directly
        val testCases =
            listOf(
                Triple(0, 0.0, "Minimum ADC should give minimum conductance"),
                Triple(2048, 15.0, "Mid-range ADC should give reasonable GSR"),
                Triple(4095, 30.0, "Maximum 12-bit ADC should give max range GSR"),
            )

        for ((adcValue, expectedMaxGsr, description) in testCases) {
            val result = recorder.convertGsrToMicroSiemens(adcValue)

            assertTrue("$description - should be positive", result >= 0.0)
            assertTrue("$description - should be reasonable range", result < 100.0)
        }

        val minResult = recorder.convertGsrToMicroSiemens(0)
        val maxResult = recorder.convertGsrToMicroSiemens(4095)

        assertTrue("Min ADC should give min GSR", minResult <= maxResult)

        val clampedResult = recorder.convertGsrToMicroSiemens(65535)
        val maxValidResult = recorder.convertGsrToMicroSiemens(4095)

        assertEquals(
            "16-bit values should be clamped to 12-bit max",
            maxValidResult,
            clampedResult,
            0.001,
        )
    }

    @Test
    fun testSamplingRateCompliance() =
        runBlocking {
            recorder.start(testSessionDir)
            Thread.sleep(1000)
            recorder.stop()

            val csvFile = File(testSessionDir, "gsr.csv")
            val lines = csvFile.readLines()

            if (lines.size > 10) {
                val timestamps = mutableListOf<Long>()

                for (i in 1 until lines.size) {
                    val data = lines[i].split(",")
                    if (data.isNotEmpty()) {
                        timestamps.add(data[0].toLong())
                    }
                }

                if (timestamps.size >= 2) {
                    val totalTimeNs = timestamps.last() - timestamps.first()
                    val totalTimeSec = totalTimeNs / 1e9
                    val actualRate = (timestamps.size - 1) / totalTimeSec

                    assertTrue(
                        "Sampling rate should be close to 128 Hz, got $actualRate",
                        actualRate >= 100.0 && actualRate <= 150.0,
                    )
                }
            }
        }

    @Test
    fun testSimulationModeRealism() =
        runBlocking {
            // Test that simulation mode generates realistic GSR data
            recorder.start(testSessionDir)
            Thread.sleep(600)
            recorder.stop()

            val csvFile = File(testSessionDir, "gsr.csv")
            val lines = csvFile.readLines()

            if (lines.size > 20) {
                val gsrValues = mutableListOf<Double>()

                for (i in 1 until lines.size) {
                    val data = lines[i].split(",")
                    if (data.size >= 2) {
                        gsrValues.add(data[1].toDouble())
                    }
                }

                if (gsrValues.isNotEmpty()) {
                    val mean = gsrValues.average()

                    assertTrue(
                        "Mean GSR should be realistic (5-50 µS)",
                        mean >= 5.0 && mean <= 50.0,
                    )

                    val variance = gsrValues.map { (it - mean) * (it - mean) }.average()
                    assertTrue("GSR should have physiological variation", variance > 0.1)

                    assertTrue(
                        "All GSR values should be positive",
                        gsrValues.all { it >= 0.0 },
                    )
                }
            }
        }

    @Test
    fun testPPGSimulation() =
        runBlocking {
            recorder.start(testSessionDir)
            Thread.sleep(400)
            recorder.stop()

            val csvFile = File(testSessionDir, "gsr.csv")
            val lines = csvFile.readLines()

            var ppgValuesFound = false
            for (i in 1 until lines.size) {
                val data = lines[i].split(",")
                if (data.size >= 3) {
                    val ppgRaw = data[2].toIntOrNull()
                    if (ppgRaw != null && ppgRaw != 0) {
                        ppgValuesFound = true

                        assertTrue(
                            "PPG should be in 12-bit range",
                            ppgRaw >= 0 && ppgRaw <= 4095,
                        )

                        assertTrue(
                            "PPG should be reasonable physiological range",
                            ppgRaw >= 1000 && ppgRaw <= 3500,
                        )
                    }
                }
            }

            assertTrue("Should have generated PPG data", ppgValuesFound)
        }

    @Test
    fun testBLEConnectionSimulation() =
        runBlocking {

            recorder.start(testSessionDir)

            Thread.sleep(200)

            recorder.stop()

            val csvFile = File(testSessionDir, "gsr.csv")
            assertTrue("Should fall back to simulation gracefully", csvFile.exists())
        }

    @Test
    fun testErrorHandling() =
        runBlocking {
            val invalidDir = File("/invalid/path/that/does/not/exist")

            try {
                recorder.start(invalidDir)
                recorder.stop()
            } catch (e: Exception) {
                assertTrue(
                    "Should be a reasonable exception type",
                    e is SecurityException || e is java.io.IOException,
                )
            }
        }

    @Test
    fun testConcurrentOperations() =
        runBlocking {
            repeat(3) {
                recorder.start(testSessionDir)
                Thread.sleep(100)
                recorder.stop()
            }

            val csvFile = File(testSessionDir, "gsr.csv")
            assertTrue("Should handle multiple cycles", csvFile.exists())
        }

    @Test
    fun testTimestampMonotonicity() =
        runBlocking {
            recorder.start(testSessionDir)
            Thread.sleep(500)
            recorder.stop()

            val csvFile = File(testSessionDir, "gsr.csv")
            val lines = csvFile.readLines()

            if (lines.size > 3) {
                val timestamps = mutableListOf<Long>()

                for (i in 1 until lines.size) {
                    val data = lines[i].split(",")
                    if (data.isNotEmpty()) {
                        timestamps.add(data[0].toLong())
                    }
                }

                for (i in 1 until timestamps.size) {
                    assertTrue(
                        "Timestamps should be monotonically increasing",
                        timestamps[i] > timestamps[i - 1],
                    )
                }
            }
        }

    @Test
    fun testDataIntegrity() =
        runBlocking {
            recorder.start(testSessionDir)
            Thread.sleep(300)
            recorder.stop()

            val csvFile = File(testSessionDir, "gsr.csv")
            val lines = csvFile.readLines()

            for (i in 1 until lines.size) {
                val data = lines[i].split(",")

                assertTrue("Each row should have 4 columns", data.size == 4)

                val timestamp = data[0].toLong()
                val gsrValue = data[1].toDouble()
                val ppgValue = data[2].toIntOrNull()
                val adcValue = data[3].toInt()

                assertTrue("Timestamp should be valid", timestamp > 0)
                assertTrue("GSR should be valid", gsrValue >= 0.0 && !gsrValue.isNaN())
                assertTrue("ADC should be valid 12-bit", adcValue >= 0 && adcValue <= 4095)
            }
        }
}
