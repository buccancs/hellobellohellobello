package com.yourcompany.sensorspoke.sensors.gsr

import org.junit.Assert.assertTrue
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertFalse
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for ShimmerDataProcessor to validate ObjectCluster conversion logic.
 * Simplified tests focusing on core functionality without Android dependencies.
 */
class ShimmerDataProcessorTest {

    private lateinit var dataProcessor: ShimmerDataProcessor

    @Before
    fun setUp() {
        dataProcessor = ShimmerDataProcessor()
    }

    @Test
    fun testCsvHeaderFormat() {
        val header = dataProcessor.getCsvHeader()
        val expectedHeader = "timestamp_ns,timestamp_ms,sample_number,gsr_kohms,gsr_raw_12bit,ppg_raw,connection_status,data_integrity"
        assertEquals("CSV header should match expected format", expectedHeader, header)
    }

    @Test
    fun testSensorSampleCreation() {
        val sample = ShimmerDataProcessor.SensorSample(
            timestampNs = 1000000000L,
            timestampMs = 1000L,
            gsrKohms = 50.5,
            gsrRaw12bit = 2048,
            ppgRaw = 1500,
            connectionStatus = "CONNECTED",
            dataIntegrity = "OK",
        )

        assertEquals("Timestamp in nanoseconds should match", 1000000000L, sample.timestampNs)
        assertEquals("Timestamp in milliseconds should match", 1000L, sample.timestampMs)
        assertEquals("GSR value should match", 50.5, sample.gsrKohms, 0.1)
        assertEquals("GSR raw value should match", 2048, sample.gsrRaw12bit)
        assertEquals("PPG raw value should match", 1500, sample.ppgRaw)
        assertEquals("Connection status should match", "CONNECTED", sample.connectionStatus)
        assertEquals("Data integrity should match", "OK", sample.dataIntegrity)
    }

    @Test
    fun testFormatSampleForCsv() {
        val sample = ShimmerDataProcessor.SensorSample(
            timestampNs = 1000000000L,
            timestampMs = 1000L,
            gsrKohms = 50.123,
            gsrRaw12bit = 2048,
            ppgRaw = 1500,
            connectionStatus = "CONNECTED",
            dataIntegrity = "OK",
        )

        val csvLine = dataProcessor.formatSampleForCsv(sample, 1)
        val expectedLine = "1000000000,1000,1,50.123,2048,1500,CONNECTED,OK"
        assertEquals("CSV formatted line should match expected format", expectedLine, csvLine)
    }

    @Test
    fun testCsvFormattingWithPrecision() {
        val sample = ShimmerDataProcessor.SensorSample(
            timestampNs = 1234567890123456L,
            timestampMs = 1234567890L,
            gsrKohms = 123.456789,
            gsrRaw12bit = 3000,
            ppgRaw = 2500,
            connectionStatus = "CONNECTED",
            dataIntegrity = "OK",
        )

        val csvLine = dataProcessor.formatSampleForCsv(sample, 42)

        assertTrue("CSV should contain properly formatted GSR value", csvLine.contains("123.457"))
        assertTrue("CSV should contain sample number", csvLine.contains("42"))
        assertTrue("CSV should contain all required fields", csvLine.split(",").size == 8)
    }

    @Test
    fun testGsrRaw12BitRange() {
        val sample = ShimmerDataProcessor.SensorSample(
            timestampNs = System.nanoTime(),
            timestampMs = System.currentTimeMillis(),
            gsrKohms = 100.0,
            gsrRaw12bit = 4095,
            ppgRaw = 2000,
            connectionStatus = "CONNECTED",
            dataIntegrity = "OK",
        )

        assertTrue("GSR raw value should be within 12-bit range", sample.gsrRaw12bit <= 4095)
        assertTrue("GSR raw value should be non-negative", sample.gsrRaw12bit >= 0)
    }
}
