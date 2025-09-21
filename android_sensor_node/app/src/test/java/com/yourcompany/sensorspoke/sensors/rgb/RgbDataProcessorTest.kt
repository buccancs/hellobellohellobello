package com.yourcompany.sensorspoke.sensors.rgb

import org.junit.Assert.assertTrue
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertFalse
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Unit tests for RgbDataProcessor to validate frame data processing and CSV formatting.
 * These tests ensure proper frame metadata handling and synchronization calculations.
 */
class RgbDataProcessorTest {

    private lateinit var dataProcessor: RgbDataProcessor

    @Before
    fun setUp() {
        dataProcessor = RgbDataProcessor()
    }

    @Test
    fun testCsvHeaderFormat() {
        val header = dataProcessor.getCsvHeader()
        val expectedHeader = "timestamp_ns,timestamp_ms,frame_number,filename,file_size_bytes,video_relative_time_ms,video_frame_estimate,sync_quality,actual_video_offset_ms"
        assertEquals("CSV header should match expected format", expectedHeader, header)
    }

    @Test
    fun testFrameDataCreation() {
        val timestampNs = 1000000000L
        val timestampMs = 1000L
        val frameNumber = 1
        val imageFile = File("/tmp/test_frame.jpg")
        val videoStartTime = 500000000L
        val actualVideoStartTime = 600000000L

        val frameData = dataProcessor.createFrameData(
            timestampNs,
            timestampMs,
            frameNumber,
            imageFile,
            videoStartTime,
            actualVideoStartTime,
        )

        assertEquals("Timestamp in nanoseconds should match", timestampNs, frameData.timestampNs)
        assertEquals("Timestamp in milliseconds should match", timestampMs, frameData.timestampMs)
        assertEquals("Frame number should match", frameNumber, frameData.frameNumber)
        assertEquals("Filename should match", "test_frame.jpg", frameData.filename)
        assertEquals("File size should be 0 for non-existent file", 0L, frameData.fileSizeBytes)

        val expectedVideoRelativeTime = ((timestampNs - actualVideoStartTime) / 1_000_000).toInt()
        assertEquals("Video relative time should be calculated correctly", expectedVideoRelativeTime, frameData.videoRelativeTimeMs)
    }

    @Test
    fun testFormatFrameDataForCsv() {
        val frameData = RgbDataProcessor.FrameData(
            timestampNs = 1000000000L,
            timestampMs = 1000L,
            frameNumber = 1,
            filename = "test_frame.jpg",
            fileSizeBytes = 123456L,
            videoRelativeTimeMs = 500,
            estimatedVideoFrame = 15,
            syncQuality = 0.95,
            actualVideoOffsetMs = 400,
        )

        val csvLine = dataProcessor.formatFrameDataForCsv(frameData)
        val expectedLine = "1000000000,1000,1,test_frame.jpg,123456,500,15,0.950,400"
        assertEquals("CSV formatted line should match expected format", expectedLine, csvLine)
    }

    @Test
    fun testSyncQualityCalculationWithoutVideoTiming() {
        val frameTimestamp = 1000000000L
        val videoBaseTime = 0L // No video timing reference

        val syncQuality = dataProcessor.calculateSyncQuality(frameTimestamp, videoBaseTime)
        assertEquals("Sync quality should be 0.0 without video timing", 0.0, syncQuality, 0.001)
    }

    @Test
    fun testSyncQualityCalculationWithVideoTiming() {
        val videoBaseTime = 1000000000L
        val frameTimestamp = videoBaseTime + 33333333L

        val syncQuality = dataProcessor.calculateSyncQuality(frameTimestamp, videoBaseTime)
        assertTrue("Sync quality should be high for perfect timing", syncQuality > 0.95)
    }

    @Test
    fun testSyncQualityCalculationWithPoorTiming() {
        val videoBaseTime = 1000000000L
        val frameTimestamp = videoBaseTime + 16666666L

        val syncQuality = dataProcessor.calculateSyncQuality(frameTimestamp, videoBaseTime)
        assertTrue("Sync quality should be lower for poor timing", syncQuality < 0.8)
    }

    @Test
    fun testTimingStatistics() {
        val videoStartTime = 1000000000L
        val actualVideoStartTime = 1100000000L
        val frameTimestampOffset = 100000000L
        val frameCount = 30

        val stats = dataProcessor.getTimingStatistics(
            videoStartTime,
            actualVideoStartTime,
            frameTimestampOffset,
            frameCount,
        )

        assertEquals("Video start time should match", videoStartTime, stats["videoStartTime"])
        assertEquals("Actual video start time should match", actualVideoStartTime, stats["actualVideoStartTime"])
        assertEquals("Frame timestamp offset should match", frameTimestampOffset, stats["frameTimestampOffset"])
        assertEquals("Frame count should match", frameCount, stats["totalFramesCaptured"])
        assertEquals("Sync quality metric should be available", true, stats["syncQualityMetricAvailable"])
    }

    @Test
    fun testFrameDataWithZeroVideoStartTime() {
        val timestampNs = 1000000000L
        val timestampMs = 1000L
        val frameNumber = 1
        val imageFile = File("/tmp/test_frame.jpg")
        val videoStartTime = 0L
        val actualVideoStartTime = 0L

        val frameData = dataProcessor.createFrameData(
            timestampNs,
            timestampMs,
            frameNumber,
            imageFile,
            videoStartTime,
            actualVideoStartTime,
        )

        assertEquals("Video relative time should be 0", 0, frameData.videoRelativeTimeMs)
        assertEquals("Estimated video frame should be 0", 0, frameData.estimatedVideoFrame)
        assertEquals("Sync quality should be 0.0", 0.0, frameData.syncQuality, 0.001)
        assertEquals("Actual video offset should be -1", -1, frameData.actualVideoOffsetMs)
    }

    @Test
    fun testEstimatedVideoFrameCalculation() {
        val timestampNs = 1000000000L
        val timestampMs = 1000L
        val frameNumber = 1
        val imageFile = File("/tmp/test_frame.jpg")
        val videoStartTime = 500000000L
        val actualVideoStartTime = 500000000L

        val frameData = dataProcessor.createFrameData(
            timestampNs,
            timestampMs,
            frameNumber,
            imageFile,
            videoStartTime,
            actualVideoStartTime,
        )

        val expectedVideoFrame = ((frameData.videoRelativeTimeMs * 30.0) / 1000.0).toInt()
        assertEquals("Estimated video frame should be calculated correctly", expectedVideoFrame, frameData.estimatedVideoFrame)
    }
}
