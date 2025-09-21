package com.yourcompany.sensorspoke.utils

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Unit tests for the Android UserExperience utilities, ensuring consistent
 * user-friendly error translation and status formatting across the platform.
 */
@RunWith(AndroidJUnit4::class)
class UserExperienceTest {
    private lateinit var context: Context

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
    }

    @Test
    fun testNetworkErrorTranslation() {
        val connectionRefusedError = "java.net.ConnectException: Connection refused"
        val translatedMessage = UserExperience.ErrorTranslator.translateNetworkError(connectionRefusedError, "network")

        assertTrue("Should provide WiFi troubleshooting", translatedMessage.contains("WiFi network"))
        assertTrue("Should mention PC Hub", translatedMessage.contains("PC Hub"))
        assertTrue("Should provide actionable steps", translatedMessage.contains("Please check"))
    }

    @Test
    fun testSensorErrorTranslation() {
        val gsrError = "Device not found"
        val translatedMessage = UserExperience.ErrorTranslator.translateSensorError("shimmer", gsrError)

        assertTrue("Should mention Shimmer device", translatedMessage.contains("Shimmer"))
        assertTrue("Should provide Bluetooth guidance", translatedMessage.contains("Bluetooth"))
        assertTrue("Should suggest power cycling", translatedMessage.contains("power cycling"))
    }

    @Test
    fun testThermalCameraErrorTranslation() {
        val thermalError = "USB device not accessible"
        val translatedMessage = UserExperience.ErrorTranslator.translateSensorError("thermal", thermalError)

        assertTrue("Should mention USB connection", translatedMessage.contains("USB"))
        assertTrue("Should suggest cable check", translatedMessage.contains("cable"))
        assertTrue("Should provide USB troubleshooting", translatedMessage.contains("USB port"))
    }

    @Test
    fun testRgbCameraErrorTranslation() {
        val cameraError = "Camera permission denied"
        val translatedMessage = UserExperience.ErrorTranslator.translateSensorError("rgb", cameraError)

        assertTrue("Should mention camera permission", translatedMessage.contains("permission"))
        assertTrue("Should suggest Settings access", translatedMessage.contains("Settings"))
        assertTrue("Should mention app restart", translatedMessage.contains("restart"))
    }

    @Test
    fun testConnectionStatusFormatting() {
        val connectedStatus = UserExperience.StatusFormatter.formatConnectionStatus(true, "192.168.1.100:8080")
        assertEquals("Connected to PC Hub: 192.168.1.100:8080", connectedStatus)

        val disconnectedStatus = UserExperience.StatusFormatter.formatConnectionStatus(false)
        assertEquals("Not connected to PC Hub", disconnectedStatus)
    }

    @Test
    fun testRecordingStatusFormatting() {
        val recordingStatus = UserExperience.StatusFormatter.formatRecordingStatus(true, "session_123", 125)
        assertTrue("Should contain session ID", recordingStatus.contains("session_123"))
        assertTrue("Should show duration", recordingStatus.contains("2:05"))

        val readyStatus = UserExperience.StatusFormatter.formatRecordingStatus(false)
        assertEquals("Ready to record", readyStatus)
    }

    @Test
    fun testSensorStatusFormatting() {
        val connectedSensor = UserExperience.StatusFormatter.formatSensorStatus("Shimmer GSR", true, false)
        assertEquals("Shimmer GSR: Connected", connectedSensor)

        val recordingSensor = UserExperience.StatusFormatter.formatSensorStatus("RGB Camera", true, true)
        assertEquals("RGB Camera: Recording", recordingSensor)

        val disconnectedSensor = UserExperience.StatusFormatter.formatSensorStatus("Thermal Camera", false, false)
        assertEquals("Thermal Camera: Disconnected", disconnectedSensor)
    }

    @Test
    fun testFileLocationFormatting() {
        val filePath = "/storage/emulated/0/SensorSpoke/recordings/session_123/rgb_video.mp4"
        val formattedLocation = UserExperience.StatusFormatter.formatFileLocation(filePath)
        assertEquals("Saved to: rgb_video.mp4", formattedLocation)
    }

    @Test
    fun testDataExportFormatting() {
        val exportInfo = UserExperience.StatusFormatter.formatDataExportInfo("HDF5", 5, 52428800L)
        assertTrue("Should contain file count", exportInfo.contains("5 files"))
        assertTrue("Should contain format", exportInfo.contains("HDF5"))
        assertTrue("Should contain size in MB", exportInfo.contains("50MB"))
    }

    @Test
    fun testQuickStartSteps() {
        val steps = UserExperience.QuickStart.getSetupSteps()

        assertEquals("Should have 6 setup steps", 6, steps.size)
        assertTrue("First step should be about WiFi", steps[0].first.contains("WiFi"))
        assertTrue("Should include permissions step", steps.any { it.first.contains("Permissions") })
        assertTrue("Should include sensor setup", steps.any { it.first.contains("Sensors") })
        assertTrue("Should include PC Hub discovery", steps.any { it.first.contains("PC Hub") })
        assertTrue("Should include recording step", steps.any { it.first.contains("Recording") })
    }

    @Test
    fun testConnectionTroubleshootingSteps() {
        val troubleshootingSteps = UserExperience.QuickStart.getConnectionTroubleshootingSteps()

        assertTrue("Should have multiple steps", troubleshootingSteps.size >= 5)
        assertTrue("Should mention PC Hub", troubleshootingSteps.any { it.contains("PC Hub") })
        assertTrue("Should mention WiFi network", troubleshootingSteps.any { it.contains("WiFi network") })
        assertTrue("Should mention firewall", troubleshootingSteps.any { it.contains("firewall") })
        assertTrue("Should suggest restart", troubleshootingSteps.any { it.contains("restart") })
    }

    @Test
    fun testPermissionExplanations() {
        val explanations = UserExperience.QuickStart.getPermissionExplanations()

        assertTrue("Should explain camera permission", explanations.containsKey("camera"))
        assertTrue("Should explain storage permission", explanations.containsKey("storage"))
        assertTrue("Should explain bluetooth permission", explanations.containsKey("bluetooth"))

        val cameraExplanation = explanations["camera"] ?: ""
        assertTrue("Camera explanation should mention video recording", cameraExplanation.contains("video"))
        assertTrue("Camera explanation should mention synchronization", cameraExplanation.contains("synchronization"))
    }

    @Test
    fun testTimeoutErrorTranslation() {
        val timeoutError = "java.net.SocketTimeoutException: Connection timeout"
        val translatedMessage = UserExperience.ErrorTranslator.translateNetworkError(timeoutError, "network")

        assertTrue("Should identify as timeout", translatedMessage.contains("timeout"))
        assertTrue("Should suggest checking PC Hub", translatedMessage.contains("PC Hub"))
        assertTrue("Should suggest checking network", translatedMessage.contains("Network"))
    }

    @Test
    fun testBluetoothErrorTranslation() {
        val bluetoothError = "Bluetooth adapter not found"
        val translatedMessage = UserExperience.ErrorTranslator.translateNetworkError(bluetoothError, "network")

        assertTrue("Should mention Bluetooth", translatedMessage.contains("Bluetooth"))
        assertTrue("Should suggest checking Bluetooth state", translatedMessage.contains("enabled"))
        assertTrue("Should mention Shimmer sensor", translatedMessage.contains("Shimmer"))
    }

    @Test
    fun testStorageErrorTranslation() {
        val storageError = "java.io.IOException: No space left on device"
        val translatedMessage = UserExperience.ErrorTranslator.translateNetworkError(storageError, "network")

        assertTrue("Should mention storage", translatedMessage.contains("Storage"))
        assertTrue("Should suggest checking space", translatedMessage.contains("storage space"))
        assertTrue("Should mention permissions", translatedMessage.contains("permissions"))
    }
}
