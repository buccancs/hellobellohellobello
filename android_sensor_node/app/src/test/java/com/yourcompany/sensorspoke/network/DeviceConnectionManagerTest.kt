package com.yourcompany.sensorspoke.network

import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertFalse
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for DeviceConnectionManager to validate unified device state management.
 * Simplified tests focusing on core functionality without Android dependencies.
 */
class DeviceConnectionManagerTest {

    private lateinit var deviceManager: DeviceConnectionManager

    @Before
    fun setUp() {
        deviceManager = DeviceConnectionManager(null)
    }

    @After
    fun tearDown() {
        deviceManager.cleanup()
    }

    @Test
    fun testInitialState() {
        assertEquals("Connected device count should be 0", 0, deviceManager.getConnectedDeviceCount())
        assertFalse("System should not be ready initially", deviceManager.isReadyForRecording())
    }

    @Test
    fun testShimmerStateUpdate() {
        val deviceDetails = DeviceConnectionManager.DeviceDetails(
            deviceType = "Shimmer3 GSR+",
            deviceName = "Test Shimmer",
            connectionState = DeviceConnectionManager.DeviceState.CONNECTED,
            batteryLevel = 80,
            dataRate = 128.0,
            isRequired = true,
        )

        deviceManager.updateShimmerState(DeviceConnectionManager.DeviceState.CONNECTED, deviceDetails)

        assertEquals("Connected device count should be 1", 1, deviceManager.getConnectedDeviceCount())
    }

    @Test
    fun testMultipleDeviceConnections() {
        deviceManager.updateShimmerState(DeviceConnectionManager.DeviceState.CONNECTED)
        deviceManager.updateRgbCameraState(DeviceConnectionManager.DeviceState.CONNECTED)
        deviceManager.updateThermalCameraState(DeviceConnectionManager.DeviceState.CONNECTED)

        assertEquals("Connected device count should be 3", 3, deviceManager.getConnectedDeviceCount())
        assertTrue("System should be ready for recording", deviceManager.isReadyForRecording())
    }

    @Test
    fun testAllDevicesConnected() {
        deviceManager.updateShimmerState(DeviceConnectionManager.DeviceState.CONNECTED)
        deviceManager.updateRgbCameraState(DeviceConnectionManager.DeviceState.CONNECTED)
        deviceManager.updateThermalCameraState(DeviceConnectionManager.DeviceState.CONNECTED)
        deviceManager.updateAudioDeviceState(DeviceConnectionManager.DeviceState.CONNECTED)
        deviceManager.updateNetworkState(DeviceConnectionManager.DeviceState.CONNECTED)

        assertEquals("Connected device count should be 5", 5, deviceManager.getConnectedDeviceCount())
        assertTrue("System should be ready for recording", deviceManager.isReadyForRecording())
    }

    @Test
    fun testPartiallyReadyState() {
        deviceManager.updateShimmerState(DeviceConnectionManager.DeviceState.CONNECTED)
        deviceManager.updateRgbCameraState(DeviceConnectionManager.DeviceState.CONNECTED)

        assertEquals("Connected device count should be 2", 2, deviceManager.getConnectedDeviceCount())
        assertTrue("System should be ready for recording in partial state", deviceManager.isReadyForRecording())
    }

    @Test
    fun testDeviceStateSummary() {
        deviceManager.updateShimmerState(DeviceConnectionManager.DeviceState.CONNECTED)
        deviceManager.updateRgbCameraState(DeviceConnectionManager.DeviceState.CONNECTING)
        deviceManager.updateThermalCameraState(DeviceConnectionManager.DeviceState.ERROR)

        val summary = deviceManager.getDeviceStateSummary()

        assertEquals("Summary should contain all devices", 5, summary.size)
        assertEquals(
            "Shimmer state should be correct",
            DeviceConnectionManager.DeviceState.CONNECTED,
            summary["shimmer"],
        )
        assertEquals(
            "RGB camera state should be correct",
            DeviceConnectionManager.DeviceState.CONNECTING,
            summary["rgb_camera"],
        )
        assertEquals(
            "Thermal camera state should be correct",
            DeviceConnectionManager.DeviceState.ERROR,
            summary["thermal_camera"],
        )
    }

    @Test
    fun testResetAllStates() {
        deviceManager.updateShimmerState(DeviceConnectionManager.DeviceState.CONNECTED)
        deviceManager.updateRgbCameraState(DeviceConnectionManager.DeviceState.CONNECTED)

        deviceManager.resetAllStates()

        assertEquals("Connected device count should be 0 after reset", 0, deviceManager.getConnectedDeviceCount())
        assertFalse("System should not be ready after reset", deviceManager.isReadyForRecording())
    }

    @Test
    fun testCleanup() {
        deviceManager.updateShimmerState(DeviceConnectionManager.DeviceState.CONNECTED)
        deviceManager.updateRgbCameraState(DeviceConnectionManager.DeviceState.CONNECTED)

        deviceManager.cleanup()

        assertEquals("Connected device count should be 0 after cleanup", 0, deviceManager.getConnectedDeviceCount())
        assertFalse("System should not be ready after cleanup", deviceManager.isReadyForRecording())
    }
}
