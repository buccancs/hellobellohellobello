package com.yourcompany.sensorspoke.utils

import android.content.Context
import android.content.pm.PackageManager
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import com.google.common.truth.Truth.assertThat

/**
 * Unit tests for EmulatorUtils to verify emulator detection and hardware capability checking.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class EmulatorUtilsTest {

    @Test
    fun testGetEnvironmentInfo_returnsValidInfo() {
        val info = EmulatorUtils.getEnvironmentInfo()
        
        // Should contain basic environment information
        assertThat(info).contains("Environment:")
        assertThat(info).contains("Model:")
        assertThat(info).contains("Fingerprint:")
        assertThat(info).contains("ABI:")
    }

    @Test
    fun testIsRunningOnEmulator_robolectricDetected() {
        // Robolectric simulates emulator environment
        val isEmulator = EmulatorUtils.isRunningOnEmulator()
        
        // Should detect Robolectric as emulator-like environment
        assertThat(isEmulator).isTrue()
    }

    @Test
    fun testHardwareCapabilities_contextNotNull() {
        val context = RuntimeEnvironment.getApplication()
        
        // These calls should not crash even if features are not available
        val hasCamera = EmulatorUtils.isCameraAvailable(context)
        val hasBluetooth = EmulatorUtils.isBluetoothAvailable(context)
        val hasUsbHost = EmulatorUtils.isUsbHostAvailable(context)
        
        // Results should be boolean (not crash)
        assertThat(hasCamera).isAnyOf(true, false)
        assertThat(hasBluetooth).isAnyOf(true, false)
        assertThat(hasUsbHost).isAnyOf(true, false)
    }
}