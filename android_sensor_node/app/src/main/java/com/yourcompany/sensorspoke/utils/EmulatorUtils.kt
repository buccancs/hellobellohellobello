package com.yourcompany.sensorspoke.utils

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build

/**
 * Utility class to detect emulator environment and check hardware capabilities
 * to gracefully handle missing features that might cause crashes.
 */
object EmulatorUtils {
    
    /**
     * Detect if the app is running on an Android emulator.
     * Uses multiple heuristics to reliably detect emulator environment.
     */
    fun isRunningOnEmulator(): Boolean {
        return (Build.FINGERPRINT.startsWith("generic") ||
                Build.FINGERPRINT.startsWith("unknown") ||
                Build.MODEL.contains("google_sdk") ||
                Build.MODEL.contains("Emulator") ||
                Build.MODEL.contains("Android SDK built for x86") ||
                Build.MANUFACTURER.contains("Genymotion") ||
                Build.BRAND.startsWith("generic") && Build.DEVICE.startsWith("generic") ||
                "google_sdk" == Build.PRODUCT)
    }
    
    /**
     * Check if camera hardware is available and functional
     */
    fun isCameraAvailable(context: Context): Boolean {
        return context.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)
    }
    
    /**
     * Check if Bluetooth is available
     */
    fun isBluetoothAvailable(context: Context): Boolean {
        return context.packageManager.hasSystemFeature(PackageManager.FEATURE_BLUETOOTH)
    }
    
    /**
     * Check if USB host is available
     */
    fun isUsbHostAvailable(context: Context): Boolean {
        return context.packageManager.hasSystemFeature(PackageManager.FEATURE_USB_HOST)
    }
    
    /**
     * Get a safe mode description for logging/debugging
     */
    fun getEnvironmentInfo(): String {
        return "Environment: ${if (isRunningOnEmulator()) "Emulator" else "Physical Device"}, " +
                "Model: ${Build.MODEL}, " +
                "Fingerprint: ${Build.FINGERPRINT}, " +
                "ABI: ${Build.SUPPORTED_ABIS.joinToString(",")}"
    }
}