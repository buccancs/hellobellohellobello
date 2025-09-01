package com.yourcompany.sensorspoke.utils

import android.content.Context
import android.widget.Toast

/**
 * UserExperience utilities for Android Sensor Node, providing user-friendly
 * error translation, status formatting, and messaging capabilities.
 *
 * This mirrors the PC Controller's user experience enhancements to provide
 * consistent, research-friendly messaging across both platforms.
 */
object UserExperience {
    /**
     * Translates technical errors into user-friendly, actionable messages
     * with Android-specific troubleshooting guidance.
     */
    object ErrorTranslator {
        fun translateNetworkError(
            error: String,
            context: String,
        ): String =
            when {
                error.contains("Connection refused") ->
                    "Unable to connect to PC Hub. Please check:\n• PC Hub is running\n• Both devices are on the same WiFi network\n• Firewall is not blocking the connection"

                error.contains("Network unreachable") ->
                    "Network connection failed. Please check:\n• WiFi is connected and stable\n• Network allows device-to-device communication\n• Try moving closer to the WiFi router"

                error.contains("timeout") || error.contains("Timeout") ->
                    "Connection timeout. Please check:\n• PC Hub is running and responsive\n• Network connection is stable\n• Try restarting both applications"

                error.contains("Permission denied") ->
                    "Permission error. Please check:\n• Camera and microphone permissions are granted\n• Storage permissions are enabled\n• Restart the app after granting permissions"

                error.contains("Bluetooth") || error.contains("BLE") ->
                    "Bluetooth connection failed. Please check:\n• Bluetooth is enabled on your device\n• Shimmer sensor is powered on and in range\n• Try restarting Bluetooth or re-pairing the device"

                error.contains("Camera") || error.contains("camera") ->
                    "Camera error occurred. Please check:\n• Camera permissions are granted\n• No other apps are using the camera\n• Try restarting the app"

                error.contains("Storage") || error.contains("storage") ->
                    "Storage error occurred. Please check:\n• Sufficient storage space is available\n• Storage permissions are granted\n• SD card is properly mounted (if using external storage)"

                else -> "An error occurred: $error\n\nIf the problem persists, please check your network connection and try restarting both the PC Hub and this app."
            }

        fun translateSensorError(
            sensorType: String,
            error: String,
        ): String =
            when (sensorType.lowercase()) {
                "shimmer", "gsr" ->
                    when {
                        error.contains("not found") || error.contains("Not found") ->
                            "GSR sensor not detected. Please check:\n• Shimmer device is powered on\n• Bluetooth is enabled\n• Device is within range (2-3 meters)\n• Try power cycling the Shimmer device"

                        error.contains("connection") || error.contains("Connection") ->
                            "GSR sensor connection lost. Please check:\n• Shimmer device battery level\n• Bluetooth connection is stable\n• Device hasn't moved out of range"

                        else -> "GSR sensor error: Please check device connection and battery level"
                    }

                "thermal" ->
                    when {
                        error.contains("USB") || error.contains("usb") ->
                            "Thermal camera connection error. Please check:\n• USB cable is properly connected\n• Camera is powered on\n• Try using a different USB port or cable"

                        error.contains("permission") || error.contains("Permission") ->
                            "Thermal camera access denied. Please check:\n• Camera permissions are granted\n• USB debugging is enabled\n• Allow USB camera access when prompted"

                        else -> "Thermal camera error: Please check USB connection and permissions"
                    }

                "rgb", "camera" ->
                    when {
                        error.contains("permission") || error.contains("Permission") ->
                            "Camera permission required. Please:\n• Grant camera permission in Settings\n• Restart the app after granting permission\n• Ensure no other apps are using the camera"

                        error.contains("unavailable") || error.contains("Unavailable") ->
                            "Camera not available. Please check:\n• Another app isn't using the camera\n• Camera hardware is functioning\n• Try restarting the device"

                        else -> "RGB camera error: Please check camera permissions and availability"
                    }

                else -> "Sensor error ($sensorType): $error"
            }
    }

    /**
     * Formats device and session status information for user display.
     */
    object StatusFormatter {
        fun formatConnectionStatus(
            isConnected: Boolean,
            serverAddress: String? = null,
        ): String =
            if (isConnected) {
                serverAddress?.let { "Connected to PC Hub: $it" } ?: "Connected to PC Hub"
            } else {
                "Not connected to PC Hub"
            }

        fun formatRecordingStatus(
            isRecording: Boolean,
            sessionId: String? = null,
            duration: Long = 0,
        ): String =
            if (isRecording) {
                val durationText =
                    if (duration > 0) {
                        val minutes = duration / 60
                        val seconds = duration % 60
                        " ($minutes:${seconds.toString().padStart(2, '0')})"
                    } else {
                        ""
                    }

                sessionId?.let { "Recording: $it$durationText" } ?: "Recording in progress$durationText"
            } else {
                "Ready to record"
            }

        fun formatSensorStatus(
            sensorName: String,
            isConnected: Boolean,
            isRecording: Boolean,
        ): String {
            val status =
                when {
                    isRecording -> "Recording"
                    isConnected -> "Connected"
                    else -> "Disconnected"
                }
            return "$sensorName: $status"
        }

        fun formatFileLocation(filePath: String): String = "Saved to: ${filePath.substringAfterLast("/")}"

        fun formatDataExportInfo(
            format: String,
            fileCount: Int,
            totalSize: Long,
        ): String {
            val sizeInMB = totalSize / (1024 * 1024)
            return "Exported $fileCount files ($format format, ${sizeInMB}MB total)"
        }
    }

    /**
     * Provides consistent messaging utilities for Android UI components.
     */
    object Messaging {
        fun showUserFriendlyError(
            context: Context,
            error: String,
            errorContext: String = "general",
        ) {
            val friendlyMessage =
                when (errorContext) {
                    "network" -> ErrorTranslator.translateNetworkError(error, errorContext)
                    "sensor" -> ErrorTranslator.translateSensorError("general", error)
                    else -> ErrorTranslator.translateNetworkError(error, errorContext)
                }

            Toast.makeText(context, friendlyMessage, Toast.LENGTH_LONG).show()
        }

        fun showStatus(
            context: Context,
            message: String,
            isLong: Boolean = false,
        ) {
            val duration = if (isLong) Toast.LENGTH_LONG else Toast.LENGTH_SHORT
            Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
        }

        fun showSuccess(
            context: Context,
            operation: String,
            result: String? = null,
        ) {
            val message = result?.let { "$operation: $it" } ?: "$operation completed successfully"
            Toast.makeText(context, "✓ $message", Toast.LENGTH_SHORT).show()
        }

        fun showProgress(
            context: Context,
            operation: String,
            progress: Int? = null,
        ) {
            val progressText = progress?.let { " ($it%)" } ?: ""
            val message = "$operation in progress$progressText..."
            Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Provides Android-specific quick start guidance and onboarding support.
     */
    object QuickStart {
        fun getSetupSteps(): List<Pair<String, String>> =
            listOf(
                "1. Connect to WiFi" to "Ensure your device is connected to the same WiFi network as the PC Hub",
                "2. Grant Permissions" to "Allow camera, microphone, and storage access when prompted",
                "3. Connect Sensors" to "Power on Shimmer GSR sensor and connect thermal camera via USB",
                "4. Find PC Hub" to "App will automatically discover the PC Hub on your network",
                "5. Start Recording" to "Tap 'Start Recording' when all sensors show as connected",
                "6. Monitor Status" to "Use the tabs to preview camera feeds and monitor sensor data",
            )

        fun getConnectionTroubleshootingSteps(): List<String> =
            listOf(
                "Check that PC Hub application is running",
                "Verify both devices are on the same WiFi network",
                "Try disabling and re-enabling WiFi on both devices",
                "Check firewall settings on PC (allow incoming connections)",
                "Try restarting both applications",
                "If using enterprise WiFi, contact IT about device-to-device communication",
            )

        fun getPermissionExplanations(): Map<String, String> =
            mapOf(
                "camera" to "Camera access is required to record RGB video and capture synchronization frames",
                "microphone" to "Microphone access is needed for audio recording during physiological measurements",
                "storage" to "Storage access is required to save recorded data and transfer files to the PC Hub",
                "bluetooth" to "Bluetooth access is needed to connect with the Shimmer GSR sensor",
                "location" to "Location access is required for WiFi network discovery and Bluetooth sensor pairing",
            )
    }
}
