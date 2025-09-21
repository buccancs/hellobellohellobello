package com.yourcompany.sensorspoke.utils

import android.Manifest
import android.app.AlertDialog
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Enhanced PermissionManager with user-friendly UI, retry logic, and comprehensive permission handling.
 * Implements all requirements from the ASD issue for Android runtime permissions.
 *
 * Key improvements:
 * - User-friendly permission dialogs with clear explanations
 * - Retry logic for denied permissions
 * - Proper API 31+ compatibility for Bluetooth permissions
 * - Enhanced USB permission handling with broadcast receiver
 * - Graceful handling of permission denials with actionable feedback
 *
 * Handles runtime permissions for:
 * - Camera (RGB video recording)
 * - Microphone (Audio recording)
 * - Bluetooth + Location (Shimmer GSR sensor)
 * - USB (Topdon TC001 thermal camera)
 * - Storage (Session data saving)
 * - Notifications (Foreground service)
 */
class PermissionManager(
    private val activity: AppCompatActivity,
) {
    companion object {
        private const val TAG = "PermissionManager"
        private const val ACTION_USB_PERMISSION = "com.yourcompany.sensorspoke.USB_PERMISSION"

        private const val TOPDON_VENDOR_ID = 0x4d54
        private const val TC001_PRODUCT_ID_1 = 0x0100
        private const val TC001_PRODUCT_ID_2 = 0x0200
    }

    private val cameraPermissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO,
    )

    private val bluetoothPermissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        arrayOf(
            Manifest.permission.BLUETOOTH_SCAN,
            Manifest.permission.BLUETOOTH_CONNECT,
            Manifest.permission.ACCESS_FINE_LOCATION,
        )
    } else {
        arrayOf(
            Manifest.permission.BLUETOOTH,
            Manifest.permission.BLUETOOTH_ADMIN,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
        )
    }

    private val storagePermissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        arrayOf(
            Manifest.permission.READ_MEDIA_IMAGES,
            Manifest.permission.READ_MEDIA_VIDEO,
        )
    } else {
        arrayOf(
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
        )
    }

    private val cameraPermissionLauncher = activity.registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { permissions ->
        handlePermissionResult("camera", permissions)
    }

    private val bluetoothPermissionLauncher = activity.registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { permissions ->
        handlePermissionResult("bluetooth", permissions)
    }

    private val storagePermissionLauncher = activity.registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { permissions ->
        handlePermissionResult("storage", permissions)
    }

    private val usbManager: UsbManager by lazy {
        activity.getSystemService(Context.USB_SERVICE) as UsbManager
    }

    private var usbPermissionDeferred: CompletableDeferred<Boolean>? = null

    private val usbReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                ACTION_USB_PERMISSION -> {
                    val granted = intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)
                    val device = intent.getParcelableExtra<UsbDevice>(UsbManager.EXTRA_DEVICE)

                    Log.i(TAG, "USB permission result: granted=$granted, device=${device?.deviceName}")

                    usbPermissionDeferred?.complete(granted)
                    usbPermissionDeferred = null

                    if (granted) {
                        UserExperience.Messaging.showSuccess(activity, "USB thermal camera permission granted")
                    } else {
                        UserExperience.Messaging.showUserFriendlyError(
                            activity,
                            "USB permission denied. Thermal camera will not be available.",
                            "permission",
                        )
                    }
                }

                UsbManager.ACTION_USB_DEVICE_ATTACHED -> {
                    val device = intent.getParcelableExtra<UsbDevice>(UsbManager.EXTRA_DEVICE)
                    if (device != null && isTopdonTC001Device(device)) {
                        Log.i(TAG, "Topdon TC001 attached: ${device.deviceName}")
                        activity.lifecycleScope.launch {
                            requestUsbPermission(device)
                        }
                    }
                }

                UsbManager.ACTION_USB_DEVICE_DETACHED -> {
                    val device = intent.getParcelableExtra<UsbDevice>(UsbManager.EXTRA_DEVICE)
                    if (device != null && isTopdonTC001Device(device)) {
                        Log.w(TAG, "Topdon TC001 detached: ${device.deviceName}")
                        UserExperience.Messaging.showStatus(
                            activity,
                            "Thermal camera disconnected. Recording will continue without thermal data.",
                            true,
                        )
                    }
                }
            }
        }
    }

    private var pendingCallbacks = mutableMapOf<String, (Boolean) -> Unit>()

    init {
        val filter = IntentFilter().apply {
            addAction(ACTION_USB_PERMISSION)
            addAction(UsbManager.ACTION_USB_DEVICE_ATTACHED)
            addAction(UsbManager.ACTION_USB_DEVICE_DETACHED)
        }
        activity.registerReceiver(usbReceiver, filter)
    }

    /**
     * Check if all required permissions are granted for recording
     */
    fun areAllPermissionsGranted(): Boolean {
        return arePermissionsGranted(cameraPermissions) &&
            arePermissionsGranted(bluetoothPermissions) &&
            arePermissionsGranted(storagePermissions) &&
            isUsbPermissionGranted()
    }

    /**
     * Request all required permissions for full recording functionality
     */
    fun requestAllPermissions(callback: (Boolean) -> Unit) {
        Log.i(TAG, "Requesting all permissions for multi-modal recording")

        val permissionResults = mutableMapOf<String, Boolean>()
        var totalPermissionGroups = 4
        var completedGroups = 0

        fun checkCompletion() {
            completedGroups++
            if (completedGroups >= totalPermissionGroups) {
                val allGranted = permissionResults.values.all { it }
                Log.i(TAG, "All permission requests completed. Success: $allGranted")
                callback(allGranted)
            }
        }

        requestCameraPermissions { granted ->
            permissionResults["camera"] = granted
            checkCompletion()
        }

        requestBluetoothPermissions { granted ->
            permissionResults["bluetooth"] = granted
            checkCompletion()
        }

        requestStoragePermissions { granted ->
            permissionResults["storage"] = granted
            checkCompletion()
        }

        activity.lifecycleScope.launch {
            val usbGranted = requestUsbPermissions()
            permissionResults["usb"] = usbGranted
            checkCompletion()
        }
    }

    /**
     * Request camera and microphone permissions for RGB video recording
     */
    fun requestCameraPermissions(callback: (Boolean) -> Unit) {
        if (arePermissionsGranted(cameraPermissions)) {
            callback(true)
            return
        }

        pendingCallbacks["camera"] = callback

        Log.i(TAG, "Requesting camera permissions")
        UserExperience.Messaging.showStatus(
            activity,
            "Camera and microphone access needed for video recording",
        )

        cameraPermissionLauncher.launch(cameraPermissions)
    }

    /**
     * Request Bluetooth and location permissions for Shimmer GSR sensor
     */
    fun requestBluetoothPermissions(callback: (Boolean) -> Unit) {
        if (arePermissionsGranted(bluetoothPermissions)) {
            callback(true)
            return
        }

        pendingCallbacks["bluetooth"] = callback

        Log.i(TAG, "Requesting Bluetooth permissions")
        UserExperience.Messaging.showStatus(
            activity,
            "Bluetooth and location access needed for GSR sensor connection",
        )

        bluetoothPermissionLauncher.launch(bluetoothPermissions)
    }

    /**
     * Request storage permissions for session data saving
     */
    fun requestStoragePermissions(callback: (Boolean) -> Unit) {
        if (arePermissionsGranted(storagePermissions)) {
            callback(true)
            return
        }

        pendingCallbacks["storage"] = callback

        Log.i(TAG, "Requesting storage permissions")
        UserExperience.Messaging.showStatus(
            activity,
            "Storage access needed to save recording sessions",
        )

        storagePermissionLauncher.launch(storagePermissions)
    }

    /**
     * Request USB permissions for Topdon TC001 thermal camera
     */
    suspend fun requestUsbPermissions(): Boolean {
        val topdonDevice = findTopdonTC001Device()
        return if (topdonDevice != null) {
            requestUsbPermission(topdonDevice)
        } else {
            Log.w(TAG, "No Topdon TC001 device found for permission request")
            true
        }
    }

    /**
     * Request permission for a specific USB device
     */
    private suspend fun requestUsbPermission(device: UsbDevice): Boolean {
        if (usbManager.hasPermission(device)) {
            Log.d(TAG, "USB permission already granted for ${device.deviceName}")
            return true
        }

        Log.i(TAG, "Requesting USB permission for device: ${device.deviceName}")

        val permissionIntent = PendingIntent.getBroadcast(
            activity,
            0,
            Intent(ACTION_USB_PERMISSION),
            PendingIntent.FLAG_UPDATE_CURRENT or
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) PendingIntent.FLAG_IMMUTABLE else 0,
        )

        usbPermissionDeferred = CompletableDeferred()
        usbManager.requestPermission(device, permissionIntent)

        return usbPermissionDeferred?.await() ?: false
    }

    /**
     * Check if USB permission is granted for Topdon device
     */
    private fun isUsbPermissionGranted(): Boolean {
        val device = findTopdonTC001Device()
        return device == null || usbManager.hasPermission(device)
    }

    /**
     * Find connected Topdon TC001 device
     */
    private fun findTopdonTC001Device(): UsbDevice? {
        return usbManager.deviceList.values.find { device ->
            isTopdonTC001Device(device)
        }
    }

    /**
     * Check if a USB device is a Topdon TC001
     */
    private fun isTopdonTC001Device(device: UsbDevice): Boolean {
        return device.vendorId == TOPDON_VENDOR_ID &&
            (device.productId == TC001_PRODUCT_ID_1 || device.productId == TC001_PRODUCT_ID_2)
    }

    /**
     * Check if a set of permissions are all granted
     */
    private fun arePermissionsGranted(permissions: Array<String>): Boolean {
        return permissions.all { permission ->
            ContextCompat.checkSelfPermission(activity, permission) == PackageManager.PERMISSION_GRANTED
        }
    }

    /**
     * Handle permission result from launchers
     */
    private fun handlePermissionResult(type: String, permissions: Map<String, Boolean>) {
        val allGranted = permissions.values.all { it }

        Log.i(TAG, "$type permissions result: $allGranted, details: $permissions")

        if (allGranted) {
            UserExperience.Messaging.showSuccess(activity, "${type.capitalize()} permissions granted")
        } else {
            val deniedPermissions = permissions.filterNot { it.value }.keys
            Log.w(TAG, "$type permissions denied: $deniedPermissions")

            val explanation = UserExperience.QuickStart.getPermissionExplanations()[type] ?: ""
            UserExperience.Messaging.showUserFriendlyError(
                activity,
                "Some $type permissions were denied: $explanation",
                "permission",
            )
        }

        pendingCallbacks[type]?.invoke(allGranted)
        pendingCallbacks.remove(type)
    }

    /**
     * Enhanced permission request with user-friendly dialog and retry logic
     */
    fun requestPermissionsWithExplanation(
        permissionType: String,
        retryCount: Int = 0,
        callback: (Boolean) -> Unit
    ) {
        val maxRetries = 2
        
        when (permissionType) {
            "camera" -> {
                if (arePermissionsGranted(cameraPermissions)) {
                    callback(true)
                    return
                }
                
                showPermissionExplanationDialog(
                    title = "Camera Permission Required",
                    message = "The RGB camera needs camera and microphone permissions to record video and audio for your research session. This data stays on your device and is only used for your study.",
                    onAccept = {
                        requestCameraPermissions { granted ->
                            if (!granted && retryCount < maxRetries) {
                                showPermissionDeniedDialog(
                                    permissionType = "camera",
                                    retryAction = { requestPermissionsWithExplanation("camera", retryCount + 1, callback) },
                                    skipAction = { callback(false) }
                                )
                            } else {
                                callback(granted)
                            }
                        }
                    },
                    onDecline = { callback(false) }
                )
            }
            
            "bluetooth" -> {
                if (arePermissionsGranted(bluetoothPermissions)) {
                    callback(true)
                    return
                }
                
                val apiNote = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    " (Android 12+ requires both Bluetooth and Location permissions)"
                } else {
                    ""
                }
                
                showPermissionExplanationDialog(
                    title = "Bluetooth & Location Permission Required",
                    message = "The GSR sensor needs Bluetooth and Location permissions to scan for and connect to your Shimmer device$apiNote. Location is required by Android for Bluetooth scanning but is not used to track your location.",
                    onAccept = {
                        requestBluetoothPermissions { granted ->
                            if (!granted && retryCount < maxRetries) {
                                showPermissionDeniedDialog(
                                    permissionType = "bluetooth",
                                    retryAction = { requestPermissionsWithExplanation("bluetooth", retryCount + 1, callback) },
                                    skipAction = { callback(false) }
                                )
                            } else {
                                callback(granted)
                            }
                        }
                    },
                    onDecline = { callback(false) }
                )
            }
            
            "storage" -> {
                if (arePermissionsGranted(storagePermissions)) {
                    callback(true)
                    return
                }
                
                showPermissionExplanationDialog(
                    title = "Storage Permission Required",
                    message = "Storage permissions are needed to save your research session data (videos, sensor readings, etc.) to your device. This data remains private on your device.",
                    onAccept = {
                        requestStoragePermissions { granted ->
                            if (!granted && retryCount < maxRetries) {
                                showPermissionDeniedDialog(
                                    permissionType = "storage",
                                    retryAction = { requestPermissionsWithExplanation("storage", retryCount + 1, callback) },
                                    skipAction = { callback(false) }  
                                )
                            } else {
                                callback(granted)
                            }
                        }
                    },
                    onDecline = { callback(false) }
                )
            }
            
            else -> {
                Log.w(TAG, "Unknown permission type: $permissionType")
                callback(false)
            }
        }
    }

    /**
     * Show user-friendly explanation dialog before requesting permissions
     */
    private fun showPermissionExplanationDialog(
        title: String,
        message: String,
        onAccept: () -> Unit,
        onDecline: () -> Unit
    ) {
        AlertDialog.Builder(activity)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("Grant Permission") { _, _ -> onAccept() }
            .setNegativeButton("Skip") { _, _ -> onDecline() }
            .setCancelable(false)
            .show()
    }

    /**
     * Show retry dialog when permissions are denied
     */
    private fun showPermissionDeniedDialog(
        permissionType: String,
        retryAction: () -> Unit,
        skipAction: () -> Unit
    ) {
        val typeDisplayName = when (permissionType) {
            "camera" -> "Camera & Microphone"
            "bluetooth" -> "Bluetooth & Location"
            "storage" -> "Storage"
            else -> permissionType.capitalize()
        }
        
        AlertDialog.Builder(activity)
            .setTitle("$typeDisplayName Permission Denied")
            .setMessage("Without $typeDisplayName permissions, some features won't work. You can:\n\n• Try again to grant permissions\n• Continue without this sensor (limited functionality)\n• Open Settings to grant permissions manually")
            .setPositiveButton("Try Again") { _, _ -> retryAction() }
            .setNeutralButton("Open Settings") { _, _ -> openAppSettings() }
            .setNegativeButton("Continue Without") { _, _ -> skipAction() }
            .setCancelable(false)
            .show()
    }

    /**
     * Open app settings for manual permission granting
     */
    private fun openAppSettings() {
        try {
            val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                data = Uri.fromParts("package", activity.packageName, null)
            }
            activity.startActivity(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open app settings: ${e.message}", e)
            UserExperience.Messaging.showUserFriendlyError(
                activity,
                "Could not open settings. Please grant permissions manually in your device settings.",
                "settings"
            )
        }
    }

    /**
     * Request all permissions with user-friendly flow
     */
    fun requestAllPermissionsWithUI(callback: (Boolean) -> Unit) {
        Log.i(TAG, "Starting user-friendly permission request flow")
        
        val permissionResults = mutableMapOf<String, Boolean>()
        val permissionTypes = listOf("camera", "bluetooth", "storage")
        var currentIndex = 0
        
        fun requestNext() {
            if (currentIndex >= permissionTypes.size) {
                activity.lifecycleScope.launch {
                    val usbGranted = requestUsbPermissions()
                    permissionResults["usb"] = usbGranted
                    
                    val allGranted = permissionResults.values.all { it }
                    Log.i(TAG, "All permission requests completed with UI. Success: $allGranted")
                    callback(allGranted)
                }
                return
            }
            
            val permissionType = permissionTypes[currentIndex]
            requestPermissionsWithExplanation(permissionType) { granted ->
                permissionResults[permissionType] = granted
                currentIndex++
                requestNext()
            }
        }
        
        requestNext()
    }

    /**
     * Check if critical permissions are granted for basic functionality
     */
    fun areCriticalPermissionsGranted(): Boolean {
        // Camera is critical for the core functionality
        return arePermissionsGranted(cameraPermissions)
    }

    /**
     * Show permission status summary dialog
     */
    fun showPermissionStatusDialog() {
        val status = getPermissionStatus()
        val message = buildString {
            appendLine("Permission Status:")
            appendLine()
            
            val cameraGranted = status["camera"] as Map<String, Any>
            appendLine("📹 Camera: ${if (cameraGranted["granted"] as Boolean) "✅ Granted" else "❌ Denied"}")
            
            val bluetoothGranted = status["bluetooth"] as Map<String, Any>
            appendLine("📡 Bluetooth: ${if (bluetoothGranted["granted"] as Boolean) "✅ Granted" else "❌ Denied"}")
            
            val storageGranted = status["storage"] as Map<String, Any>
            appendLine("💾 Storage: ${if (storageGranted["granted"] as Boolean) "✅ Granted" else "❌ Denied"}")
            
            val usbGranted = status["usb"] as Map<String, Any>
            appendLine("🔌 USB: ${if (usbGranted["granted"] as Boolean) "✅ Granted" else "❌ Denied"}")
            
            if (usbGranted["device_found"] as Boolean) {
                appendLine("   Thermal camera: ${usbGranted["device_info"]}")
            }
        }
        
        AlertDialog.Builder(activity)
            .setTitle("Sensor Permissions")
            .setMessage(message)
            .setPositiveButton("Grant Missing") { _, _ -> 
                requestAllPermissionsWithUI { } 
            }
            .setNeutralButton("Settings") { _, _ -> 
                openAppSettings() 
            }
            .setNegativeButton("Close") { _, _ -> }
            .show()
    }

    /**
     * Clean up resources
     */
    fun cleanup() {
        try {
            activity.unregisterReceiver(usbReceiver)
        } catch (e: Exception) {
            Log.w(TAG, "Error unregistering USB receiver: ${e.message}")
        }

        pendingCallbacks.clear()
        usbPermissionDeferred?.cancel()
        usbPermissionDeferred = null
    }

    /**
     * Get detailed permission status for debugging
     */
    fun getPermissionStatus(): Map<String, Any> {
        return mapOf(
            "camera" to mapOf(
                "granted" to arePermissionsGranted(cameraPermissions),
                "permissions" to cameraPermissions.map {
                    it to (ContextCompat.checkSelfPermission(activity, it) == PackageManager.PERMISSION_GRANTED)
                }.toMap(),
            ),
            "bluetooth" to mapOf(
                "granted" to arePermissionsGranted(bluetoothPermissions),
                "permissions" to bluetoothPermissions.map {
                    it to (ContextCompat.checkSelfPermission(activity, it) == PackageManager.PERMISSION_GRANTED)
                }.toMap(),
            ),
            "storage" to mapOf(
                "granted" to arePermissionsGranted(storagePermissions),
                "permissions" to storagePermissions.map {
                    it to (ContextCompat.checkSelfPermission(activity, it) == PackageManager.PERMISSION_GRANTED)
                }.toMap(),
            ),
            "usb" to mapOf(
                "granted" to isUsbPermissionGranted(),
                "device_found" to (findTopdonTC001Device() != null),
                "device_info" to (
                    findTopdonTC001Device()?.let {
                        "${it.deviceName} (${it.vendorId}:${it.productId})"
                    } ?: "None"
                    ),
            ),
        )
    }
}
