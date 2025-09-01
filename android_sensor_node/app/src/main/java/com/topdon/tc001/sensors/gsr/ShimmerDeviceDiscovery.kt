package com.topdon.tc001.sensors.gsr

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.topdon.tc001.sensors.gsr.shimmer.ShimmerBluetoothDialog
import com.topdon.tc001.sensors.gsr.shimmer.ShimmerHardwareType
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

/**
 * Shimmer device discovery and connection management utility
 * 
 * Provides functionality for:
 * - Bluetooth permission checking and requesting
 * - Shimmer device discovery using official ShimmerBluetoothDialog
 * - Device validation and connection status tracking
 * 
 * Based on ShimmerAndroidAPI best practices
 */
class ShimmerDeviceDiscovery(private val context: Context) {
    
    companion object {
        private const val TAG = "ShimmerDeviceDiscovery"
        
        // Shimmer device name patterns for filtering
        private val SHIMMER_DEVICE_PATTERNS = arrayOf(
            "Shimmer3",
            "shimmer3",
            "SHIMMER3",
            "GSR",
            "shimmer"
        )
    }
    
    private val bluetoothManager: BluetoothManager by lazy { 
        context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager 
    }
    private val bluetoothAdapter: BluetoothAdapter? by lazy { bluetoothManager.adapter }
    
    private val _discoveredDevices = MutableStateFlow<List<ShimmerDeviceInfo>>(emptyList())
    val discoveredDevices: Flow<List<ShimmerDeviceInfo>> = _discoveredDevices.asStateFlow()
    
    private val _isScanning = MutableStateFlow(false)
    val isScanning: Flow<Boolean> = _isScanning.asStateFlow()
    
    /**
     * Check if all required Bluetooth permissions are granted
     */
    fun hasRequiredPermissions(): Boolean {
        val requiredPermissions = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
            arrayOf(
                Manifest.permission.BLUETOOTH_CONNECT,
                Manifest.permission.BLUETOOTH_SCAN,
                Manifest.permission.ACCESS_FINE_LOCATION
            )
        } else {
            arrayOf(
                Manifest.permission.BLUETOOTH,
                Manifest.permission.BLUETOOTH_ADMIN,
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
            )
        }
        
        return requiredPermissions.all { permission ->
            ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
        }
    }
    
    /**
     * Get the list of required Bluetooth permissions for the current Android version
     */
    fun getRequiredPermissions(): Array<String> {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
            arrayOf(
                Manifest.permission.BLUETOOTH_CONNECT,
                Manifest.permission.BLUETOOTH_SCAN,
                Manifest.permission.ACCESS_FINE_LOCATION
            )
        } else {
            arrayOf(
                Manifest.permission.BLUETOOTH,
                Manifest.permission.BLUETOOTH_ADMIN,
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
            )
        }
    }
    
    /**
     * Check if Bluetooth is enabled and available
     */
    fun isBluetoothAvailable(): Boolean {
        val adapter = bluetoothAdapter ?: return false
        return adapter.isEnabled
    }
    
    /**
     * Create intent for Shimmer device selection dialog
     * Uses the official ShimmerBluetoothDialog from ShimmerAndroidAPI
     */
    fun createDeviceSelectionIntent(): Intent {
        return Intent(context, ShimmerBluetoothDialog::class.java)
    }
    
    /**
     * Validate if a device name/address corresponds to a Shimmer device
     */
    fun isShimmerDevice(deviceName: String?, deviceAddress: String?): Boolean {
        if (deviceName == null && deviceAddress == null) return false
        
        // Check device name patterns
        deviceName?.let { name ->
            if (SHIMMER_DEVICE_PATTERNS.any { pattern ->
                name.contains(pattern, ignoreCase = true)
            }) {
                return true
            }
        }
        
        // Additional validation could be added based on device address patterns
        // if Shimmer devices follow specific MAC address patterns
        
        return false
    }
    
    /**
     * Get paired Shimmer devices
     */
    suspend fun getPairedShimmerDevices(): List<ShimmerDeviceInfo> = withContext(Dispatchers.IO) {
        try {
            val adapter = bluetoothAdapter ?: return@withContext emptyList()
            
            if (!hasRequiredPermissions()) {
                Log.w(TAG, "Missing Bluetooth permissions")
                return@withContext emptyList()
            }
            
            val pairedDevices = if (ActivityCompat.checkSelfPermission(
                    context,
                    Manifest.permission.BLUETOOTH_CONNECT
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                adapter.bondedDevices
            } else {
                emptySet()
            }
            
            val shimmerDevices = pairedDevices?.filter { device ->
                isShimmerDevice(device.name, device.address)
            }?.map { device ->
                ShimmerDeviceInfo(
                    deviceName = device.name ?: "Unknown Shimmer",
                    deviceAddress = device.address,
                    connectionState = ShimmerConnectionState.DISCONNECTED,
                    streamingState = ShimmerStreamingState.STOPPED
                )
            } ?: emptyList()
            
            Log.d(TAG, "Found ${shimmerDevices.size} paired Shimmer devices")
            shimmerDevices
            
        } catch (e: Exception) {
            Log.e(TAG, "Error getting paired Shimmer devices", e)
            emptyList()
        }
    }
    
    /**
     * Extract device information from ShimmerBluetoothDialog result
     */
    fun extractDeviceFromIntent(data: Intent): ShimmerDeviceInfo? {
        return try {
            val deviceAddress = data.getStringExtra(ShimmerBluetoothDialog.EXTRA_DEVICE_ADDRESS)
            val deviceName = data.getStringExtra(ShimmerBluetoothDialog.EXTRA_DEVICE_NAME)
            
            if (deviceAddress != null) {
                ShimmerDeviceInfo(
                    deviceName = deviceName ?: "Shimmer Device",
                    deviceAddress = deviceAddress,
                    connectionState = ShimmerConnectionState.DISCONNECTED,
                    streamingState = ShimmerStreamingState.STOPPED
                )
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting device from intent", e)
            null
        }
    }
    
    /**
     * Determine Shimmer device type from device name
     */
    fun getShimmerHardwareType(deviceName: String?): ShimmerHardwareType {
        return if (deviceName?.contains("3R", ignoreCase = true) == true) {
            ShimmerHardwareType.SHIMMER_3R
        } else {
            ShimmerHardwareType.SHIMMER_3
        }
    }
    
    /**
     * Format device information for display
     */
    fun formatDeviceInfo(deviceInfo: ShimmerDeviceInfo): String {
        return "${deviceInfo.deviceName} (${deviceInfo.deviceAddress})"
    }
    
    /**
     * Log current Bluetooth state for debugging
     */
    fun logBluetoothState() {
        try {
            val adapter = bluetoothAdapter
            Log.d(TAG, "Bluetooth adapter: ${if (adapter != null) "Available" else "Not available"}")
            Log.d(TAG, "Bluetooth enabled: ${adapter?.isEnabled ?: false}")
            Log.d(TAG, "Required permissions granted: ${hasRequiredPermissions()}")
            
            if (hasRequiredPermissions()) {
                val pairedCount = if (ActivityCompat.checkSelfPermission(
                        context,
                        Manifest.permission.BLUETOOTH_CONNECT
                    ) == PackageManager.PERMISSION_GRANTED
                ) {
                    adapter?.bondedDevices?.size ?: 0
                } else {
                    0
                }
                Log.d(TAG, "Paired devices: $pairedCount")
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error logging Bluetooth state", e)
        }
    }
}