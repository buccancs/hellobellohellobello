package com.topdon.tc001.sensors.gsr.shimmer

import android.Manifest
import android.app.Activity
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.topdon.tc001.R

/**
 * Bluetooth device selection dialog for Shimmer devices
 * Compatible with ShimmerAndroidAPI ShimmerBluetoothDialog
 * 
 * Provides device discovery and selection functionality for Shimmer3 GSR+ devices
 */
class ShimmerBluetoothDialog : AppCompatActivity() {
    
    companion object {
        private const val TAG = "ShimmerBluetoothDialog"
        
        // Intent extras (compatible with ShimmerAndroidAPI)
        const val EXTRA_DEVICE_ADDRESS = "device_address"
        const val EXTRA_DEVICE_NAME = "device_name"
        
        // Request codes
        private const val REQUEST_ENABLE_BT = 1001
        private const val REQUEST_PERMISSIONS = 1002
        
        // Shimmer device patterns for filtering
        private val SHIMMER_PATTERNS = arrayOf(
            "shimmer", "SHIMMER", "Shimmer",
            "GSR", "gsr"
        )
    }
    
    private lateinit var bluetoothAdapter: BluetoothAdapter
    private lateinit var deviceListView: ListView
    private lateinit var deviceAdapter: ArrayAdapter<String>
    private val deviceList = mutableListOf<String>()
    private val deviceMap = mutableMapOf<String, BluetoothDevice>()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Set simple layout for device selection
        setContentView(createDeviceSelectionLayout())
        
        setupBluetoothAdapter()
        setupUI()
        
        // Check permissions and start device discovery
        if (checkPermissions()) {
            loadPairedDevices()
        } else {
            requestPermissions()
        }
    }
    
    private fun createDeviceSelectionLayout(): View {
        // Create simple layout programmatically
        val layout = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
        }
        
        val titleText = TextView(this).apply {
            text = "Select Shimmer Device"
            textSize = 18f
            setPadding(0, 0, 0, 24)
            setTypeface(null, android.graphics.Typeface.BOLD)
        }
        layout.addView(titleText)
        
        deviceListView = ListView(this).apply {
            layoutParams = android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        layout.addView(deviceListView)
        
        return layout
    }
    
    private fun setupBluetoothAdapter() {
        val bluetoothManager = getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        bluetoothAdapter = bluetoothManager.adapter
        
        if (!bluetoothAdapter.isEnabled) {
            val enableBtIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
            if (ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.BLUETOOTH_CONNECT
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                startActivityForResult(enableBtIntent, REQUEST_ENABLE_BT)
            }
        }
    }
    
    private fun setupUI() {
        deviceAdapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, deviceList)
        deviceListView.adapter = deviceAdapter
        
        deviceListView.onItemClickListener = AdapterView.OnItemClickListener { _, _, position, _ ->
            val deviceInfo = deviceList[position]
            val device = deviceMap[deviceInfo]
            
            if (device != null) {
                selectDevice(device)
            }
        }
    }
    
    private fun checkPermissions(): Boolean {
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
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
        }
    }
    
    private fun requestPermissions() {
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
        
        ActivityCompat.requestPermissions(this, requiredPermissions, REQUEST_PERMISSIONS)
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        if (requestCode == REQUEST_PERMISSIONS) {
            if (grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                loadPairedDevices()
            } else {
                Toast.makeText(this, "Bluetooth permissions required", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }
    
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        if (requestCode == REQUEST_ENABLE_BT) {
            if (resultCode == Activity.RESULT_OK) {
                loadPairedDevices()
            } else {
                Toast.makeText(this, "Bluetooth is required", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }
    
    private fun loadPairedDevices() {
        try {
            if (!checkPermissions()) {
                Log.w(TAG, "Missing permissions for device discovery")
                return
            }
            
            val pairedDevices = bluetoothAdapter.bondedDevices
            deviceList.clear()
            deviceMap.clear()
            
            var shimmerDeviceCount = 0
            
            pairedDevices?.forEach { device ->
                if (isShimmerDevice(device.name)) {
                    val deviceInfo = "${device.name ?: "Unknown"}\n${device.address}"
                    deviceList.add(deviceInfo)
                    deviceMap[deviceInfo] = device
                    shimmerDeviceCount++
                }
            }
            
            if (shimmerDeviceCount == 0) {
                // Add all paired devices if no Shimmer devices found
                pairedDevices?.forEach { device ->
                    val deviceInfo = "${device.name ?: "Unknown"}\n${device.address}"
                    deviceList.add(deviceInfo)
                    deviceMap[deviceInfo] = device
                }
                
                if (deviceList.isEmpty()) {
                    deviceList.add("No paired devices found")
                }
            }
            
            deviceAdapter.notifyDataSetChanged()
            
            Log.d(TAG, "Found ${shimmerDeviceCount} Shimmer devices, ${deviceList.size} total devices")
            
        } catch (e: SecurityException) {
            Log.e(TAG, "Security exception loading paired devices", e)
            Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show()
            finish()
        } catch (e: Exception) {
            Log.e(TAG, "Error loading paired devices", e)
            Toast.makeText(this, "Error loading devices", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun isShimmerDevice(deviceName: String?): Boolean {
        if (deviceName == null) return false
        
        return SHIMMER_PATTERNS.any { pattern ->
            deviceName.contains(pattern, ignoreCase = true)
        }
    }
    
    private fun selectDevice(device: BluetoothDevice) {
        Log.d(TAG, "Device selected: ${device.name} (${device.address})")
        
        // Return selected device information
        val resultIntent = Intent().apply {
            putExtra(EXTRA_DEVICE_ADDRESS, device.address)
            putExtra(EXTRA_DEVICE_NAME, device.name ?: "Unknown Shimmer")
        }
        
        setResult(Activity.RESULT_OK, resultIntent)
        finish()
    }
}