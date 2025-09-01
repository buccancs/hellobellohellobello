package com.topdon.tc001.fragment

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.lifecycle.lifecycleScope
import com.topdon.tc001.R
import com.topdon.tc001.databinding.FragmentMainBinding
import com.topdon.tc001.sensors.SensorType
import com.topdon.tc001.ui.viewmodel.MainViewModel
import kotlinx.coroutines.launch

/**
 * Main fragment integrating GSR sensor functionality with IRCamera-style UI
 * 
 * Key features:
 * - Shimmer3 GSR+ device connection and management
 * - Real-time GSR data recording with proper timestamping
 * - IRCamera-style device connection interface
 * - Modern BLE permissions handling
 */
class MainFragment : Fragment() {

    private var _binding: FragmentMainBinding? = null
    private val binding get() = _binding!!
    
    private val viewModel: MainViewModel by viewModels()
    
    private lateinit var bluetoothAdapter: BluetoothAdapter
    private lateinit var bluetoothEnableLauncher: ActivityResultLauncher<Intent>
    private lateinit var permissionLauncher: ActivityResultLauncher<Array<String>>
    
    private val discoveredDevices = mutableListOf<BluetoothDevice>()
    private lateinit var deviceAdapter: ArrayAdapter<String>

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        _binding = FragmentMainBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        initializeBluetooth()
        initializePermissionLaunchers()
        initView()
        observeViewModel()
    }

    private fun initializeBluetooth() {
        val bluetoothManager = requireContext().getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        bluetoothAdapter = bluetoothManager.adapter
    }
    
    private fun initializePermissionLaunchers() {
        // Bluetooth enable launcher
        bluetoothEnableLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == android.app.Activity.RESULT_OK) {
                checkPermissionsAndScan()
            } else {
                showToast("Bluetooth is required for GSR sensor connection")
            }
        }
        
        // Permission launcher
        permissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { permissions ->
            val allGranted = permissions.all { it.value }
            if (allGranted) {
                startDeviceScanning()
            } else {
                showToast("Permissions are required for GSR sensor connection")
            }
        }
    }

    private fun initView() {
        // Initialize device list adapter
        deviceAdapter = ArrayAdapter(requireContext(), android.R.layout.simple_list_item_1)
        binding.deviceList.adapter = deviceAdapter
        
        // Device list click handler
        binding.deviceList.setOnItemClickListener { _, _, position, _ ->
            if (position < discoveredDevices.size) {
                val device = discoveredDevices[position]
                connectToGsrDevice(device)
            }
        }
        
        // Control buttons
        binding.btnScanDevices.setOnClickListener {
            scanForGsrDevices()
        }
        
        binding.btnStartRecording.setOnClickListener {
            viewModel.startRecording()
        }
        
        binding.btnStopRecording.setOnClickListener {
            viewModel.stopRecording()
        }
        
        binding.btnDisconnectGsr.setOnClickListener {
            viewModel.disconnectSensor(SensorType.GSR_SHIMMER)
        }
    }

    private fun observeViewModel() {
        // Recording state
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.isRecording.collect { isRecording ->
                updateRecordingState(isRecording)
            }
        }
        
        // Connected sensors
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.connectedSensors.collect { sensors ->
                updateSensorConnectionState(sensors)
            }
        }
        
        // GSR connection state
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.isConnectingGsr.collect { connecting ->
                binding.btnScanDevices.isEnabled = !connecting
                binding.progressConnection.visibility = if (connecting) View.VISIBLE else View.GONE
            }
        }
        
        // Sensor info
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.sensorInfoMap.collect { infoMap ->
                updateSensorInfo(infoMap)
            }
        }
        
        // Status messages
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.statusMessage.collect { message ->
                showToast(message)
            }
        }
        
        // Error messages
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.errorMessage.collect { error ->
                showToast("Error: $error")
            }
        }
        
        // Current session
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.currentSession.collect { session ->
                if (session != null) {
                    binding.tvSessionInfo.text = "Session: ${session.sessionId}"
                    binding.tvSessionInfo.visibility = View.VISIBLE
                } else {
                    binding.tvSessionInfo.visibility = View.GONE
                }
            }
        }
    }
    
    private fun scanForGsrDevices() {
        if (!::bluetoothAdapter.isInitialized) {
            showToast("Bluetooth not available")
            return
        }
        
        if (!bluetoothAdapter.isEnabled) {
            val enableBtIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
            bluetoothEnableLauncher.launch(enableBtIntent)
            return
        }
        
        checkPermissionsAndScan()
    }
    
    private fun checkPermissionsAndScan() {
        val requiredPermissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            arrayOf(
                Manifest.permission.BLUETOOTH_SCAN,
                Manifest.permission.BLUETOOTH_CONNECT
            )
        } else {
            arrayOf(
                Manifest.permission.BLUETOOTH,
                Manifest.permission.BLUETOOTH_ADMIN,
                Manifest.permission.ACCESS_FINE_LOCATION
            )
        }
        
        val missingPermissions = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(requireContext(), it) != PackageManager.PERMISSION_GRANTED
        }
        
        if (missingPermissions.isEmpty()) {
            startDeviceScanning()
        } else {
            permissionLauncher.launch(missingPermissions.toTypedArray())
        }
    }
    
    private fun startDeviceScanning() {
        try {
            discoveredDevices.clear()
            deviceAdapter.clear()
            
            // Get paired devices first
            val pairedDevices = bluetoothAdapter.bondedDevices
            pairedDevices?.forEach { device ->
                if (isShimmerDevice(device)) {
                    discoveredDevices.add(device)
                    deviceAdapter.add("${device.name ?: "Unknown"} (${device.address}) - Paired")
                }
            }
            
            deviceAdapter.notifyDataSetChanged()
            binding.tvDeviceStatus.text = "Found ${discoveredDevices.size} paired Shimmer device(s)"
            
            showToast("Scanning for GSR devices...")
            
        } catch (e: SecurityException) {
            showToast("Permission error: Cannot scan for devices")
        } catch (e: Exception) {
            showToast("Error scanning devices: ${e.message}")
        }
    }
    
    private fun isShimmerDevice(device: BluetoothDevice): Boolean {
        val deviceName = device.name?.lowercase()
        return deviceName?.contains("shimmer") == true || 
               deviceName?.contains("gsr") == true
    }
    
    private fun connectToGsrDevice(device: BluetoothDevice) {
        try {
            binding.tvDeviceStatus.text = "Connecting to ${device.name ?: "Unknown"}..."
            viewModel.connectGsrSensor(device.address)
        } catch (e: Exception) {
            showToast("Error connecting to device: ${e.message}")
        }
    }
    
    private fun updateRecordingState(isRecording: Boolean) {
        binding.btnStartRecording.isEnabled = !isRecording && viewModel.isGsrConnected()
        binding.btnStopRecording.isEnabled = isRecording
        binding.btnScanDevices.isEnabled = !isRecording
        
        if (isRecording) {
            binding.tvRecordingStatus.text = "Recording GSR data..."
            binding.tvRecordingStatus.setTextColor(ContextCompat.getColor(requireContext(), android.R.color.holo_red_dark))
        } else {
            binding.tvRecordingStatus.text = "Ready to record"
            binding.tvRecordingStatus.setTextColor(ContextCompat.getColor(requireContext(), android.R.color.darker_gray))
        }
    }
    
    private fun updateSensorConnectionState(connectedSensors: Set<SensorType>) {
        val gsrConnected = connectedSensors.contains(SensorType.GSR_SHIMMER)
        
        binding.btnDisconnectGsr.isEnabled = gsrConnected
        binding.btnStartRecording.isEnabled = gsrConnected && !viewModel.isRecording.value
        
        if (gsrConnected) {
            binding.tvConnectionStatus.text = "GSR sensor connected"
            binding.tvConnectionStatus.setTextColor(ContextCompat.getColor(requireContext(), android.R.color.holo_green_dark))
            binding.imgConnectionStatus.setImageResource(R.drawable.ic_check_circle)
        } else {
            binding.tvConnectionStatus.text = "No GSR sensor connected"
            binding.tvConnectionStatus.setTextColor(ContextCompat.getColor(requireContext(), android.R.color.darker_gray))
            binding.imgConnectionStatus.setImageResource(R.drawable.ic_error)
        }
    }
    
    private fun updateSensorInfo(infoMap: Map<SensorType, com.topdon.tc001.sensors.SensorInfo>) {
        val gsrInfo = infoMap[SensorType.GSR_SHIMMER]
        if (gsrInfo != null) {
            val infoText = buildString {
                append("Device: ${gsrInfo.deviceName}\n")
                append("Address: ${gsrInfo.deviceAddress}\n")
                gsrInfo.batteryLevel?.let { append("Battery: $it%\n") }
                gsrInfo.signalStrength?.let { append("Signal: $it dBm\n") }
                gsrInfo.lastDataTimestamp?.let { append("Last data: ${System.currentTimeMillis() - it}ms ago") }
            }
            binding.tvSensorInfo.text = infoText
            binding.tvSensorInfo.visibility = View.VISIBLE
        } else {
            binding.tvSensorInfo.visibility = View.GONE
        }
    }
    
    private fun showToast(message: String) {
        Toast.makeText(requireContext(), message, Toast.LENGTH_SHORT).show()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}