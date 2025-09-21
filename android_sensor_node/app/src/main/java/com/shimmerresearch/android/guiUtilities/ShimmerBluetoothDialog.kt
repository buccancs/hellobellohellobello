package com.shimmerresearch.android.guiUtilities

import android.annotation.SuppressLint
import android.app.Activity
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.yourcompany.sensorspoke.R

/**
 * Shimmer Bluetooth device selection dialog from ShimmerAndroidAPI
 */
class ShimmerBluetoothDialog : AppCompatActivity() {
    companion object {
        const val EXTRA_DEVICE_ADDRESS = "device_address"
        const val EXTRA_DEVICE_NAME = "device_name"
        private const val TAG = "ShimmerBluetoothDialog"
    }

    private lateinit var deviceAdapter: DeviceListAdapter
    private val pairedDevices = mutableListOf<BluetoothDevice>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.dialog_shimmer_bluetooth)

        setupRecyclerView()
        loadPairedDevices()
    }

    private fun setupRecyclerView() {
        val recyclerView = findViewById<RecyclerView>(R.id.recyclerViewDevices)
        deviceAdapter =
            DeviceListAdapter { device ->
                selectDevice(device)
            }

        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = deviceAdapter
    }

    private fun loadPairedDevices() {
        try {
            val bluetoothAdapter = BluetoothAdapter.getDefaultAdapter()

            if (bluetoothAdapter == null) {
                Log.e(TAG, "Bluetooth adapter not available")
                finish()
                return
            }

            if (!bluetoothAdapter.isEnabled) {
                Log.w(TAG, "Bluetooth is not enabled")
                // Could request to enable Bluetooth here
            }

            val paired = bluetoothAdapter.bondedDevices
            pairedDevices.clear()

            for (device in paired) {
                if (device.name?.contains("Shimmer", ignoreCase = true) == true ||
                    device.name?.contains("GSR", ignoreCase = true) == true
                ) {
                    pairedDevices.add(device)
                    Log.d(TAG, "Found Shimmer device: ${device.name} (${device.address})")
                }
            }

            deviceAdapter.updateDevices(pairedDevices)
        } catch (e: SecurityException) {
            Log.e(TAG, "Bluetooth permissions not granted", e)
        } catch (e: Exception) {
            Log.e(TAG, "Error loading paired devices", e)
        }
    }

    @SuppressLint("MissingPermission")
    private fun selectDevice(device: BluetoothDevice) {
        val deviceName = try {
            device.name
        } catch (e: SecurityException) {
            "Unknown Device"
        }

        Log.i(TAG, "Selected device: $deviceName (${device.address})")

        val resultIntent =
            Intent().apply {
                putExtra(EXTRA_DEVICE_ADDRESS, device.address)
                putExtra(EXTRA_DEVICE_NAME, deviceName ?: "Unknown Device")
            }

        setResult(Activity.RESULT_OK, resultIntent)
        finish()
    }
}

/**
 * RecyclerView adapter for Bluetooth devices
 */
class DeviceListAdapter(
    private val onDeviceSelected: (BluetoothDevice) -> Unit,
) : RecyclerView.Adapter<DeviceListAdapter.DeviceViewHolder>() {
    private val devices = mutableListOf<BluetoothDevice>()

    fun updateDevices(newDevices: List<BluetoothDevice>) {
        devices.clear()
        devices.addAll(newDevices)
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(
        parent: ViewGroup,
        viewType: Int,
    ): DeviceViewHolder {
        val view =
            LayoutInflater
                .from(parent.context)
                .inflate(R.layout.item_bluetooth_device, parent, false)
        return DeviceViewHolder(view)
    }

    override fun onBindViewHolder(
        holder: DeviceViewHolder,
        position: Int,
    ) {
        holder.bind(devices[position])
    }

    override fun getItemCount(): Int = devices.size

    inner class DeviceViewHolder(
        itemView: View,
    ) : RecyclerView.ViewHolder(itemView) {
        private val deviceName: TextView = itemView.findViewById(R.id.textDeviceName)
        private val deviceAddress: TextView = itemView.findViewById(R.id.textDeviceAddress)

        @SuppressLint("MissingPermission")
        fun bind(device: BluetoothDevice) {
            val name = try {
                device.name ?: "Unknown Device"
            } catch (e: SecurityException) {
                "Unknown Device"
            }

            deviceName.text = name
            deviceAddress.text = device.address

            itemView.setOnClickListener {
                onDeviceSelected(device)
            }
        }
    }
}
