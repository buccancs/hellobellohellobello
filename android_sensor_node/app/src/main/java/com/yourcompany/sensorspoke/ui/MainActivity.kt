package com.yourcompany.sensorspoke.ui

import android.Manifest
import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.yourcompany.sensorspoke.controller.RecordingController
import com.yourcompany.sensorspoke.sensors.gsr.ShimmerRecorder
import com.yourcompany.sensorspoke.sensors.rgb.RgbCameraRecorder
import com.yourcompany.sensorspoke.sensors.thermal.ThermalCameraRecorder
import com.yourcompany.sensorspoke.service.RecordingService
import com.yourcompany.sensorspoke.utils.EmulatorUtils
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : ComponentActivity() {
    private val vm: MainViewModel by viewModels()

    private var controller: RecordingController? = null
    private var rootLayout: LinearLayout? = null

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startRecording()
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
            }
        }

    private val controlReceiver =
        object : BroadcastReceiver() {
            override fun onReceive(
                context: Context?,
                intent: Intent?,
            ) {
                val action = intent?.action ?: return
                when (action) {
                    RecordingService.ACTION_START_RECORDING -> {
                        val sessionId = intent.getStringExtra(RecordingService.EXTRA_SESSION_ID)
                        lifecycleScope.launch {
                            try {
                                ensureController().startSession(sessionId)
                            } catch (_: Exception) {
                            }
                        }
                    }

                    RecordingService.ACTION_STOP_RECORDING -> {
                        lifecycleScope.launch { runCatching { controller?.stopSession() } }
                    }

                    RecordingService.ACTION_FLASH_SYNC -> {
                        val ts = intent.getLongExtra(RecordingService.EXTRA_FLASH_TS_NS, 0L)
                        showFlashOverlay()
                        logFlashEvent(ts)
                    }
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Log environment info for debugging
        android.util.Log.i("MainActivity", "Starting app: ${EmulatorUtils.getEnvironmentInfo()}")
        
        // Check if running on emulator and log hardware capabilities
        val isEmulator = EmulatorUtils.isRunningOnEmulator()
        val hasCamera = EmulatorUtils.isCameraAvailable(this)
        val hasBluetooth = EmulatorUtils.isBluetoothAvailable(this)
        val hasUsbHost = EmulatorUtils.isUsbHostAvailable(this)
        
        android.util.Log.i("MainActivity", "Hardware capabilities - Camera: $hasCamera, Bluetooth: $hasBluetooth, USB: $hasUsbHost")
        
        if (isEmulator) {
            android.util.Log.i("MainActivity", "Running on emulator - will use simulation mode for sensors")
        }
        
        // Simple UI with Start/Stop buttons
        val layout =
            LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
            }
        rootLayout = layout
        val startBtn = Button(this).apply { text = "Start Recording" }
        val stopBtn = Button(this).apply { text = "Stop Recording" }
        layout.addView(startBtn)
        layout.addView(stopBtn)
        setContentView(layout)

        // Ensure background service for NSD + TCP server is running (skip during unit tests)
        if (!isRunningUnderTest()) {
            try {
                val svcIntent = Intent(this, RecordingService::class.java)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    startForegroundService(svcIntent)
                } else {
                    startService(svcIntent)
                }
            } catch (e: Exception) {
                android.util.Log.e("MainActivity", "Failed to start recording service", e)
                Toast.makeText(this, "Warning: Background service failed to start", Toast.LENGTH_LONG).show()
            }
        }

        startBtn.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED
            ) {
                requestCameraPermission.launch(Manifest.permission.CAMERA)
            } else {
                startRecording()
            }
        }
        stopBtn.setOnClickListener { stopRecording() }
    }

    @SuppressLint("UnspecifiedRegisterReceiverFlag")
    override fun onStart() {
        super.onStart()
        val filter =
            IntentFilter().apply {
                addAction(RecordingService.ACTION_START_RECORDING)
                addAction(RecordingService.ACTION_STOP_RECORDING)
                addAction(RecordingService.ACTION_FLASH_SYNC)
            }
        if (Build.VERSION.SDK_INT >= 33) {
            registerReceiver(controlReceiver, filter, RECEIVER_NOT_EXPORTED)
        } else {
            @Suppress("DEPRECATION")
            registerReceiver(controlReceiver, filter)
        }
    }

    override fun onStop() {
        super.onStop()
        runCatching { unregisterReceiver(controlReceiver) }
    }

    private fun ensureController(): RecordingController {
        val existing = controller
        if (existing != null) return existing
        val c = RecordingController(applicationContext)
        
        // Register recorders based on hardware availability
        val isEmulator = EmulatorUtils.isRunningOnEmulator()
        val hasCamera = EmulatorUtils.isCameraAvailable(this)
        val hasBluetooth = EmulatorUtils.isBluetoothAvailable(this)
        val hasUsbHost = EmulatorUtils.isUsbHostAvailable(this)
        
        try {
            // RGB camera recorder - only if camera is available or we're on emulator (will simulate)
            if (hasCamera || isEmulator) {
                c.register("rgb", RgbCameraRecorder(applicationContext, this))
                android.util.Log.i("MainActivity", "Registered RGB camera recorder")
            } else {
                android.util.Log.w("MainActivity", "Camera not available - skipping RGB recorder")
            }
        } catch (e: Exception) {
            android.util.Log.e("MainActivity", "Failed to register RGB camera recorder", e)
        }
        
        try {
            // Thermal camera recorder - register but it will handle USB availability internally
            c.register("thermal", ThermalCameraRecorder())
            android.util.Log.i("MainActivity", "Registered thermal camera recorder")
        } catch (e: Exception) {
            android.util.Log.e("MainActivity", "Failed to register thermal camera recorder", e)
        }
        
        try {
            // GSR recorder - register but it will handle Bluetooth availability internally
            c.register("gsr", ShimmerRecorder())
            android.util.Log.i("MainActivity", "Registered GSR recorder")
        } catch (e: Exception) {
            android.util.Log.e("MainActivity", "Failed to register GSR recorder", e)
        }
        
        controller = c
        return c
    }

    private fun startRecording() {
        lifecycleScope.launch {
            try {
                ensureController().startSession()
                Toast.makeText(this@MainActivity, "Recording started", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, "Failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun stopRecording() {
        lifecycleScope.launch {
            try {
                controller?.stopSession()
                Toast.makeText(this@MainActivity, "Recording stopped", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, "Failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun showFlashOverlay() {
        val parent = rootLayout ?: return
        val flash =
            View(this).apply {
                setBackgroundColor(Color.WHITE)
                layoutParams =
                    ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
                alpha = 1f
            }
        parent.addView(flash)
        flash.postDelayed({ parent.removeView(flash) }, 150)
    }

    private fun logFlashEvent(tsNs: Long) {
        try {
            val dir = getExternalFilesDir(null) ?: filesDir
            val f = File(dir, "flash_sync_events.csv")
            if (!f.exists()) {
                f.writeText("timestamp_ns\n")
            }
            f.appendText("$tsNs\n")
        } catch (_: Exception) {
        }
    }

    private fun isRunningUnderTest(): Boolean {
        return try {
            Class.forName("org.robolectric.Robolectric")
            true
        } catch (_: Throwable) {
            false
        }
    }
}
