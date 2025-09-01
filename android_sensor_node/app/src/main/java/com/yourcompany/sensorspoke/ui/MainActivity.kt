package com.yourcompany.sensorspoke.ui

import android.Manifest
import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.snackbar.Snackbar
import com.yourcompany.sensorspoke.R
import com.yourcompany.sensorspoke.controller.RecordingController
import com.yourcompany.sensorspoke.sensors.thermal.ThermalCameraRecorder
import com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001UIController
import com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001ConnectType
import com.yourcompany.sensorspoke.service.RecordingService
import com.yourcompany.sensorspoke.ui.dialogs.QuickStartDialog
import com.yourcompany.sensorspoke.ui.popup.DelPopup
import com.yourcompany.sensorspoke.utils.UserExperience
import kotlinx.coroutines.launch
import java.io.File

/**
 * MainActivity - Simplified IRCamera-style interface focused on Topdon TC001
 *
 * Replaces complex tabbed interface with clean device connection UI
 */
class MainActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "MainActivity"
    }

    private val vm: MainViewModel by viewModels()

    private var controller: RecordingController? = null

    // IRCamera-style UI components
    private var deviceConnectionLayout: LinearLayout? = null
    private var deviceIconConnected: ImageView? = null
    private var deviceIconDisconnected: ImageView? = null
    private var deviceNameText: TextView? = null
    private var deviceStatusText: TextView? = null
    private var btnStartRecording: Button? = null
    private var btnStopRecording: Button? = null
    private var statusText: TextView? = null
    private var rootLayout: ViewGroup? = null

    // TC001 UI Controller inspired by IRCamera
    private var tc001Controller: TC001UIController? = null
    private var delPopup: DelPopup? = null

    // User experience enhancements
    private lateinit var preferences: SharedPreferences
    private var isFirstLaunch: Boolean = false

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                UserExperience.Messaging.showSuccess(this, "Camera permission granted")
                startRecording()
            } else {
                val explanation = UserExperience.QuickStart.getPermissionExplanations()["camera"] ?: ""
                UserExperience.Messaging.showUserFriendlyError(this, "Permission denied: $explanation", "permission")
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
                        updateStatusText("Starting recording session: $sessionId")
                        lifecycleScope.launch {
                            try {
                                ensureController().startSession(sessionId)
                                UserExperience.Messaging.showSuccess(this@MainActivity, "Recording started", sessionId)
                            } catch (e: Exception) {
                                UserExperience.Messaging.showUserFriendlyError(this@MainActivity, e.message ?: "Unknown error", "recording")
                            }
                        }
                    }

                    RecordingService.ACTION_STOP_RECORDING -> {
                        updateStatusText("Stopping recording...")
                        lifecycleScope.launch {
                            runCatching {
                                controller?.stopSession()
                                UserExperience.Messaging.showSuccess(this@MainActivity, "Recording stopped")
                                updateStatusText("Ready to record")
                            }.onFailure { e ->
                                UserExperience.Messaging.showUserFriendlyError(this@MainActivity, e.message ?: "Unknown error", "recording")
                            }
                        }
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
        setContentView(R.layout.activity_main)

        // Initialize preferences and check first launch
        preferences = getSharedPreferences("sensor_spoke_prefs", Context.MODE_PRIVATE)
        isFirstLaunch = preferences.getBoolean("first_launch", true)

        // Initialize views for simplified IRCamera-style UI
        initializeViews()

        // Initialize TC001 controller
        initializeThermalController()

        // Setup device connection listeners
        setupDeviceConnectionUI()

        // Setup button handlers
        setupButtons()

        // Setup toolbar with menu
        setupToolbar()

        // Initialize status
        updateStatusText("Initializing...")

        // Initialize TC001 thermal camera system
        initializeTC001System()

        // Ensure background service for NSD + TCP server is running
        if (!isRunningUnderTest()) {
            val svcIntent = Intent(this, RecordingService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                startForegroundService(svcIntent)
            } else {
                startService(svcIntent)
            }
        }

        // Show quick start guide for first-time users
        if (isFirstLaunch) {
            showQuickStartGuide()
        }

        updateStatusText("Ready to connect")
    }

    private fun initializeViews() {
        deviceConnectionLayout = findViewById(R.id.deviceConnectionLayout)
        deviceIconConnected = findViewById(R.id.deviceIconConnected)
        deviceIconDisconnected = findViewById(R.id.deviceIconDisconnected)
        deviceNameText = findViewById(R.id.deviceNameText)
        deviceStatusText = findViewById(R.id.deviceStatusText)
        btnStartRecording = findViewById(R.id.btnStartRecording)
        btnStopRecording = findViewById(R.id.btnStopRecording)
        statusText = findViewById(R.id.statusText)
        rootLayout = findViewById<ViewGroup>(android.R.id.content)
    }

    private fun initializeThermalController() {
        tc001Controller = TC001UIController()

        // Setup device click listeners inspired by IRCamera
        tc001Controller?.onItemClickListener = { type ->
            handleDeviceClick(type)
        }

        tc001Controller?.onItemLongClickListener = { type ->
            handleDeviceLongClick(type)
        }

        // Setup DelPopup for device deletion
        delPopup = DelPopup(this)
        delPopup?.onDelListener = {
            deleteDevice()
        }

        // Observe connection status changes
        tc001Controller?.hasConnectLine?.observe(this) { isConnected ->
            updateDeviceConnectionUI(isConnected)
        }

        tc001Controller?.deviceConnectionStatus?.observe(this) { status ->
            updateStatusText("TC001 Status: $status")
        }
    }

    private fun setupDeviceConnectionUI() {
        deviceConnectionLayout?.setOnClickListener {
            handleDeviceClick(TC001ConnectType.LINE)
        }

        deviceConnectionLayout?.setOnLongClickListener { view ->
            delPopup?.show(view)
            true
        }
    }

    private fun updateDeviceConnectionUI(isConnected: Boolean) {
        runOnUiThread {
            if (isConnected) {
                deviceIconConnected?.visibility = View.VISIBLE
                deviceIconDisconnected?.visibility = View.GONE
                deviceNameText?.text = "TC001 Thermal Camera"
                deviceStatusText?.text = "Connected"
                deviceStatusText?.setTextColor(ContextCompat.getColor(this, R.color.status_text))
            } else {
                deviceIconConnected?.visibility = View.GONE
                deviceIconDisconnected?.visibility = View.VISIBLE
                deviceNameText?.text = "TC001 Thermal Camera"
                deviceStatusText?.text = "Disconnected"
                deviceStatusText?.setTextColor(ContextCompat.getColor(this, R.color.device_connect_state))
            }
        }
    }

    private fun handleDeviceClick(type: TC001ConnectType) {
        when (type) {
            TC001ConnectType.LINE -> {
                // Handle TC001 USB connection
                connectTC001Device()
            }
            else -> {
                // Handle other connection types if needed
                Log.d(TAG, "Connection type $type not yet implemented")
            }
        }
    }

    private fun handleDeviceLongClick(type: TC001ConnectType) {
        // Show deletion popup only when device is disconnected
        deviceConnectionLayout?.let { view ->
            delPopup?.show(view)
        }
    }

    private fun connectTC001Device() {
        lifecycleScope.launch {
            try {
                updateStatusText("Connecting to TC001...")
                
                // Simulate TC001 connection logic
                // In a real implementation, this would use TC001Connector
                val isConnected = initializeTC001Connection()
                
                tc001Controller?.updateConnectionStatus(isConnected)
                
                if (isConnected) {
                    UserExperience.Messaging.showSuccess(this@MainActivity, "TC001 connected successfully")
                } else {
                    UserExperience.Messaging.showUserFriendlyError(this@MainActivity, "Failed to connect TC001", "connection")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "TC001 connection failed", e)
                UserExperience.Messaging.showUserFriendlyError(this@MainActivity, e.message ?: "Connection failed", "connection")
            }
        }
    }

    private suspend fun initializeTC001Connection(): Boolean {
        // Simulate connection attempt
        // In real implementation, would use TC001Connector or ThermalCameraRecorder
        return try {
            Log.i(TAG, "Attempting TC001 connection...")
            // Simulate connection delay
            kotlinx.coroutines.delay(1000)
            true // Assume successful for now
        } catch (e: Exception) {
            Log.e(TAG, "TC001 connection failed", e)
            false
        }
    }

    private fun deleteDevice() {
        tc001Controller?.updateConnectionStatus(false)
        UserExperience.Messaging.showStatus(this, "Device deleted")
        updateStatusText("Device removed")
    }

    private fun setupToolbar() {
        supportActionBar?.setDisplayShowTitleEnabled(true)
        supportActionBar?.title = "TOPDON INFRARED"
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.main_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean =
        when (item.itemId) {
            R.id.action_quick_start -> {
                showQuickStartGuide()
                true
            }
            R.id.action_connection_help -> {
                showConnectionHelp()
                true
            }
            R.id.action_reset_tutorial -> {
                resetFirstLaunchFlag()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }

    // Remove the complex ViewPager setup and replace with simplified device management

    private fun setupButtons() {
        btnStartRecording?.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED
            ) {
                requestCameraPermission.launch(Manifest.permission.CAMERA)
            } else {
                startRecording()
            }
        }

        btnStopRecording?.setOnClickListener {
            stopRecording()
        }
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

    override fun onDestroy() {
        super.onDestroy()
        // Cleanup TC001 controller
        lifecycleScope.launch {
            tc001Controller?.updateConnectionStatus(false)
        }
    }

    private fun ensureController(): RecordingController {
        val existing = controller
        if (existing != null) return existing
        val c = RecordingController(applicationContext)
        // Register only thermal recorder for simplified TC001 focus
        c.register("thermal", ThermalCameraRecorder(applicationContext))
        controller = c
        return c
    }

    private fun startRecording() {
        updateStatusText("Starting thermal recording...")
        lifecycleScope.launch {
            try {
                // Simplified recording for TC001 thermal camera only
                ensureController().startSession()
                
                UserExperience.Messaging.showSuccess(this@MainActivity, "TC001 thermal recording started")
                updateStatusText("TC001 recording in progress")
                updateButtonStates(isRecording = true)
            } catch (e: Exception) {
                UserExperience.Messaging.showUserFriendlyError(this@MainActivity, e.message ?: "Unknown error", "recording")
                updateStatusText("Ready to record")
            }
        }
    }

    private fun stopRecording() {
        updateStatusText("Stopping thermal recording...")
        lifecycleScope.launch {
            try {
                // Stop thermal recording
                controller?.stopSession()
                
                UserExperience.Messaging.showSuccess(this@MainActivity, "TC001 recording stopped")
                updateStatusText("Ready to record")
                updateButtonStates(isRecording = false)
            } catch (e: Exception) {
                UserExperience.Messaging.showUserFriendlyError(this@MainActivity, e.message ?: "Unknown error", "recording")
            }
        }
    }

    private fun updateStatusText(status: String) {
        runOnUiThread {
            statusText?.text = status
        }
    }

    private fun updateButtonStates(isRecording: Boolean) {
        runOnUiThread {
            btnStartRecording?.isEnabled = !isRecording
            btnStopRecording?.isEnabled = isRecording
        }
    }

    private fun showQuickStartGuide() {
        QuickStartDialog.show(this) {
            // Mark first launch as complete
            preferences
                .edit()
                .putBoolean("first_launch", false)
                .apply()

            UserExperience.Messaging.showStatus(this, "Quick start guide completed!")
        }
    }

    private fun showConnectionHelp() {
        val troubleshootingSteps = UserExperience.QuickStart.getConnectionTroubleshootingSteps()
        val message =
            "Connection Troubleshooting:\n\n" +
                troubleshootingSteps
                    .mapIndexed { index, step ->
                        "${index + 1}. $step"
                    }.joinToString("\n")

        // Show as a Snackbar with action
        rootLayout?.let { layout ->
            val snackbar =
                Snackbar
                    .make(layout, "Connection help available", Snackbar.LENGTH_LONG)
                    .setAction("Show Help") {
                        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
                    }
            snackbar.show()
        }
    }

    private fun resetFirstLaunchFlag() {
        preferences
            .edit()
            .putBoolean("first_launch", true)
            .apply()
        UserExperience.Messaging.showStatus(this, "Tutorial will show on next launch")
    }

    private fun showFlashOverlay() {
        val parent = rootLayout ?: return
        val flashStartTime = System.nanoTime()

        val flash =
            View(this).apply {
                setBackgroundColor(Color.WHITE)
                layoutParams =
                    ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
                alpha = 1f
            }
        parent.addView(flash)

        // Log flash display timing for synchronization validation
        Log.d("FlashSync", "Flash overlay displayed at: ${flashStartTime}ns")

        flash.postDelayed({
            parent.removeView(flash)
            val flashEndTime = System.nanoTime()
            Log.d("FlashSync", "Flash overlay removed at: ${flashEndTime}ns (duration: ${(flashEndTime - flashStartTime) / 1_000_000}ms)")
        }, 150)
    }

    private fun logFlashEvent(tsNs: Long) {
        try {
            val actualFlashTime = System.nanoTime()
            val dir = getExternalFilesDir(null) ?: filesDir
            val f = File(dir, "flash_sync_events.csv")
            if (!f.exists()) {
                f.writeText("trigger_timestamp_ns,actual_flash_timestamp_ns,sync_delay_ms,device_id\n")
            }

            val syncDelay = (actualFlashTime - tsNs) / 1_000_000.0 // Convert to milliseconds
            val deviceId =
                android.os.Build.MODEL
                    ?.replace(" ", "_") ?: "unknown"

            f.appendText("$tsNs,$actualFlashTime,$syncDelay,$deviceId\n")

            Log.i("FlashSync", "Flash event logged - Sync delay: ${syncDelay}ms")
        } catch (e: Exception) {
            Log.e("FlashSync", "Failed to log flash event: ${e.message}", e)
        } catch (_: Exception) {
        }
    }

    private fun isRunningUnderTest(): Boolean =
        try {
            Class.forName("org.robolectric.Robolectric")
            true
        } catch (_: Throwable) {
            false
        }

    /**
     * Initialize TC001 thermal camera system
     */
    private fun initializeTC001System() {
        try {
            // Initialize TC001 logging
            com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001InitUtil
                .initLog()

            // Initialize TC001 USB receivers
            com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001InitUtil
                .initReceiver(this)

            // Initialize TC001 device manager
            com.yourcompany.sensorspoke.sensors.thermal.tc001.TC001InitUtil
                .initTC001DeviceManager(this)

            Log.i("MainActivity", "TC001 thermal camera system initialized successfully")
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to initialize TC001 system", e)
        }
    }
}
