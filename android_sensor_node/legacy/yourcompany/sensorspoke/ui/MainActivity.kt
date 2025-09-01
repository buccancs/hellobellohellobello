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
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.viewpager2.widget.ViewPager2
import com.google.android.material.snackbar.Snackbar
import com.google.android.material.tabs.TabLayout
import com.google.android.material.tabs.TabLayoutMediator
import com.yourcompany.sensorspoke.R
import com.yourcompany.sensorspoke.controller.RecordingController
import com.yourcompany.sensorspoke.sensors.audio.AudioRecorder
import com.yourcompany.sensorspoke.sensors.coordination.MultiModalSensorCoordinator
import com.yourcompany.sensorspoke.sensors.gsr.ShimmerRecorder
import com.yourcompany.sensorspoke.sensors.rgb.RgbCameraRecorder
import com.yourcompany.sensorspoke.sensors.thermal.ThermalCameraRecorder
import com.yourcompany.sensorspoke.service.RecordingService
import com.yourcompany.sensorspoke.ui.adapters.MainPagerAdapter
import com.yourcompany.sensorspoke.ui.dialogs.QuickStartDialog
import com.yourcompany.sensorspoke.ui.navigation.NavigationController
import com.yourcompany.sensorspoke.ui.navigation.ThermalNavigationState
import com.yourcompany.sensorspoke.utils.UserExperience
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : AppCompatActivity() {
    private val vm: MainViewModel by viewModels()

    private var controller: RecordingController? = null

    // Full integration: MultiModalSensorCoordinator for comprehensive sensor management
    private var multiModalCoordinator: MultiModalSensorCoordinator? = null

    private var viewPager: ViewPager2? = null
    private var tabLayout: TabLayout? = null
    private var btnStartRecording: Button? = null
    private var btnStopRecording: Button? = null
    private var statusText: TextView? = null
    private var rootLayout: ViewGroup? = null

    // Enhanced navigation controller from IRCamera architecture
    private var navigationController: NavigationController? = null

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

        // Initialize views
        viewPager = findViewById(R.id.viewPager)
        tabLayout = findViewById(R.id.tabLayout)
        btnStartRecording = findViewById(R.id.btnStartRecording)
        btnStopRecording = findViewById(R.id.btnStopRecording)
        statusText = findViewById(R.id.statusText)
        rootLayout = findViewById<ViewGroup>(android.R.id.content)

        // Setup ViewPager with fragments
        setupViewPager()

        // Setup button handlers
        setupButtons()

        // Setup toolbar with menu
        setupToolbar()

        // Initialize status
        updateStatusText("Initializing...")

        // Initialize TC001 thermal camera system
        initializeTC001System()

        // Ensure background service for NSD + TCP server is running (skip during unit tests)
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

    private fun setupToolbar() {
        supportActionBar?.setDisplayShowTitleEnabled(true)
        supportActionBar?.title = "Sensor Spoke"
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
            R.id.action_thermal_settings -> {
                // Navigate to thermal settings using enhanced navigation
                navigationController?.navigateToThermalCamera(ThermalNavigationState.SETTINGS)
                true
            }
            else -> super.onOptionsItemSelected(item)
        }

    private fun setupViewPager() {
        val adapter = MainPagerAdapter(this)
        viewPager?.adapter = adapter

        // Initialize enhanced navigation controller
        viewPager?.let { vp ->
            navigationController = NavigationController(this, vp)
        }

        // Connect TabLayout with ViewPager2
        tabLayout?.let { tabLayout ->
            viewPager?.let { viewPager ->
                TabLayoutMediator(tabLayout, viewPager) { tab, position ->
                    tab.text = MainPagerAdapter.TAB_TITLES[position]
                }.attach()

                // Register page change callback for enhanced navigation tracking
                viewPager.registerOnPageChangeCallback(
                    object : ViewPager2.OnPageChangeCallback() {
                        override fun onPageSelected(position: Int) {
                            super.onPageSelected(position)
                            navigationController?.updateCurrentTab(position)
                        }
                    },
                )
            }
        }
    }

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
        // Cleanup MultiModalSensorCoordinator for full integration
        lifecycleScope.launch {
            multiModalCoordinator?.stopRecording()
            multiModalCoordinator = null
        }
    }

    private fun ensureController(): RecordingController {
        val existing = controller
        if (existing != null) return existing
        val c = RecordingController(applicationContext)
        // Register recorders
        c.register("rgb", RgbCameraRecorder(applicationContext, this))
        c.register("thermal", ThermalCameraRecorder(applicationContext))
        c.register("gsr", ShimmerRecorder(applicationContext))
        c.register("audio", AudioRecorder(applicationContext)) // FR5: Audio recording support
        controller = c
        return c
    }

    /**
     * Ensure MultiModalSensorCoordinator is initialized - Full Integration
     */
    private suspend fun ensureMultiModalCoordinator(): MultiModalSensorCoordinator {
        val existing = multiModalCoordinator
        if (existing != null) return existing

        val coordinator = MultiModalSensorCoordinator(applicationContext, this)

        // Initialize the complete multi-modal system
        val initResult = coordinator.initializeSystem()
        if (initResult) {
            Log.i("MainActivity", "MultiModalSensorCoordinator initialized successfully - Full Integration active")
        } else {
            Log.w("MainActivity", "MultiModalSensorCoordinator initialization failed, falling back to individual recorders")
        }

        multiModalCoordinator = coordinator
        return coordinator
    }

    private fun startRecording() {
        updateStatusText("Starting recording...")
        lifecycleScope.launch {
            try {
                // Full Integration: Use MultiModalSensorCoordinator for comprehensive sensor management
                val coordinator = ensureMultiModalCoordinator()

                // Start coordinated multi-modal recording with session directory
                val sessionDir = File(applicationContext.filesDir, "sessions")
                if (!sessionDir.exists()) sessionDir.mkdirs()

                val startResult = coordinator.startRecording(sessionDir)

                if (startResult) {
                    UserExperience.Messaging.showSuccess(this@MainActivity, "Full multi-modal recording started")
                    updateStatusText("Full integration recording in progress")
                    updateButtonStates(isRecording = true)
                } else {
                    // Fallback to individual controller if coordinator fails
                    Log.w("MainActivity", "Coordinator failed, falling back to individual recorders")
                    ensureController().startSession()
                    UserExperience.Messaging.showSuccess(this@MainActivity, "Recording started (fallback mode)")
                    updateStatusText("Recording in progress (fallback)")
                    updateButtonStates(isRecording = true)
                }
            } catch (e: Exception) {
                UserExperience.Messaging.showUserFriendlyError(this@MainActivity, e.message ?: "Unknown error", "recording")
                updateStatusText("Ready to record")
            }
        }
    }

    private fun stopRecording() {
        updateStatusText("Stopping recording...")
        lifecycleScope.launch {
            try {
                // Full Integration: Stop coordinated recording first, then fallback if needed
                val coordinator = multiModalCoordinator
                if (coordinator != null) {
                    val stopResult = coordinator.stopRecording()
                    if (stopResult) {
                        UserExperience.Messaging.showSuccess(this@MainActivity, "Full multi-modal recording stopped")
                        updateStatusText("Ready to record")
                        updateButtonStates(isRecording = false)
                        return@launch
                    } else {
                        Log.w("MainActivity", "Coordinator stop failed, trying individual controller")
                    }
                }

                // Fallback to individual controller
                controller?.stopSession()
                UserExperience.Messaging.showSuccess(this@MainActivity, "Recording stopped")
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
     * Navigate to thermal camera preview - Enhanced integration method
     */
    fun navigateToThermalPreview() {
        // Use NavigationController to navigate to thermal camera
        navigationController?.navigateToThermalCamera(ThermalNavigationState.PREVIEW)
    }

    /**
     * Navigate to thermal camera settings
     */
    fun navigateToThermalSettings() {
        navigationController?.navigateToThermalCamera(ThermalNavigationState.SETTINGS)
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
