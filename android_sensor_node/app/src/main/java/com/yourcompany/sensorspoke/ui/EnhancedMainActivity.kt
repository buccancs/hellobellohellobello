package com.yourcompany.sensorspoke.ui

import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.ServiceConnection
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.viewpager2.adapter.FragmentStateAdapter
import androidx.viewpager2.widget.ViewPager2
import com.google.android.material.tabs.TabLayout
import com.google.android.material.tabs.TabLayoutMediator
import com.yourcompany.sensorspoke.R
import com.yourcompany.sensorspoke.controller.RecordingController
import com.yourcompany.sensorspoke.data.MultiModalSensorCoordinator
import com.yourcompany.sensorspoke.sensors.audio.AudioRecorder
import com.yourcompany.sensorspoke.sensors.gsr.ShimmerRecorder
import com.yourcompany.sensorspoke.sensors.rgb.RgbCameraRecorder
import com.yourcompany.sensorspoke.sensors.thermal.ThermalCameraRecorder
import com.yourcompany.sensorspoke.service.RecordingService
import com.yourcompany.sensorspoke.ui.fragments.DashboardFragment
import com.yourcompany.sensorspoke.ui.fragments.RgbPreviewFragment
import com.yourcompany.sensorspoke.ui.fragments.SensorStatusFragment
import com.yourcompany.sensorspoke.ui.fragments.SessionManagementFragment
import com.yourcompany.sensorspoke.ui.fragments.ThermalPreviewFragment
import com.yourcompany.sensorspoke.utils.PermissionManager
import kotlinx.coroutines.launch

/**
 * Phase 4: Enhanced Main Activity with Tabbed Interface
 *
 * Advanced UI for the Multi-Modal Physiological Sensing Platform featuring:
 * - Real-time sensor data visualization
 * - Tabbed interface with sensor previews
 * - Live connection status monitoring
 * - Session management with visual feedback
 * - Enhanced control interface for research applications
 */
class EnhancedMainActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "EnhancedMainActivity"
    }

    private var controller: RecordingController? = null
    private var multiModalCoordinator: MultiModalSensorCoordinator? = null
    private lateinit var permissionManager: PermissionManager
    private lateinit var mainViewModel: MainViewModel

    private var recordingService: RecordingService? = null
    private var isServiceBound = false

    private lateinit var viewPager: ViewPager2
    private lateinit var tabLayout: TabLayout

    private val fragments = listOf<Pair<String, () -> Fragment>>(
        "Dashboard" to { DashboardFragment() },
        "RGB Camera" to { RgbPreviewFragment() },
        "Thermal" to { ThermalPreviewFragment() },
        "Sensors" to { SensorStatusFragment() },
        "Sessions" to { SessionManagementFragment() },
    )

    private val commandReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                RecordingService.ACTION_START_RECORDING -> {
                    val sessionId = intent.getStringExtra(RecordingService.EXTRA_SESSION_ID) ?: ""
                    handleRemoteStartRecording(sessionId)
                }
                RecordingService.ACTION_STOP_RECORDING -> {
                    handleRemoteStopRecording()
                }
                RecordingService.ACTION_FLASH_SYNC -> {
                    val flashTimestamp = intent.getLongExtra(RecordingService.EXTRA_FLASH_TS_NS, 0)
                    handleFlashSync(flashTimestamp)
                }
            }
        }
    }

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            isServiceBound = true
            Log.i(TAG, "RecordingService connected")
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            isServiceBound = false
            Log.i(TAG, "RecordingService disconnected")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_enhanced_main)

        Log.i(TAG, "Enhanced MainActivity created - Phase 4 Multi-Modal Interface")

        permissionManager = PermissionManager(this)
        mainViewModel = ViewModelProvider(this)[MainViewModel::class.java]

        setupTabbedInterface()

        lifecycleScope.launch {
            initializeComponents()
        }

        startRecordingService()

        registerCommandReceiver()

        requestNecessaryPermissions()
    }

    private fun setupTabbedInterface() {
        viewPager = findViewById(R.id.viewPager)
        tabLayout = findViewById(R.id.tabLayout)

        viewPager.adapter = object : FragmentStateAdapter(this) {
            override fun getItemCount(): Int = fragments.size

            override fun createFragment(position: Int): Fragment {
                return fragments[position].second()
            }
        }

        TabLayoutMediator(tabLayout, viewPager) { tab, position ->
            tab.text = fragments[position].first
        }.attach()

        Log.i(TAG, "Tabbed interface initialized with ${fragments.size} tabs")
    }

    private suspend fun initializeComponents() {
        try {
            controller = ensureController()

            multiModalCoordinator = ensureMultiModalCoordinator()

            // Update ViewModel with controller
            controller?.let { mainViewModel.initialize(it) }

            Log.i(TAG, "All components initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize components: ${e.message}", e)
            showError("Failed to initialize sensor components: ${e.message}")
        }
    }

    private fun ensureController(): RecordingController {
        val existing = controller
        if (existing != null) return existing

        val c = RecordingController(applicationContext)

        c.register("rgb", RgbCameraRecorder(applicationContext, this))
        c.register("thermal", ThermalCameraRecorder(applicationContext))
        c.register("gsr", ShimmerRecorder(applicationContext))
        c.register("audio", AudioRecorder(applicationContext))

        controller = c
        Log.i(TAG, "RecordingController initialized with all sensor modalities")
        return c
    }

    private suspend fun ensureMultiModalCoordinator(): MultiModalSensorCoordinator {
        val existing = multiModalCoordinator
        if (existing != null) return existing

        val coordinator = MultiModalSensorCoordinator(applicationContext, this)

        val initResult = coordinator.initializeSystem()
        if (initResult) {
            Log.i(TAG, "MultiModalSensorCoordinator initialized successfully - Full Integration active")
        } else {
            Log.w(TAG, "MultiModalSensorCoordinator initialization failed, falling back to individual recorders")
        }

        multiModalCoordinator = coordinator
        return coordinator
    }

    private fun startRecordingService() {
        val serviceIntent = Intent(this, RecordingService::class.java)
        ContextCompat.startForegroundService(this, serviceIntent)
        bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)
        Log.i(TAG, "RecordingService started and bound")
    }

    private fun registerCommandReceiver() {
        val filter = IntentFilter().apply {
            addAction(RecordingService.ACTION_START_RECORDING)
            addAction(RecordingService.ACTION_STOP_RECORDING)
            addAction(RecordingService.ACTION_FLASH_SYNC)
        }
        registerReceiver(commandReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        Log.i(TAG, "Command receiver registered")
    }

    private fun requestNecessaryPermissions() {
        lifecycleScope.launch {
            try {
                permissionManager.requestCameraPermissions { success ->
                    Log.d(TAG, "Camera permissions: $success")
                }
                permissionManager.requestBluetoothPermissions { success ->
                    Log.d(TAG, "Bluetooth permissions: $success")
                }
                permissionManager.requestStoragePermissions { success ->
                    Log.d(TAG, "Storage permissions: $success")
                }

                Log.i(TAG, "Permission requests initiated")
            } catch (e: Exception) {
                Log.e(TAG, "Error requesting permissions: ${e.message}", e)
                showError("Failed to request necessary permissions")
            }
        }
    }

    private fun handleRemoteStartRecording(sessionId: String) {
        lifecycleScope.launch {
            try {
                Log.i(TAG, "Remote start recording command received: $sessionId")

                val c = controller ?: ensureController()
                c.startSession(sessionId)

                mainViewModel.startRecording(sessionId)

                runOnUiThread {
                    Toast.makeText(
                        this@EnhancedMainActivity,
                        "Recording started: $sessionId",
                        Toast.LENGTH_SHORT,
                    ).show()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start remote recording: ${e.message}", e)
                showError("Failed to start recording: ${e.message}")
            }
        }
    }

    private fun handleRemoteStopRecording() {
        lifecycleScope.launch {
            try {
                Log.i(TAG, "Remote stop recording command received")

                val c = controller ?: return@launch
                c.stopSession()

                mainViewModel.stopRecording()

                runOnUiThread {
                    Toast.makeText(
                        this@EnhancedMainActivity,
                        "Recording stopped",
                        Toast.LENGTH_SHORT,
                    ).show()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to stop remote recording: ${e.message}", e)
                showError("Failed to stop recording: ${e.message}")
            }
        }
    }

    private fun handleFlashSync(flashTimestamp: Long) {
        Log.i(TAG, "Flash sync command received: $flashTimestamp")

        lifecycleScope.launch {
            try {
                runOnUiThread {
                    val originalBackground = window.decorView.background
                    window.decorView.setBackgroundColor(android.graphics.Color.WHITE)
                    window.decorView.postDelayed({
                        window.decorView.background = originalBackground
                    }, 100)
                }

                Log.i(TAG, "Flash sync executed at timestamp: $flashTimestamp")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to execute flash sync: ${e.message}", e)
            }
        }
    }

    private fun showError(message: String) {
        runOnUiThread {
            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        try {
            unregisterReceiver(commandReceiver)
        } catch (e: Exception) {
            Log.w(TAG, "Error unregistering receiver: ${e.message}")
        }

        if (isServiceBound) {
            unbindService(serviceConnection)
        }

        multiModalCoordinator?.cleanup()

        Log.i(TAG, "Enhanced MainActivity destroyed")
    }

    fun getRecordingController(): RecordingController? = controller
    fun getMultiModalCoordinator(): MultiModalSensorCoordinator? = multiModalCoordinator
    fun getMainViewModel(): MainViewModel = mainViewModel
}
