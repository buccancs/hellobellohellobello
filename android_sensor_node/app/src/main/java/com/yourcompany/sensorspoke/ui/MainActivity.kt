package com.yourcompany.sensorspoke.ui

import android.annotation.SuppressLint
import android.app.AlertDialog
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.SharedPreferences
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
import com.yourcompany.sensorspoke.network.ConnectionManager
import com.yourcompany.sensorspoke.sensors.audio.AudioRecorder
import com.yourcompany.sensorspoke.sensors.coordination.MultiModalSensorCoordinator
import com.yourcompany.sensorspoke.sensors.gsr.ShimmerRecorder
import com.yourcompany.sensorspoke.sensors.rgb.RgbCameraRecorder
import com.yourcompany.sensorspoke.sensors.thermal.ThermalCameraRecorder
import com.yourcompany.sensorspoke.service.RecordingService
import com.yourcompany.sensorspoke.ui.adapters.MainPagerAdapter
import com.yourcompany.sensorspoke.ui.components.SensorStatusIndicator
import com.yourcompany.sensorspoke.ui.dialogs.QuickStartDialog
import com.yourcompany.sensorspoke.ui.models.SensorStatus
import com.yourcompany.sensorspoke.ui.navigation.NavigationController
import com.yourcompany.sensorspoke.ui.navigation.ThermalNavigationState
import com.yourcompany.sensorspoke.utils.PermissionManager
import com.yourcompany.sensorspoke.utils.UserExperience
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "MainActivity"
    }

    private val vm: MainViewModel by viewModels()

    private var controller: RecordingController? = null
    private var connectionManager: ConnectionManager? = null

    private var multiModalCoordinator: MultiModalSensorCoordinator? = null

    private var viewPager: ViewPager2? = null
    private var tabLayout: TabLayout? = null
    private var btnStartRecording: Button? = null
    private var btnStopRecording: Button? = null
    private var statusText: TextView? = null
    private var recordingTimeText: TextView? = null
    private var rootLayout: ViewGroup? = null

    private var rgbStatusIndicator: View? = null
    private var rgbStatusText: TextView? = null
    private var thermalStatusIndicator: View? = null
    private var thermalStatusText: TextView? = null
    private var gsrStatusIndicator: View? = null
    private var gsrStatusText: TextView? = null
    private var pcStatusIndicator: View? = null
    private var pcStatusText: TextView? = null

    // Sensor status indicator components
    private var rgbSensorStatus: SensorStatusIndicator? = null
    private var thermalSensorStatus: SensorStatusIndicator? = null
    private var gsrSensorStatus: SensorStatusIndicator? = null
    private var pcSensorStatus: SensorStatusIndicator? = null

    private var navigationController: NavigationController? = null

    private lateinit var preferences: SharedPreferences
    private var isFirstLaunch: Boolean = false

    private lateinit var permissionManager: PermissionManager

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
                        vm.updateUiState { copy(statusText = "Starting recording session: $sessionId") }
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
                        vm.updateUiState { copy(statusText = "Stopping recording...") }
                        lifecycleScope.launch {
                            runCatching {
                                controller?.stopSession()
                                UserExperience.Messaging.showSuccess(this@MainActivity, "Recording stopped")
                                vm.updateUiState { copy(statusText = "Ready to record") }
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

        initializeViews()

        setupViewPager()

        setupButtons()

        setupToolbar()

        permissionManager = PermissionManager(this)

        connectionManager = createConnectionManager()

        // Initialize ViewModel with orchestrator and connection manager
        initializeViewModel()

        setupUiStateObservers()

        vm.updateUiState { copy(statusText = "Initializing...") }

        initializeTC001System()

        if (!isRunningUnderTest()) {
            val svcIntent = Intent(this, RecordingService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                startForegroundService(svcIntent)
            } else {
                startService(svcIntent)
            }
        }

        if (isFirstLaunch) {
            showQuickStartGuide()
        }

        if (!permissionManager.areAllPermissionsGranted()) {
            requestAllPermissions()
        }

        initializeSensorStatusMonitoring()

        vm.updateUiState { copy(statusText = "Ready to connect") }
    }

    /**
     * Initialize all UI view references
     */
    private fun initializeViews() {
        viewPager = findViewById(R.id.viewPager)
        tabLayout = findViewById(R.id.tabLayout)
        btnStartRecording = findViewById(R.id.btnStartRecording)
        btnStopRecording = findViewById(R.id.btnStopRecording)
        statusText = findViewById(R.id.statusText)
        recordingTimeText = findViewById(R.id.recordingTimeText)
        rootLayout = findViewById<ViewGroup>(android.R.id.content)

        rgbStatusIndicator = findViewById(R.id.rgbStatusIndicator)
        rgbStatusText = findViewById(R.id.rgbStatusText)
        thermalStatusIndicator = findViewById(R.id.thermalStatusIndicator)
        thermalStatusText = findViewById(R.id.thermalStatusText)
        gsrStatusIndicator = findViewById(R.id.gsrStatusIndicator)
        gsrStatusText = findViewById(R.id.gsrStatusText)
        pcStatusIndicator = findViewById(R.id.pcStatusIndicator)
        pcStatusText = findViewById(R.id.pcStatusText)
    }

    /**
     * Create and configure connection manager
     */
    private fun createConnectionManager(): ConnectionManager? {
        return try {
            val networkClient = com.yourcompany.sensorspoke.network.NetworkClient(this)
            val manager = ConnectionManager(this, networkClient)

            manager.onConnectionEstablished = { address, port ->
                runOnUiThread {
                    vm.updatePcConnectionStatus(true, "$address:$port")
                    showToast("Connected to PC Hub")
                }
            }

            manager.onConnectionLost = {
                runOnUiThread {
                    vm.updatePcConnectionStatus(false)
                    showToast("Connection to PC Hub lost")
                }
            }

            manager.onConnectionRestored = {
                runOnUiThread {
                    vm.updatePcConnectionStatus(true)
                    showToast("Connection restored")
                }
            }

            manager.onReconnectFailed = {
                runOnUiThread {
                    vm.showError("Failed to reconnect to PC Hub")
                }
            }

            manager
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create connection manager", e)
            null
        }
    }

    /**
     * Initialize ViewModel with orchestrator and connection manager
     */
    private fun initializeViewModel() {
        val orchestrator = ensureController()
        connectionManager?.let { connManager ->
            vm.initialize(orchestrator, connManager)
        } ?: run {
            vm.initialize(orchestrator)
        }
    }

    /**
     * Setup UI state observers for reactive updates
     */
    private fun setupUiStateObservers() {
        lifecycleScope.launch {
            vm.uiState.collect { uiState ->
                updateUI(uiState)
            }
        }

        lifecycleScope.launch {
            vm.showErrorDialog.collect { showDialog ->
                if (showDialog) {
                    vm.errorMessage.value?.let { message ->
                        showErrorDialog(message)
                    }
                }
            }
        }

        lifecycleScope.launch {
            vm.sensorStatus.collect { sensorMap ->
                updateSensorStatusIndicators(sensorMap)
            }
        }

        lifecycleScope.launch {
            vm.recordingState.collect { state ->
                updateButtonsForRecordingState(state)
            }
        }
    }

    /**
     * Update UI based on comprehensive UI state
     */
    private fun updateUI(state: MainViewModel.MainUiState) {
        statusText?.text = state.statusText
        updateStatusTextColor(state.statusText)

        if (state.isRecording) {
            recordingTimeText?.visibility = View.VISIBLE
            recordingTimeText?.text = "Recording: ${state.recordingElapsedTime}"
        } else {
            recordingTimeText?.visibility = View.GONE
        }

        btnStartRecording?.isEnabled = state.startButtonEnabled
        btnStopRecording?.isEnabled = state.stopButtonEnabled

        btnStartRecording?.text = if (state.isRecording) "Recording..." else "Start Recording"

        updateSensorIndicator(rgbStatusIndicator, rgbStatusText, state.isCameraConnected, "RGB")
        updateSensorIndicator(thermalStatusIndicator, thermalStatusText, state.isThermalConnected, "Thermal", state.thermalStatus.isSimulated)
        updateSensorIndicator(gsrStatusIndicator, gsrStatusText, state.isShimmerConnected, "GSR")
        updateSensorIndicator(pcStatusIndicator, pcStatusText, state.isPcConnected, "PC Link")

        if (state.thermalStatus.isSimulated) {
            thermalStatusText?.text = "Thermal (Sim)"
        }
    }

    /**
     * Update individual sensor status indicator
     */
    private fun updateSensorIndicator(indicator: View?, textView: TextView?, isConnected: Boolean, label: String, isSimulated: Boolean = false) {
        indicator?.background = ContextCompat.getDrawable(
            this,
            when {
                isConnected && isSimulated -> R.drawable.status_indicator_orange
                isConnected -> R.drawable.status_indicator_green
                else -> R.drawable.status_indicator_red
            },
        )

        textView?.text = when {
            isConnected && isSimulated -> "$label (Sim)"
            isConnected -> label
            else -> label
        }
    }

    /**
     * Update sensor status indicators from sensor map
     */
    private fun updateSensorStatusIndicators(sensorMap: Map<String, MainViewModel.SensorStatus>) {
        sensorMap["rgb"]?.let { status ->
            vm.updateRgbCameraStatus(status.isActive)
        }

        sensorMap["thermal"]?.let { status ->
            vm.updateThermalStatus(status.isActive, status.isSimulated)
        }

        sensorMap["gsr"]?.let { status ->
            vm.updateGsrStatus(status.isActive)
        }
    }

    /**
     * Update button states based on recording state
     */
    private fun updateButtonsForRecordingState(state: MainViewModel.RecordingState) {
        when (state) {
            MainViewModel.RecordingState.IDLE -> {
                btnStartRecording?.isEnabled = permissionManager.areAllPermissionsGranted()
                btnStopRecording?.isEnabled = false
                btnStartRecording?.text = "Start Recording"
            }
            MainViewModel.RecordingState.PREPARING -> {
                btnStartRecording?.isEnabled = false
                btnStopRecording?.isEnabled = false
                btnStartRecording?.text = "Preparing..."
            }
            MainViewModel.RecordingState.RECORDING -> {
                btnStartRecording?.isEnabled = false
                btnStopRecording?.isEnabled = true
                btnStartRecording?.text = "Recording..."
            }
            MainViewModel.RecordingState.STOPPING -> {
                btnStartRecording?.isEnabled = false
                btnStopRecording?.isEnabled = false
                btnStartRecording?.text = "Stopping..."
            }
            MainViewModel.RecordingState.ERROR -> {
                btnStartRecording?.isEnabled = true
                btnStopRecording?.isEnabled = false
                btnStartRecording?.text = "Start Recording"
            }
        }
    }

    /**
     * Initialize sensor status monitoring
     */
    private fun initializeSensorStatusMonitoring() {
        lifecycleScope.launch {
            while (true) {
                try {
                    monitorRgbCameraStatus()

                    monitorThermalCameraStatus()

                    monitorGsrSensorStatus()
                } catch (e: Exception) {
                    Log.e(TAG, "Error monitoring sensor status", e)
                }

                kotlinx.coroutines.delay(3000)
            }
        }
    }

    /**
     * Monitor RGB camera status
     */
    private suspend fun monitorRgbCameraStatus() {
        try {
            val cameraManager = getSystemService(Context.CAMERA_SERVICE) as android.hardware.camera2.CameraManager
            val cameraIds = cameraManager.cameraIdList
            val isAvailable = cameraIds.isNotEmpty()

            vm.updateRgbCameraStatus(isAvailable, isAvailable)
        } catch (e: Exception) {
            vm.updateRgbCameraStatus(false)
        }
    }

    /**
     * Monitor thermal camera status
     */
    private suspend fun monitorThermalCameraStatus() {
        try {
            val isHardwareConnected = checkThermalHardware()
            val isSimulated = !isHardwareConnected

            vm.updateThermalStatus(true, isSimulated, if (isHardwareConnected) "TC001" else null)
        } catch (e: Exception) {
            vm.updateThermalStatus(false, false)
        }
    }

    /**
     * Monitor GSR sensor status
     */
    private suspend fun monitorGsrSensorStatus() {
        try {
            val isConnected = checkShimmerConnection()

            vm.updateGsrStatus(isConnected, if (isConnected) "Shimmer3" else null)
        } catch (e: Exception) {
            vm.updateGsrStatus(false)
        }
    }

    /**
     * Check thermal hardware availability
     */
    private fun checkThermalHardware(): Boolean {
        // For now, return false to simulate simulation mode
        return false
    }

    /**
     * Check Shimmer connection status
     */
    private fun checkShimmerConnection(): Boolean {
        return false
    }

    /**
     * Show error dialog with proper material design
     */
    private fun showErrorDialog(message: String) {
        AlertDialog.Builder(this)
            .setTitle("Error")
            .setMessage(message)
            .setPositiveButton("OK") { dialog, _ ->
                dialog.dismiss()
                vm.clearError()
            }
            .setCancelable(false)
            .show()
    }

    /**
     * Show toast message
     */
    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    /**
     * Update status text color based on content
     */
    private fun updateStatusTextColor(message: String) {
        statusText?.setTextColor(
            ContextCompat.getColor(
                this,
                when {
                    message.contains("error", ignoreCase = true) ||
                        message.contains("failed", ignoreCase = true) -> android.R.color.holo_red_dark
                    message.contains("recording", ignoreCase = true) -> android.R.color.holo_red_light
                    message.contains("ready", ignoreCase = true) ||
                        message.contains("connected", ignoreCase = true) ||
                        message.contains("success", ignoreCase = true) -> android.R.color.holo_green_dark
                    message.contains("checking", ignoreCase = true) ||
                        message.contains("starting", ignoreCase = true) ||
                        message.contains("stopping", ignoreCase = true) -> android.R.color.holo_blue_dark
                    else -> android.R.color.primary_text_light
                },
            ),
        )
    }

    /**
     * Initialize sensor status indicators with default states
     */
    private fun initializeSensorStatusIndicators() {
        rgbSensorStatus?.updateStatus(
            "RGB",
            SensorStatusIndicator.SensorStatus(
                name = "RGB Camera",
                isActive = false,
                isHealthy = false,
                statusMessage = "Offline",
            ),
        )

        thermalSensorStatus?.updateStatus(
            "Thermal",
            SensorStatusIndicator.SensorStatus(
                name = "Thermal Camera",
                isActive = false,
                isHealthy = false,
                statusMessage = "Offline",
            ),
        )

        gsrSensorStatus?.updateStatus(
            "GSR",
            SensorStatusIndicator.SensorStatus(
                name = "GSR Sensor",
                isActive = false,
                isHealthy = false,
                statusMessage = "Disconnected",
            ),
        )

        pcSensorStatus?.updateStatus(
            "PC",
            SensorStatusIndicator.SensorStatus(
                name = "PC Link",
                isActive = false,
                isHealthy = false,
                statusMessage = "Not Connected",
            ),
        )

        // Then update with ViewModel data
        updateSensorStatus()
    }

    /**
     * Update sensor status from current ViewModel state
     */
    private fun updateSensorStatus() {
        // This will be called to refresh sensor status indicators
        // The actual updates happen via the uiState flow in observeUiState()
    }

    /**
     * Observe UI state changes and update the interface accordingly
     */
    private fun observeUiState() {
        lifecycleScope.launch {
            vm.uiState.collect { state ->
                statusText?.text = state.statusText

                if (state.isRecording && state.recordingElapsedTime != "00:00") {
                    recordingTimeText?.visibility = View.VISIBLE
                    recordingTimeText?.text = state.recordingElapsedTime
                } else {
                    recordingTimeText?.visibility = View.GONE
                }

                btnStartRecording?.isEnabled = state.startButtonEnabled
                btnStopRecording?.isEnabled = state.stopButtonEnabled

                btnStartRecording?.text = if (state.isRecording) "Recording..." else "Start Recording"

                rgbSensorStatus?.updateStatus("RGB Camera", 
                    state.sensorStatus["rgb"]?.let { sensorStatus ->
                        SensorStatusIndicator.SensorStatus(
                            name = "RGB Camera",
                            isActive = sensorStatus.isActive,
                            isHealthy = sensorStatus.isHealthy,
                            statusMessage = sensorStatus.statusMessage
                        )
                    } ?: SensorStatusIndicator.SensorStatus(
                        name = "RGB Camera",
                        isActive = state.isCameraConnected,
                        isHealthy = state.isCameraConnected,
                        statusMessage = if (state.isCameraConnected) "Connected" else "Disconnected"
                    )
                )
                
                thermalSensorStatus?.updateStatus("Thermal", 
                    state.sensorStatus["thermal"]?.let { sensorStatus ->
                        SensorStatusIndicator.SensorStatus(
                            name = "Thermal Camera",
                            isActive = sensorStatus.isActive,
                            isHealthy = sensorStatus.isHealthy,
                            statusMessage = sensorStatus.statusMessage
                        )
                    } ?: SensorStatusIndicator.SensorStatus(
                        name = "Thermal Camera",
                        isActive = state.isThermalConnected,
                        isHealthy = state.isThermalConnected,
                        statusMessage = if (state.isThermalConnected) "Connected" else "Disconnected"
                    )
                )
                
                gsrSensorStatus?.updateStatus("GSR", 
                    state.sensorStatus["gsr"]?.let { sensorStatus ->
                        SensorStatusIndicator.SensorStatus(
                            name = "GSR Sensor",
                            isActive = sensorStatus.isActive,
                            isHealthy = sensorStatus.isHealthy,
                            statusMessage = sensorStatus.statusMessage
                        )
                    } ?: SensorStatusIndicator.SensorStatus(
                        name = "GSR Sensor",
                        isActive = state.isShimmerConnected,
                        isHealthy = state.isShimmerConnected,
                        statusMessage = if (state.isShimmerConnected) "Connected" else "Disconnected"
                    )
                )
                
                pcSensorStatus?.updateStatus("PC Link", SensorStatusIndicator.SensorStatus(
                    name = "PC Link",
                    isActive = state.isPcConnected,
                    isHealthy = state.isPcConnected,
                    statusMessage = if (state.isPcConnected) "Connected" else "Disconnected"
                ))

                if (state.showErrorDialog && !state.errorMessage.isNullOrEmpty()) {
                    showErrorDialog(state.errorMessage)
                }

                if (state.thermalStatus.isSimulated && state.isThermalConnected) {
                    showThermalSimulationOverlay()
                }
            }
        }
    }

    /**
     * Format recording time in HH:MM:SS format
     */
    private fun formatRecordingTime(seconds: Long): String {
        val hours = TimeUnit.SECONDS.toHours(seconds)
        val minutes = TimeUnit.SECONDS.toMinutes(seconds) % 60
        val secs = seconds % 60
        return String.format("%02d:%02d:%02d", hours, minutes, secs)
    }

    /**
     * Convert SensorStatus to SensorStatusIndicator.SensorStatus
     */
    private fun SensorStatus.toSensorStatusIndicator(): SensorStatusIndicator.SensorStatus {
        return SensorStatusIndicator.SensorStatus(
            name = this.name,
            isActive = this.isActive,
            isHealthy = this.isHealthy,
            lastUpdate = this.lastUpdate,
            statusMessage = this.statusMessage,
        )
    }



    /**
     * Show thermal simulation overlay when thermal camera is in simulation mode
     */
    private fun showThermalSimulationOverlay() {
        rootLayout?.let { layout ->
            val snackbar = Snackbar.make(
                layout,
                "Thermal Camera Simulation (device not found)",
                Snackbar.LENGTH_LONG,
            )
            snackbar.show()
        }
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

        viewPager?.let { vp ->
            navigationController = NavigationController(this, vp)
        }

        tabLayout?.let { tabLayout ->
            viewPager?.let { viewPager ->
                TabLayoutMediator(tabLayout, viewPager) { tab, position ->
                    tab.text = MainPagerAdapter.TAB_TITLES[position]
                }.attach()

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
            vm.updateUiState { copy(statusText = "Checking permissions...") }
            btnStartRecording?.isEnabled = false

            if (permissionManager.areAllPermissionsGranted()) {
                startRecording()
            } else {
                vm.showToast("Checking permissions for all sensors")
                requestAllPermissions()
            }
        }

        btnStopRecording?.setOnClickListener {
            vm.updateUiState { copy(statusText = "Stopping recording...") }
            btnStopRecording?.isEnabled = false

            stopRecording()
        }
    }

    private fun startRecording() {
        lifecycleScope.launch {
            try {
                // Use ViewModel to start recording
                vm.startRecording()

                // Fallback to coordinator/controller if ViewModel doesn't handle it
                val coordinator = ensureMultiModalCoordinator()
                val sessionDir = File(applicationContext.filesDir, "sessions")
                if (!sessionDir.exists()) sessionDir.mkdirs()

                val startResult = coordinator.startRecording(sessionDir)

                if (startResult) {
                    showToast("Full multi-modal recording started")
                    vm.onRecordingCompleted("Full integration recording in progress")
                } else {
                    Log.w(TAG, "Coordinator failed, falling back to individual recorders")
                    ensureController().startSession()
                    showToast("Recording started (fallback mode)")
                    vm.onRecordingCompleted("Recording in progress (fallback)")
                }
            } catch (e: Exception) {
                vm.showError("Failed to start recording: ${e.message}")
                Log.e(TAG, "Recording start failed", e)
            }
        }
    }

    private fun stopRecording() {
        lifecycleScope.launch {
            try {
                // Use ViewModel to stop recording
                vm.stopRecording()

                // Fallback to coordinator/controller if ViewModel doesn't handle it
                val coordinator = multiModalCoordinator
                if (coordinator != null) {
                    val stopResult = coordinator.stopRecording()
                    if (stopResult) {
                        showToast("Full multi-modal recording stopped")
                        vm.onRecordingCompleted("Files saved to sessions directory")
                        return@launch
                    } else {
                        Log.w(TAG, "Coordinator stop failed, trying individual controller")
                    }
                }

                controller?.stopSession()
                showToast("Recording stopped")
                vm.onRecordingCompleted("Files saved successfully")
            } catch (e: Exception) {
                vm.showError("Failed to stop recording: ${e.message}")
                Log.e(TAG, "Recording stop failed", e)
                // Re-enable buttons on error
                btnStartRecording?.isEnabled = true
                btnStopRecording?.isEnabled = true
            }
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
        lifecycleScope.launch {
            multiModalCoordinator?.stopRecording()
            multiModalCoordinator = null
        }

        permissionManager.cleanup()
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
        return c
    }

    /**
     * Ensure MultiModalSensorCoordinator is initialized - Full Integration
     */
    private suspend fun ensureMultiModalCoordinator(): MultiModalSensorCoordinator {
        val existing = multiModalCoordinator
        if (existing != null) return existing

        val coordinator = MultiModalSensorCoordinator(applicationContext, this)

        val initResult = coordinator.initializeSystem()
        if (initResult) {
            Log.i("MainActivity", "MultiModalSensorCoordinator initialized successfully - Full Integration active")
        } else {
            Log.w("MainActivity", "MultiModalSensorCoordinator initialization failed, falling back to individual recorders")
        }

        multiModalCoordinator = coordinator
        return coordinator
    }

    private fun showQuickStartGuide() {
        QuickStartDialog.show(this) {
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

            val syncDelay = (actualFlashTime - tsNs) / 1_000_000.0
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

    /**
     * Navigate to thermal camera preview - Enhanced integration method
     */
    fun navigateToThermalPreview() {
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
            com.yourcompany.sensorspoke.sensors.thermal.TC001InitUtil
                .initLog()

            com.yourcompany.sensorspoke.sensors.thermal.TC001InitUtil
                .initReceiver(this)

            com.yourcompany.sensorspoke.sensors.thermal.TC001InitUtil
                .initTC001DeviceManager(this)

            Log.i("MainActivity", "TC001 thermal camera system initialized successfully")
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to initialize TC001 system", e)
        }
    }

    private fun requestAllPermissions() {
        vm.updateUiState { copy(statusText = "Requesting permissions...") }

        permissionManager.requestAllPermissions { allGranted ->
            if (allGranted) {
                vm.updateUiState {
                    copy(
                        statusText = "All permissions granted - Ready to record",
                        startButtonEnabled = true,
                    )
                }
                showToast("All sensor permissions granted. Ready to start recording!")
            } else {
                vm.updateUiState {
                    copy(statusText = "Some permissions denied - Limited functionality")
                }
                vm.showError("Some permissions were denied. Recording may not include all sensors.")
                Log.d(TAG, "Permission status: ${permissionManager.getPermissionStatus()}")
            }
        }
    }

    private fun showUserFriendlyError(message: String, context: String = "") {
        val fullMessage = if (context.isNotEmpty()) {
            "Error in $context: $message"
        } else {
            message
        }
        showToast(fullMessage)
        vm.showError(fullMessage)
    }

    private fun isRunningUnderTest(): Boolean {
        return try {
            Class.forName("androidx.test.espresso.Espresso")
            true
        } catch (e: ClassNotFoundException) {
            false
        }
    }
}
