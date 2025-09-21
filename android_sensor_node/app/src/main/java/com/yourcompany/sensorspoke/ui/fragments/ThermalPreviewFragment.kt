package com.yourcompany.sensorspoke.ui.fragments

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.hardware.usb.UsbManager
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.yourcompany.sensorspoke.R
import com.yourcompany.sensorspoke.sensors.thermal.TopdonThermalPalette
import com.yourcompany.sensorspoke.sensors.thermal.TC001IntegrationManager
import com.yourcompany.sensorspoke.sensors.thermal.TC001IntegrationState
import com.yourcompany.sensorspoke.sensors.thermal.TC001TemperatureData
import com.yourcompany.sensorspoke.ui.components.TC001SystemStatusView
import com.yourcompany.sensorspoke.ui.components.ThermalControlsView
import com.yourcompany.sensorspoke.ui.navigation.ThermalNavigationState
import kotlinx.coroutines.launch
import kotlin.math.sqrt

/**
 * Enhanced ThermalPreviewFragment with IRCamera-style architecture
 *
 * Provides comprehensive thermal camera interface with:
 * - Real-time thermal preview with enhanced processing
 * - Advanced thermal controls (palette, emissivity, ranges)
 * - Navigation state handling for different thermal modes
 * - Professional temperature measurement display
 * - TC001 device integration and management
 */
class ThermalPreviewFragment : Fragment() {
    private var thermalImageView: ImageView? = null
    private var statusText: TextView? = null
    private var temperatureRangeText: TextView? = null
    private var frameCountText: TextView? = null
    private var controlsScrollView: ScrollView? = null
    private var thermalControlsView: ThermalControlsView? = null

    private var tc001SystemStatusView: TC001SystemStatusView? = null

    private var tc001IntegrationManager: TC001IntegrationManager? = null

    private var frameCount = 0
    private var currentNavigationState = ThermalNavigationState.PREVIEW
    private var currentPalette = TopdonThermalPalette.IRON
    private var isControlsVisible = false

    /**
     * Setup TC001 system status view for comprehensive monitoring
     */
    private fun setupTC001SystemStatusView(view: View) {
        tc001SystemStatusView = TC001SystemStatusView(requireContext())

        controlsScrollView?.let { scrollView ->
            (scrollView.getChildAt(0) as? LinearLayout)?.addView(tc001SystemStatusView, 0)
        } ?: run {
            (view as? ViewGroup)?.addView(tc001SystemStatusView)
        }

        tc001SystemStatusView?.updateSystemStatus(
            TC001IntegrationState.UNINITIALIZED,
            "System Initializing",
        )
        tc001SystemStatusView?.updateConnectionStatus(false)
        tc001SystemStatusView?.updateDataProcessingStatus(false)
        tc001SystemStatusView?.updateTemperatureStatus(null)
    }

    /**
     * Initialize TC001 integration components for enhanced thermal functionality
     */
    private fun initializeTC001Integration() {
        requireContext().let { context ->
            tc001IntegrationManager = TC001IntegrationManager(context)

            lifecycleScope.launch {
                val initResult = tc001IntegrationManager!!.initializeSystem()
                if (initResult) {
                    setupTC001Observers()
                    statusText?.text = "TC001 System Initialized"

                    val startResult = tc001IntegrationManager!!.startSystem()
                    if (startResult) {
                        statusText?.text = "TC001 System Running"
                    }
                } else {
                    statusText?.text = "TC001 System Error"
                }
            }
        }
    }

    /**
     * Setup TC001 system observers
     */
    private fun setupTC001Observers() {
        tc001IntegrationManager?.let { manager ->
            manager.integrationState.observe(viewLifecycleOwner) { state ->
                updateTC001IntegrationStatus(state)
            }

            manager.systemStatus.observe(viewLifecycleOwner) { status ->
                statusText?.text = status
            }

            manager.getDataManager()?.let { dataManager ->
                dataManager.thermalBitmap.observe(viewLifecycleOwner) { bitmap ->
                    bitmap?.let {
                        thermalImageView?.setImageBitmap(it)
                    }
                }

                dataManager.temperatureData.observe(viewLifecycleOwner) { tempData ->
                    tempData?.let {
                        updateTemperatureDisplay(it)
                        tc001SystemStatusView?.updateTemperatureStatus(it.centerTemperature)
                    }
                }
            }
        }
    }

    /**
     * Update TC001 integration status
     */
    private fun updateTC001IntegrationStatus(state: TC001IntegrationState) {
        val statusMessage =
            when (state) {
                TC001IntegrationState.UNINITIALIZED -> "TC001 Uninitialized"
                TC001IntegrationState.INITIALIZING -> "TC001 Initializing..."
                TC001IntegrationState.INITIALIZED -> "TC001 Initialized"
                TC001IntegrationState.STARTING -> "TC001 Starting..."
                TC001IntegrationState.RUNNING -> "TC001 Running"
                TC001IntegrationState.STOPPING -> "TC001 Stopping..."
                TC001IntegrationState.CONNECTION_FAILED -> "TC001 Connection Failed"
                TC001IntegrationState.ERROR -> "TC001 Error"
            }

        statusText?.text = statusMessage

        val isReady = tc001IntegrationManager?.isSystemReady() ?: false
        thermalControlsView?.updateDeviceStatus(statusMessage, isReady)
    }

    /**
     * Update temperature display from TC001 data
     */
    private fun updateTemperatureDisplay(tempData: TC001TemperatureData) {
        thermalControlsView?.updateCurrentTemperature(tempData.centerTemperature)
        temperatureRangeText?.text =
            "Range: ${String.format("%.1f", tempData.minTemperature)}°C - ${String.format("%.1f", tempData.maxTemperature)}°C"
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View? = inflater.inflate(R.layout.fragment_thermal_preview, container, false)

    override fun onViewCreated(
        view: View,
        savedInstanceState: Bundle?,
    ) {
        super.onViewCreated(view, savedInstanceState)

        thermalImageView = view.findViewById(R.id.thermalImageView)
        statusText = view.findViewById(R.id.statusText)
        temperatureRangeText = view.findViewById(R.id.temperatureRangeText)
        frameCountText = view.findViewById(R.id.frameCountText)
        controlsScrollView = view.findViewById(R.id.controlsScrollView)

        setupThermalControls(view)

        setupTC001SystemStatusView(view)

        initializeTC001Integration()

        setupGestureHandling()

        updateInitialStatus()

        createEnhancedThermalImage()
    }

    /**
     * Initialize UI status indicators with proper system state
     */
    private fun updateInitialStatus() {
        statusText?.text = "Thermal camera ready"
        temperatureRangeText?.text = "Temperature Range: -20°C to 150°C"
        frameCountText?.text = "Frames: 0"

        val usbManager = requireContext().getSystemService(Context.USB_SERVICE) as UsbManager
        val hasTC001 = usbManager.deviceList.values.any { device ->
            device.productName?.contains("TC001", ignoreCase = true) == true
        }

        if (hasTC001) {
            statusText?.text = "TC001 thermal camera detected"
        } else {
            statusText?.text = "Simulation mode - No TC001 detected"
        }
    }

    override fun onResume() {
        super.onResume()
        startEnhancedThermalSimulation()
    }

    /**
     * Handle navigation state changes from NavigationController
     */
    fun handleNavigationState(state: ThermalNavigationState) {
        currentNavigationState = state

        when (state) {
            ThermalNavigationState.PREVIEW -> {
                showThermalPreview()
                hideControls()
            }
            ThermalNavigationState.SETTINGS -> {
                showThermalPreview()
                showControls()
            }
            ThermalNavigationState.CALIBRATION -> {
                showThermalPreview()
                showControls()
                statusText?.text = "Calibration mode active"
            }
            ThermalNavigationState.ANALYSIS -> {
                showThermalPreview()
                showControls()
                statusText?.text = "Analysis mode active"
            }
        }
    }

    private fun setupThermalControls(view: View) {
        thermalControlsView = ThermalControlsView(requireContext())

        thermalControlsView?.apply {
            onPaletteChanged = { palette ->
                currentPalette = palette
                updateThermalPalette(palette)
                tc001IntegrationManager?.getDataManager()?.updatePalette(palette)
            }

            onEmissivityChanged = { emissivity ->
                updateEmissivity(emissivity)
                tc001IntegrationManager?.getDataManager()?.updateEmissivity(emissivity)
            }

            onTemperatureRangeChanged = { minTemp, maxTemp ->
                updateTemperatureRange(minTemp, maxTemp)
                tc001IntegrationManager?.getDataManager()?.updateTemperatureRange(minTemp, maxTemp)
            }

            onAutoGainToggled = { enabled ->
                updateAutoGain(enabled)
                tc001IntegrationManager?.getUIController()?.onAutoGainToggled(enabled)
            }

            onTemperatureCompensationToggled = { enabled ->
                updateTemperatureCompensation(enabled)
                tc001IntegrationManager?.getUIController()?.onTemperatureCompensationToggled(enabled)
            }
        }

        controlsScrollView?.addView(thermalControlsView)

        controlsScrollView?.visibility = View.GONE
    }

    private fun setupGestureHandling() {
        thermalImageView?.setOnLongClickListener {
            toggleControls()
            true
        }

        thermalImageView?.setOnClickListener {
            cycleThermalPalette()
        }
    }

    private fun showThermalPreview() {
        thermalImageView?.visibility = View.VISIBLE
    }

    private fun showControls() {
        controlsScrollView?.visibility = View.VISIBLE
        isControlsVisible = true
        statusText?.text = "Thermal settings available"
    }

    private fun hideControls() {
        controlsScrollView?.visibility = View.GONE
        isControlsVisible = false
        statusText?.text = "Thermal preview active"
    }

    private fun toggleControls() {
        if (isControlsVisible) {
            hideControls()
        } else {
            showControls()
        }
    }

    private fun cycleThermalPalette() {
        currentPalette =
            when (currentPalette) {
                TopdonThermalPalette.IRON -> TopdonThermalPalette.RAINBOW
                TopdonThermalPalette.RAINBOW -> TopdonThermalPalette.GRAYSCALE
                TopdonThermalPalette.GRAYSCALE -> TopdonThermalPalette.HOT
                TopdonThermalPalette.HOT -> TopdonThermalPalette.COOL
                TopdonThermalPalette.COOL -> TopdonThermalPalette.WHITE_HOT
                TopdonThermalPalette.WHITE_HOT -> TopdonThermalPalette.BLACK_HOT
                TopdonThermalPalette.BLACK_HOT -> TopdonThermalPalette.RED
                TopdonThermalPalette.RED -> TopdonThermalPalette.GREEN
                TopdonThermalPalette.GREEN -> TopdonThermalPalette.BLUE
                TopdonThermalPalette.BLUE -> TopdonThermalPalette.IRON
            }

        updateThermalPalette(currentPalette)
        createEnhancedThermalImage()
    }

    /**
     * Creates enhanced thermal image with better processing inspired by IRCamera
     */
    private fun createEnhancedThermalImage() {
        lifecycleScope.launch {
            val bitmap = generateAdvancedThermalBitmap(currentPalette)
            thermalImageView?.setImageBitmap(bitmap)
        }
    }

    private fun generateAdvancedThermalBitmap(palette: TopdonThermalPalette): Bitmap {
        val width = 256
        val height = 192
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val centerX = width / 2f
                val centerY = height / 2f
                val distance = kotlin.math.sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY))
                val maxDistance = kotlin.math.sqrt(centerX * centerX + centerY * centerY)
                val normalized = 1f - (distance / maxDistance).coerceIn(0f, 1f)

                val noise = (kotlin.math.sin(x * 0.1) * kotlin.math.cos(y * 0.1) * 0.1).toFloat()
                val adjustedNormalized = (normalized + noise).coerceIn(0f, 1f)

                val color = mapTemperatureToColor(adjustedNormalized, palette)
                bitmap.setPixel(x, y, color)
            }
        }

        return bitmap
    }

    private fun mapTemperatureToColor(
        normalized: Float,
        palette: TopdonThermalPalette,
    ): Int =
        when (palette) {
            TopdonThermalPalette.IRON -> {
                val red = (normalized * 255).toInt()
                val green = if (normalized > 0.4f) ((normalized - 0.4f) * 1.67f * 255).toInt().coerceAtMost(255) else 0
                val blue = if (normalized > 0.8f) ((normalized - 0.8f) * 5f * 255).toInt().coerceAtMost(255) else 0
                (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
            }
            TopdonThermalPalette.RAINBOW -> {
                val hue = normalized * 240f
                val saturation = 0.9f
                val value = 0.9f
                Color.HSVToColor(floatArrayOf(hue, saturation, value))
            }
            TopdonThermalPalette.GRAYSCALE -> {
                val gray = (normalized * 255 * 0.8f + 32).toInt().coerceIn(0, 255)
                (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
            }
            TopdonThermalPalette.HOT -> {
                val red = (255 * (normalized + 0.5f).coerceAtMost(1f)).toInt()
                val green = (255 * (normalized * 2f - 1f).coerceIn(0f, 1f)).toInt()
                val blue = (255 * (normalized - 0.75f).coerceAtLeast(0f) * 4f).toInt()
                (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
            }
            TopdonThermalPalette.COOL -> {
                val red = (255 * (1f - normalized)).toInt()
                val green = (255 * (1f - normalized * 0.5f)).toInt()
                val blue = 255
                (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
            }
            TopdonThermalPalette.WHITE_HOT -> {
                val gray = (normalized * 255).toInt()
                (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
            }
            TopdonThermalPalette.BLACK_HOT -> {
                val gray = (255 * (1f - normalized)).toInt()
                (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
            }
            TopdonThermalPalette.RED -> {
                val red = (normalized * 255).toInt()
                (0xFF shl 24) or (red shl 16)
            }
            TopdonThermalPalette.GREEN -> {
                val green = (normalized * 255).toInt()
                (0xFF shl 24) or (green shl 8)
            }
            TopdonThermalPalette.BLUE -> {
                val blue = (normalized * 255).toInt()
                (0xFF shl 24) or blue
            }
        }

    /**
     * Enhanced thermal data simulation with more realistic patterns
     */
    private fun startEnhancedThermalSimulation() {
        statusText?.text = "Enhanced thermal simulation active"
        temperatureRangeText?.text = "Temperature Range: 18.0°C - 42.3°C"

        val handler = android.os.Handler(android.os.Looper.getMainLooper())
        val updateRunnable =
            object : Runnable {
                override fun run() {
                    frameCount++
                    frameCountText?.text = "Frames: $frameCount"

                    val currentTemp = 25.0f + kotlin.math.sin(frameCount * 0.1f).toFloat() * 8.0f
                    thermalControlsView?.updateCurrentTemperature(currentTemp)

                    val isConnected = frameCount % 100 < 80
                    val status = if (isConnected) "TC001 Connected" else "TC001 Disconnected"
                    thermalControlsView?.updateDeviceStatus(status, isConnected)

                    if (frameCount % 30 == 0) {
                        createEnhancedThermalImage()
                    }

                    if (isAdded && isVisible) {
                        handler.postDelayed(this, 100)
                    }
                }
            }
        handler.post(updateRunnable)
    }

    private fun updateThermalPalette(palette: TopdonThermalPalette) {
        statusText?.text = "Palette changed to: ${palette.name}"
    }

    private fun updateEmissivity(emissivity: Float) {
        // In production: Update TC001 emissivity setting
        statusText?.text = "Emissivity set to: ${String.format("%.2f", emissivity)}"
    }

    private fun updateTemperatureRange(
        minTemp: Float,
        maxTemp: Float,
    ) {
        temperatureRangeText?.text = "Temperature Range: ${String.format("%.1f", minTemp)}°C - ${String.format("%.1f", maxTemp)}°C"
    }

    private fun updateAutoGain(enabled: Boolean) {
        // In production: Enable/disable TC001 auto-gain control
        statusText?.text = "Auto Gain Control: ${if (enabled) "Enabled" else "Disabled"}"
    }

    private fun updateTemperatureCompensation(enabled: Boolean) {
        // In production: Enable/disable TC001 temperature compensation
        statusText?.text = "Temperature Compensation: ${if (enabled) "Enabled" else "Disabled"}"
    }

    companion object {
        fun newInstance(): ThermalPreviewFragment = ThermalPreviewFragment()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        tc001IntegrationManager?.cleanup()
    }
}
