package com.yourcompany.sensorspoke.ui.components

import android.content.Context
import android.graphics.Color
import android.util.AttributeSet
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.TextView
import androidx.cardview.widget.CardView
import com.yourcompany.sensorspoke.sensors.thermal.TC001IntegrationState

/**
 * TC001SystemStatusView - Comprehensive thermal system status display
 *
 * Provides real-time status monitoring for the complete TC001 thermal integration:
 * - Integration manager status
 * - Device connection status
 * - Data processing status
 * - Temperature measurement status
 * - System health indicators
 */
class TC001SystemStatusView
@JvmOverloads
constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
) : CardView(context, attrs, defStyleAttr) {
    private val statusContainer: LinearLayout
    private val systemStatusText: TextView
    private val connectionStatusText: TextView
    private val dataProcessingStatusText: TextView
    private val temperatureStatusText: TextView
    private val systemHealthIndicator: TextView

    init {
        radius = 12f
        cardElevation = 4f
        setCardBackgroundColor(Color.parseColor("#1E1E1E"))

        statusContainer =
            LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(16, 12, 16, 12)
            }

        systemStatusText =
            TextView(context).apply {
                text = "TC001 System: Initializing"
                setTextColor(Color.WHITE)
                textSize = 14f
                gravity = Gravity.CENTER
            }
        statusContainer.addView(systemStatusText)

        connectionStatusText =
            TextView(context).apply {
                text = "Device: Not Connected"
                setTextColor(Color.parseColor("#FFA726"))
                textSize = 12f
                gravity = Gravity.CENTER
                setPadding(0, 4, 0, 0)
            }
        statusContainer.addView(connectionStatusText)

        dataProcessingStatusText =
            TextView(context).apply {
                text = "Processing: Stopped"
                setTextColor(Color.parseColor("#66BB6A"))
                textSize = 12f
                gravity = Gravity.CENTER
                setPadding(0, 4, 0, 0)
            }
        statusContainer.addView(dataProcessingStatusText)

        temperatureStatusText =
            TextView(context).apply {
                text = "Temperature: N/A"
                setTextColor(Color.parseColor("#42A5F5"))
                textSize = 12f
                gravity = Gravity.CENTER
                setPadding(0, 4, 0, 0)
            }
        statusContainer.addView(temperatureStatusText)

        systemHealthIndicator =
            TextView(context).apply {
                text = "● System Status"
                setTextColor(Color.parseColor("#757575"))
                textSize = 10f
                gravity = Gravity.CENTER
                setPadding(0, 8, 0, 0)
            }
        statusContainer.addView(systemHealthIndicator)

        addView(statusContainer)
    }

    /**
     * Update system integration status
     */
    fun updateSystemStatus(
        state: TC001IntegrationState,
        statusMessage: String,
    ) {
        systemStatusText.text = "TC001 System: $statusMessage"

        val (color, healthIndicator) =
            when (state) {
                TC001IntegrationState.UNINITIALIZED -> Color.parseColor("#757575") to "● Uninitialized"
                TC001IntegrationState.INITIALIZING -> Color.parseColor("#FFA726") to "● Initializing"
                TC001IntegrationState.INITIALIZED -> Color.parseColor("#66BB6A") to "● Ready"
                TC001IntegrationState.STARTING -> Color.parseColor("#42A5F5") to "● Starting"
                TC001IntegrationState.RUNNING -> Color.parseColor("#4CAF50") to "● Running"
                TC001IntegrationState.STOPPING -> Color.parseColor("#FF9800") to "● Stopping"
                TC001IntegrationState.CONNECTION_FAILED -> Color.parseColor("#F44336") to "● Connection Failed"
                TC001IntegrationState.ERROR -> Color.parseColor("#D32F2F") to "● Error"
            }

        systemStatusText.setTextColor(color)
        systemHealthIndicator.text = healthIndicator
        systemHealthIndicator.setTextColor(color)
    }

    /**
     * Update device connection status
     */
    fun updateConnectionStatus(
        isConnected: Boolean,
        deviceName: String = "TC001",
    ) {
        val status = if (isConnected) "Connected: $deviceName" else "Disconnected"
        val color = if (isConnected) Color.parseColor("#4CAF50") else Color.parseColor("#F44336")

        connectionStatusText.text = "Device: $status"
        connectionStatusText.setTextColor(color)
    }

    /**
     * Update data processing status
     */
    fun updateDataProcessingStatus(
        isProcessing: Boolean,
        frameRate: Int = 0,
    ) {
        val status = if (isProcessing) "Active ($frameRate FPS)" else "Stopped"
        val color = if (isProcessing) Color.parseColor("#66BB6A") else Color.parseColor("#757575")

        dataProcessingStatusText.text = "Processing: $status"
        dataProcessingStatusText.setTextColor(color)
    }

    /**
     * Update temperature measurement status
     */
    fun updateTemperatureStatus(
        temp: Float?,
        unit: String = "°C",
    ) {
        val status = if (temp != null) "${String.format("%.1f", temp)}$unit" else "N/A"
        val color = if (temp != null) Color.parseColor("#42A5F5") else Color.parseColor("#757575")

        temperatureStatusText.text = "Temperature: $status"
        temperatureStatusText.setTextColor(color)
    }

    /**
     * Update overall system health
     */
    fun updateSystemHealth(isHealthy: Boolean) {
        val color = if (isHealthy) Color.parseColor("#4CAF50") else Color.parseColor("#F44336")
        setCardBackgroundColor(if (isHealthy) Color.parseColor("#1E2E1E") else Color.parseColor("#2E1E1E"))
    }
}
