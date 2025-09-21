package com.yourcompany.sensorspoke.ui.components

import android.content.Context
import android.util.AttributeSet
import android.view.LayoutInflater
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.yourcompany.sensorspoke.R

/**
 * Custom UI component for displaying real-time sensor status with visual indicators.
 * Shows colored status dots and descriptive text for each sensor type.
 */
class SensorStatusIndicator @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
) : LinearLayout(context, attrs, defStyleAttr) {

    private val statusDot: ImageView
    private val sensorLabel: TextView
    private val statusText: TextView

    init {
        orientation = HORIZONTAL
        LayoutInflater.from(context).inflate(R.layout.component_sensor_status, this, true)

        statusDot = findViewById(R.id.statusDot)
        sensorLabel = findViewById(R.id.sensorLabel)
        statusText = findViewById(R.id.statusText)
    }

    /**
     * Update the sensor status with visual indicators
     */
    fun updateStatus(sensorName: String, status: SensorStatus) {
        sensorLabel.text = sensorName
        statusText.text = status.statusMessage

        val dotColor = when {
            status.isActive && status.isHealthy -> ContextCompat.getColor(context, android.R.color.holo_green_light)
            status.isActive && !status.isHealthy -> ContextCompat.getColor(context, android.R.color.holo_orange_light)
            !status.isActive -> ContextCompat.getColor(context, android.R.color.holo_red_light)
            else -> ContextCompat.getColor(context, android.R.color.darker_gray)
        }

        statusDot.setColorFilter(dotColor)

        val textColor = when {
            status.isActive && status.isHealthy -> ContextCompat.getColor(context, android.R.color.black)
            status.isActive && !status.isHealthy -> ContextCompat.getColor(context, android.R.color.holo_orange_dark)
            !status.isActive -> ContextCompat.getColor(context, android.R.color.holo_red_dark)
            else -> ContextCompat.getColor(context, android.R.color.darker_gray)
        }

        statusText.setTextColor(textColor)
    }

    /**
     * Sensor status data class matching MainViewModel structure
     */
    data class SensorStatus(
        val name: String,
        val isActive: Boolean,
        val isHealthy: Boolean,
        val lastUpdate: Long = System.currentTimeMillis(),
        val statusMessage: String = "",
    )
}
