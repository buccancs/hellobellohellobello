package com.yourcompany.sensorspoke.ui.components

import android.content.Context
import android.util.AttributeSet
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.widget.SwitchCompat
import com.yourcompany.sensorspoke.R
import com.yourcompany.sensorspoke.sensors.thermal.TopdonThermalPalette
import java.util.Locale

/**
 * ThermalControlsView - Enhanced thermal camera controls inspired by IRCamera
 *
 * Provides comprehensive thermal camera control interface with IRCamera-style UI
 */
class ThermalControlsView
@JvmOverloads
constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
) : LinearLayout(context, attrs, defStyleAttr) {
    private val paletteSpinner: Spinner
    private val emissivitySlider: SeekBar
    private val emissivityValue: TextView
    private val minTempInput: EditText
    private val maxTempInput: EditText
    private val autoGainToggle: SwitchCompat
    private val temperatureCompensationToggle: SwitchCompat
    private val currentTempDisplay: TextView
    private val tempRangeDisplay: TextView
    private val deviceStatusIndicator: View
    private val deviceStatusText: TextView

    var onPaletteChanged: ((TopdonThermalPalette) -> Unit)? = null
    var onEmissivityChanged: ((Float) -> Unit)? = null
    var onTemperatureRangeChanged: ((Float, Float) -> Unit)? = null
    var onAutoGainToggled: ((Boolean) -> Unit)? = null
    var onTemperatureCompensationToggled: ((Boolean) -> Unit)? = null

    init {
        orientation = VERTICAL
        setPadding(16, 16, 16, 16)

        deviceStatusText =
            TextView(context).apply {
                text = "Device Status: Disconnected"
                textSize = 14f
                setPadding(0, 0, 0, 8)
            }
        addView(deviceStatusText)

        deviceStatusIndicator =
            View(context).apply {
                layoutParams =
                    LayoutParams(50, 20).apply {
                        bottomMargin = 16
                    }
                setBackgroundColor(context.getColor(android.R.color.holo_red_dark))
            }
        addView(deviceStatusIndicator)

        addView(
            TextView(context).apply {
                text = "Thermal Palette:"
                textSize = 12f
                setPadding(0, 8, 0, 4)
            },
        )

        paletteSpinner =
            Spinner(context).apply {
                val palettes = arrayOf("Iron", "Rainbow", "Grayscale")
                val adapter = ArrayAdapter(context, android.R.layout.simple_spinner_item, palettes)
                adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
                this.adapter = adapter
            }
        addView(paletteSpinner)

        addView(
            TextView(context).apply {
                text = "Emissivity:"
                textSize = 12f
                setPadding(0, 16, 0, 4)
            },
        )

        emissivityValue =
            TextView(context).apply {
                text = "0.95"
                textSize = 12f
            }
        addView(emissivityValue)

        emissivitySlider =
            SeekBar(context).apply {
                max = 90
                progress = 85
            }
        addView(emissivitySlider)

        addView(
            TextView(context).apply {
                text = "Min Temperature (°C):"
                textSize = 12f
                setPadding(0, 16, 0, 4)
            },
        )

        minTempInput =
            EditText(context).apply {
                setText("-20")
                inputType = android.text.InputType.TYPE_CLASS_NUMBER or android.text.InputType.TYPE_NUMBER_FLAG_SIGNED
            }
        addView(minTempInput)

        addView(
            TextView(context).apply {
                text = "Max Temperature (°C):"
                textSize = 12f
                setPadding(0, 8, 0, 4)
            },
        )

        maxTempInput =
            EditText(context).apply {
                setText("150")
                inputType = android.text.InputType.TYPE_CLASS_NUMBER or android.text.InputType.TYPE_NUMBER_FLAG_SIGNED
            }
        addView(maxTempInput)

        tempRangeDisplay =
            TextView(context).apply {
                text = "-20.0°C - 150.0°C"
                textSize = 12f
                setPadding(0, 8, 0, 0)
            }
        addView(tempRangeDisplay)

        autoGainToggle =
            SwitchCompat(context).apply {
                text = "Auto Gain Control"
                isChecked = true
            }
        addView(autoGainToggle)

        temperatureCompensationToggle =
            SwitchCompat(context).apply {
                text = "Temperature Compensation"
                isChecked = true
            }
        addView(temperatureCompensationToggle)

        currentTempDisplay =
            TextView(context).apply {
                text = "Current: N/A"
                textSize = 16f
                setPadding(0, 16, 0, 0)
                setTypeface(null, android.graphics.Typeface.BOLD)
            }
        addView(currentTempDisplay)

        setupControls()
    }

    private fun setupControls() {
        paletteSpinner.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long,
                ) {
                    val palette =
                        when (position) {
                            0 -> TopdonThermalPalette.IRON
                            1 -> TopdonThermalPalette.RAINBOW
                            2 -> TopdonThermalPalette.GRAYSCALE
                            else -> TopdonThermalPalette.IRON
                        }
                    onPaletteChanged?.invoke(palette)
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }

        emissivitySlider.setOnSeekBarChangeListener(
            object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(
                    seekBar: SeekBar?,
                    progress: Int,
                    fromUser: Boolean,
                ) {
                    val emissivity = (progress + 10) / 100.0f
                    emissivityValue.text = String.format(java.util.Locale.ROOT, "%.2f", emissivity)
                    if (fromUser) {
                        onEmissivityChanged?.invoke(emissivity)
                    }
                }

                override fun onStartTrackingTouch(seekBar: SeekBar?) {}

                override fun onStopTrackingTouch(seekBar: SeekBar?) {}
            },
        )

        minTempInput.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) updateTemperatureRange()
        }
        maxTempInput.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) updateTemperatureRange()
        }

        autoGainToggle.setOnCheckedChangeListener { _, isChecked ->
            onAutoGainToggled?.invoke(isChecked)
        }
        temperatureCompensationToggle.setOnCheckedChangeListener { _, isChecked ->
            onTemperatureCompensationToggled?.invoke(isChecked)
        }
    }

    private fun updateTemperatureRange() {
        try {
            val minTemp = minTempInput.text.toString().toFloatOrNull() ?: -20f
            val maxTemp = maxTempInput.text.toString().toFloatOrNull() ?: 150f

            if (minTemp < maxTemp) {
                onTemperatureRangeChanged?.invoke(minTemp, maxTemp)
                tempRangeDisplay.text = "${String.format(java.util.Locale.ROOT, "%.1f", minTemp)}°C - ${String.format(java.util.Locale.ROOT, "%.1f", maxTemp)}°C"
            }
        } catch (e: Exception) {
        }
    }

    fun updateCurrentTemperature(temp: Float) {
        currentTempDisplay.text = "Current: ${String.format(java.util.Locale.ROOT, "%.1f", temp)}°C"
    }

    fun updateDeviceStatus(
        status: String,
        isConnected: Boolean,
    ) {
        deviceStatusText.text = "Device Status: $status"
        deviceStatusIndicator.setBackgroundColor(
            if (isConnected) {
                context.getColor(android.R.color.holo_green_dark)
            } else {
                context.getColor(android.R.color.holo_red_dark)
            },
        )

        // Enable/disable controls based on connection
        val alpha = if (isConnected) 1.0f else 0.5f
        paletteSpinner.alpha = alpha
        emissivitySlider.alpha = alpha
        minTempInput.alpha = alpha
        maxTempInput.alpha = alpha
        autoGainToggle.alpha = alpha
        temperatureCompensationToggle.alpha = alpha

        paletteSpinner.isEnabled = isConnected
        emissivitySlider.isEnabled = isConnected
        minTempInput.isEnabled = isConnected
        maxTempInput.isEnabled = isConnected
        autoGainToggle.isEnabled = isConnected
        temperatureCompensationToggle.isEnabled = isConnected
    }
}
