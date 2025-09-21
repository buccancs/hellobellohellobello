package com.yourcompany.sensorspoke.ui.models

/**
 * UI State data class for MainActivity holding all sensor states and UI feedback information.
 * Used by MainViewModel to provide reactive state updates to the UI.
 */
data class MainUiState(
    val isRecording: Boolean = false,
    val isInitialized: Boolean = false,
    val currentSessionId: String? = null,
    val recordingDurationSeconds: Long = 0,

    val isCameraConnected: Boolean = false,
    val isThermalConnected: Boolean = false,
    val isShimmerConnected: Boolean = false,
    val isPcConnected: Boolean = false,

    val cameraStatus: SensorStatus = SensorStatus("RGB Camera", false, false, "Offline"),
    val thermalStatus: SensorStatus = SensorStatus("Thermal Camera", false, false, "Offline"),
    val shimmerStatus: SensorStatus = SensorStatus("GSR Sensor", false, false, "Disconnected"),
    val pcStatus: SensorStatus = SensorStatus("PC Link", false, false, "Not Connected"),

    val statusText: String = "Initializing...",
    val errorMessage: String? = null,
    val showErrorDialog: Boolean = false,
    val showProgress: Boolean = false,
    val progressMessage: String = "",

    val startButtonEnabled: Boolean = false,
    val stopButtonEnabled: Boolean = false,

    val rgbPreviewAvailable: Boolean = false,
    val thermalPreviewAvailable: Boolean = false,
    val thermalIsSimulated: Boolean = false,
)

/**
 * Individual sensor status information
 */
data class SensorStatus(
    val name: String,
    val isActive: Boolean,
    val isHealthy: Boolean,
    val statusMessage: String,
    val lastUpdate: Long = System.currentTimeMillis(),
)
