package com.yourcompany.sensorspoke.sensors

import java.io.File

/**
 * Common interface for all sensor recorders.
 * All I/O should be done using Kotlin coroutines.
 */
interface SensorRecorder {
    /**
     * Starts recording into the provided session directory (sensor-specific subfolder).
     */
    suspend fun start(sessionDir: File)

    /**
     * Stops recording and releases resources.
     */
    suspend fun stop()
}
