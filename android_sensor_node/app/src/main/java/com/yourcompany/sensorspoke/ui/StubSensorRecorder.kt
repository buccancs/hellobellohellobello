package com.yourcompany.sensorspoke.sensors

import android.util.Log
import kotlinx.coroutines.delay
import java.io.File

/**
 * Stub sensor recorder for Phase 1 testing.
 * Creates sample files and logs to verify session directory creation.
 */
class StubSensorRecorder : SensorRecorder {
    companion object {
        private const val TAG = "StubSensorRecorder"
    }

    private var isRecording = false
    private var sessionDir: File? = null

    override suspend fun start(sessionDir: File) {
        this.sessionDir = sessionDir

        delay(100)

        val testFile = File(sessionDir, "stub_sensor_test.log")
        testFile.writeText("Stub sensor recording started at ${System.currentTimeMillis()}\n")

        isRecording = true
        Log.i(TAG, "Stub sensor started recording in: ${sessionDir.absolutePath}")

        val dataFile = File(sessionDir, "stub_data.csv")
        dataFile.writeText("timestamp,value\n")
        dataFile.appendText("${System.currentTimeMillis()},42.0\n")
    }

    override suspend fun stop() {
        if (!isRecording) return

        delay(50)

        sessionDir?.let { dir ->
            val testFile = File(dir, "stub_sensor_test.log")
            testFile.appendText("Stub sensor recording stopped at ${System.currentTimeMillis()}\n")

            val dataFile = File(dir, "stub_data.csv")
            dataFile.appendText("${System.currentTimeMillis()},0.0\n")
        }

        isRecording = false
        Log.i(TAG, "Stub sensor stopped recording")
    }
}
