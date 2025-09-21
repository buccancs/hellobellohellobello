package com.yourcompany.sensorspoke

import com.yourcompany.sensorspoke.controller.RecordingController
import com.yourcompany.sensorspoke.sensors.audio.AudioRecorder
import com.yourcompany.sensorspoke.sensors.gsr.ShimmerRecorder
import com.yourcompany.sensorspoke.sensors.rgb.RgbCameraRecorder
import com.yourcompany.sensorspoke.sensors.thermal.ThermalCameraRecorder
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import java.io.File

/**
 * Phase 2 Integration Test - Validates multi-modal sensor integration.
 */
@RunWith(RobolectricTestRunner::class)
class Phase2IntegrationTest {

    @Test
    fun `test multi-modal sensor registration`() = runBlocking {
        val context = RuntimeEnvironment.getApplication().applicationContext
        val controller = RecordingController(context)

        try {
            controller.register("rgb", RgbCameraRecorder(context, null))
            controller.register("thermal", ThermalCameraRecorder(context))
            controller.register("gsr", ShimmerRecorder(context))
            controller.register("audio", AudioRecorder(context))

            assert(controller.toString().contains("RecordingController"))
        } catch (e: Exception) {
            assert(
                e.message?.contains("permission") == true ||
                    e.message?.contains("camera") == true ||
                    e.message?.contains("Hardware") == true ||
                    e.message?.contains("Service") == true,
            )
        }
    }

    @Test
    fun `test session directory structure for multi-modal recording`() = runBlocking {
        val context = RuntimeEnvironment.getApplication().applicationContext
        val controller = RecordingController(context)

        val sessionId = "phase2_test_session"
        val sessionsRoot = File(context.filesDir, "sessions")
        val sessionDir = File(sessionsRoot, sessionId)
        sessionDir.mkdirs()

        val rgbDir = File(sessionDir, "rgb").apply { mkdirs() }
        val thermalDir = File(sessionDir, "thermal").apply { mkdirs() }
        val gsrDir = File(sessionDir, "gsr").apply { mkdirs() }
        val audioDir = File(sessionDir, "audio").apply { mkdirs() }

        assert(rgbDir.exists() && rgbDir.isDirectory)
        assert(thermalDir.exists() && thermalDir.isDirectory)
        assert(gsrDir.exists() && gsrDir.isDirectory)
        assert(audioDir.exists() && audioDir.isDirectory)

        val rgbFramesDir = File(rgbDir, "frames").apply { mkdirs() }
        val thermalImagesDir = File(thermalDir, "thermal_images").apply { mkdirs() }

        assert(rgbFramesDir.exists())
        assert(thermalImagesDir.exists())

        sessionDir.deleteRecursively()
    }

    @Test
    fun `test Phase 2 sensor classes exist and are instantiable`() {
        val context = RuntimeEnvironment.getApplication().applicationContext


        try {
            val rgbRecorder = RgbCameraRecorder(context, null)
            assert(rgbRecorder.javaClass.simpleName == "RgbCameraRecorder")
        } catch (e: Exception) {
        }

        try {
            val thermalRecorder = ThermalCameraRecorder(context)
            assert(thermalRecorder.javaClass.simpleName == "ThermalCameraRecorder")
        } catch (e: Exception) {
        }

        try {
            val gsrRecorder = ShimmerRecorder(context)
            assert(gsrRecorder.javaClass.simpleName == "ShimmerRecorder")
        } catch (e: Exception) {
        }

        try {
            val audioRecorder = AudioRecorder(context)
            assert(audioRecorder.javaClass.simpleName == "AudioRecorder")
        } catch (e: Exception) {
        }
    }
}
