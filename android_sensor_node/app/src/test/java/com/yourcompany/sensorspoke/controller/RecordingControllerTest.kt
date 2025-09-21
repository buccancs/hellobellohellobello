package com.yourcompany.sensorspoke.controller

import com.yourcompany.sensorspoke.sensors.SensorRecorder
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files

class RecordingControllerTest {
    private class FakeRecorder : SensorRecorder {
        var startedDir: File? = null
        var stopCount: Int = 0

        override suspend fun start(sessionDir: File) {
            startedDir = sessionDir
            if (!sessionDir.exists()) sessionDir.mkdirs()
        }

        override suspend fun stop() {
            stopCount++
        }
    }

    @Test
    fun startStopSession_createsDirs_and_transitionsState() =
        runBlocking {
            val tmpRoot = Files.createTempDirectory("rc_test_root").toFile()
            val sessionsRoot = File(tmpRoot, "sessions").apply { mkdirs() }
            val controller = RecordingController(context = null, sessionsRootOverride = sessionsRoot)
            val fake = FakeRecorder()
            controller.register("fake", fake)

            val sessionId = "test_session"
            controller.startSession(sessionId)

            assertEquals(RecordingController.State.RECORDING, controller.state.value)
            assertEquals(sessionId, controller.currentSessionId.value)

            val sessionDir = File(sessionsRoot, sessionId)
            assertTrue("Session dir should exist", sessionDir.exists())
            val subDir = File(sessionDir, "fake")
            assertTrue("Recorder subdir should exist", subDir.exists())
            assertEquals(subDir.absolutePath, fake.startedDir?.absolutePath)

            controller.stopSession()
            assertEquals(RecordingController.State.IDLE, controller.state.value)
            assertEquals(null, controller.currentSessionId.value)
            assertTrue("stop() should be called at least once", fake.stopCount > 0)
        }
}
