package com.yourcompany.sensorspoke.sensors.thermal

import android.content.Context
import com.google.common.truth.Truth.assertThat
import io.mockk.mockk
import kotlinx.coroutines.runBlocking
import org.junit.Test
import java.io.File
import kotlin.io.path.createTempDirectory

class ThermalCameraRecorderTest {
    @Test
    fun start_createsCsvWithHeader_and_stop_closes() =
        runBlocking {
            val tmp = createTempDirectory("thermal_test_").toFile()
            try {
                val mockContext = mockk<Context>(relaxed = true)
                val recorder = ThermalCameraRecorder(mockContext)
                recorder.start(tmp)

                val csv = File(tmp, "thermal_data.csv")
                assertThat(csv.exists()).isTrue()

                val firstLine = csv.bufferedReader().use { it.readLine() }
                assertThat(firstLine).isNotNull()

                assertThat(firstLine!!.startsWith("timestamp_ns,timestamp_ms,frame_number")).isTrue()
                assertThat(firstLine.contains("temperature_celsius")).isTrue()
                assertThat(firstLine.contains("filename")).isTrue()

                recorder.stop()
            } finally {
                tmp.deleteRecursively()
            }
        }
}
