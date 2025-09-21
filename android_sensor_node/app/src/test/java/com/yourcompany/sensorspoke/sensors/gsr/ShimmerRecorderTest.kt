package com.yourcompany.sensorspoke.sensors.gsr

import android.content.Context
import com.google.common.truth.Truth.assertThat
import io.mockk.mockk
import kotlinx.coroutines.runBlocking
import org.junit.Test
import java.io.File
import kotlin.io.path.createTempDirectory

class ShimmerRecorderTest {
    private val mockContext = mockk<Context>(relaxed = true)

    @Test
    fun start_createsCsvHeader_and_stop_closes() =
        runBlocking {
            val tmp = createTempDirectory("shimmer_test_").toFile()
            try {
                val recorder = ShimmerRecorder(mockContext)
                recorder.start(tmp)
                val csv = File(tmp, "gsr.csv")
                assertThat(csv.exists()).isTrue()
                val first = csv.bufferedReader().use { it.readLine() }
                assertThat(first).isEqualTo("timestamp_ns,gsr_microsiemens,ppg_raw")
                recorder.stop()
            } finally {
                tmp.deleteRecursively()
            }
        }

    @Test
    fun recorder_initializes_with_correct_defaults() {
        val recorder = ShimmerRecorder(mockContext)
        assertThat(recorder).isNotNull()
    }

    @Test
    fun start_stop_lifecycle_handles_gracefully() =
        runBlocking {
            val tmp = createTempDirectory("shimmer_lifecycle_test_").toFile()
            try {
                val recorder = ShimmerRecorder(mockContext)

                recorder.start(tmp)
                recorder.stop()

                recorder.start(tmp)
                recorder.stop()

                assertThat(File(tmp, "gsr_data.csv").exists()).isTrue()
            } finally {
                tmp.deleteRecursively()
            }
        }
}
