package com.yourcompany.sensorspoke.sensors.rgb

import android.content.Context
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.test.core.app.ApplicationProvider
import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.runBlocking
import org.junit.Ignore
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@Ignore("CameraX classes are not available in Robolectric unit tests; this recorder is validated via instrumentation/integration tests.")
@RunWith(RobolectricTestRunner::class)
class RgbCameraRecorderTest {
    private class TestOwner : LifecycleOwner {
        private val registry = LifecycleRegistry(this)

        init {
            registry.currentState = Lifecycle.State.CREATED
        }

        override val lifecycle: Lifecycle
            get() = registry
    }

    @Test
    fun stop_is_safe_without_start_and_idempotent() =
        runBlocking {
            val context: Context = ApplicationProvider.getApplicationContext()
            val owner = TestOwner()
            val recorder = RgbCameraRecorder(context, owner)
            recorder.stop()
            recorder.stop()
            assertThat(true).isTrue()
        }
}
