package com.yourcompany.sensorspoke.ui

import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import com.google.common.truth.Truth.assertThat
import org.junit.Ignore
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner

@Ignore("UI smoke test placeholder; Robolectric availability varies in CI; will be enabled with proper shadows/instrumentation.")
@RunWith(RobolectricTestRunner::class)
class MainActivityTest {
    @Test
    fun smoke_launch_hasStartAndStopButtons() {
        val controller = Robolectric.buildActivity(MainActivity::class.java).setup()
        val activity = controller.get()

        val content = activity.findViewById<ViewGroup>(android.R.id.content)
        assertThat(content.childCount).isAtLeast(1)
        val root = content.getChildAt(0)
        assertThat(root).isInstanceOf(LinearLayout::class.java)
        val rootLayout = root as LinearLayout
        assertThat(rootLayout.childCount).isAtLeast(2)

        val startBtn = rootLayout.getChildAt(0)
        val stopBtn = rootLayout.getChildAt(1)
        assertThat(startBtn).isInstanceOf(Button::class.java)
        assertThat(stopBtn).isInstanceOf(Button::class.java)
        assertThat((startBtn as Button).text.toString()).contains("Start Recording")
        assertThat((stopBtn as Button).text.toString()).contains("Stop Recording")
    }
}
