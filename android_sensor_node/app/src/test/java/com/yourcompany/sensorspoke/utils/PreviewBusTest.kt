package com.yourcompany.sensorspoke.utils

import com.google.common.truth.Truth.assertThat
import org.junit.Test
import java.util.concurrent.CopyOnWriteArrayList

class PreviewBusTest {
    @Test
    fun subscribe_emit_unsubscribe_works() {
        val received = CopyOnWriteArrayList<Pair<Int, Long>>()
        val listener: (ByteArray, Long) -> Unit = { bytes, ts ->
            received.add(Pair(bytes.size, ts))
        }
        PreviewBus.subscribe(listener)
        val payload1 = ByteArray(10) { 1 }
        val t1 = TimeManager.nowNanos()
        PreviewBus.emit(payload1, t1)
        assertThat(received.size).isAtLeast(1)
        val (len1, ts1) = received[0]
        assertThat(len1).isEqualTo(10)
        assertThat(ts1).isEqualTo(t1)

        PreviewBus.unsubscribe(listener)
        val before = received.size
        val payload2 = ByteArray(7) { 2 }
        val t2 = TimeManager.nowNanos()
        PreviewBus.emit(payload2, t2)
        assertThat(received.size).isEqualTo(before)
    }
}
