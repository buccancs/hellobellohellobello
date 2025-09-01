package com.yourcompany.sensorspoke.utils

import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicReference

/**
 * PreviewBus is a simple in-process event bus to distribute downsampled JPEG
 * preview frames from the camera recorder to interested subscribers (e.g.,
 * the RecordingService which forwards frames to the PC Hub).
 *
 * Enhanced with frame storage for testing and debugging purposes.
 */
object PreviewBus {
    private val listeners = CopyOnWriteArrayList<(bytes: ByteArray, timestampNs: Long) -> Unit>()

    // Enhanced: Store current frame for retrieval and testing
    private val currentFrameRef = AtomicReference<ByteArray?>()
    private val currentTimestampRef = AtomicReference<Long?>(null)

    fun subscribe(listener: (ByteArray, Long) -> Unit) {
        listeners.add(listener)
    }

    fun unsubscribe(listener: (ByteArray, Long) -> Unit) {
        listeners.remove(listener)
    }

    fun emit(
        bytes: ByteArray,
        timestampNs: Long,
    ) {
        // Store current frame for retrieval
        currentFrameRef.set(bytes)
        currentTimestampRef.set(timestampNs)

        for (l in listeners) {
            runCatching { l(bytes, timestampNs) }
        }
    }

    /**
     * Publish frame (alias for emit for test compatibility)
     */
    fun publishFrame(bytes: ByteArray) {
        emit(bytes, System.nanoTime())
    }

    /**
     * Get current frame for testing and debugging
     */
    fun getCurrentFrame(): ByteArray? = currentFrameRef.get()

    /**
     * Get current frame timestamp
     */
    fun getCurrentTimestamp(): Long? = currentTimestampRef.get()

    /**
     * Clear current frame data
     */
    fun clearCurrentFrame() {
        currentFrameRef.set(null)
        currentTimestampRef.set(null)
    }
}
