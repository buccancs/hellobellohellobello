package com.yourcompany.sensorspoke.network

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * HeartbeatManager: Manages sending heartbeats to PC Hub for fault tolerance (FR8).
 *
 * This class handles:
 * - Periodic heartbeat transmission
 * - Connection health monitoring
 * - Automatic reconnection logic
 * - Local recording continuation during disconnection
 */
class HeartbeatManager(
    private val context: Context,
    private val deviceId: String,
    private val networkClient: NetworkClient,
    private val heartbeatIntervalMs: Long = 3000L,
    private val maxReconnectAttempts: Int = 10,
    private val reconnectBackoffMs: Long = 5000L,
) {
    companion object {
        private const val TAG = "HeartbeatManager"
        private const val HEARTBEAT_TYPE = "heartbeat"
    }

    private val handler = Handler(Looper.getMainLooper())
    private var heartbeatRunnable: Runnable? = null

    private val isRunning = AtomicBoolean(false)
    private val isConnected = AtomicBoolean(false)
    private val reconnectAttempts = AtomicInteger(0)
    private val lastHeartbeatTime = AtomicLong(0)

    var onConnectionLost: (() -> Unit)? = null
    var onConnectionRestored: (() -> Unit)? = null
    var onMaxReconnectAttemptsReached: (() -> Unit)? = null

    /**
     * Start sending periodic heartbeats.
     */
    fun startHeartbeats() {
        if (isRunning.getAndSet(true)) {
            Log.w(TAG, "Heartbeats already running")
            return
        }

        Log.i(TAG, "Starting heartbeat transmission for device: $deviceId")
        scheduleNextHeartbeat()
    }

    /**
     * Stop sending heartbeats.
     */
    fun stopHeartbeats() {
        if (!isRunning.getAndSet(false)) {
            return
        }

        Log.i(TAG, "Stopping heartbeat transmission")
        heartbeatRunnable?.let { handler.removeCallbacks(it) }
        heartbeatRunnable = null
    }

    /**
     * Manually trigger a heartbeat (useful for connection testing).
     */
    fun sendImmediateHeartbeat(metadata: Map<String, Any>? = null) {
        if (!isRunning.get()) {
            Log.w(TAG, "Cannot send heartbeat - manager not running")
            return
        }

        sendHeartbeat(metadata)
    }

    /**
     * Mark connection as lost and start reconnection process.
     */
    fun markConnectionLost() {
        if (isConnected.getAndSet(false)) {
            Log.w(TAG, "Connection lost for device: $deviceId")
            onConnectionLost?.invoke()

            scheduleReconnectAttempt()
        }
    }

    /**
     * Mark connection as restored and reset reconnection state.
     */
    fun markConnectionRestored() {
        if (!isConnected.getAndSet(true)) {
            Log.i(TAG, "Connection restored for device: $deviceId")
            reconnectAttempts.set(0)
            onConnectionRestored?.invoke()

            if (isRunning.get()) {
                scheduleNextHeartbeat()
            }
        }
    }

    /**
     * Get current connection status.
     */
    fun isConnectionHealthy(): Boolean = isConnected.get()

    /**
     * Get current reconnection attempt count.
     */
    fun getReconnectAttempts(): Int = reconnectAttempts.get()

    /**
     * Get time since last successful heartbeat (in milliseconds).
     */
    fun getTimeSinceLastHeartbeat(): Long {
        val lastTime = lastHeartbeatTime.get()
        return if (lastTime > 0) System.currentTimeMillis() - lastTime else -1
    }

    private fun scheduleNextHeartbeat() {
        if (!isRunning.get()) return

        heartbeatRunnable?.let { handler.removeCallbacks(it) }

        heartbeatRunnable =
            Runnable {
                if (isRunning.get() && isConnected.get()) {
                    sendHeartbeat()
                    scheduleNextHeartbeat()
                }
            }

        handler.postDelayed(heartbeatRunnable!!, heartbeatIntervalMs)
    }

    private fun sendHeartbeat(additionalMetadata: Map<String, Any>? = null) {
        try {
            val metadata =
                mutableMapOf<String, Any>().apply {
                    put("battery_level", getBatteryLevel())
                    put("recording_active", isRecordingActive())
                    put("storage_available_mb", getAvailableStorageMB())
                    put("uptime_ms", getUptimeMs())

                    additionalMetadata?.let { putAll(it) }
                }

            val heartbeatMessage = createHeartbeatMessage(metadata)
            val success = networkClient.sendMessage(heartbeatMessage)

            if (success) {
                lastHeartbeatTime.set(System.currentTimeMillis())
                Log.d(TAG, "Heartbeat sent successfully")
            } else {
                Log.w(TAG, "Failed to send heartbeat - connection may be lost")
                markConnectionLost()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error sending heartbeat", e)
            markConnectionLost()
        }
    }

    private fun scheduleReconnectAttempt() {
        val attempt = reconnectAttempts.incrementAndGet()

        if (attempt > maxReconnectAttempts) {
            Log.e(TAG, "Max reconnection attempts reached ($maxReconnectAttempts)")
            onMaxReconnectAttemptsReached?.invoke()
            return
        }

        Log.i(TAG, "Scheduling reconnection attempt $attempt in ${reconnectBackoffMs}ms")

        handler.postDelayed({
            if (isRunning.get() && !isConnected.get()) {
                attemptReconnection()
            }
        }, reconnectBackoffMs)
    }

    private fun attemptReconnection() {
        Log.i(TAG, "Attempting reconnection (attempt ${reconnectAttempts.get()})")

        try {
            if (networkClient.reconnect()) {
                markConnectionRestored()
            } else {
                scheduleReconnectAttempt()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Reconnection attempt failed", e)
            scheduleReconnectAttempt()
        }
    }

    private fun createHeartbeatMessage(metadata: Map<String, Any>): String {
        val message =
            JSONObject().apply {
                put("v", 1)
                put("type", HEARTBEAT_TYPE)
                put("device_id", deviceId)
                put("timestamp_ns", System.nanoTime())

                val metadataObj = JSONObject()
                metadata.forEach { (key, value) ->
                    metadataObj.put(key, value)
                }
                put("metadata", metadataObj)
            }

        return message.toString()
    }

    /**
     * Get device battery level (0-100).
     */
    private fun getBatteryLevel(): Int =
        try {
            val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as android.os.BatteryManager
            batteryManager.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get battery level", e)
            -1
        }

    /**
     * Check if recording is currently active.
     */
    private fun isRecordingActive(): Boolean =
        try {
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
            activityManager.getRunningServices(Integer.MAX_VALUE).any { serviceInfo ->
                serviceInfo.service.className.contains("RecordingService")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to check recording state", e)
            false
        }

    /**
     * Get available storage in MB.
     */
    private fun getAvailableStorageMB(): Long =
        try {
            val statsFs = android.os.StatFs(context.filesDir.path)
            val availableBytes = statsFs.availableBytes
            availableBytes / (1024 * 1024)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get storage info", e)
            -1L
        }

    /**
     * Get device uptime in milliseconds.
     */
    private fun getUptimeMs(): Long = android.os.SystemClock.elapsedRealtime()

    /**
     * Create a status summary for debugging.
     */
    fun getStatusSummary(): Map<String, Any> =
        mapOf(
            "device_id" to deviceId,
            "is_running" to isRunning.get(),
            "is_connected" to isConnected.get(),
            "reconnect_attempts" to reconnectAttempts.get(),
            "time_since_last_heartbeat_ms" to getTimeSinceLastHeartbeat(),
            "heartbeat_interval_ms" to heartbeatIntervalMs,
        )
}
