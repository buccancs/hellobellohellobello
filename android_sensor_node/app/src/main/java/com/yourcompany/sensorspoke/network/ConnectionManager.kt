package com.yourcompany.sensorspoke.network

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.InetAddress
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Phase 3: Advanced Connection Manager with Automatic Recovery
 *
 * Manages robust network connections with the PC Hub including:
 * - Automatic reconnection with exponential backoff
 * - Connection health monitoring
 * - Session state persistence across disconnections
 * - Enhanced error handling and recovery
 */
class ConnectionManager(
    private val context: Context,
    private val networkClient: NetworkClient,
) {
    companion object {
        private const val TAG = "ConnectionManager"
        private const val HEARTBEAT_INTERVAL_MS = 3000L
        private const val CONNECTION_TIMEOUT_MS = 10000L
        private const val MAX_RECONNECT_ATTEMPTS = 10
        private const val INITIAL_BACKOFF_MS = 1000L
        private const val MAX_BACKOFF_MS = 30000L
    }

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var heartbeatJob: Job? = null
    private var reconnectJob: Job? = null

    private val isConnected = AtomicBoolean(false)
    private val isReconnecting = AtomicBoolean(false)
    private val reconnectAttempts = AtomicInteger(0)
    private val lastHeartbeatTime = AtomicLong(0)

    private var hubAddress: InetAddress? = null
    private var hubPort: Int = 0

    private var currentSessionId: String? = null
    private var wasRecording: Boolean = false

    var onConnectionEstablished: ((InetAddress, Int) -> Unit)? = null
    var onConnectionLost: (() -> Unit)? = null
    var onConnectionRestored: (() -> Unit)? = null
    var onReconnectFailed: (() -> Unit)? = null

    /**
     * Establish connection with PC Hub
     */
    fun connectToHub(address: InetAddress, port: Int) {
        hubAddress = address
        hubPort = port

        Log.i(TAG, "Connecting to PC Hub at $address:$port")

        scope.launch {
            try {
                networkClient.connect(address.hostAddress, port)
                onConnectionSuccessful()
            } catch (e: Exception) {
                Log.e(TAG, "Initial connection failed: ${e.message}", e)
                startReconnection()
            }
        }
    }

    /**
     * Handle successful connection
     */
    private fun onConnectionSuccessful() {
        isConnected.set(true)
        isReconnecting.set(false)
        reconnectAttempts.set(0)

        startHeartbeat()

        hubAddress?.let { addr ->
            onConnectionEstablished?.invoke(addr, hubPort)
        }

        if (reconnectAttempts.get() > 0) {
            onConnectionRestored?.invoke()

            currentSessionId?.let { sessionId ->
                if (wasRecording) {
                    scope.launch {
                        sendRejoinMessage(sessionId)
                    }
                }
            }
        }

        Log.i(TAG, "Connection established with PC Hub")
    }

    /**
     * Handle connection loss
     */
    fun onConnectionLost() {
        if (!isConnected.get()) return

        isConnected.set(false)
        stopHeartbeat()

        Log.w(TAG, "Connection to PC Hub lost")
        onConnectionLost?.invoke()

        startReconnection()
    }

    /**
     * Start automatic reconnection with exponential backoff
     */
    private fun startReconnection() {
        if (isReconnecting.get()) return

        isReconnecting.set(true)

        reconnectJob = scope.launch {
            val hubAddr = hubAddress
            if (hubAddr == null) {
                Log.e(TAG, "Cannot reconnect - hub address not set")
                return@launch
            }

            while (isReconnecting.get() && reconnectAttempts.get() < MAX_RECONNECT_ATTEMPTS) {
                val attempt = reconnectAttempts.incrementAndGet()
                val backoffMs = calculateBackoff(attempt)

                Log.i(TAG, "Reconnection attempt $attempt/$MAX_RECONNECT_ATTEMPTS in ${backoffMs}ms")
                delay(backoffMs)

                if (!isReconnecting.get()) break

                try {
                    networkClient.connect(hubAddr.hostAddress, hubPort)
                    onConnectionSuccessful()
                    return@launch
                } catch (e: Exception) {
                    Log.w(TAG, "Reconnection attempt $attempt failed: ${e.message}")
                }
            }

            if (reconnectAttempts.get() >= MAX_RECONNECT_ATTEMPTS) {
                Log.e(TAG, "Max reconnection attempts reached")
                isReconnecting.set(false)
                onReconnectFailed?.invoke()
            }
        }
    }

    /**
     * Calculate exponential backoff delay
     */
    private fun calculateBackoff(attempt: Int): Long {
        val backoff = INITIAL_BACKOFF_MS * (1L shl (attempt - 1))
        return minOf(backoff, MAX_BACKOFF_MS)
    }

    /**
     * Start heartbeat monitoring
     */
    private fun startHeartbeat() {
        stopHeartbeat()

        heartbeatJob = scope.launch {
            while (isConnected.get()) {
                try {
                    sendHeartbeat()
                    delay(HEARTBEAT_INTERVAL_MS)
                } catch (e: Exception) {
                    Log.e(TAG, "Heartbeat failed: ${e.message}", e)
                    onConnectionLost()
                    break
                }
            }
        }
    }

    /**
     * Stop heartbeat monitoring
     */
    private fun stopHeartbeat() {
        heartbeatJob?.cancel()
        heartbeatJob = null
    }

    /**
     * Send heartbeat message to PC Hub
     */
    private suspend fun sendHeartbeat() {
        val heartbeat = EnhancedProtocol.createHeartbeatMessage()

        heartbeat.put(
            "connection_stats",
            JSONObject().apply {
                put("uptime_ms", System.currentTimeMillis() - lastHeartbeatTime.get())
                put("reconnect_attempts", reconnectAttempts.get())
                put("session_active", currentSessionId != null)
            },
        )

        networkClient.sendMessage(heartbeat.toString())
        lastHeartbeatTime.set(System.currentTimeMillis())
    }

    /**
     * Send rejoin message for session recovery
     */
    private suspend fun sendRejoinMessage(sessionId: String) {
        val rejoinMessage = JSONObject().apply {
            put("v", EnhancedProtocol.PROTOCOL_VERSION)
            put("type", "command")
            put("command", "session_rejoin")
            put("session_id", sessionId)
            put("was_recording", wasRecording)
            put("timestamp", System.currentTimeMillis())
            put("device_id", android.os.Build.MODEL)
        }

        try {
            networkClient.sendMessage(rejoinMessage.toString())
            Log.i(TAG, "Sent session rejoin message for session: $sessionId")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send rejoin message: ${e.message}", e)
        }
    }

    /**
     * Update session state for persistence
     */
    fun updateSessionState(sessionId: String?, isRecording: Boolean) {
        currentSessionId = sessionId
        wasRecording = isRecording

        Log.d(TAG, "Session state updated - ID: $sessionId, Recording: $isRecording")
    }

    /**
     * Get current connection status
     */
    fun getConnectionStatus(): ConnectionStatus {
        return ConnectionStatus(
            isConnected = isConnected.get(),
            isReconnecting = isReconnecting.get(),
            reconnectAttempts = reconnectAttempts.get(),
            lastHeartbeatTime = lastHeartbeatTime.get(),
            hubAddress = hubAddress?.hostAddress,
            hubPort = hubPort,
            currentSessionId = currentSessionId,
            wasRecording = wasRecording,
        )
    }

    /**
     * Force disconnection
     */
    fun disconnect() {
        isConnected.set(false)
        isReconnecting.set(false)
        stopHeartbeat()
        reconnectJob?.cancel()

        Log.i(TAG, "Disconnected from PC Hub")
    }

    /**
     * Cleanup resources
     */
    fun cleanup() {
        disconnect()
        scope.cancel()
    }

    /**
     * Data class for connection status
     */
    data class ConnectionStatus(
        val isConnected: Boolean,
        val isReconnecting: Boolean,
        val reconnectAttempts: Int,
        val lastHeartbeatTime: Long,
        val hubAddress: String?,
        val hubPort: Int,
        val currentSessionId: String?,
        val wasRecording: Boolean,
    ) {
        fun getStatusDescription(): String = when {
            isConnected -> "Connected to $hubAddress:$hubPort"
            isReconnecting -> "Reconnecting... (attempt $reconnectAttempts)"
            else -> "Disconnected"
        }
    }
}

/**
 * Extension for NetworkClient to support suspending message sending
 */
suspend fun NetworkClient.sendMessage(message: String): Boolean = withContext(Dispatchers.IO) {
    sendMessage(message)
}
