package com.yourcompany.sensorspoke.network

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.nio.charset.StandardCharsets

/**
 * Phase 3: Time Synchronization Service for Multi-Modal Data Alignment
 *
 * Implements NTP-like protocol over UDP for high-precision timestamp coordination
 * between PC Hub and Android Spoke. Essential for synchronized multi-modal recording
 * with scientific accuracy requirements.
 */
class TimeSyncService(
    private val context: Context,
) {
    companion object {
        private const val TAG = "TimeSyncService"
        private const val DEFAULT_TIME_SERVER_PORT = 8081
        private const val SYNC_REQUEST_TIMEOUT_MS = 5000L
        private const val SYNC_INTERVAL_MS = 30000L
        private const val MAX_SYNC_ATTEMPTS = 3
        private const val ACCEPTABLE_SYNC_ACCURACY_MS = 5L
    }

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var syncJob: Job? = null
    private var isActive = false

    private var clockOffsetNs: Long = 0L
    private var lastSyncTimestamp: Long = 0L
    private var syncAccuracyMs: Double = Double.MAX_VALUE
    private var pcHubAddress: InetAddress? = null
    private var timeServerPort: Int = DEFAULT_TIME_SERVER_PORT

    private var successfulSyncs: Int = 0
    private var failedSyncs: Int = 0
    private var averageRoundTripMs: Double = 0.0

    /**
     * Start time synchronization service with PC Hub
     */
    fun startSync(hubAddress: InetAddress, port: Int = DEFAULT_TIME_SERVER_PORT) {
        if (isActive) {
            Log.w(TAG, "Time sync already active")
            return
        }

        pcHubAddress = hubAddress
        timeServerPort = port
        isActive = true

        Log.i(TAG, "Starting time synchronization with PC Hub at $hubAddress:$port")

        syncJob = scope.launch {
            while (isActive) {
                try {
                    performTimeSync()
                    delay(SYNC_INTERVAL_MS)
                } catch (e: Exception) {
                    Log.e(TAG, "Time sync error: ${e.message}", e)
                    failedSyncs++
                    delay(SYNC_INTERVAL_MS / 2)
                }
            }
        }
    }

    /**
     * Stop time synchronization service
     */
    fun stopSync() {
        isActive = false
        syncJob?.cancel()
        syncJob = null
        Log.i(TAG, "Time synchronization stopped")
    }

    /**
     * Get current synchronized timestamp in nanoseconds
     */
    fun getSyncedTimestampNs(): Long {
        return System.nanoTime() + clockOffsetNs
    }

    /**
     * Get current synchronized timestamp in milliseconds
     */
    fun getSyncedTimestampMs(): Long {
        return getSyncedTimestampNs() / 1_000_000L
    }

    /**
     * Check if time synchronization is accurate enough for recording
     */
    fun isSyncAccurate(): Boolean {
        return syncAccuracyMs <= ACCEPTABLE_SYNC_ACCURACY_MS &&
            (System.currentTimeMillis() - lastSyncTimestamp) < SYNC_INTERVAL_MS * 2
    }

    /**
     * Get synchronization statistics for monitoring
     */
    fun getSyncStats(): TimeSyncStats {
        return TimeSyncStats(
            isActive = isActive,
            clockOffsetNs = clockOffsetNs,
            syncAccuracyMs = syncAccuracyMs,
            lastSyncTimestamp = lastSyncTimestamp,
            successfulSyncs = successfulSyncs,
            failedSyncs = failedSyncs,
            averageRoundTripMs = averageRoundTripMs,
        )
    }

    /**
     * Perform NTP-like time synchronization with PC Hub
     */
    private suspend fun performTimeSync() = withContext(Dispatchers.IO) {
        val hubAddr = pcHubAddress ?: return@withContext

        var bestOffset: Long = 0L
        var bestRoundTrip: Long = Long.MAX_VALUE
        var successCount = 0

        repeat(MAX_SYNC_ATTEMPTS) { attempt ->
            try {
                val result = performSingleSync(hubAddr)
                if (result != null) {
                    if (result.roundTripTimeNs < bestRoundTrip) {
                        bestOffset = result.clockOffsetNs
                        bestRoundTrip = result.roundTripTimeNs
                    }
                    successCount++
                }

                if (attempt < MAX_SYNC_ATTEMPTS - 1) {
                    delay(100)
                }
            } catch (e: Exception) {
                Log.w(TAG, "Sync attempt ${attempt + 1} failed: ${e.message}")
            }
        }

        if (successCount > 0) {
            clockOffsetNs = bestOffset
            syncAccuracyMs = bestRoundTrip / 2_000_000.0
            lastSyncTimestamp = System.currentTimeMillis()
            successfulSyncs++
            averageRoundTripMs = (averageRoundTripMs * (successfulSyncs - 1) + bestRoundTrip / 1_000_000.0) / successfulSyncs

            Log.i(TAG, "Time sync successful - Offset: ${clockOffsetNs}ns, Accuracy: ${syncAccuracyMs}ms")
        } else {
            failedSyncs++
            Log.e(TAG, "All sync attempts failed")
        }
    }

    /**
     * Perform single time synchronization exchange
     */
    private suspend fun performSingleSync(hubAddress: InetAddress): TimeSyncResult? = withContext(Dispatchers.IO) {
        var socket: DatagramSocket? = null

        try {
            socket = DatagramSocket()
            socket.soTimeout = SYNC_REQUEST_TIMEOUT_MS.toInt()

            val t1 = System.nanoTime()

            val request = JSONObject().apply {
                put("type", "TIME_SYNC_REQUEST")
                put("client_timestamp", t1)
                put("request_id", System.nanoTime().toString())
            }

            val requestData = request.toString().toByteArray(StandardCharsets.UTF_8)
            val requestPacket = DatagramPacket(requestData, requestData.size, hubAddress, timeServerPort)

            socket.send(requestPacket)

            val responseBuffer = ByteArray(1024)
            val responsePacket = DatagramPacket(responseBuffer, responseBuffer.size)
            socket.receive(responsePacket)

            val t4 = System.nanoTime()

            val responseStr = String(responsePacket.data, 0, responsePacket.length, StandardCharsets.UTF_8)
            val response = JSONObject(responseStr)

            if (response.getString("type") == "TIME_SYNC_RESPONSE") {
                val t2 = response.getLong("server_receive_timestamp")
                val t3 = response.getLong("server_send_timestamp")

                val clockOffset = ((t2 - t1) + (t3 - t4)) / 2
                val roundTripTime = (t4 - t1) - (t3 - t2)

                return@withContext TimeSyncResult(clockOffset, roundTripTime)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Single sync failed: ${e.message}")
        } finally {
            socket?.close()
        }

        return@withContext null
    }

    /**
     * Data class for time sync results
     */
    private data class TimeSyncResult(
        val clockOffsetNs: Long,
        val roundTripTimeNs: Long,
    )

    /**
     * Data class for synchronization statistics
     */
    data class TimeSyncStats(
        val isActive: Boolean,
        val clockOffsetNs: Long,
        val syncAccuracyMs: Double,
        val lastSyncTimestamp: Long,
        val successfulSyncs: Int,
        val failedSyncs: Int,
        val averageRoundTripMs: Double,
    ) {
        fun getSyncQuality(): String = when {
            !isActive -> "Inactive"
            syncAccuracyMs <= 1.0 -> "Excellent (≤1ms)"
            syncAccuracyMs <= 5.0 -> "Good (≤5ms)"
            syncAccuracyMs <= 10.0 -> "Fair (≤10ms)"
            else -> "Poor (>${syncAccuracyMs.toInt()}ms)"
        }
    }

    fun cleanup() {
        stopSync()
        scope.cancel()
    }
}
