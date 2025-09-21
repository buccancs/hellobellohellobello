package com.yourcompany.sensorspoke.controller

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * CrashRecoveryManager handles detection and recovery of interrupted recording sessions.
 *
 * This manager is responsible for:
 * - Detecting unfinished sessions on app startup
 * - Marking crashed sessions appropriately
 * - Cleaning up stale resources
 * - Providing recovery status information
 */
class CrashRecoveryManager(
    private val context: Context,
    private val sessionsRootOverride: File? = null,
) {

    companion object {
        private const val TAG = "CrashRecoveryManager"
        private const val RECOVERY_MARKER_FILE = "recovery_marker.json"
    }

    /**
     * Data class for recovery results
     */
    data class RecoveryResult(
        val totalSessionsScanned: Int,
        val crashedSessionsFound: Int,
        val crashedSessionIds: List<String>,
        val recoveryErrors: List<String>,
    )

    /**
     * Perform crash recovery on app startup
     */
    suspend fun performRecovery(): RecoveryResult {
        Log.i(TAG, "Starting crash recovery process")

        val errors = mutableListOf<String>()
        val crashedSessions = mutableListOf<String>()
        var scannedCount = 0

        try {
            val sessionsRoot = getSessionsRoot()
            if (!sessionsRoot.exists()) {
                Log.i(TAG, "Sessions directory does not exist - no recovery needed")
                return RecoveryResult(0, 0, emptyList(), emptyList())
            }

            val sessionDirs = sessionsRoot.listFiles { file -> file.isDirectory } ?: emptyArray()
            scannedCount = sessionDirs.size

            Log.i(TAG, "Scanning $scannedCount session directories for crashed sessions")

            for (sessionDir in sessionDirs) {
                try {
                    val sessionId = sessionDir.name
                    val recoveryResult = recoverSession(sessionDir)

                    if (recoveryResult.wasCrashed) {
                        crashedSessions.add(sessionId)
                        Log.w(TAG, "Recovered crashed session: $sessionId")
                    }
                } catch (e: Exception) {
                    val error = "Failed to recover session ${sessionDir.name}: ${e.message}"
                    errors.add(error)
                    Log.e(TAG, error, e)
                }
            }

            createRecoveryMarker(crashedSessions.size)
        } catch (e: Exception) {
            val error = "Fatal error during crash recovery: ${e.message}"
            errors.add(error)
            Log.e(TAG, error, e)
        }

        val result = RecoveryResult(
            totalSessionsScanned = scannedCount,
            crashedSessionsFound = crashedSessions.size,
            crashedSessionIds = crashedSessions,
            recoveryErrors = errors,
        )

        Log.i(TAG, "Crash recovery completed: $result")
        return result
    }

    /**
     * Recover a single session directory
     */
    private fun recoverSession(sessionDir: File): SessionRecoveryResult {
        val metadataFile = File(sessionDir, "session_metadata.json")

        if (!metadataFile.exists()) {
            createCrashedMetadata(sessionDir, "NO_METADATA")
            return SessionRecoveryResult(wasCrashed = true, reason = "No metadata file found")
        }

        return try {
            val metadata = metadataFile.readText(Charsets.UTF_8)
            val json = JSONObject(metadata)
            val status = json.optString("session_status", "UNKNOWN")

            when (status) {
                "STARTED" -> {
                    markSessionAsCrashed(metadataFile, json, "INTERRUPTED")
                    SessionRecoveryResult(wasCrashed = true, reason = "Session was interrupted")
                }
                "COMPLETED", "COMPLETED_WITH_ERRORS", "CRASHED" -> {
                    SessionRecoveryResult(wasCrashed = false, reason = "Session was properly finished")
                }
                else -> {
                    markSessionAsCrashed(metadataFile, json, "UNKNOWN_STATUS")
                    SessionRecoveryResult(wasCrashed = true, reason = "Unknown session status: $status")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error reading session metadata for ${sessionDir.name}: ${e.message}", e)
            SessionRecoveryResult(wasCrashed = false, reason = "Error reading metadata: ${e.message}")
        }
    }

    /**
     * Mark a session as crashed in its metadata
     */
    private fun markSessionAsCrashed(metadataFile: File, originalMetadata: JSONObject, reason: String) {
        try {
            val crashTimestamp = System.currentTimeMillis()
            val dateFormatter = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", java.util.Locale.US)

            val updatedMetadata = originalMetadata
                .put("session_status", "CRASHED")
                .put("crash_detected_at", crashTimestamp)
                .put("crash_reason", reason)
                .put("recovery_timestamp", dateFormatter.format(java.util.Date(crashTimestamp)))
                .put("recovery_note", "Session was interrupted and recovered on app restart")

            metadataFile.writeText(updatedMetadata.toString(2), Charsets.UTF_8)

            Log.i(TAG, "Marked session as crashed: ${metadataFile.parentFile?.name}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to mark session as crashed: ${e.message}", e)
        }
    }

    /**
     * Create metadata for sessions that have no metadata file
     */
    private fun createCrashedMetadata(sessionDir: File, reason: String) {
        try {
            val metadataFile = File(sessionDir, "session_metadata.json")
            val crashTimestamp = System.currentTimeMillis()
            val dateFormatter = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", java.util.Locale.US)

            val metadata = JSONObject()
                .put("session_id", sessionDir.name)
                .put("session_status", "CRASHED")
                .put("crash_detected_at", crashTimestamp)
                .put("crash_reason", reason)
                .put("recovery_timestamp", dateFormatter.format(java.util.Date(crashTimestamp)))
                .put("recovery_note", "Session directory found without proper metadata - marked as crashed during recovery")

            metadataFile.writeText(metadata.toString(2), Charsets.UTF_8)

            Log.i(TAG, "Created crashed metadata for session: ${sessionDir.name}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create crashed metadata: ${e.message}", e)
        }
    }

    /**
     * Create a recovery marker file to track recovery operations
     */
    private fun createRecoveryMarker(crashedSessionCount: Int) {
        try {
            val sessionsRoot = getSessionsRoot()
            val markerFile = File(sessionsRoot, RECOVERY_MARKER_FILE)

            val marker = JSONObject()
                .put("last_recovery_timestamp", System.currentTimeMillis())
                .put("crashed_sessions_recovered", crashedSessionCount)
                .put("recovery_version", "1.0")

            markerFile.writeText(marker.toString(2), Charsets.UTF_8)

            Log.d(TAG, "Created recovery marker file")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create recovery marker: ${e.message}", e)
        }
    }

    /**
     * Get the sessions root directory
     */
    private fun getSessionsRoot(): File {
        return sessionsRootOverride ?: run {
            val base = context.getExternalFilesDir(null) ?: context.filesDir
            File(base, "sessions")
        }
    }

    /**
     * Get information about the last recovery operation
     */
    fun getLastRecoveryInfo(): RecoveryInfo? {
        return try {
            val sessionsRoot = getSessionsRoot()
            val markerFile = File(sessionsRoot, RECOVERY_MARKER_FILE)

            if (markerFile.exists()) {
                val marker = JSONObject(markerFile.readText(Charsets.UTF_8))
                RecoveryInfo(
                    lastRecoveryTimestamp = marker.getLong("last_recovery_timestamp"),
                    crashedSessionsRecovered = marker.getInt("crashed_sessions_recovered"),
                )
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error reading recovery info: ${e.message}", e)
            null
        }
    }

    /**
     * Clean up old crashed sessions (older than specified days)
     */
    suspend fun cleanupOldCrashedSessions(olderThanDays: Int = 30): Int {
        Log.i(TAG, "Cleaning up crashed sessions older than $olderThanDays days")

        val cutoffTime = System.currentTimeMillis() - (olderThanDays * 24 * 60 * 60 * 1000L)
        var cleanedCount = 0

        try {
            val sessionsRoot = getSessionsRoot()
            val sessionDirs = sessionsRoot.listFiles { file -> file.isDirectory } ?: return 0

            for (sessionDir in sessionDirs) {
                try {
                    val metadataFile = File(sessionDir, "session_metadata.json")
                    if (metadataFile.exists()) {
                        val metadata = JSONObject(metadataFile.readText(Charsets.UTF_8))
                        val status = metadata.optString("session_status", "")
                        val crashTime = metadata.optLong("crash_detected_at", 0L)

                        if (status == "CRASHED" && crashTime > 0 && crashTime < cutoffTime) {
                            if (sessionDir.deleteRecursively()) {
                                cleanedCount++
                                Log.d(TAG, "Cleaned up old crashed session: ${sessionDir.name}")
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error cleaning session ${sessionDir.name}: ${e.message}", e)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup: ${e.message}", e)
        }

        Log.i(TAG, "Cleaned up $cleanedCount old crashed sessions")
        return cleanedCount
    }

    /**
     * Result of recovering a single session
     */
    private data class SessionRecoveryResult(
        val wasCrashed: Boolean,
        val reason: String,
    )

    /**
     * Information about the last recovery operation
     */
    data class RecoveryInfo(
        val lastRecoveryTimestamp: Long,
        val crashedSessionsRecovered: Int,
    )
}
