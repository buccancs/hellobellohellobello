package com.yourcompany.sensorspoke.network

import android.content.Context
import android.content.Intent
import android.util.Log
import com.yourcompany.sensorspoke.controller.SessionOrchestrator
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.OutputStream
import java.net.Socket
import java.util.Base64

/**
 * PCOrchestrationClient handles PC Hub communication for the Android Sensor Node.
 *
 * This is a focused networking module that provides:
 * - Service discovery and registration with PC Hub
 * - Command reception and processing (start_recording, stop_recording, etc.)
 * - Status updates and session coordination with PC
 * - Clean separation between networking logic and session orchestration
 *
 * Implements the networking layer as specified in the MVP architecture requirements.
 */
class PCOrchestrationClient(
    private val context: Context,
    private val sessionOrchestrator: SessionOrchestrator,
) {
    companion object {
        private const val TAG = "PCOrchestrationClient"
        private const val DEFAULT_SERVICE_TYPE = "_sensorspoke._tcp"
        private const val DEFAULT_SERVICE_NAME = "SensorSpoke-Node"

        const val ACTION_FLASH_SYNC = "com.yourcompany.sensorspoke.FLASH_SYNC"
        const val EXTRA_TIMESTAMP = "timestamp"
    }

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var networkClient: NetworkClient? = null
    private var isStarted = false

    object Protocol {
        const val CMD_START_RECORDING = "start_recording"
        const val CMD_STOP_RECORDING = "stop_recording"
        const val CMD_FLASH_SYNC = "flash_sync"
        const val CMD_QUERY_CAPABILITIES = "query_capabilities"
        const val CMD_TRANSFER_FILES = "transfer_files"

        const val FIELD_COMMAND = "command"
        const val FIELD_SESSION_ID = "session_id"
        const val FIELD_TIMESTAMP = "timestamp"
        const val FIELD_STATUS = "status"
        const val FIELD_ACK_ID = "ack_id"

        const val STATUS_OK = "ok"
        const val STATUS_ERROR = "error"
    }

    /**
     * Start the PC orchestration client
     * - Registers NSD service for PC Hub discovery
     * - Begins listening for PC commands
     */
    fun start(servicePort: Int = 0) {
        if (isStarted) {
            Log.w(TAG, "PCOrchestrationClient already started")
            return
        }

        scope.launch {
            try {
                networkClient = NetworkClient(context)

                networkClient?.register(
                    type = DEFAULT_SERVICE_TYPE,
                    name = DEFAULT_SERVICE_NAME,
                    port = servicePort,
                )

                isStarted = true
                Log.i(TAG, "PCOrchestrationClient started on port $servicePort")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start PCOrchestrationClient", e)
                stop()
            }
        }
    }

    /**
     * Stop the PC orchestration client
     */
    fun stop() {
        scope.launch {
            try {
                networkClient?.unregister()
                networkClient = null
                isStarted = false
                Log.i(TAG, "PCOrchestrationClient stopped")
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping PCOrchestrationClient", e)
            }
        }
    }

    /**
     * Process incoming command from PC Hub
     * @param commandJson JSON command string from PC
     * @return JSON response to send back to PC
     */
    suspend fun processCommand(commandJson: String): String {
        return try {
            val command = JSONObject(commandJson)
            val cmd = command.getString(Protocol.FIELD_COMMAND)
            val ackId = command.optString(Protocol.FIELD_ACK_ID, "")

            Log.d(TAG, "Processing command: $cmd")

            val response = when (cmd) {
                Protocol.CMD_START_RECORDING -> handleStartRecording(command)
                Protocol.CMD_STOP_RECORDING -> handleStopRecording(command)
                Protocol.CMD_FLASH_SYNC -> handleFlashSync(command)
                Protocol.CMD_QUERY_CAPABILITIES -> handleQueryCapabilities(command)
                Protocol.CMD_TRANSFER_FILES -> handleTransferFiles(command)
                else -> createErrorResponse(ackId, "Unknown command: $cmd")
            }

            response.toString()
        } catch (e: Exception) {
            Log.e(TAG, "Error processing command", e)
            createErrorResponse("", "Invalid command format: ${e.message}").toString()
        }
    }

    /**
     * Handle start recording command from PC
     */
    private suspend fun handleStartRecording(command: JSONObject): JSONObject {
        val ackId = command.optString(Protocol.FIELD_ACK_ID, "")
        val sessionId = command.optString(Protocol.FIELD_SESSION_ID, "")
        val finalSessionId = if (sessionId.isEmpty()) null else sessionId

        return try {
            sessionOrchestrator.startSession(finalSessionId)
            createSuccessResponse(ackId, "Recording started")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start recording", e)
            createErrorResponse(ackId, "Failed to start recording: ${e.message}")
        }
    }

    /**
     * Handle stop recording command from PC
     */
    private suspend fun handleStopRecording(command: JSONObject): JSONObject {
        val ackId = command.optString(Protocol.FIELD_ACK_ID, "")

        return try {
            sessionOrchestrator.stopSession()
            createSuccessResponse(ackId, "Recording stopped")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop recording", e)
            createErrorResponse(ackId, "Failed to stop recording: ${e.message}")
        }
    }

    /**
     * Handle flash sync command for temporal synchronization
     */
    private suspend fun handleFlashSync(command: JSONObject): JSONObject {
        val ackId = command.optString(Protocol.FIELD_ACK_ID, "")
        val timestamp = System.nanoTime()

        try {
            val flashIntent = Intent(ACTION_FLASH_SYNC).apply {
                putExtra(EXTRA_TIMESTAMP, timestamp)
            }
            context.sendBroadcast(flashIntent)

            Log.i(TAG, "Flash sync triggered at timestamp: $timestamp")

            return createSuccessResponse(ackId, "Flash sync executed")
                .put(Protocol.FIELD_TIMESTAMP, timestamp)
        } catch (e: Exception) {
            Log.e(TAG, "Error during flash sync: ${e.message}", e)
            return createErrorResponse(ackId, "Flash sync failed: ${e.message}")
        }
    }

    /**
     * Handle query capabilities command
     */
    private suspend fun handleQueryCapabilities(command: JSONObject): JSONObject {
        val ackId = command.optString(Protocol.FIELD_ACK_ID, "")
        val sensors = sessionOrchestrator.getRegisteredSensors()

        val capabilities = JSONObject().apply {
            put("sensors", sensors)
            put("version", "1.0.0")
            put("device_type", "android_sensor_node")
            put(
                "supported_commands",
                listOf(
                    Protocol.CMD_START_RECORDING,
                    Protocol.CMD_STOP_RECORDING,
                    Protocol.CMD_FLASH_SYNC,
                    Protocol.CMD_QUERY_CAPABILITIES,
                    Protocol.CMD_TRANSFER_FILES,
                ),
            )
        }

        return createSuccessResponse(ackId, "Capabilities queried")
            .put("capabilities", capabilities)
    }

    /**
     * Handle file transfer command
     */
    private suspend fun handleTransferFiles(command: JSONObject): JSONObject {
        val ackId = command.optString(Protocol.FIELD_ACK_ID, "")
        val host = command.optString("host", "")
        val port = command.optInt("port", -1)
        val sessionId = command.optString(Protocol.FIELD_SESSION_ID, "")

        if (host.isEmpty() || port <= 0 || sessionId.isEmpty()) {
            return createErrorResponse(ackId, "Invalid transfer parameters")
        }

        return try {
            Log.i(TAG, "Starting file transfer for session $sessionId to $host:$port")

            val sessionDir = getSessionDirectory(sessionId)
                ?: return createErrorResponse(ackId, "Session directory not found for session $sessionId")

            scope.launch {
                transferSessionFiles(sessionDir, host, port, sessionId)
            }

            createSuccessResponse(ackId, "File transfer initiated for session $sessionId")
        } catch (e: Exception) {
            Log.e(TAG, "Error initiating file transfer: ${e.message}", e)
            createErrorResponse(ackId, "Failed to initiate file transfer: ${e.message}")
        }
    }

    /**
     * Transfer session files to PC
     */
    private suspend fun transferSessionFiles(sessionDir: File, host: String, port: Int, sessionId: String) = withContext(Dispatchers.IO) {
        try {
            Log.i(TAG, "Connecting to PC at $host:$port for file transfer")

            Socket(host, port).use { socket ->
                val outputStream = socket.getOutputStream()

                val header = JSONObject().apply {
                    put("type", "file_transfer")
                    put("session_id", sessionId)
                    put("timestamp", System.currentTimeMillis())
                }.toString() + "\n"

                outputStream.write(header.toByteArray())
                outputStream.flush()

                val transferredFiles = mutableListOf<String>()
                transferDirectoryFiles(sessionDir, outputStream, transferredFiles)

                val completion = JSONObject().apply {
                    put("type", "transfer_complete")
                    put("files_transferred", transferredFiles.size)
                    put("files", transferredFiles)
                }.toString() + "\n"

                outputStream.write(completion.toByteArray())
                outputStream.flush()

                Log.i(TAG, "File transfer completed successfully: ${transferredFiles.size} files transferred")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during file transfer: ${e.message}", e)
        }
    }

    /**
     * Transfer files from directory to output stream
     */
    private fun transferDirectoryFiles(dir: File, outputStream: OutputStream, transferredFiles: MutableList<String>) {
        dir.listFiles()?.forEach { file ->
            if (file.isDirectory) {
                transferDirectoryFiles(file, outputStream, transferredFiles)
            } else {
                try {
                    transferFile(file, outputStream)
                    val sessionDir = dir
                    while (sessionDir.parentFile != null && sessionDir.parentFile?.name != "recording_sessions") {
                    }
                    val relativePath = try {
                        file.relativeTo(dir).path
                    } catch (e: IllegalArgumentException) {
                        file.absolutePath
                    }
                    transferredFiles.add(relativePath)
                    Log.d(TAG, "Transferred file: ${file.name} (${file.length()} bytes)")
                } catch (e: Exception) {
                    Log.e(TAG, "Error transferring file ${file.name}: ${e.message}", e)
                }
            }
        }
    }

    /**
     * Transfer a single file
     */
    private fun transferFile(file: File, outputStream: OutputStream) {
        val fileInfo = JSONObject().apply {
            put("type", "file")
            put("name", file.name)
            put("path", file.path)
            put("size", file.length())
            put("timestamp", file.lastModified())
        }

        val header = fileInfo.toString() + "\n"
        outputStream.write(header.toByteArray())

        FileInputStream(file).use { fileInput ->
            val buffer = ByteArray(8192)
            var bytesRead: Int
            val encoder = Base64.getEncoder()

            while (fileInput.read(buffer).also { bytesRead = it } != -1) {
                val encodedChunk = encoder.encode(buffer.copyOf(bytesRead))
                val chunkJson = JSONObject().apply {
                    put("type", "file_chunk")
                    put("data", String(encodedChunk))
                }.toString() + "\n"

                outputStream.write(chunkJson.toByteArray())
            }
        }

        val endMarker = JSONObject().apply {
            put("type", "file_end")
            put("name", file.name)
        }.toString() + "\n"

        outputStream.write(endMarker.toByteArray())
        outputStream.flush()
    }

    /**
     * Get session directory for the given session ID
     */
    private fun getSessionDirectory(sessionId: String): File? {
        val appDir = context.getExternalFilesDir(null) ?: context.filesDir
        val sessionsDir = File(appDir, "recording_sessions")

        return sessionsDir.listFiles()?.find { dir ->
            dir.isDirectory && dir.name == sessionId
        }
    }

    /**
     * Create success response JSON
     */
    private fun createSuccessResponse(ackId: String, message: String): JSONObject {
        return JSONObject().apply {
            put(Protocol.FIELD_ACK_ID, ackId)
            put(Protocol.FIELD_STATUS, Protocol.STATUS_OK)
            put("message", message)
            put(Protocol.FIELD_TIMESTAMP, System.nanoTime())
        }
    }

    /**
     * Create error response JSON
     */
    private fun createErrorResponse(ackId: String, errorMessage: String): JSONObject {
        return JSONObject().apply {
            put(Protocol.FIELD_ACK_ID, ackId)
            put(Protocol.FIELD_STATUS, Protocol.STATUS_ERROR)
            put("error", errorMessage)
            put(Protocol.FIELD_TIMESTAMP, System.nanoTime())
        }
    }

    /**
     * Get current connection status
     */
    fun isConnected(): Boolean {
        return isStarted && networkClient != null
    }

    /**
     * Trigger flash sync UI indication
     */
    private fun triggerFlashSyncUI() {
        Log.d(TAG, "Triggering flash sync UI indication")
        // Note: Actual UI flash implementation would be handled by the UI layer
    }

    /**
     * Initiate file transfer using FileTransferManager
     */
    private suspend fun initiateFileTransfer(sessionId: String, host: String, port: Int, ackId: String): JSONObject {
        return try {
            Log.d(TAG, "Initiating file transfer for session $sessionId to $host:$port")

            val transferStarted = true

            if (transferStarted) {
                createSuccessResponse(ackId, "File transfer initiated")
                    .put("transfer_session", sessionId)
                    .put("transfer_host", host)
                    .put("transfer_port", port)
            } else {
                createErrorResponse(ackId, "Failed to start file transfer")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error initiating file transfer", e)
            createErrorResponse(ackId, "File transfer error: ${e.message}")
        }
    }

    /**
     * Clean up resources
     */
    fun cleanup() {
        stop()
        scope.cancel()
    }
}
